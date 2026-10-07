"""Collect one closed K normal dataset with verified local raw ACK and complete installed GT."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tarfile

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command

COUNTS = {'RGBNT201':(3951,836,836,2647),'MSVR310':(1032,591,1055,705),'RGBNT100':(8675,1715,8575,6357)}
REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=tuple(COUNTS),required=True)
    dataset = parser.parse_args().dataset
    train,query,gallery,steps = COUNTS[dataset]
    pf = PROJECT/'results/preflight'
    load = lambda path:json.loads(path.read_text(encoding='utf-8'))
    actual = load(pf/('r201k_relation_local_pi_'+dataset+'_actual_session_20261007.json'))
    assert actual['status']=='ACTUAL_K_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW'
    assert actual['exit_code']==0 and actual['successful_updates']==2*steps and actual['additional_epochs']==100 and actual['native_updates']==6
    assert len(actual['archives'])==2
    review = load(pf/'r201k_normal_deployment_cpu_source_review_20261007.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    assert {'results/preflight/r201k_dataset_normal_complete_intake_20261007.py','analyze_r201k_dataset_normal.py'} <= set(review['sources_sha256'])
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
    for row in actual['archives'].values():
        path=Path(row['local'])
        assert path.stat().st_size==row['file']['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['file']['sha256']
    package=pf/('r201k_'+dataset+'_closed_normal_text_20261007.tar.gz')
    local=PROJECT/'results/r201k_relation_local_pi_20261007'/dataset
    proof=pf/('r201k_'+dataset+'_normal_terminal_text_intake_20261007.json')
    assert not package.exists() and not local.exists() and not proof.exists()
    remote = REMOTE+'/runs/r201k_relation_local_pi_20261007/'+dataset
    package_name = '/tmp/r201k_'+dataset+'_closed_normal_text_20261007.tar.gz'
    reuse = review['directly_reused_sources_sha256']
    code = f'''import hashlib,json,tarfile
from pathlib import Path
project=Path({REMOTE!r});root=Path({remote!r});package=Path({package_name!r})
assert all(hashlib.sha256((project/name).read_bytes()).hexdigest()==sha for name,sha in {reuse!r}.items() if not name.startswith('results/'))
controller=json.loads((root/'controller_result.json').read_text())
assert controller['status']=='COMPLETE' and controller['dataset']=={dataset!r} and controller['controls']==2
assert controller['additional_epochs']==100 and controller['successful_updates']=={2*steps!r} and controller['native_updates']==6
assert controller['paired_sampling_exact'] and controller['normal_archives_local_verified']==2
for job in controller['runs']:
 name=job['name'];run=root/'training'/name;data=json.loads((run/'result.json').read_text())
 assert data['status']=='COMPLETE' and data['epochs']==50 and data['steps']==data['optimizer_steps']=={steps!r} and data['amp_skipped_steps']==0
 assert (data['train_records'],data['query_records'],data['gallery_records'])=={(train,query,gallery)!r}
 assert data['training_coverage']==dict(eligible={train!r},visited={train!r},unvisited=[])
 assert data['training_heldout_identities']==0 and data['descriptor_dim']==5120
 assert '  NUM_INSTANCE: '+str({{'RGBNT201':8,'MSVR310':4,'RGBNT100':16}}[{dataset!r}]) in data['config'].splitlines()
 assert [path.name for path in run.glob('*.pth')]==['best.pth']
 assert json.loads((root/'training'/(name+'_exit.json')).read_text())['exit_code']==0
 audit=json.loads((run/'normal_cpu_audit.json').read_text())
 assert audit['status']=='PASS' and audit['max_metric_error']<1e-8 and audit['CMC50_and_per_query_and_groups']
 assert audit['installed_split_counts']==dict(train={train!r},query={query!r},gallery={gallery!r})
 assert json.loads((run/'normal_local_archive.json').read_text())['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
 assert not (run/'best_official_arrays.npz').exists()
files={{}};assert not package.exists()
with tarfile.open(package,'w:gz') as archive:
 for path in sorted(root.rglob('*')):
  if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log','.txt'):
   name=path.relative_to(root).as_posix();data=path.read_bytes();files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
   archive.add(path,arcname=name,recursive=False)
print(json.dumps(dict(files=files,package=dict(bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()))))'''
    inventory=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    command(['scp',*OPTIONS,'2026:'+package_name,str(package)])
    assert package.stat().st_size==inventory['package']['bytes'] and hashlib.sha256(package.read_bytes()).hexdigest()==inventory['package']['sha256']
    local.mkdir(parents=True)
    with tarfile.open(package,'r:gz') as archive:
        assert set(archive.getnames())==set(inventory['files'])
        archive.extractall(local,filter='data')
    assert all((local/name).stat().st_size==row['bytes'] and hashlib.sha256((local/name).read_bytes()).hexdigest()==row['sha256'] for name,row in inventory['files'].items())
    command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input='from pathlib import Path\nPath('+repr(package_name)+').unlink()\n')
    package.unlink()
    result=dict(status='ACTUAL_K_DATASET_TWO_FULL50_NORMAL_GT_TEXT_VERIFIED',dataset=dataset,verified_at=datetime.now().isoformat(timespec='seconds'),
        files=len(inventory['files']),manifest=inventory['files'],remote_root=remote,local_root=str(local),raw_archives_local_verified=2,new_neural_calls=0,new_optimizer_updates=0)
    proof.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('status','dataset','verified_at','files')}))


if __name__=='__main__':
    main()
