"""Collect four detached AE jobs after all NN/audit/state workers have completed."""
from pathlib import Path
from datetime import datetime
import hashlib,json,os,shlex,shutil,subprocess,sys,tarfile
PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
TAG='r201ae_MS_strong_staged_20261012';ROOT=REMOTE+'/runs/'+TAG
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')/TAG


def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    started=load(pf/'r201ae_MS_strong_staged_actual_session_20261012.json')
    assert started['status']=='ACTUAL_AE_DETACHED_REMOTE_CONTROLLER_STARTED_NOT_NATIVE_OR_FORMAL_CONFIRMED'
    review=load(pf/'r201ae_MS_strong_staged_source_review_20261012.json');assert review['status']=='PASS' and not review['blocking_findings']
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
    local=PROJECT/'results'/TAG/'MSVR310';proof=pf/'r201ae_MS_strong_staged_terminal_text_intake_20261012.json'
    package=pf/(TAG+'_closed_text.tar.gz');remote_package='/tmp/'+package.name;remote_root=ROOT+'/MSVR310'
    assert not local.exists() and not proof.exists() and not package.exists() and not ARCHIVE.exists()
    code=f'''import json
from pathlib import Path
r=Path({remote_root!r});load=lambda p:json.loads(p.read_text())
owner=load(r.parent/'remote_owner.json');assert not Path('/proc/'+str(owner['pid'])).exists()
c=load(r/'controller_result.json');native=load(r/'native_acceptance.json');raw=load(r/'raw_manifest.json')
assert c['status']=='COMPLETE' and c['controls']==4 and c['successful_updates']==2820 and c['native_updates']==12
assert c['paired_sampling_exact'] and c['normal_raws_preserved']==c['state_raws_preserved']==4 and not c['local_transfer_completed']
assert native['status']=='PASS' and native['controls']==4 and native['actual_updates']==12 and native['phase_boundary_actual']
assert raw['status']=='REMOTE_RAW_PRESERVED_PENDING_LOCAL_TRANSFER' and len(raw['files'])==8
for job in c['runs']:
 name=job['name'];p=r/'training'/name;d=load(p/'result.json')
 assert d['status']=='COMPLETE' and d['epochs']==50 and d['steps']==d['optimizer_steps']==705
 assert d['amp_skipped_steps']==d['training_heldout_identities']==0
 assert (d['train_records'],d['query_records'],d['gallery_records'])==(1032,591,1055)
 assert d['training_coverage']==dict(eligible=1032,visited=1032,unvisited=[])
 assert [x.name for x in p.glob('*.pth')]==['best.pth']
 for phase in ('training','audit','states'):
  assert load(r/phase/(name+'_exit.json'))['exit_code']==0
  assert not Path('/proc/'+str(load(r/phase/(name+'_launch.json'))['pid'])).exists()
 assert load(p/'normal_cpu_audit.json')['status']=='PASS'
 assert load(r/'states'/name/'result.json')['full11_six_metrics_equal_selected_normal']
print(json.dumps(dict(controller=c,native=native,raw=raw,owner=owner)))'''
    ready=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    assert shutil.disk_usage(ARCHIVE.parent).free>sum(row['bytes'] for row in ready['raw']['files'].values())+1024**3
    archives={}
    for key,row in ready['raw']['files'].items():
        file=dict(bytes=row['bytes'],sha256=row['sha256'])
        destination=copy_verified(row['path'],file,Path(ROOT),ARCHIVE)
        archives[key]=dict(file=file,local=destination)
    assert all(Path(row['local']).stat().st_size==row['file']['bytes'] and hashlib.sha256(Path(row['local']).read_bytes()).hexdigest()==row['file']['sha256'] for row in archives.values())
    code=f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({remote_root!r});package=Path({remote_package!r});assert not package.exists()
for key,row in {ready['raw']['files']!r}.items():
 p=Path(row['path']);assert p.resolve().is_relative_to(r.resolve())
 assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 p.unlink();kind=key.rsplit('/',1)[1]
 (p.parent/(kind+'_local_archive.json')).write_text(json.dumps(dict(status='LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED',file=dict(bytes=row['bytes'],sha256=row['sha256']))))
manifest={{}}
with tarfile.open(package,'w:gz') as archive:
 for p in sorted(r.rglob('*')):
  if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.log','.txt'):
   n=p.relative_to(r).as_posix();data=p.read_bytes();manifest[n]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());archive.add(p,arcname=n,recursive=False)
 for filename in ('remote_owner.json','controller.log'):
  p=r.parent/filename;n=filename;data=p.read_bytes();manifest[n]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());archive.add(p,arcname=n,recursive=False)
print(json.dumps(dict(manifest=manifest,package=dict(bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()))))'''
    inventory=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    command(['scp',*OPTIONS,'2026:'+remote_package,str(package)])
    assert package.stat().st_size==inventory['package']['bytes'] and hashlib.sha256(package.read_bytes()).hexdigest()==inventory['package']['sha256']
    local.mkdir(parents=True)
    with tarfile.open(package,'r:gz') as archive:
        assert set(archive.getnames())==set(inventory['manifest']);archive.extractall(local,filter='data')
    assert all((local/n).stat().st_size==row['bytes'] and hashlib.sha256((local/n).read_bytes()).hexdigest()==row['sha256'] for n,row in inventory['manifest'].items())
    command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input='from pathlib import Path\nPath('+repr(remote_package)+').unlink()\n')
    package.unlink()
    record=dict(status='ACTUAL_AE_FOUR_FULL50_NORMAL_STATES_TEXT_VERIFIED',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        collector_pid=os.getpid(),local_root=str(local),remote_root=remote_root,manifest=inventory['manifest'],archives=archives,
        raw_archives_local_verified=8,all_NN_and_CPU_children_closed=True,new_neural_calls=0,new_optimizer_updates=0)
    proof.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,'-X','utf8','-B','-S',str(PROJECT/'analyze_r201ae_MS_strong_staged_normal.py')],check=True)
    started.update(status='ACTUAL_AE_FOUR_FULL50_NORMAL_GT_RAW_STATES_CPU_COMPLETE',finished=record['at'],formal_updates=2820,native_updates=12,
        local_transfer_completed=True,archives=archives,all_NN_and_CPU_children_closed=True,missing_evaluated=False,new49=False)
    (pf/'r201ae_MS_strong_staged_actual_session_20261012.json').write_text(json.dumps(started,indent=2)+'\n',encoding='utf-8')
    print(started['status'],len(record['manifest']),flush=True)


if __name__=='__main__':main()
