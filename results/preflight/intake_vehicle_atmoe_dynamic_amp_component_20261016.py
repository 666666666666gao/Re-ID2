"""Archive one genuinely completed component; leave the other training untouched."""
import argparse
from datetime import datetime,timezone,timedelta
from pathlib import Path
import hashlib,importlib.util,json,shlex,shutil,subprocess,tarfile,time

P=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
PF=P/'results/preflight'
T=Path('C:/Users/gb/.codex_tmp')
D=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/vehicle_atmoe_head_dynamic_amp_20261016')
spec=importlib.util.spec_from_file_location('transport',T/'publish_RGBNT100_N5_s44_closed_s45_started_20261015.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
CAPTURES=[]
DATASET=None


def command(argv,*,input=None):
    number=len(CAPTURES)+1
    prefix=T/f'vehicle_head_dynamic_amp_{DATASET}_intake_{number:02d}_20261016'
    out,err,receipt=[prefix.with_suffix(s) for s in ('.stdout.log','.stderr.log','.json')]
    assert not any(file.exists() for file in (out,err,receipt))
    started=time.time()
    receipt.write_text(json.dumps(dict(status='ATTEMPT_STARTED',started=started,stdout=str(out),stderr=str(err)),indent=2)+'\n','utf-8')
    with out.open('x',encoding='utf-8') as stdout,err.open('x',encoding='utf-8') as stderr:
        child=subprocess.run(argv,input=input,text=True,encoding='utf-8',stdout=stdout,stderr=stderr,timeout=300)
    row=dict(status='RETURNED',started=started,finished=time.time(),exit_code=child.returncode,stdout=str(out),stderr=str(err))
    receipt.write_text(json.dumps(row,indent=2)+'\n','utf-8');CAPTURES.append(row);child.check_returncode()
    return out.read_text('utf-8')


def main():
    global DATASET
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=('RGBNT100','MSVR310'),required=True)
    DATASET=parser.parse_args().dataset
    output=PF/f'vehicle_atmoe_head_dynamic_amp_{DATASET}_intake_actual_20261016.json'
    assert not output.exists()
    review=prior.load(PF/'vehicle_atmoe_head_dynamic_amp_intake_source_review_20261016.json')
    assert review['status']=='FINAL' and review['verdict']=='PASS' and not review['blocking_findings']
    request=prior.load(PF/'vehicle_atmoe_head_dynamic_amp_intake_source_request_20261016.json')
    assert review['request']==str(PF/'vehicle_atmoe_head_dynamic_amp_intake_source_request_20261016.json')
    assert all(prior.sha(Path(row['path']))==row['sha256'] for row in request['sources'])
    node=prior.load(PF/f'vehicle_atmoe_dynamic_amp_{DATASET}_completion_node_actual_20261016.json')
    assert node['dataset']==DATASET and node['actual']['component'] is not None
    assert all(not row['live'] and row['exit']['exit_code']==0 for row in node['actual']['children'])
    assert {row['phase'] for row in node['actual']['children']}=={'training','audit'}
    session=prior.load(PF/'vehicle_atmoe_head_dynamic_amp_actual_session_20261016.json')
    root=session['remote_root'];python='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
    export=f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({root!r});e=r/'experiments';dataset={DATASET!r}
c=json.loads((e/(dataset+'_component_complete.json')).read_text());assert c['dataset']==dataset
run=e/'training'/c['name'];data=json.loads((run/'result.json').read_text());audit=json.loads((run/'normal_cpu_audit.json').read_text())
assert data['status']=='COMPLETE' and data['epochs']==50 and audit['status']=='PASS'
for phase in ('training','audit'):
 launch=json.loads((e/phase/(c['name']+'_launch.json')).read_text());ended=json.loads((e/phase/(c['name']+'_exit.json')).read_text())
 assert ended['exit_code']==0 and not Path('/proc/'+str(launch['pid'])).exists()
files=[e/(dataset+'_component_complete.json'),e/'controller_launch.json',e/'native_acceptance.json']
files += [f for f in run.rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.jsonl','.log')]
for phase in ('training','audit'):
 files += [e/phase/(c['name']+suffix) for suffix in ('.log','_launch.json','_exit.json')]
manifest={{f.relative_to(e).as_posix():dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files}}
raw={{}}
for key,filename in (('raw','best_official_arrays.npz'),('best','best.pth')):
 f=run/filename;v=dict(path=str(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest());assert v==c[key];raw[key]=v
package=r/(dataset+'_selected_texts.tar.gz');assert not package.exists()
with tarfile.open(package,'w:gz') as tar:
 for name in manifest:tar.add(e/name,arcname=name,recursive=False)
print(json.dumps(dict(dataset=dataset,component=c,manifest=manifest,raw=raw,text_archive=dict(path=str(package),bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()),other_dataset_progress_reads=0,new_neural_calls=0)))'''
    exported=json.loads(command(['ssh',*prior.OPTIONS,'2026',shlex.quote(python)+' -'],input=export))
    needed=sum(row['bytes'] for row in exported['raw'].values())
    assert shutil.disk_usage(D.anchor).free>needed+100000000
    text_bytes=sum(row['bytes'] for row in exported['manifest'].values())
    assert shutil.disk_usage(P).free>text_bytes+exported['text_archive']['bytes']+100000000
    local=P/'results/vehicle_atmoe_head_dynamic_amp_20261016'/DATASET
    assert not local.exists();local.mkdir(parents=True)
    package=T/f'vehicle_head_dynamic_amp_{DATASET}_texts_20261016.tar.gz';assert not package.exists()
    command(['scp',*prior.OPTIONS,'2026:'+exported['text_archive']['path'],str(package)])
    assert package.stat().st_size==exported['text_archive']['bytes'] and prior.sha(package)==exported['text_archive']['sha256']
    with tarfile.open(package) as tar:
        assert set(tar.getnames())==set(exported['manifest'])
        for name in exported['manifest']:
            target=local/name;assert target.resolve().is_relative_to(local.resolve())
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(name).read())
    assert all((local/name).stat().st_size==row['bytes'] and prior.sha(local/name)==row['sha256'] for name,row in exported['manifest'].items())
    archives={}
    for key,row in exported['raw'].items():
        dest=D/exported['component']['name']/key/Path(row['path']).name
        assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True)
        command(['scp',*prior.OPTIONS,'2026:'+row['path'],str(dest)])
        assert dest.stat().st_size==row['bytes'] and prior.sha(dest)==row['sha256']
        archives[key]=dict(local=str(dest),remote=row['path'],bytes=row['bytes'],sha256=row['sha256'])
    # Only remove the transfer package after all local bytes are proved. Preserve both binaries for fresh result audit.
    retire=f'''import hashlib,json
from pathlib import Path
p=Path({exported['text_archive']['path']!r});assert p.stat().st_size=={exported['text_archive']['bytes']!r} and hashlib.sha256(p.read_bytes()).hexdigest()=={exported['text_archive']['sha256']!r}
assert all(Path(row['remote']).stat().st_size==row['bytes'] and hashlib.sha256(Path(row['remote']).read_bytes()).hexdigest()==row['sha256'] for row in {archives!r}.values())
p.unlink();print(json.dumps(dict(only_text_transfer_package_removed=True,remote_raw_and_best_preserved=True,new_neural_calls=0)))'''
    cleared=json.loads(command(['ssh',*prior.OPTIONS,'2026',shlex.quote(python)+' -'],input=retire))
    package.unlink()
    proof=dict(status='ACTUAL_ONE_VEHICLE_HEAD_FULL50_GT_COMPONENT_ARCHIVED',dataset=DATASET,
        at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),component=exported['component'],
        local_text_root=str(local),text_manifest=exported['manifest'],archives=archives,actual_transport_captures=CAPTURES,
        cleared=cleared,D_free_after=shutil.disk_usage(D.anchor).free,C_free_after=shutil.disk_usage(P).free,
        selected_completed_component_only=True,other_dataset_progress_reads=0,new_neural_calls=0,NN_restarts=0,
        result_audit='Fresh independent audit pending; no success or scientific claim established by byte-exact transfer.',
        goal='ACTIVE_UNMET')
    output.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(dict(status=proof['status'],dataset=DATASET,text_files=len(exported['manifest']),D_archived_bytes=needed,other_dataset_progress_reads=0)),flush=True)


if __name__=='__main__':main()
