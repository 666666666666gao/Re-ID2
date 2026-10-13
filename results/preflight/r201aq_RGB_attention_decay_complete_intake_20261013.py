"""Collect terminal AQ normal evidence; archive raw on D before clearing exact remote duplicates."""
from datetime import datetime
from pathlib import Path
import hashlib,json,os,shlex,shutil,subprocess,sys,tarfile
PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
TAG='r201aq_RGB_attention_decay_20261013';ROOT=REMOTE+'/runs/'+TAG
PF=PROJECT/'results/preflight';LOCAL=PROJECT/'results'/TAG
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')/TAG
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
remote=lambda code:json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))


def main():
    started=load(PF/(TAG+'_actual_session.json'));owner=started['remote_owner']['pid']
    terminal=remote(f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({ROOT!r});p=r/'RGBNT100';assert not Path('/proc/{owner}').exists()
d=json.loads((p/'controller_result.json').read_text());assert d['status']=='COMPLETE' and d['successful_updates']==13126 and d['native_updates']==6
assert d['AG_training_record_order_exact'] and d['AL_training_record_order_exact'] and d['correction_budget']==.10 and d['attention_packed_QKV_decay']==0. and d['attention_decay_targets']==5 and d['physical_gpus']==[2,3] and d['states_cases']==8
for phase in ('native','training','audit','states'):
 for f in (p/phase).glob('*_launch.json'):
  launch=json.loads(f.read_text());assert not Path('/proc/'+str(launch['pid'])).exists()
  assert json.loads(f.with_name(f.name.replace('_launch.json','_exit.json')).read_text())['exit_code']==0
manifest={{str(f.relative_to(r)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in r.rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.log','.jsonl')}}
pkg=r/'text_intake.tar.gz';assert not pkg.exists()
with tarfile.open(pkg,'w:gz') as tar:
 for n in manifest:tar.add(r/n,arcname=n)
print(json.dumps(dict(controller=d,manifest=manifest,package=dict(bytes=pkg.stat().st_size,sha256=hashlib.sha256(pkg.read_bytes()).hexdigest()),all_children_and_controller_absent=True)))''')
    assert not LOCAL.exists() and not ARCHIVE.exists()
    assert shutil.disk_usage(ARCHIVE.parent).free>sum(r[key]['bytes'] for r in terminal['controller']['runs'] for key in ('raw','states_raw'))+500_000_000
    LOCAL.mkdir();ARCHIVE.mkdir();package=LOCAL/'text_intake.tar.gz'
    command(['scp',*OPTIONS,'2026:'+ROOT+'/text_intake.tar.gz',str(package)])
    assert package.stat().st_size==terminal['package']['bytes'] and sha(package)==terminal['package']['sha256']
    with tarfile.open(package) as tar:tar.extractall(LOCAL,filter='data')
    assert all((LOCAL/n).stat().st_size==r['bytes'] and sha(LOCAL/n)==r['sha256'] for n,r in terminal['manifest'].items())
    archives={}
    raw_rows=[dict(name=row['name']+'/'+key,raw=row[key]) for row in terminal['controller']['runs'] for key in ('raw','states_raw')]
    for row in raw_rows:
        name=row['name'];raw=row['raw'];folder=ARCHIVE/name;folder.mkdir(parents=True);destination=folder/('best_official_arrays.npz' if name.endswith('/raw') else 'normal_states.npz')
        subprocess.run(['scp',*OPTIONS,'2026:'+raw['path'],str(destination)],check=True,capture_output=True,text=True,encoding='utf-8',timeout=900)
        assert destination.stat().st_size==raw['bytes'] and sha(destination)==raw['sha256'] and sha(destination)==raw['sha256']
        archives[name]=dict(local=str(destination),remote=raw['path'],bytes=raw['bytes'],sha256=raw['sha256'])
    subprocess.run([sys.executable,'-X','utf8','-B','-S',str(PROJECT/'analyze_r201aq_RGB_attention_decay_normal.py')],cwd=PROJECT,check=True)
    protected=started['protected_weights']
    cleared=remote(f'''import hashlib,json,shutil
from pathlib import Path
p=Path({REMOTE!r});r=Path({ROOT!r});owned={{v['remote']:dict(bytes=v['bytes'],sha256=v['sha256']) for v in {archives!r}.values()}}
owned[str(r/'text_intake.tar.gz')]={terminal['package']!r}
assert not Path('/proc/{owner}').exists()
assert all(Path(n).resolve().is_relative_to(r.resolve()) and Path(n).stat().st_size==v['bytes'] and hashlib.sha256(Path(n).read_bytes()).hexdigest()==v['sha256'] for n,v in owned.items())
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
for n in owned:Path(n).unlink()
print(json.dumps(dict(matching_owned_duplicates_cleared=len(owned),bytes=sum(v['bytes'] for v in owned.values()),protected_bests_exact=True,remote_project_free=shutil.disk_usage(p).free)))''')
    package.unlink()
    for name,row in archives.items():
        model,key=name.split('/')
        (LOCAL/'RGBNT100'/('training' if key=='raw' else 'states')/model/'local_archive.json').write_text(json.dumps(dict(status='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED',**row),indent=2)+'\n',encoding='utf-8')
    proof=dict(status='ACTUAL_AQ_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        collector_pid=os.getpid(),terminal=terminal,archives=archives,cleared=cleared,D_free=shutil.disk_usage(ARCHIVE).free,
        formal_updates=13126,native_updates=6,state_cases=8,unique_best_deleted=0,missing_evaluated=False,new49=False,goal='ACTIVE_UNMET')
    (PF/(TAG+'_complete_intake.json')).write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    started.update(status=proof['status'],completion=str(PF/(TAG+'_complete_intake.json')),remote_controller_absent=True)
    (PF/(TAG+'_actual_session.json')).write_text(json.dumps(started,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=proof['status'],archives=archives,cleared=cleared)),flush=True)


if __name__=='__main__':main()
