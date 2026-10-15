"""Future terminal-only F-local intake; no new NN or automatic retry."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

P = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
T = Path('C:/Users/gb/.codex_tmp')
PF = P/'results/preflight'
spec = importlib.util.spec_from_file_location('transport',T/'publish_RGBNT100_N5_s44_closed_s45_started_20261015.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
load,sha,command,OPTIONS = prior.load,prior.sha,prior.command,prior.OPTIONS
TAG = 'rgbnt100_F_local_grid2_AV_20261016'
REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT = REMOTE+'/runs/'+TAG
LOCAL = P/'results'/TAG
ARCHIVE = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')/TAG


def remote(code):
    return json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))


def main():
    output = PF/'rgbnt100_F_local_pair_complete_intake_20261016.json'
    assert not output.exists()
    schedule = load(PF/'rgbnt100_F_local_pair_near_end_schedule_20261016.json')
    assert time.time() >= schedule['local_receipt_node_epoch']
    session = load(PF/'rgbnt100_F_local_pair_actual_session_20261016.json')
    owner = session['remote_owner']['pid']
    assert session['remote_root'] == ROOT
    reviewed = load(PF/'rgbnt100_local_band_pair_source_review_request_20261016.json')
    seals = reviewed['sources_sha256'] | reviewed['directly_reused_sources_sha256']
    assert len(seals) == 352 and all(sha(P/n) == h for n,h in seals.items())
    terminal = remote(f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({ROOT!r});p=r/'RGBNT100';assert not Path('/proc/{owner}').exists()
d=json.loads((p/'controller_result.json').read_text())
assert d['status']=='COMPLETE' and d['parent_recipe']=='AV' and d['local_reading_grid']==2
assert d['seed']==d['teacher_seed']==42 and d['physical_gpus']==[2,3] and d['maximum_NN']==2
assert d['native_updates']==6 and d['parity_forward_record_executions']==64 and d['successful_updates']==13126
assert d['state_cases']==8 and d['paired_sampling_exact'] and d['full_training_coverage'] and d['normal_GT_audited']
assert not d['missing_evaluated'] and len(d['runs'])==2
children=[]
for phase in ('parity','native','training','audit','states'):
 for f in (p/phase).glob('*_launch.json'):
  launch=json.loads(f.read_text());assert not Path('/proc/'+str(launch['pid'])).exists()
  ended=json.loads(f.with_name(f.name.replace('_launch.json','_exit.json')).read_text());assert ended['exit_code']==0
  children.append(dict(phase=phase,name=f.name,pid=launch['pid'],exit=ended))
assert len(children)==10
manifest={{str(f.relative_to(r)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in r.rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.log','.jsonl')}}
pkg=r/'text_intake.tar.gz';assert not pkg.exists()
with tarfile.open(pkg,'w:gz') as tar:
 for n in manifest:tar.add(r/n,arcname=n)
print(json.dumps(dict(controller=d,children=children,manifest=manifest,package=dict(bytes=pkg.stat().st_size,sha256=hashlib.sha256(pkg.read_bytes()).hexdigest()),all_children_and_controller_absent=True,controller_OS_exit='UNATTESTED')))''')
    assert not LOCAL.exists() and not ARCHIVE.exists()
    required = sum(run[key]['bytes'] for run in terminal['controller']['runs'] for key in ('raw','states_raw','best'))
    assert shutil.disk_usage(ARCHIVE.parent).free > required+500_000_000
    LOCAL.mkdir();ARCHIVE.mkdir()
    package = LOCAL/'text_intake.tar.gz'
    command(['scp',*OPTIONS,'2026:'+ROOT+'/text_intake.tar.gz',str(package)])
    assert package.stat().st_size == terminal['package']['bytes'] and sha(package) == terminal['package']['sha256']
    with tarfile.open(package) as tar:
        assert set(tar.getnames()) == set(terminal['manifest'])
        tar.extractall(LOCAL,filter='data')
    assert all((LOCAL/n).stat().st_size == row['bytes'] and sha(LOCAL/n) == row['sha256'] for n,row in terminal['manifest'].items())
    archives = {}
    for run in terminal['controller']['runs']:
        for key,filename in [('raw','best_official_arrays.npz'),('states_raw','normal_states.npz'),('best','best.pth')]:
            row = run[key]
            folder = ARCHIVE/run['name']/key
            folder.mkdir(parents=True)
            destination = folder/filename
            subprocess.run(['scp',*OPTIONS,'2026:'+row['path'],str(destination)],check=True,capture_output=True,text=True,encoding='utf-8',timeout=900)
            assert destination.stat().st_size == row['bytes'] and sha(destination) == row['sha256']
            archives[run['name']+'/'+key] = dict(local=str(destination),remote=row['path'],bytes=row['bytes'],sha256=row['sha256'])
    assert len(archives) == 6
    subprocess.run([sys.executable,'-X','utf8','-B','-S',str(P/'analyze_rgbnt100_local_band_normal.py')],cwd=P,check=True)
    assert all(sha(P/n) == h for n,h in seals.items())
    protected = session['protected_weights']
    cleared = remote(f'''import hashlib,json,shutil
from pathlib import Path
p=Path({REMOTE!r});r=Path({ROOT!r});owned={{v['remote']:dict(bytes=v['bytes'],sha256=v['sha256']) for v in {archives!r}.values()}}
owned[str(r/'text_intake.tar.gz')]={terminal['package']!r}
assert not Path('/proc/{owner}').exists()
assert all(Path(n).resolve().is_relative_to(r.resolve()) and Path(n).stat().st_size==v['bytes'] and hashlib.sha256(Path(n).read_bytes()).hexdigest()==v['sha256'] for n,v in owned.items())
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
for n in owned:Path(n).unlink()
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
print(json.dumps(dict(matching_owned_duplicates_cleared=len(owned),bytes=sum(v['bytes'] for v in owned.values()),protected_bests_exact=True,remote_project_free=shutil.disk_usage(p).free)))''')
    package.unlink()
    for name,row in archives.items():
        model,key = name.split('/')
        folder = LOCAL/'RGBNT100'/('states' if key == 'states_raw' else 'training')/model
        (folder/('local_best_archive.json' if key == 'best' else 'local_archive.json')).write_text(json.dumps(dict(status='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED',**row),indent=2)+'\n','utf-8')
    proof = dict(status='ACTUAL_F_LOCAL_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE',at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),collector_pid=os.getpid(),terminal=terminal,archives=archives,cleared=cleared,D_free=shutil.disk_usage(ARCHIVE).free,
                 formal_updates=13126,native_updates=6,parity_forward_record_executions=64,state_cases=8,selected_bests_D_archived=2,unique_best_deleted=0,source_seals_unchanged=len(seals),missing_evaluated=False,new49=False,goal='ACTIVE_UNMET')
    output.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps({k:proof[k] for k in ('status','formal_updates','state_cases','D_free','cleared')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
