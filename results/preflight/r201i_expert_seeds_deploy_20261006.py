"""Wait for the original three normal datasets; run only extra expert seeds43/44 once."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified

REMOTE='/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/r201i_expert_seeds_20261006'
LOCAL=PROJECT/'results/r201i_expert_seeds_20261006'
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201i_expert_seeds_20261006')


def main():
    pf=PROJECT/'results/preflight'
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    proof=pf/'r201i_expert_seeds_actual_session_20261006.json'
    assert not proof.exists() and not LOCAL.exists() and not ARCHIVE.exists()
    review=load(pf/'r201i_expert_seeds_source_review_20261006.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    expected={'identity_coordinate_expert_seeds.py','run_r201i_expert_seeds.py','launch_r201i_expert_seeds.py','analyze_r201i_expert_seeds.py','results/preflight/r201i_expert_seeds_deploy_20261006.py'}
    sources,reuse=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert set(sources)==expected
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    state=dict(status='WAIT_ORIGINAL_I_THREE_NORMAL_COMPLETE',started=datetime.now().isoformat(timespec='seconds'),receiver_pid=os.getpid(),new_training_updates=0,new_neural_calls=0,seeds=[43,44],fixed_anchor_seed=42)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(state),flush=True)
    original=pf/'r201i_primary_margin_actual_session_20261006.json'
    while not original.exists():
        time.sleep(240)
    completed=load(original)
    assert completed['status']=='ACTUAL_I_THREE_NORMAL_DATASETS_SIX_FULL50_GT_RAW_COMPLETE' and completed['exit_code']==0
    assert completed['additional_epochs']==300 and completed['successful_updates']==19418
    # Only the unfinished original RGBNT100 CPU intake remains; do not replay closed 201/MSVR.
    assert not (pf/'r201i_RGBNT100_normal_terminal_text_intake_20261006.json').exists()
    subprocess.run([sys.executable,'-X','utf8','-B','-S',str(pf/'r201i_dataset_normal_complete_intake_20261006.py'),'--dataset','RGBNT100'],cwd=PROJECT,check=True)
    subprocess.run([sys.executable,'-X','utf8','-B','-S','analyze_r201i_dataset_normal.py','--dataset','RGBNT100'],cwd=PROJECT,check=True)
    assert all(load(PROJECT/'results/r201i_primary_margin_20261006'/ds/'normal_analysis/result.json')['status']=='ACTUAL_I_DATASET_TWO_FULL50_NORMAL_CPU_READOUT' for ds in ('RGBNT201','MSVR310','RGBNT100'))
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    assert shutil.disk_usage(ARCHIVE.parent).free>500_000_000
    code=f'''import hashlib,shutil
from pathlib import Path
root=Path({REMOTE!r});assert not Path({ROOT!r}).exists()
assert shutil.disk_usage(root).free>3_500_000_000
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {reuse!r}.items() if not name.startswith('results/') and name!='analyze_r201i_dataset_normal.py')
print('ACTUAL_ORIGINAL_CLOSED_EXPERT_SEEDS_STORAGE_READY')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    neural=('identity_coordinate_expert_seeds.py','run_r201i_expert_seeds.py','launch_r201i_expert_seeds.py')
    for name in neural:
        command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+REMOTE+'/'+name])
    code=f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r});assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {dict((n,sources[n]) for n in neural)!r}.items())
print('ACTUAL_REVIEWED_EXPERT_SEED_SOURCES_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    LOCAL.mkdir(parents=True)
    outcomes={}
    for seed in (43,44):
        output=ROOT+'/s'+str(seed)
        argv=[PYTHON,'-u','launch_r201i_expert_seeds.py','--dataset','RGBNT201','--seed',str(seed),
            '--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
            '--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',output]
        process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026','cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        archives,cleared,native,finished={},set(),False,False
        state.update(status='EXTRA_SEED_ORIGINAL_CONTROLLER_STARTED',expert_seed=seed,started_current_seed=datetime.now().isoformat(timespec='seconds'))
        state.pop('new_neural_calls',None)
        proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        print('ACTUAL_EXPERT_SEED_CONTROLLER_STARTED',seed,flush=True)
        for line in process.stdout:
            print(line,end='',flush=True)
            if not line.startswith('R201I_STREAM '):continue
            message=json.loads(line[len('R201I_STREAM '):]);event=message['event']
            if event=='NATIVE_PASS':
                assert not native and message['controls']==2 and message['actual_updates']==6
                native=True
                state.update(status='ACTUAL_EXTRA_SEED_NATIVE6_PASS_FORMAL_RUNNING',native_updates_closed=6*(len(outcomes)+1))
                proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
            elif event=='NORMAL_READY':
                name=message['name']
                assert native and name.endswith('_s'+str(seed)) and name not in archives
                assert shutil.disk_usage(ARCHIVE.parent).free>message['file']['bytes']
                local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVE)
                archives[name]=dict(file=message['file'],local=local)
                process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED',name=name,file=message['file']))+'\n');process.stdin.flush()
            elif event=='NORMAL_CLEARED':
                assert message['name'] in archives and message['name'] not in cleared
                cleared.add(message['name'])
            else:
                assert event=='CONTROLLER_COMPLETE' and not finished and message['controls']==2
                finished=True;updates=message['successful_updates']
        process.stdin.close()
        assert process.wait()==0 and native and finished and set(archives)==cleared and len(archives)==2
        package=pf/('r201i_expert_s'+str(seed)+'_text_20261006.tar.gz')
        remote_package='/tmp/'+package.name
        code=f'''import hashlib,json,tarfile
from pathlib import Path
root=Path({output!r});controller=json.loads((root/'controller_result.json').read_text())
assert controller['status']=='COMPLETE' and controller['expert_seed']=={seed} and controller['anchor_seed']==42
assert controller['successful_updates']=={updates} and controller['paired_sampling_exact']
for job in controller['runs']:
 run=root/'training'/job['name'];d=json.loads((run/'result.json').read_text())
 assert d['status']=='COMPLETE' and d['epochs']==50 and d['steps']==d['optimizer_steps'] and d['amp_skipped_steps']==0
 assert d['arguments']['seed']=={seed} and d['anchor']['anchor_seed']==42
 assert (d['train_records'],d['query_records'],d['gallery_records'])==(3951,836,836)
 assert d['training_coverage']==dict(eligible=3951,visited=3951,unvisited=[])
 assert json.loads((run/'normal_cpu_audit.json').read_text())['status']=='PASS'
 assert [p.name for p in run.glob('*.pth')]==['best.pth'] and not (run/'best_official_arrays.npz').exists()
files={{}};package=Path({remote_package!r});assert not package.exists()
with tarfile.open(package,'w:gz') as tar:
 for path in sorted(root.rglob('*')):
  if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log','.txt'):
   name=path.relative_to(root).as_posix();data=path.read_bytes();files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());tar.add(path,arcname=name,recursive=False)
print(json.dumps(dict(files=files,package=dict(bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()))))'''
        inventory=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
        command(['scp',*OPTIONS,'2026:'+remote_package,str(package)])
        assert package.stat().st_size==inventory['package']['bytes'] and hashlib.sha256(package.read_bytes()).hexdigest()==inventory['package']['sha256']
        local=LOCAL/('s'+str(seed));local.mkdir()
        with tarfile.open(package,'r:gz') as tar:
            assert set(tar.getnames())==set(inventory['files']);tar.extractall(local,filter='data')
        assert all((local/n).stat().st_size==r['bytes'] and hashlib.sha256((local/n).read_bytes()).hexdigest()==r['sha256'] for n,r in inventory['files'].items())
        command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input='from pathlib import Path\nPath('+repr(remote_package)+').unlink()\n');package.unlink()
        outcomes[str(seed)]=dict(status='COMPLETE',expert_seed=seed,fixed_anchor_seed=42,additional_epochs=100,updates=updates,archives=archives,text_manifest=inventory['files'],finished=datetime.now().isoformat(timespec='seconds'))
        state.update(new_training_updates=sum(row['updates'] for row in outcomes.values()),closed_extra_seeds=list(outcomes),closed_extra_epochs=100*len(outcomes))
        proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        (pf/('r201i_expert_s'+str(seed)+'_actual_20261006.json')).write_text(json.dumps(outcomes[str(seed)],indent=2)+'\n',encoding='utf-8')
        print('ACTUAL_EXPERT_SEED_FULL50_PAIR_CLOSED',seed,flush=True)
    result=dict(status='COMPLETE',seeds=[42,43,44],seed42_reused=True,fixed_anchor_seed=42,new_epochs=200,native_updates=12,outcomes=outcomes,finished=datetime.now().isoformat(timespec='seconds'))
    (LOCAL/'controller_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,'-X','utf8','-B','-S','analyze_r201i_expert_seeds.py'],cwd=PROJECT,check=True)
    state.update(status='ACTUAL_I_THREE_EXPERT_SEEDS_NORMAL_GT_RAW_CPU_COMPLETE',result=result)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(state['status'],flush=True)


if __name__=='__main__':main()
