"""Four paired seed waves after the existing K42 all49 receiver closes."""
from datetime import datetime
import hashlib,json,os,shlex,shutil,subprocess,sys,tarfile
from pathlib import Path

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified

REMOTE='/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/r201k_expert_N5_20261007'
LOCAL=PROJECT/'results/r201k_expert_N5_20261007'
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201k_expert_N5_20261007')
SEEDS=(43,44,45,46)
VARIANTS=('frequency_shared','axis_shared')
WEIGHT_BYTES=405242089
RESERVE=1024**3+100_000_000


def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    proof=pf/'r201k_expert_N5_actual_session_20261007.json'
    assert not proof.exists() and not LOCAL.exists() and not ARCHIVE.exists()
    review=load(pf/'r201k_expert_N5_source_review_20261007.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources,reuse=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert set(sources)=={'identity_coordinate_k_expert_seeds.py','run_r201k_expert_N5.py','launch_r201k_expert_N5.py','analyze_r201k_expert_N5.py','results/preflight/r201k_expert_N5_deploy_20261007.py'}
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    plan=load(pf/'r201k_expert_N5_plan_20261007.json')
    assert plan['expert_seeds']==[42,43,44,45,46] and plan['new_training_jobs']==8 and plan['anchor_seed']==42
    completed=load(pf/'r201k_relation_local_pi_actual_session_20261007.json')
    assert completed['status']=='ACTUAL_K_THREE_NORMAL_DATASETS_SIX_FULL50_GT_RAW_COMPLETE' and completed['exit_code']==0
    assert (completed['additional_epochs'],completed['successful_updates'],completed['native_updates'])==(300,19418,18)
    missing=load(pf/'r201k_missing49_actual_session_20261007.json')
    assert missing['status']=='ACTUAL_K_ALL_DECLARED_FIXED_BEST_MISSING49_GT_RAW_CPU_COMPLETE'
    assert missing['phases']['full']['conditions']==294 and missing['phases']['full']['state_cases']==1176
    assert missing['phases']['full']['exit_code']==0 and missing['phases']['native']['exit_code']==0
    assert missing['new_optimizer_updates']==0 and missing['native_duplicate_raw_retired']==6 and missing['restored_normal_copies_cleared']
    assert load(PROJECT/'results/r201k_missing49_20261007/missing_analysis/result.json')['status']=='ACTUAL_K_THREE_DATASET_FIXED_NORMAL_BEST_ALL49_CPU_COMPLETE'
    assert all(load(PROJECT/'results/r201k_relation_local_pi_20261007'/ds/'normal_analysis/result.json')['status']=='ACTUAL_K_DATASET_TWO_FULL50_NORMAL_CPU_READOUT' for ds in ('RGBNT201','MSVR310','RGBNT100'))
    assert shutil.disk_usage(ARCHIVE.parent).free>8*35_000_000+1024**3
    code=f'''import hashlib,shutil
from pathlib import Path
root=Path({REMOTE!r});assert not Path({ROOT!r}).exists()
assert shutil.disk_usage(root).free>8*{WEIGHT_BYTES}+{RESERVE}
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {reuse!r}.items() if not n.startswith('results/'))
print('ACTUAL_K49_CLOSED_N5_EIGHT_SELECTED_WEIGHTS_CAPACITY_READY')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    neural=('identity_coordinate_k_expert_seeds.py','run_r201k_expert_N5.py','launch_r201k_expert_N5.py')
    for name in neural:command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+REMOTE+'/'+name])
    code=f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r});assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {dict((n,sources[n]) for n in neural)!r}.items())
print('ACTUAL_REVIEWED_K_N5_SOURCES_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    LOCAL.mkdir(parents=True)
    jobs=[dict(id='RGBNT201_r201k_'+v+'_s'+str(seed),seed=seed,variant=v,gpu=2 if v=='frequency_shared' else 3,status='pending',phase='not_started') for seed in SEEDS for v in VARIANTS]
    manifest=dict(project='RGBNT201_K_expert_N5',server='2026',cwd=REMOTE,python=PYTHON,seeds=list(SEEDS),fixed_anchor_seed=42,gpus=[2,3],max_parallel=2,ordered_waves=list(SEEDS),jobs=jobs,preconditions='K42 all3 normal and full49/GT/rawACK/CPU COMPLETE; per-wave paired native3 before full50',queue_implementation='Existing bidirectional paired controller/stream ACK protocol; ordered four waves with explicit local queue_state, no additional screen scheduler',oom_retry=False,automatic_retry=False,temperature_power_control=False,seed42_reused=True)
    (LOCAL/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    state=dict(status='ACTUAL_K49_CLOSED_PREDECLARED_N5_QUEUE_READY',started=datetime.now().isoformat(timespec='seconds'),receiver_pid=os.getpid(),new_training_updates=0,new_neural_calls=0,seeds=list(SEEDS),fixed_anchor_seed=42,jobs=jobs)
    def save_state():
        content=json.dumps(state,indent=2)+'\n';proof.write_text(content,encoding='utf-8');(LOCAL/'queue_state.json').write_text(content,encoding='utf-8')
    save_state();outcomes={}
    for seed in SEEDS:
        remaining=2*(len(SEEDS)-len(outcomes))
        code=f'''import shutil
from pathlib import Path
assert shutil.disk_usage(Path({REMOTE!r})).free>{remaining}*{WEIGHT_BYTES}+{RESERVE}
print('CURRENT_K_N5_REMAINING_WEIGHTS_CAPACITY_READY')'''
        print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
        output=ROOT+'/s'+str(seed)
        argv=[PYTHON,'-u','launch_r201k_expert_N5.py','--dataset','RGBNT201','--seed',str(seed),
            '--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
            '--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',output]
        process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026','cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        archives,cleared,native,finished={},set(),False,False
        state.update(status='ACTUAL_K_N5_ORIGINAL_SEED_CONTROLLER_STARTED',expert_seed=seed,started_current_seed=datetime.now().isoformat(timespec='seconds'));state.pop('new_neural_calls',None)
        for job in jobs:
            if job['seed']==seed:job.update(status='running',phase='paired_native')
        save_state();print('ACTUAL_K_N5_EXPERT_SEED_CONTROLLER_STARTED',seed,flush=True)
        with (LOCAL/('s'+str(seed)+'_receiver.log')).open('x',encoding='utf-8') as log:
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
                if not line.startswith('R201K_N5_STREAM '):continue
                message=json.loads(line[len('R201K_N5_STREAM '):]);event=message['event']
                if event=='NATIVE_PASS':
                    assert not native and message['controls']==2 and message['actual_updates']==6 and message['PI_outlet']=='shared_relation_local'
                    native=True;state.update(status='ACTUAL_K_N5_SEED_NATIVE6_PASS_FORMAL_RUNNING',native_updates_closed=6*(len(outcomes)+1))
                    for job in jobs:
                        if job['seed']==seed:job['phase']='formal50'
                    save_state()
                elif event=='NORMAL_READY':
                    name=message['name'];assert native and name.endswith('_s'+str(seed)) and name not in archives
                    assert shutil.disk_usage(ARCHIVE.parent).free>message['file']['bytes']+1024**3
                    local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVE);archives[name]=dict(file=message['file'],local=local)
                    process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED',name=name,file=message['file']))+'\n');process.stdin.flush()
                elif event=='NORMAL_CLEARED':
                    assert message['name'] in archives and message['name'] not in cleared;cleared.add(message['name'])
                else:
                    assert event=='CONTROLLER_COMPLETE' and not finished and message['controls']==2
                    finished=True;updates=message['successful_updates']
        process.stdin.close();returncode=process.wait()
        if returncode!=0:
            state.update(status='ACTUAL_K_N5_SEED_CONTROLLER_FAILED_NO_RETRY',failed_seed=seed,exit_code=returncode)
            for job in jobs:
                if job['seed']==seed:job.update(status='failed',phase='original_controller_failed')
            save_state()
        assert returncode==0 and native and finished and set(archives)==cleared and len(archives)==2
        package=pf/('r201k_N5_expert_s'+str(seed)+'_text_20261007.tar.gz');remote_package='/tmp/'+package.name
        code=f'''import hashlib,json,tarfile
from pathlib import Path
root=Path({output!r});controller=json.loads((root/'controller_result.json').read_text())
assert controller['status']=='COMPLETE' and controller['expert_seed']=={seed} and controller['anchor_seed']==42
assert controller['successful_updates']=={updates} and controller['native_updates']==6 and controller['paired_sampling_exact']
for job in controller['runs']:
 run=root/'training'/job['name'];d=json.loads((run/'result.json').read_text())
 assert d['status']=='COMPLETE' and d['epochs']==50 and d['steps']==d['optimizer_steps'] and d['amp_skipped_steps']==0
 assert d['arguments']['seed']=={seed} and d['anchor']['anchor_seed']==42 and d['anchor']['expert_seed']=={seed}
 assert d['method_revision'].startswith('R201K single PI-outlet factor') and d['anchor']['method']==d['method_revision']
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
        assert all((local/n).stat().st_size==row['bytes'] and hashlib.sha256((local/n).read_bytes()).hexdigest()==row['sha256'] for n,row in inventory['files'].items())
        command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input='from pathlib import Path\nPath('+repr(remote_package)+').unlink()\n');package.unlink()
        outcomes[str(seed)]=dict(status='COMPLETE',expert_seed=seed,fixed_anchor_seed=42,additional_epochs=100,updates=updates,archives=archives,text_manifest=inventory['files'],finished=datetime.now().isoformat(timespec='seconds'))
        for job in jobs:
            if job['seed']==seed:job.update(status='completed',phase='full50_GT_raw_text_complete')
        state.update(new_training_updates=sum(row['updates'] for row in outcomes.values()),closed_extra_seeds=list(outcomes),closed_extra_epochs=100*len(outcomes));save_state()
        (pf/('r201k_N5_expert_s'+str(seed)+'_actual_20261007.json')).write_text(json.dumps(outcomes[str(seed)],indent=2)+'\n',encoding='utf-8')
        print('ACTUAL_K_N5_EXPERT_SEED_FULL50_PAIR_CLOSED',seed,flush=True)
    assert all(job['status']=='completed' for job in jobs) and len(outcomes)==4
    result=dict(status='COMPLETE',seeds=[42,43,44,45,46],seed42_reused=True,fixed_anchor_seed=42,new_epochs=400,new_models=8,native_updates=24,new_training_updates=state['new_training_updates'],outcomes=outcomes,finished=datetime.now().isoformat(timespec='seconds'))
    (LOCAL/'controller_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,'-X','utf8','-B','-S','analyze_r201k_expert_N5.py'],cwd=PROJECT,check=True)
    assert load(LOCAL/'normal_seed_analysis/result.json')['status']=='ACTUAL_K_RGBNT201_FIVE_EXPERT_SEEDS_NORMAL_CPU_COMPLETE'
    state.update(status='ACTUAL_K_FIVE_EXPERT_SEEDS_NORMAL_GT_RAW_CPU_COMPLETE',result=result);save_state();print(state['status'],flush=True)


if __name__=='__main__':main()
