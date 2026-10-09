"""Four fixed N42 paper-six jobs after weak normal closure; retained K201 six rows reused."""
from datetime import datetime
import hashlib,json,shlex,shutil,subprocess,sys,tarfile
from pathlib import Path

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified

REMOTE='/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/r201n_paper_missing6_20261009'
LOCAL=PROJECT/'results/r201n_paper_missing6_20261009'
ARCHIVE_BASES={'frequency_shared':Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts'),'axis_shared':Path('E:/ReID2-experiment-artifacts')}
ARCHIVES={variant:base/'r201n_paper_missing6_20261009' for variant,base in ARCHIVE_BASES.items()}


def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    proof=pf/'r201n_paper_missing6_actual_session_20261009.json'
    assert not proof.exists() and not LOCAL.exists() and all(not path.exists() for path in ARCHIVES.values())
    review=load(pf/'r201n_paper_missing6_split_receivers_source_review_20261010.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources,reuse=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert set(sources)=={'evaluate_r201n_paper_missing6_stream.py','launch_r201n_paper_missing6_stream.py','analyze_r201n_paper_missing6.py','results/preflight/r201n_paper_missing6_deploy_20261009.py'}
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    original=load(pf/'r201n_bounded_identity_shift_actual_session_20261009.json')
    assert original['status']=='ACTUAL_N_TWO_WEAK_FOUR_FULL50_GT_RAW_COMPLETE_SPLIT_RECEIVERS'
    assert original['native_updates_closed']==12 and original['new_formal_updates_closed']==15126 and original['additional_epochs']==200
    assert set(original['formal_datasets_closed'])=={'MSVR310','RGBNT100'}
    assert original['original_receiver_exit_code']==1 and not original['original_controller_complete']
    recovered=load(PROJECT/original['MSVR_recovered_receipt'])
    assert recovered['status']=='ACTUAL_N_ORIGINAL_MSVR_POST_TRANSPORT_GT_RAW_TEXT_CPU_COMPLETE'
    msvr=recovered['collection']
    assert msvr['successful_updates']==2000 and msvr['native_updates']==6 and msvr['additional_epochs']==100
    assert msvr['original_receiver_exit_code']==1 and not msvr['original_controller_complete'] and msvr['paired_sampling_exact']
    rgb100=load(PROJECT/original['RGBNT100_receipt'])
    assert rgb100['status']=='ACTUAL_N_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW' and rgb100['exit_code']==0
    assert rgb100['dataset']=='RGBNT100' and rgb100['successful_updates']==13126 and rgb100['native_updates']==6 and rgb100['additional_epochs']==100
    assert len(msvr['archives'])==len(rgb100['archives'])==2
    normal_datasets={'MSVR310':msvr,'RGBNT100':rgb100}
    protocol=load(pf/'demo_missing_protocol_primary_six_20261009.json')
    assert protocol['new_missing_evaluation_scope'].startswith('Paper six only')
    assert hashlib.sha256((PROJECT/protocol['source_csv']).read_bytes()).hexdigest()==protocol['source_csv_sha256']
    replay=load(pf/'r201l_uniform_K8_sampler_cpu_replay_20261007.json')
    assert replay['status']=='ACTUAL_EXISTING_SAMPLER_CPU_FULL50_OLD_AND_K8_REPLAY_COMPLETE'
    state=dict(status='N_WEAK_NORMAL_TERMINAL_PAPER6_SOURCE_CHECKED_NO_NN_STARTED',started=datetime.now().isoformat(timespec='seconds'),new_neural_calls=0,new_optimizer_updates=0,
        normal_collection_scope='Two actual receivers: MSVR original interrupted and independently recovered; RGBNT100 continuation complete',
        original_MSVR_receiver_exit_code=1,original_MSVR_controller_complete=False)
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    jobs=[]
    def add(dataset,variant):
        name=dataset+'_r201n_'+variant+'_s42'
        folder=PROJECT/'results/r201n_bounded_identity_shift_20261009'/dataset/'training'/name;data=load(folder/'result.json')
        assert data['status']=='COMPLETE' and data['epochs']==50 and data['optimizer_steps']==data['steps'] and data['amp_skipped_steps']==0
        assert data['arguments']['seed']==42 and data['arguments']['dataset']==dataset
        assert data['optimizer_steps']==replay['datasets'][dataset]['8']['total_steps'] and '  NUM_INSTANCE: 8' in data['config'].splitlines()
        assert data['method_revision'].startswith('R201N one-factor final unit5120 shift bound') and data['anchor']['method']==data['method_revision']
        assert data['anchor']['final_unit_descriptor_bound']['epsilon']==.10 and data['anchor']['final_unit_descriptor_bound']['parameter_change']==0
        assert data['training_heldout_identities']==0 and data['descriptor_dim']==5120
        assert load(folder/'normal_cpu_audit.json')['status']=='PASS'
        raw=normal_datasets[dataset]['archives'][name];local=Path(raw['local'])
        assert local.resolve().is_relative_to(ARCHIVE_BASES['frequency_shared'].resolve())
        assert local.stat().st_size==raw['file']['bytes'] and hashlib.sha256(local.read_bytes()).hexdigest()==raw['file']['sha256']
        assert data['arguments']['output'].startswith(REMOTE+'/runs/r201n_bounded_identity_shift_20261009/')
        jobs.append(dict(name=name,dataset=dataset,variant=variant,seed=42,gpu={'frequency_shared':2,'axis_shared':3}[variant],run=data['arguments']['output'],normal_local=str(local),normal_file=raw['file'],selected_epoch=data['best']['epoch'],full_metrics=data['full_metrics'],selection_scope='Nweak42 normal-best, epsilon.10 final bound on unchanged M margin.6; K201margin.3 retained',forward_graph='relation_local_shared_PI'))
    for dataset in ('MSVR310','RGBNT100'):
        assert load(PROJECT/'results/r201n_bounded_identity_shift_20261009'/dataset/'normal_analysis/result.json')['status']=='ACTUAL_N_DATASET_TWO_FULL50_NORMAL_CPU_READOUT'
        for variant in ('frequency_shared','axis_shared'):add(dataset,variant)
    controls=len(jobs);assert controls==4 and len({j['name'] for j in jobs})==4
    capacity={}
    for variant,base in ARCHIVE_BASES.items():
        free=shutil.disk_usage(base).free;required=4*1024**3
        assert free>required
        capacity[variant]=dict(root=str(ARCHIVES[variant]),free=free,required=required,
            limits='4GiB exceeds seven retained four-state distance sets per dataset/lane plus2GiB reserve; every actual file size still gated')
    LOCAL.mkdir(parents=True)
    registry=dict(jobs=jobs,reused_RGBNT201_paper6_csv=protocol['output_csv'],
        selected_at=datetime.now().isoformat(timespec='seconds'),missing_reselection=False,
        source_graph='relation_local_shared_PI',explicit_archive_roots={v:str(path) for v,path in ARCHIVES.items()},capacity=capacity,
        limits='Only paper-six same source masks, K201selected42 preserved; new Nweak42 epsilon.10 on fixed .6 recipe. No missing epoch/seed selection; old201 results reused exactly, not new N201.')
    registry_path=LOCAL/'jobs.json';registry_path.write_text(json.dumps(registry,indent=2)+'\n',encoding='utf-8')
    variants={job['name']:job['variant'] for job in jobs}
    restores={j['run']+'/best_official_arrays_restore_N_paper6.npz':j['normal_file'] for j in jobs}
    neural=('evaluate_r201n_paper_missing6_stream.py','launch_r201n_paper_missing6_stream.py')
    code=f'''import hashlib,shutil
from pathlib import Path
root=Path({REMOTE!r});assert not Path({ROOT!r}).exists()
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {reuse!r}.items() if not n.startswith('results/'))
assert all(not Path(n).exists() for n in {restores!r})
assert shutil.disk_usage(root).free>sum(f['bytes'] for f in {restores!r}.values())+1_600_000_000
print('ACTUAL_N_WEAK_NORMAL_CLOSED_PAPER6_CAPACITY_READY')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    for job in jobs:
        command(['scp',*OPTIONS,job['normal_local'],'2026:'+job['run']+'/best_official_arrays_restore_N_paper6.npz'])
    for name in neural:
        command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+REMOTE+'/'+name])
    remote_registry=REMOTE+'/runs/r201n_paper_missing6_jobs_20261009.json'
    command(['scp',*OPTIONS,str(registry_path),'2026:'+remote_registry])
    code=f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r});assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {dict((n,sources[n]) for n in neural)!r}.items())
assert all(Path(n).stat().st_size==f['bytes'] and hashlib.sha256(Path(n).read_bytes()).hexdigest()==f['sha256'] for n,f in {restores!r}.items())
print('ACTUAL_REVIEWED_MISSING_SOURCES_AND_FIXED_NORMAL_RESTORES_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    phases={}
    for mode in ('native','full'):
        state.update(status='ACTUAL_N_FIXED_SELECTED_PAPER6_'+mode.upper()+'_STARTED',controls=controls,registry=str(registry_path));state.pop('new_neural_calls',None)
        proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        argv=[PYTHON,'-u','launch_r201n_paper_missing6_stream.py','--jobs',remote_registry,'--output',ROOT,'--mode',mode]
        process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026','cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        raw,cleared,finished={},set(),False
        for line in process.stdout:
            if not line.startswith('PAPER_MISSING6_STREAM '):print(line,end='',flush=True);continue
            message=json.loads(line[len('PAPER_MISSING6_STREAM '):]);event=message['event']
            if event=='RAW_READY':
                key=message['job']+'/'+message['condition'];assert message['installed_gt_cases']==4 and key not in raw
                variant=variants[message['job']]
                assert shutil.disk_usage(ARCHIVE_BASES[variant]).free>message['file']['bytes']+1024**3
                local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVES[variant])
                raw[key]=dict(file=message['file'],local=local,remote=message['path'])
                process.stdin.write(json.dumps(dict(event='ARCHIVED',job=message['job'],condition=message['condition'],file=message['file']))+'\n');process.stdin.flush()
            elif event=='RAW_CLEARED':
                key=message['job']+'/'+message['condition'];assert key in raw and key not in cleared
                cleared.add(key);print('N_PAPER6_RAW_LOCAL_VERIFIED_REMOTE_CLEARED',mode,key,flush=True)
            else:
                expected=controls*(1 if mode=='native' else 6)
                assert event=='CONTROLLER_COMPLETE' and not finished and message['controls']==controls and message['conditions']==expected and message['cases']==4*expected and message['optimizer_updates']==0
                finished=True
        process.stdin.close();assert process.wait()==0 and finished and set(raw)==cleared and len(raw)==expected
        phases[mode]=dict(status='COMPLETE',finished=datetime.now().isoformat(timespec='seconds'),exit_code=0,controls=controls,conditions=expected,state_cases=4*expected,raw=raw)
        (pf/('r201n_paper_missing6_'+mode+'_actual_20261009.json')).write_text(json.dumps(phases[mode],indent=2)+'\n',encoding='utf-8')
    code=f'''import hashlib,json,tarfile
from pathlib import Path
root=Path({ROOT!r});package=Path('/tmp/r201n_paper_missing6_closed_text_20261009.tar.gz');assert not package.exists()
files={{}}
with tarfile.open(package,'w:gz') as tar:
 for path in sorted(root.rglob('*')):
  if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log','.txt'):
   n=path.relative_to(root).as_posix();data=path.read_bytes();files[n]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());tar.add(path,arcname=n,recursive=False)
print(json.dumps(dict(files=files,package=dict(bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()))))'''
    inventory=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    package=pf/'r201n_paper_missing6_closed_text_20261009.tar.gz';assert not package.exists()
    command(['scp',*OPTIONS,'2026:/tmp/'+package.name,str(package)])
    assert package.stat().st_size==inventory['package']['bytes'] and hashlib.sha256(package.read_bytes()).hexdigest()==inventory['package']['sha256']
    with tarfile.open(package,'r:gz') as tar:
        assert set(tar.getnames())==set(inventory['files']);tar.extractall(LOCAL,filter='data')
    assert all((LOCAL/n).stat().st_size==r['bytes'] and hashlib.sha256((LOCAL/n).read_bytes()).hexdigest()==r['sha256'] for n,r in inventory['files'].items())
    code=f'''import hashlib
from pathlib import Path
for n,f in {restores!r}.items():
 p=Path(n);assert p.name=='best_official_arrays_restore_N_paper6.npz' and p.resolve().is_relative_to(Path({REMOTE!r}).resolve())
 assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];p.unlink()
Path('/tmp/r201n_paper_missing6_closed_text_20261009.tar.gz').unlink()
print('ONLY_SELECTED_RESTORED_ARRAY_COPIES_CLEARED')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True);package.unlink()
    subprocess.run([sys.executable,'-X','utf8','-B','-S','analyze_r201n_paper_missing6.py'],cwd=PROJECT,check=True)
    state.update(status='ACTUAL_N_ALL_DECLARED_FIXED_BEST_PAPER6_GT_RAW_CPU_COMPLETE_K201_RETAINED',
        source_graph='relation_local_shared_PI',archive_roots={v:str(path) for v,path in ARCHIVES.items()},
        finished=datetime.now().isoformat(timespec='seconds'),phases=phases,text_manifest=inventory['files'],
        native_raw_retained=True,new_full_conditions=24,new_full_state_cases=96,
        reused_RGBNT201_paper_conditions=12,reused_RGBNT201_state_cases=48,
        combined_full_conditions=36,combined_state_cases=144,restored_normal_copies_cleared=True,
        unique_raw_lost=0,new_optimizer_updates=0,new_49_evaluation=False,final_unit_shift_epsilon=.10)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(state['status'],flush=True)


if __name__=='__main__':main()
