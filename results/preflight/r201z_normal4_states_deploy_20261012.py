"""Two fixed Z42 MSVR adapting-modality normal-four-state banks0/1 after original Z normal closure; preserve liveX2/3."""
from datetime import datetime
import hashlib,json,shlex,shutil,subprocess,sys,tarfile
from pathlib import Path

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified

REMOTE='/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/r201z_normal4_states_20261012'
LOCAL=PROJECT/'results/r201z_normal4_states_20261012'
ARCHIVE_BASES={variant:Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts') for variant in ('frequency_shared','axis_shared')}
ARCHIVES={variant:base/'r201z_normal4_states_20261012' for variant,base in ARCHIVE_BASES.items()}


def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    proof=pf/'r201z_normal4_states_actual_session_20261012.json'
    assert not proof.exists() and not LOCAL.exists() and all(not path.exists() for path in ARCHIVES.values())
    review=load(pf/'r201z_normal4_states_source_review_20261012.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources,reuse=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert set(sources)=={'evaluate_r201z_normal4_states.py','launch_r201z_normal4_states.py','analyze_r201z_normal4_states.py','results/preflight/r201z_normal4_states_deploy_20261012.py','results/preflight/r201z_normal4_states_plan_20261012.json'}
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    terminal=load(pf/'r201z_original_owner_normal_closed_actual_20261012.json')
    assert terminal['status']=='ACTUAL_Z_ORIGINAL_TWO_FULL50_NORMAL_GT_RAW_CPU_TRUE_EXIT0_CLOSED' and terminal['exit_code']==0
    assert terminal['formal_updates']==1410 and terminal['native_updates']==6 and terminal['terminal_consumed'] and terminal['original_process_absent'] and terminal['normal_GT_raw_CPU_complete']
    original=load(pf/'r201z_MS_modality_adapt_actual_session_20261012.json')
    assert original['status']=='ACTUAL_Z_MS_NORMAL_TWO_FULL50_GT_RAW_CPU_COMPLETE' and original['exit_code']==0
    assert original['native_updates']==6 and original['successful_updates']==1410 and original['additional_epochs']==100
    assert set(original['datasets'])=={'MSVR310'}
    normal_datasets=original['datasets'];value=normal_datasets['MSVR310']
    assert value['status']=='ACTUAL_Z_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW' and value['exit_code']==0
    assert value['successful_updates']==1410 and value['native_updates']==6 and value['additional_epochs']==100 and len(value['archives'])==2
    replay=load(pf/'r201l_uniform_K8_sampler_cpu_replay_20261007.json')
    assert replay['status']=='ACTUAL_EXISTING_SAMPLER_CPU_FULL50_OLD_AND_K8_REPLAY_COMPLETE'
    state=dict(status='Z_ORIGINAL_NORMAL_TERMINAL_NORMAL4_SOURCE_CHECKED_NO_NN_STARTED',started=datetime.now().isoformat(timespec='seconds'),new_neural_calls=0,new_optimizer_updates=0,
        normal_collection_scope='Original Z receiver mustclose MSVRfull50/GT/raw/CPU first; XRGB1002/3 unchanged; no model/remote work until sourcePASS+trueZexit0 closure')
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    jobs=[]
    def add(dataset,variant):
        name=dataset+'_r201z_'+variant+'_s42'
        folder=PROJECT/'results/r201z_MS_modality_adapt_20261012'/dataset/'training'/name;data=load(folder/'result.json')
        assert data['status']=='COMPLETE' and data['epochs']==50 and data['optimizer_steps']==data['steps'] and data['amp_skipped_steps']==0
        assert data['arguments']['seed']==42 and data['arguments']['dataset']==dataset
        assert data['optimizer_steps']==replay['datasets'][dataset]['4']['total_steps'] and '  NUM_INSTANCE: 4' in data['config'].splitlines()
        assert data['method_revision'].startswith('Z MS-only existing-modality-parameter gradient factor') and data['anchor']['method']==data['method_revision']
        assert data['anchor']['M_F_auxiliary_weight']==.001 and data['anchor']['parent_contribution_regression_weight']==0.
        assert data['anchor']['freeze_identity_encoder'] is False and data['anchor']['partial_identity_adaptation']['adapted_parameters']==9958912
        assert (data['parameters'],data['trainable_parameters'])==(100821638,13081734)
        assert data['anchor']['final_unit_descriptor_bound']['epsilon']==.10 and data['anchor']['final_unit_descriptor_bound']['parameter_change']==0
        assert data['training_heldout_identities']==0 and data['descriptor_dim']==5120
        assert load(folder/'normal_cpu_audit.json')['status']=='PASS'
        raw=normal_datasets[dataset]['archives'][name];local=Path(raw['local'])
        assert local.resolve().is_relative_to(ARCHIVE_BASES['frequency_shared'].resolve())
        assert local.stat().st_size==raw['file']['bytes'] and hashlib.sha256(local.read_bytes()).hexdigest()==raw['file']['sha256']
        assert data['arguments']['output'].startswith(REMOTE+'/runs/r201z_MS_modality_adapt_20261012/')
        jobs.append(dict(name=name,dataset=dataset,variant=variant,seed=42,gpu={'frequency_shared':0,'axis_shared':1}[variant],run=data['arguments']['output'],normal_local=str(local),normal_file=raw['file'],selected_epoch=data['best']['epoch'],full_metrics=data['full_metrics'],selection_scope='ZMS42 fixednormalbest only; adapting00/currentbase cap.10, CE.05/auxcfg.25/hinge.6/MFaux.001, no cached X/K evaluation',forward_graph='relation_local_shared_PI'))
    for dataset in ('MSVR310',):
        assert load(PROJECT/'results/r201z_MS_modality_adapt_20261012'/dataset/'normal_analysis/result.json')['status']=='ACTUAL_Z_DATASET_TWO_FULL50_NORMAL_CPU_READOUT'
        for variant in ('frequency_shared','axis_shared'):add(dataset,variant)
    controls=len(jobs);assert controls==2 and len({j['name'] for j in jobs})==2
    capacity={}
    for variant,base in ARCHIVE_BASES.items():
        free=shutil.disk_usage(base).free;required=4*1024**3
        assert free>required
        capacity[variant]=dict(root=str(ARCHIVES[variant]),free=free,required=required,
            limits='4GiB reserved for two normal-four-state distance archives; actual file size gated before ACK')
    LOCAL.mkdir(parents=True)
    registry=dict(jobs=jobs,
        selected_at=datetime.now().isoformat(timespec='seconds'),missing_reselection=False,
        source_graph='relation_local_shared_PI',explicit_archive_roots={v:str(path) for v,path in ARCHIVES.items()},capacity=capacity,
        limits='Only Znormal RNT quartet at fixedsavedbest, adaptedGF/frozen4/GFbuffers/CE.05/auxcfg.25/reg0/MFaux.001/P16K4/hinge.6/cap.10 currentbase. 00maydifferteacher/controls. No missing/reselection/newoptimizer/weights/49; X/K untouched.')
    registry_path=LOCAL/'jobs.json';registry_path.write_text(json.dumps(registry,indent=2)+'\n',encoding='utf-8')
    variants={job['name']:job['variant'] for job in jobs}
    restores={j['run']+'/best_official_arrays_restore_Z_normal4.npz':j['normal_file'] for j in jobs}
    neural=('evaluate_r201z_normal4_states.py','launch_r201z_normal4_states.py')
    code=f'''import hashlib,shutil
from pathlib import Path
root=Path({REMOTE!r});assert not Path({ROOT!r}).exists()
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {reuse!r}.items() if not n.startswith('results/'))
assert all(not Path(n).exists() for n in {restores!r})
assert shutil.disk_usage(root).free>sum(f['bytes'] for f in {restores!r}.values())+700_000_000
print('ACTUAL_Z_ORIGINAL_NORMAL_CLOSED_NORMAL4_CAPACITY_READY')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    for job in jobs:
        command(['scp',*OPTIONS,job['normal_local'],'2026:'+job['run']+'/best_official_arrays_restore_Z_normal4.npz'])
    for name in neural:
        command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+REMOTE+'/'+name])
    remote_registry=REMOTE+'/runs/r201z_normal4_states_jobs_20261012.json'
    command(['scp',*OPTIONS,str(registry_path),'2026:'+remote_registry])
    code=f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r});assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {dict((n,sources[n]) for n in neural)!r}.items())
assert all(Path(n).stat().st_size==f['bytes'] and hashlib.sha256(Path(n).read_bytes()).hexdigest()==f['sha256'] for n,f in {restores!r}.items())
print('ACTUAL_REVIEWED_NORMAL4_SOURCES_AND_FIXED_NORMAL_RESTORES_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    phases={}
    for mode in ('normal4',):
        state.update(status='ACTUAL_Z_FIXED_SELECTED_NORMAL4_'+mode.upper()+'_STARTED',controls=controls,registry=str(registry_path));state.pop('new_neural_calls',None)
        proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        argv=[PYTHON,'-u','launch_r201z_normal4_states.py','--jobs',remote_registry,'--output',ROOT]
        process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026','cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        raw,cleared,finished={},set(),False
        for line in process.stdout:
            if not line.startswith('NORMAL4_STREAM '):print(line,end='',flush=True);continue
            message=json.loads(line[len('NORMAL4_STREAM '):]);event=message['event']
            if event=='RAW_READY':
                key=message['job']+'/'+message['condition'];assert message['installed_gt_cases']==4 and key not in raw
                variant=variants[message['job']]
                assert shutil.disk_usage(ARCHIVE_BASES[variant]).free>message['file']['bytes']+1024**3
                local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVES[variant])
                raw[key]=dict(file=message['file'],local=local,remote=message['path'])
                process.stdin.write(json.dumps(dict(event='ARCHIVED',job=message['job'],condition=message['condition'],file=message['file']))+'\n');process.stdin.flush()
            elif event=='RAW_CLEARED':
                key=message['job']+'/'+message['condition'];assert key in raw and key not in cleared
                cleared.add(key);print('Z_NORMAL4_RAW_LOCAL_VERIFIED_REMOTE_CLEARED',mode,key,flush=True)
            else:
                expected=controls
                assert event=='CONTROLLER_COMPLETE' and not finished and message['controls']==controls and message['conditions']==expected and message['cases']==4*expected and message['optimizer_updates']==0
                finished=True
        process.stdin.close();assert process.wait()==0 and finished and set(raw)==cleared and len(raw)==expected
        phases[mode]=dict(status='COMPLETE',finished=datetime.now().isoformat(timespec='seconds'),exit_code=0,controls=controls,conditions=expected,state_cases=4*expected,raw=raw)
        (pf/('r201z_normal4_states_'+mode+'_actual_20261012.json')).write_text(json.dumps(phases[mode],indent=2)+'\n',encoding='utf-8')
    code=f'''import hashlib,json,tarfile
from pathlib import Path
root=Path({ROOT!r});package=Path('/tmp/r201z_normal4_states_closed_text_20261012.tar.gz');assert not package.exists()
files={{}}
with tarfile.open(package,'w:gz') as tar:
 for path in sorted(root.rglob('*')):
  if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log','.txt'):
   n=path.relative_to(root).as_posix();data=path.read_bytes();files[n]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());tar.add(path,arcname=n,recursive=False)
print(json.dumps(dict(files=files,package=dict(bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()))))'''
    inventory=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    package=pf/'r201z_normal4_states_closed_text_20261012.tar.gz';assert not package.exists()
    command(['scp',*OPTIONS,'2026:/tmp/'+package.name,str(package)])
    assert package.stat().st_size==inventory['package']['bytes'] and hashlib.sha256(package.read_bytes()).hexdigest()==inventory['package']['sha256']
    with tarfile.open(package,'r:gz') as tar:
        assert set(tar.getnames())==set(inventory['files']);tar.extractall(LOCAL,filter='data')
    assert all((LOCAL/n).stat().st_size==r['bytes'] and hashlib.sha256((LOCAL/n).read_bytes()).hexdigest()==r['sha256'] for n,r in inventory['files'].items())
    code=f'''import hashlib
from pathlib import Path
for n,f in {restores!r}.items():
 p=Path(n);assert p.name=='best_official_arrays_restore_Z_normal4.npz' and p.resolve().is_relative_to(Path({REMOTE!r}).resolve())
 assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];p.unlink()
Path('/tmp/r201z_normal4_states_closed_text_20261012.tar.gz').unlink()
print('ONLY_SELECTED_RESTORED_ARRAY_COPIES_CLEARED')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True);package.unlink()
    stream=dict(status='ACTUAL_Z_TWO_NORMAL_FOUR_STATES_GT_RAW_TEXT_COMPLETE',exit_code=0,normal_conditions=2,state_cases=8,optimizer_updates=0,phases=phases)
    (pf/'r201z_normal4_stream_actual_20261012.json').write_text(json.dumps(stream,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,'-X','utf8','-B','-S','analyze_r201z_normal4_states.py'],cwd=PROJECT,check=True)
    state.update(status='ACTUAL_Z_SELECTED_MSVZ_NORMAL4_GT_RAW_CPU_COMPLETE',finished=datetime.now().isoformat(timespec='seconds'),exit_code=0,phases=phases,text_manifest=inventory['files'],normal_conditions=2,state_cases=8,missing_conditions=0,restored_normal_copies_cleared=True,unique_raw_lost=0,new_optimizer_updates=0,new_49_evaluation=False)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(state['status'],flush=True)


if __name__=='__main__':main()
