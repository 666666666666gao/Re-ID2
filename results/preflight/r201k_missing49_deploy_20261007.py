"""Full K42 four-state inference only after all three K normal datasets close."""
from datetime import datetime
import hashlib,json,shlex,shutil,subprocess,sys,tarfile
from pathlib import Path

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified

REMOTE='/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/r201k_missing49_20261007'
LOCAL=PROJECT/'results/r201k_missing49_20261007'
ARCHIVE_BASES={'frequency_shared':Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts'),'axis_shared':Path('E:/ReID2-experiment-artifacts')}
ARCHIVES={variant:base/'r201k_missing49_20261007' for variant,base in ARCHIVE_BASES.items()}


def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    proof=pf/'r201k_missing49_actual_session_20261007.json'
    assert not proof.exists() and not LOCAL.exists() and all(not path.exists() for path in ARCHIVES.values())
    review=load(pf/'r201k_missing49_source_review_20261007.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources,reuse=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert set(sources)=={'evaluate_r201k_missing49_stream.py','launch_r201k_missing49_stream.py','analyze_r201k_missing49.py','results/preflight/r201k_missing49_deploy_20261007.py'}
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    original=load(pf/'r201k_relation_local_pi_actual_session_20261007.json')
    assert original['status']=='ACTUAL_K_THREE_NORMAL_DATASETS_SIX_FULL50_GT_RAW_COMPLETE' and original['exit_code']==0
    assert original['native_updates']==18 and original['successful_updates']==19418 and original['additional_epochs']==300
    assert set(original['datasets'])=={'RGBNT201','MSVR310','RGBNT100'}
    state=dict(status='K_NORMAL3_TERMINAL_MISSING_SOURCE_CHECKED_NO_NN_STARTED',started=datetime.now().isoformat(timespec='seconds'),new_neural_calls=0,new_optimizer_updates=0)
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    jobs=[]
    def add(dataset,variant):
        name=dataset+'_r201k_'+variant+'_s42'
        folder=PROJECT/'results/r201k_relation_local_pi_20261007'/dataset/'training'/name;data=load(folder/'result.json')
        assert data['status']=='COMPLETE' and data['epochs']==50 and data['optimizer_steps']==data['steps'] and data['amp_skipped_steps']==0
        assert data['arguments']['seed']==42 and data['arguments']['dataset']==dataset
        assert data['method_revision'].startswith('R201K single PI-outlet factor') and data['anchor']['method']==data['method_revision']
        assert data['training_heldout_identities']==0 and data['descriptor_dim']==5120
        assert load(folder/'normal_cpu_audit.json')['status']=='PASS'
        raw=original['datasets'][dataset]['archives'][name];local=Path(raw['local'])
        assert local.resolve().is_relative_to(ARCHIVE_BASES['frequency_shared'].resolve())
        assert local.stat().st_size==raw['file']['bytes'] and hashlib.sha256(local.read_bytes()).hexdigest()==raw['file']['sha256']
        assert data['arguments']['output'].startswith(REMOTE+'/runs/r201k_relation_local_pi_20261007/')
        jobs.append(dict(name=name,dataset=dataset,variant=variant,seed=42,gpu={'frequency_shared':2,'axis_shared':3}[variant],run=data['arguments']['output'],normal_local=str(local),normal_file=raw['file'],selected_epoch=data['best']['epoch'],full_metrics=data['full_metrics'],selection_scope='unified_K_seed42_normal_best',forward_graph='relation_local_shared_PI'))
    for dataset in ('RGBNT201','MSVR310','RGBNT100'):
        assert load(PROJECT/'results/r201k_relation_local_pi_20261007'/dataset/'normal_analysis/result.json')['status']=='ACTUAL_K_DATASET_TWO_FULL50_NORMAL_CPU_READOUT'
        for variant in ('frequency_shared','axis_shared'):add(dataset,variant)
    controls=len(jobs);assert controls==6 and len({j['name'] for j in jobs})==6
    historical_full=load(pf/'r201i_missing49_full_actual_20261007.json')['raw']
    historical_native=load(pf/'r201i_missing49_native_actual_20261007.json')['raw']
    capacity={}
    for variant,base in ARCHIVE_BASES.items():
        measured_full=sum(item['file']['bytes'] for key,item in historical_full.items() if key.split('/')[0].endswith('_'+variant+'_s42'))
        measured_native=sum(item['file']['bytes'] for key,item in historical_native.items() if key.split('/')[0].endswith('_'+variant+'_s42'))
        required=measured_full+measured_native+2*1024**3
        free=shutil.disk_usage(base).free;assert free>required
        capacity[variant]=dict(root=str(ARCHIVES[variant]),free=free,required=required,empirical_I_full_bytes=measured_full,empirical_I_native_bytes=measured_native,extra_reserve_bytes=2*1024**3)
    LOCAL.mkdir(parents=True)
    registry=dict(jobs=jobs,selected_at=datetime.now().isoformat(timespec='seconds'),missing_reselection=False,source_graph='relation_local_shared_PI',explicit_archive_roots={v:str(path) for v,path in ARCHIVES.items()},capacity=capacity,limits='Fixed K42 normal selection; no missing seed/epoch selection or BestofN. Runtime perfile sizes still gated; empirical I archive sizes are an estimate.')
    registry_path=LOCAL/'jobs.json';registry_path.write_text(json.dumps(registry,indent=2)+'\n',encoding='utf-8')
    variants={job['name']:job['variant'] for job in jobs}
    restores={j['run']+'/best_official_arrays_restore_K_missing49.npz':j['normal_file'] for j in jobs}
    neural=('evaluate_r201k_missing49_stream.py','launch_r201k_missing49_stream.py')
    code=f'''import hashlib,shutil
from pathlib import Path
root=Path({REMOTE!r});assert not Path({ROOT!r}).exists()
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {reuse!r}.items() if not n.startswith('results/'))
assert all(not Path(n).exists() for n in {restores!r})
assert shutil.disk_usage(root).free>sum(f['bytes'] for f in {restores!r}.values())+1_600_000_000
print('ACTUAL_NORMAL_SEEDS_CLOSED_MISSING_CAPACITY_READY')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    for job in jobs:
        command(['scp',*OPTIONS,job['normal_local'],'2026:'+job['run']+'/best_official_arrays_restore_K_missing49.npz'])
    for name in neural:
        command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+REMOTE+'/'+name])
    remote_registry=REMOTE+'/runs/r201k_missing49_jobs_20261007.json'
    command(['scp',*OPTIONS,str(registry_path),'2026:'+remote_registry])
    code=f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r});assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {dict((n,sources[n]) for n in neural)!r}.items())
assert all(Path(n).stat().st_size==f['bytes'] and hashlib.sha256(Path(n).read_bytes()).hexdigest()==f['sha256'] for n,f in {restores!r}.items())
print('ACTUAL_REVIEWED_MISSING_SOURCES_AND_FIXED_NORMAL_RESTORES_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    phases={}
    for mode in ('native','full'):
        state.update(status='ACTUAL_K_FIXED_SELECTED_MISSING_'+mode.upper()+'_STARTED',controls=controls,registry=str(registry_path));state.pop('new_neural_calls',None)
        proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        argv=[PYTHON,'-u','launch_r201k_missing49_stream.py','--jobs',remote_registry,'--output',ROOT,'--mode',mode]
        process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026','cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        raw,cleared,finished={},set(),False
        for line in process.stdout:
            if not line.startswith('IDENTITY_MISSING_STREAM '):print(line,end='',flush=True);continue
            message=json.loads(line[len('IDENTITY_MISSING_STREAM '):]);event=message['event']
            if event=='RAW_READY':
                key=message['job']+'/'+message['condition'];assert message['installed_gt_cases']==4 and key not in raw
                variant=variants[message['job']]
                assert shutil.disk_usage(ARCHIVE_BASES[variant]).free>message['file']['bytes']+1024**3
                local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVES[variant])
                raw[key]=dict(file=message['file'],local=local,remote=message['path'])
                process.stdin.write(json.dumps(dict(event='ARCHIVED',job=message['job'],condition=message['condition'],file=message['file']))+'\n');process.stdin.flush()
            elif event=='RAW_CLEARED':
                key=message['job']+'/'+message['condition'];assert key in raw and key not in cleared
                cleared.add(key);print('K_MISSING_RAW_LOCAL_VERIFIED_REMOTE_CLEARED',mode,key,flush=True)
            else:
                expected=controls*(1 if mode=='native' else 49)
                assert event=='CONTROLLER_COMPLETE' and not finished and message['controls']==controls and message['conditions']==expected and message['cases']==4*expected and message['optimizer_updates']==0
                finished=True
        process.stdin.close();assert process.wait()==0 and finished and set(raw)==cleared and len(raw)==expected
        phases[mode]=dict(status='COMPLETE',finished=datetime.now().isoformat(timespec='seconds'),exit_code=0,controls=controls,conditions=expected,state_cases=4*expected,raw=raw)
        (pf/('r201k_missing49_'+mode+'_actual_20261007.json')).write_text(json.dumps(phases[mode],indent=2)+'\n',encoding='utf-8')
    code=f'''import hashlib,json,tarfile
from pathlib import Path
root=Path({ROOT!r});package=Path('/tmp/r201k_missing49_closed_text_20261007.tar.gz');assert not package.exists()
files={{}}
with tarfile.open(package,'w:gz') as tar:
 for path in sorted(root.rglob('*')):
  if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log','.txt'):
   n=path.relative_to(root).as_posix();data=path.read_bytes();files[n]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());tar.add(path,arcname=n,recursive=False)
print(json.dumps(dict(files=files,package=dict(bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()))))'''
    inventory=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    package=pf/'r201k_missing49_closed_text_20261007.tar.gz';assert not package.exists()
    command(['scp',*OPTIONS,'2026:/tmp/'+package.name,str(package)])
    assert package.stat().st_size==inventory['package']['bytes'] and hashlib.sha256(package.read_bytes()).hexdigest()==inventory['package']['sha256']
    with tarfile.open(package,'r:gz') as tar:
        assert set(tar.getnames())==set(inventory['files']);tar.extractall(LOCAL,filter='data')
    assert all((LOCAL/n).stat().st_size==r['bytes'] and hashlib.sha256((LOCAL/n).read_bytes()).hexdigest()==r['sha256'] for n,r in inventory['files'].items())
    for key,duplicate in phases['native']['raw'].items():
        original_raw=phases['full']['raw'][key]
        assert duplicate['file']==original_raw['file']
        path=Path(duplicate['local']);variant=variants[key.split('/')[0]];assert path.resolve().is_relative_to(ARCHIVES[variant].resolve()) and '/native/' in path.as_posix()
        assert path.stat().st_size==duplicate['file']['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==duplicate['file']['sha256']
        full_path=Path(original_raw['local']);assert full_path.stat().st_size==original_raw['file']['bytes'] and hashlib.sha256(full_path.read_bytes()).hexdigest()==original_raw['file']['sha256']
        path.unlink()
    code=f'''import hashlib
from pathlib import Path
for n,f in {restores!r}.items():
 p=Path(n);assert p.name=='best_official_arrays_restore_K_missing49.npz' and p.resolve().is_relative_to(Path({REMOTE!r}).resolve())
 assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];p.unlink()
Path('/tmp/r201k_missing49_closed_text_20261007.tar.gz').unlink()
print('ONLY_SELECTED_RESTORED_ARRAY_COPIES_CLEARED')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True);package.unlink()
    subprocess.run([sys.executable,'-X','utf8','-B','-S','analyze_r201k_missing49.py'],cwd=PROJECT,check=True)
    state.update(status='ACTUAL_K_ALL_DECLARED_FIXED_BEST_MISSING49_GT_RAW_CPU_COMPLETE',source_graph='relation_local_shared_PI',archive_roots={v:str(path) for v,path in ARCHIVES.items()},finished=datetime.now().isoformat(timespec='seconds'),phases=phases,text_manifest=inventory['files'],native_duplicate_raw_retired=controls,restored_normal_copies_cleared=True,unique_raw_lost=0,new_optimizer_updates=0)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(state['status'],flush=True)


if __name__=='__main__':main()
