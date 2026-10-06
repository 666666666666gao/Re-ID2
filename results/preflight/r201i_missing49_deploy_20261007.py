"""A single continuation after the registered normal-seed queue, with fixed normal selection."""
from datetime import datetime
import hashlib,json,os,shlex,shutil,subprocess,sys,tarfile,time
from pathlib import Path

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified

REMOTE='/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/r201i_missing49_20261007'
LOCAL=PROJECT/'results/r201i_missing49_20261007'
ARCHIVE_BASE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')
ARCHIVE=ARCHIVE_BASE/'r201i_missing49_20261007'


def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    proof=pf/'r201i_missing49_actual_session_20261007.json'
    assert not proof.exists() and not LOCAL.exists() and not ARCHIVE.exists()
    review=load(pf/'r201i_missing49_source_review_20261007.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources,reuse=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert set(sources)=={'evaluate_r201i_missing49_stream.py','launch_r201i_missing49_stream.py','analyze_r201i_missing49.py','results/preflight/r201i_missing49_deploy_20261007.py'}
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    state=dict(status='WAIT_REGISTERED_I_NORMAL_THREE_EXPERT_SEEDS_COMPLETE',started=datetime.now().isoformat(timespec='seconds'),receiver_pid=os.getpid(),new_neural_calls=0,new_optimizer_updates=0)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');print(json.dumps(state),flush=True)
    seed_proof=pf/'r201i_expert_seeds_actual_session_20261006.json'
    while load(seed_proof)['status']!='ACTUAL_I_THREE_EXPERT_SEEDS_NORMAL_GT_RAW_CPU_COMPLETE':
        time.sleep(240)
    original=load(pf/'r201i_primary_margin_actual_session_20261006.json')
    assert original['status']=='ACTUAL_I_THREE_NORMAL_DATASETS_SIX_FULL50_GT_RAW_COMPLETE' and original['exit_code']==0
    summary=load(PROJECT/'results/r201i_expert_seeds_20261006/normal_seed_analysis/result.json')
    assert summary['status']=='ACTUAL_I_RGBNT201_THREE_EXPERT_SEEDS_NORMAL_CPU_COMPLETE' and summary['seeds']==[42,43,44]
    assert summary['fixed_anchor_seed']==42
    relocation=load(pf/'own_D_closed_G100_frequency49_to_E_actual_20261007.json')
    assert relocation['status']=='ACTUAL_CLOSED_G100_FREQUENCY49_RAW_RELOCATED_D_TO_E_SIZE_SHA_VERIFIED' and relocation['files']==49 and relocation['lost_raw_files']==0
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    jobs=[]
    def add(dataset,variant,seed,selection):
        name=dataset+'_r201i_'+variant+'_s'+str(seed)
        normal=PROJECT/'results/r201i_primary_margin_20261006'/dataset if seed==42 else PROJECT/'results/r201i_expert_seeds_20261006'/('s'+str(seed))
        folder=normal/'training'/name;data=load(folder/'result.json')
        assert data['status']=='COMPLETE' and data['epochs']==50 and data['optimizer_steps']==data['steps'] and data['amp_skipped_steps']==0
        assert data['arguments']['seed']==seed and data['arguments']['dataset']==dataset
        assert load(folder/'normal_cpu_audit.json')['status']=='PASS'
        receipts=original['datasets'][dataset]['archives'] if seed==42 else load(pf/('r201i_expert_s'+str(seed)+'_actual_20261006.json'))['archives']
        raw=receipts[name];local=Path(raw['local'])
        assert local.resolve().is_relative_to(ARCHIVE_BASE.resolve())
        assert local.stat().st_size==raw['file']['bytes'] and hashlib.sha256(local.read_bytes()).hexdigest()==raw['file']['sha256']
        assert data['arguments']['output'].startswith(REMOTE+'/runs/')
        jobs.append(dict(name=name,dataset=dataset,variant=variant,seed=seed,gpu={'frequency_shared':2,'axis_shared':3}[variant],run=data['arguments']['output'],normal_local=str(local),normal_file=raw['file'],selected_epoch=data['best']['epoch'],full_metrics=data['full_metrics'],selection_scope=selection,normal_best_of3_selected=False))
    for dataset in ('RGBNT201','MSVR310','RGBNT100'):
        assert load(PROJECT/'results/r201i_primary_margin_20261006'/dataset/'normal_analysis/result.json')['status']=='ACTUAL_I_DATASET_TWO_FULL50_NORMAL_CPU_READOUT'
        for variant in ('frequency_shared','axis_shared'):add(dataset,variant,42,'unified_I_seed42')
    for selected in summary['best_of_three']:
        variant,seed=selected['model'],selected['seed'];name='RGBNT201_r201i_'+variant+'_s'+str(seed)
        if seed!=42:add('RGBNT201',variant,seed,'RGBNT201_normal_BestOf3')
        job=next(j for j in jobs if j['name']==name);job['normal_best_of3_selected']=True
        assert job['selected_epoch']==selected['selected_epoch'] and all(job['full_metrics'][k]==selected[k] for k in ('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20'))
    controls=len(jobs);assert 6<=controls<=8 and len({j['name'] for j in jobs})==controls
    assert shutil.disk_usage(ARCHIVE_BASE).free>22_000_000_000
    LOCAL.mkdir(parents=True)
    registry=dict(jobs=jobs,normal_best_of3=summary['best_of_three'],selected_at=datetime.now().isoformat(timespec='seconds'),missing_reselection=False)
    registry_path=LOCAL/'jobs.json';registry_path.write_text(json.dumps(registry,indent=2)+'\n',encoding='utf-8')
    restores={j['run']+'/best_official_arrays_restore_I_missing49.npz':j['normal_file'] for j in jobs}
    neural=('evaluate_r201i_missing49_stream.py','launch_r201i_missing49_stream.py')
    code=f'''import hashlib,shutil
from pathlib import Path
root=Path({REMOTE!r});assert not Path({ROOT!r}).exists()
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {reuse!r}.items() if not n.startswith('results/'))
assert all(not Path(n).exists() for n in {restores!r})
assert shutil.disk_usage(root).free>sum(f['bytes'] for f in {restores!r}.values())+1_600_000_000
print('ACTUAL_NORMAL_SEEDS_CLOSED_MISSING_CAPACITY_READY')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    for job in jobs:
        command(['scp',*OPTIONS,job['normal_local'],'2026:'+job['run']+'/best_official_arrays_restore_I_missing49.npz'])
    for name in neural:
        command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+REMOTE+'/'+name])
    remote_registry=REMOTE+'/runs/r201i_missing49_jobs_20261007.json'
    command(['scp',*OPTIONS,str(registry_path),'2026:'+remote_registry])
    code=f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r});assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {dict((n,sources[n]) for n in neural)!r}.items())
assert all(Path(n).stat().st_size==f['bytes'] and hashlib.sha256(Path(n).read_bytes()).hexdigest()==f['sha256'] for n,f in {restores!r}.items())
print('ACTUAL_REVIEWED_MISSING_SOURCES_AND_FIXED_NORMAL_RESTORES_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    phases={}
    for mode in ('native','full'):
        state.update(status='ACTUAL_I_FIXED_SELECTED_MISSING_'+mode.upper()+'_STARTED',controls=controls,registry=str(registry_path));state.pop('new_neural_calls',None)
        proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        argv=[PYTHON,'-u','launch_r201i_missing49_stream.py','--jobs',remote_registry,'--output',ROOT,'--mode',mode]
        process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026','cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        raw,cleared,finished={},set(),False
        for line in process.stdout:
            if not line.startswith('IDENTITY_MISSING_STREAM '):print(line,end='',flush=True);continue
            message=json.loads(line[len('IDENTITY_MISSING_STREAM '):]);event=message['event']
            if event=='RAW_READY':
                key=message['job']+'/'+message['condition'];assert message['installed_gt_cases']==4 and key not in raw
                assert shutil.disk_usage(ARCHIVE_BASE).free>message['file']['bytes']+1024**3
                local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVE)
                raw[key]=dict(file=message['file'],local=local,remote=message['path'])
                process.stdin.write(json.dumps(dict(event='ARCHIVED',job=message['job'],condition=message['condition'],file=message['file']))+'\n');process.stdin.flush()
            elif event=='RAW_CLEARED':
                key=message['job']+'/'+message['condition'];assert key in raw and key not in cleared
                cleared.add(key);print('I_MISSING_RAW_LOCAL_VERIFIED_REMOTE_CLEARED',mode,key,flush=True)
            else:
                expected=controls*(1 if mode=='native' else 49)
                assert event=='CONTROLLER_COMPLETE' and not finished and message['controls']==controls and message['conditions']==expected and message['cases']==4*expected and message['optimizer_updates']==0
                finished=True
        process.stdin.close();assert process.wait()==0 and finished and set(raw)==cleared and len(raw)==expected
        phases[mode]=dict(status='COMPLETE',finished=datetime.now().isoformat(timespec='seconds'),exit_code=0,controls=controls,conditions=expected,state_cases=4*expected,raw=raw)
        (pf/('r201i_missing49_'+mode+'_actual_20261007.json')).write_text(json.dumps(phases[mode],indent=2)+'\n',encoding='utf-8')
    code=f'''import hashlib,json,tarfile
from pathlib import Path
root=Path({ROOT!r});package=Path('/tmp/r201i_missing49_closed_text_20261007.tar.gz');assert not package.exists()
files={{}}
with tarfile.open(package,'w:gz') as tar:
 for path in sorted(root.rglob('*')):
  if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log','.txt'):
   n=path.relative_to(root).as_posix();data=path.read_bytes();files[n]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());tar.add(path,arcname=n,recursive=False)
print(json.dumps(dict(files=files,package=dict(bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()))))'''
    inventory=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    package=pf/'r201i_missing49_closed_text_20261007.tar.gz';assert not package.exists()
    command(['scp',*OPTIONS,'2026:/tmp/'+package.name,str(package)])
    assert package.stat().st_size==inventory['package']['bytes'] and hashlib.sha256(package.read_bytes()).hexdigest()==inventory['package']['sha256']
    with tarfile.open(package,'r:gz') as tar:
        assert set(tar.getnames())==set(inventory['files']);tar.extractall(LOCAL,filter='data')
    assert all((LOCAL/n).stat().st_size==r['bytes'] and hashlib.sha256((LOCAL/n).read_bytes()).hexdigest()==r['sha256'] for n,r in inventory['files'].items())
    for key,duplicate in phases['native']['raw'].items():
        original_raw=phases['full']['raw'][key]
        assert duplicate['file']==original_raw['file']
        path=Path(duplicate['local']);assert path.resolve().is_relative_to(ARCHIVE.resolve()) and '/native/' in path.as_posix()
        assert path.stat().st_size==duplicate['file']['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==duplicate['file']['sha256']
        full_path=Path(original_raw['local']);assert full_path.stat().st_size==original_raw['file']['bytes'] and hashlib.sha256(full_path.read_bytes()).hexdigest()==original_raw['file']['sha256']
        path.unlink()
    code=f'''import hashlib
from pathlib import Path
for n,f in {restores!r}.items():
 p=Path(n);assert p.name=='best_official_arrays_restore_I_missing49.npz' and p.resolve().is_relative_to(Path({REMOTE!r}).resolve())
 assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];p.unlink()
Path('/tmp/r201i_missing49_closed_text_20261007.tar.gz').unlink()
print('ONLY_SELECTED_RESTORED_ARRAY_COPIES_CLEARED')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True);package.unlink()
    subprocess.run([sys.executable,'-X','utf8','-B','-S','analyze_r201i_missing49.py'],cwd=PROJECT,check=True)
    state.update(status='ACTUAL_I_ALL_DECLARED_FIXED_BEST_MISSING49_GT_RAW_CPU_COMPLETE',finished=datetime.now().isoformat(timespec='seconds'),phases=phases,text_manifest=inventory['files'],native_duplicate_raw_retired=controls,restored_normal_copies_cleared=True,unique_raw_lost=0,new_optimizer_updates=0)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(state['status'],flush=True)


if __name__=='__main__':main()
