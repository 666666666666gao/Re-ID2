"""CPU-only resume of the observed incomplete archive; never rerun the model."""
from datetime import datetime
import hashlib,json,os,shlex,subprocess,sys,tarfile
from pathlib import Path

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/selected_expert_outlets_20261011'
LOCAL=PROJECT/'results/selected_expert_outlets_20261011'
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/selected_expert_outlets_20261011')
PREFIX='OUTLET_STREAM '


def remote(code):
    return command(['ssh',*OPTIONS,'2026','cd '+shlex.quote(REMOTE)+' && '+shlex.quote(PYTHON)+' -'],input=code)


def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    proof=pf/'selected_expert_outlets_cpu_archive_recovery_actual_20261011.json';assert not proof.exists()
    original=load(pf/'selected_expert_outlets_original_owner_closed_actual_20261011.json');assert original['exit_code']==1 and original['handle_consumed']
    review=load(pf/'selected_expert_outlets_archive_recovery_source_review_20261011.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
    interrupted=load(pf/'selected_expert_outlets_archive_timeout_actual_20261011.json')
    job='RGBNT100_r201p_axis_shared_s42';folder=ROOT+'/'+job;raw=folder+'/raw.npz'
    ready=[json.loads(line[len(PREFIX):]) for line in interrupted['remote']['logs'][job+'.log'].splitlines() if line.startswith(PREFIX)]
    assert len(ready)==1 and ready[0]['event']=='RAW_READY' and ready[0]['installed_gt_cases']==13
    expected=ready[0]['file'];path=ARCHIVE/job/'raw.npz'
    offset=path.stat().st_size;assert 0<offset<expected['bytes']
    local_prefix=hashlib.sha256(path.read_bytes()).hexdigest()
    code=f'''import hashlib,json,subprocess
from pathlib import Path
p=Path({raw!r});assert p.stat().st_size=={expected['bytes']!r}
assert hashlib.sha256(p.read_bytes()).hexdigest()=={expected['sha256']!r}
with p.open('rb') as f:prefix=hashlib.sha256(f.read({offset!r})).hexdigest()
assert prefix=={local_prefix!r}
for pid in (1498702,1498703,1498704,1498705):assert subprocess.run(['ps','-p',str(pid)],stdout=subprocess.DEVNULL).returncode==1
print(json.dumps(dict(prefix_bytes={offset!r},prefix_exact=True)))'''
    observed=json.loads(remote(code))
    state=dict(status='ACTUAL_EXISTING_RAW_PREFIX_VERIFIED_CPU_TAIL_ARCHIVE_STARTED',started=datetime.now().astimezone().isoformat(timespec='seconds'),receiver_pid=os.getpid(),original_exit_code=1,original_axis_worker_exit_code=-9,new_neural_calls=0,optimizer_updates=0,new_weights=0,existing_prefix=observed,file=expected,chunks=[])
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    index=0
    while offset<expected['bytes']:
        length=min(32*1024**2,expected['bytes']-offset)
        chunk_remote=folder+'/archive_tail_'+str(index)+'_20261011.bin'
        chunk_local=path.parent/('archive_tail_'+str(index)+'_20261011.bin');assert not chunk_local.exists()
        code=f'''import hashlib,json
from pathlib import Path
p=Path({raw!r});out=Path({chunk_remote!r});assert not out.exists()
with p.open('rb') as f:f.seek({offset!r});data=f.read({length!r})
assert len(data)=={length!r};out.write_bytes(data)
print(json.dumps(dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())))'''
        info=json.loads(remote(code));command(['scp',*OPTIONS,'2026:'+chunk_remote,str(chunk_local)])
        data=chunk_local.read_bytes();assert len(data)==info['bytes'] and hashlib.sha256(data).hexdigest()==info['sha256']
        assert path.stat().st_size==offset
        with path.open('ab') as f:f.write(data)
        offset+=length;assert path.stat().st_size==offset
        code=f'''import hashlib
from pathlib import Path
p=Path({chunk_remote!r});assert p.name=={chunk_local.name!r} and p.parent==Path({folder!r})
assert p.stat().st_size=={info['bytes']!r} and hashlib.sha256(p.read_bytes()).hexdigest()=={info['sha256']!r};p.unlink()
print('VERIFIED_OWN_ARCHIVE_TAIL_CHUNK_CLEARED')'''
        print(remote(code),flush=True);chunk_local.unlink()
        state['chunks'].append(dict(offset_after=offset,file=info));proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');index+=1
    assert path.stat().st_size==expected['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==expected['sha256']
    registry=load(LOCAL/'jobs.json')
    restored={j['normal_restore']:j['normal_file'] for j in registry['jobs']}
    code=f'''import hashlib,json
from pathlib import Path
import numpy as np
from audit_full_official_condition import installed_context
from audit_full_official49 import recount,verify,METRICS
r=Path({ROOT!r});folder=r/{job!r};trained,dataset,installed=installed_context(Path({next(j['run'] for j in registry['jobs'] if j['name']==job)!r}))
audit=json.loads((folder/'independent_cpu_audit.json').read_text());assert audit['status']=='PASS'
with np.load(folder/'raw.npz') as saved,np.load({next(j['normal_restore'] for j in registry['jobs'] if j['name']==job)!r}) as normal:
 assert np.array_equal(saved['distance_normal11'],normal['distances'])
 for role in ('query','gallery'):
  for field,key in (('ids','identity'),('cameras','camera'),('scenes','scene'),('names','name')):assert np.array_equal(saved[role+'_'+field],np.asarray([x[key] for x in installed[role]]))
 for stage,value in audit['measured'].items():
  rows=recount(saved['distance_'+stage],installed['query'],installed['gallery'],dataset)
  report=json.loads((folder/(stage+'.json')).read_text())
  checked,error=verify(rows,report,folder/(stage+'.csv'),len(installed['gallery']));assert checked==value and error<1e-8
 assert all(audit['measured']['normal11'][k]==trained['full_metrics'][k] for k in METRICS)
for restore,info in {restored!r}.items():
 p=Path(restore);assert p.stat().st_size==info['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
result=dict(status='COMPLETE_VIA_CPU_ARCHIVE_RECOVERY',dataset=dataset,variant=trained['arguments']['variant'],selected_epoch=trained['best']['epoch'],model_arguments=trained['arguments'],stages=list(audit['measured']),measurements=audit['measured'],comparisons=audit['changes'],dimensions={{k:5120 if k in ('base00','normal11') else 3584 if k in ('M_projection_input','F_projection_input','M_projection_output','F_projection_output','M_unit_direction','F_unit_direction') else 512 for k in audit['measured']}},query_records=trained['query_records'],gallery_records=trained['gallery_records'],train_records=trained['train_records'],normal_features_and_distance_exact=True,state_tensor_versions_unchanged=True,optimizer_updates=0,new_weights=0,new49=False,original_receiver_exit_code=1,original_neural_worker_exit_code=-9,recovery_new_neural_calls=0,raw=dict(file={expected!r},local_verified_cpu_recovery=True,original_ACK_sent=False,server_copy_cleared=True),evidence='Original RAW_READY requires all normal feature/distance exact and tensor-version/frozen checks before this point; CPU independently rechecked all13 GT and exactnormal distance after tail transfer. Worker killed at archive wait, not rerun. No fabricated original exit0 or original ACK.',limits='Independent residual descriptors only, fixed original checkpoint and normal scope; CPU archive completion is explicitly separate from original neural process exit.')
(folder/'archive_recovered_result.json').write_text(json.dumps(result,indent=2)+'\\n')
p=folder/'raw.npz';assert hashlib.sha256(p.read_bytes()).hexdigest()=={expected['sha256']!r};p.unlink()
print('ACTUAL_RGB_AXIS_EXISTING_RAW_CPU_RECOUNT_AND_ARCHIVE_RECOVERED_WITHOUT_NEURAL_CALL')'''
    print(remote(code),flush=True)
    for j in registry['jobs']:
        j['completion_file']='archive_recovered_result.json' if j['name']==job else 'result.json'
        j['completion_status']='COMPLETE_VIA_CPU_ARCHIVE_RECOVERY' if j['name']==job else 'COMPLETE'
    (LOCAL/'jobs.json').write_text(json.dumps(registry,indent=2)+'\n',encoding='utf-8')
    code=f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({ROOT!r});package=Path('/tmp/selected_expert_outlets_recovered_text_20261011.tar.gz');assert not package.exists();files={{}}
with tarfile.open(package,'w:gz') as tar:
 for p in sorted(r.rglob('*')):
  if p.is_file() and p.suffix in ('.json','.csv','.log','.txt'):
   n=p.relative_to(r).as_posix();data=p.read_bytes();files[n]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());tar.add(p,arcname=n,recursive=False)
print(json.dumps(dict(files=files,package=dict(bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()))))'''
    inventory=json.loads(remote(code));package=pf/'selected_expert_outlets_recovered_text_20261011.tar.gz';assert not package.exists()
    command(['scp',*OPTIONS,'2026:/tmp/'+package.name,str(package)])
    assert package.stat().st_size==inventory['package']['bytes'] and hashlib.sha256(package.read_bytes()).hexdigest()==inventory['package']['sha256']
    with tarfile.open(package,'r:gz') as tar:
        assert set(tar.getnames())==set(inventory['files']);tar.extractall(LOCAL,filter='data')
    assert all((LOCAL/n).stat().st_size==f['bytes'] and hashlib.sha256((LOCAL/n).read_bytes()).hexdigest()==f['sha256'] for n,f in inventory['files'].items())
    archives={}
    for j in registry['jobs']:
        result=load(LOCAL/j['name']/j['completion_file']);info=result['raw']['file'];p=ARCHIVE/j['name']/'raw.npz'
        assert p.stat().st_size==info['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
        archives[j['name']]=dict(file=info,local=str(p),server_copy_cleared=True)
    code=f'''import hashlib
from pathlib import Path
for n,f in {restored!r}.items():
 p=Path(n);assert p.name=='best_official_arrays_restore_outlets_20261011.npz' and p.resolve().is_relative_to(Path({REMOTE!r}).resolve())
 assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];p.unlink()
Path('/tmp/selected_expert_outlets_recovered_text_20261011.tar.gz').unlink()
print('OWN_RECOVERY_RESTORES_AND_PACKAGE_CLEARED')'''
    print(remote(code),flush=True);package.unlink()
    subprocess.run([sys.executable,'-X','utf8','-B','-S','analyze_selected_expert_outlets_recovered.py'],cwd=PROJECT,check=True)
    state.update(status='ACTUAL_ORIGINAL_OUTLET_ARCHIVE_AND_CPU_RECOVERED_WITHOUT_NEURAL_RERUN',finished=datetime.now().astimezone().isoformat(timespec='seconds'),exit_code=0,archives=archives,stage_cases=52,text_manifest=inventory['files'],normal_restores_cleared=True,unique_raw_lost=0)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    actual=pf/'selected_expert_outlets_actual_session_20261011.json';snapshot=pf/'selected_expert_outlets_original_started_snapshot_20261011.json';assert not snapshot.exists();snapshot.write_bytes(actual.read_bytes())
    aggregate=load(actual);aggregate.update(status='ACTUAL_FOUR_SELECTED_P_R_FULL_NORMAL_OUTLET_GT_RAW_CPU_COMPLETE_VIA_ARCHIVE_RECOVERY',finished=state['finished'],exit_code=0,original_receiver_exit_code=1,completion_mode='Three original exit0 + RGBaxis completeGT/RAW_READY, transfer timeout/kill; CPU-only tail/fullGT recovery0',raw=archives,text_manifest=inventory['files'],stage_cases=52,normal_restores_cleared=True,unique_raw_lost=0,cpu_status=load(LOCAL/'outlet_analysis/result.json')['status'])
    actual.write_text(json.dumps(aggregate,indent=2)+'\n',encoding='utf-8');print(aggregate['status'],flush=True)


if __name__=='__main__':main()
