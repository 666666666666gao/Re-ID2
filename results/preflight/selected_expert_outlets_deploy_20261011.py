"""Four-card fixed-best normal outlet diagnostics after all original owners closed."""
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

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
TAG='selected_expert_outlets_20261011'
ROOT=REMOTE+'/runs/'+TAG
LOCAL=PROJECT/'results'/TAG
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')/TAG
PREFIX='OUTLET_STREAM '


def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    proof=pf/'selected_expert_outlets_actual_session_20261011.json'
    assert not proof.exists() and not LOCAL.exists() and not ARCHIVE.exists()
    for owner in ('r201p_resume2_original_normal_owner_closed_actual_20261011.json',
                  'r201p_post_resume2_paper_original_queue_closed_actual_20261011.json',
                  'r201q_paper_original_owner_closed_actual_20261011.json',
                  'r201r_primary_ce_original_owner_closed_actual_20261011.json',
                  'r201r_selected_gradients_original_owner_closed_actual_20261011.json'):
        assert load(pf/owner)['exit_code']==0
    review=load(pf/'selected_expert_outlets_source_review_20261011.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources,reuse=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert set(sources)=={'evaluate_selected_expert_outlets.py','launch_selected_expert_outlets.py','analyze_selected_expert_outlets.py',
                         'results/preflight/selected_expert_outlets_deploy_20261011.py','results/preflight/selected_expert_outlets_plan_20261011.json'}
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    jobs=[]
    for dataset,stage,date,gpus in (('MSVR310','r201r_primary_ce','20261011',(0,1)),('RGBNT100','r201p_auxiliary_balance','20261010',(2,3))):
        normal=load(pf/(stage+'_'+dataset+'_actual_session_20261011.json' if dataset=='MSVR310' else stage+'_'+dataset+'_actual_session_20261010.json'))
        assert normal['exit_code']==0 and normal['additional_epochs']==100 and len(normal['archives'])==2
        for variant,gpu in zip(('frequency_shared','axis_shared'),gpus):
            label='r201r' if dataset=='MSVR310' else 'r201p'
            name=dataset+'_'+label+'_'+variant+'_s42'
            run=PROJECT/'results'/(stage+'_'+date)/dataset/'training'/name
            trained=load(run/'result.json');raw=normal['archives'][name];path=Path(raw['local'])
            assert trained['status']=='COMPLETE' and trained['epochs']==50 and trained['amp_skipped_steps']==trained['training_heldout_identities']==0
            assert trained['arguments']['seed']==42 and trained['arguments']['dataset']==dataset and trained['descriptor_dim']==5120
            assert load(run/'normal_cpu_audit.json')['status']=='PASS'
            assert path.resolve().is_relative_to(ARCHIVE.parent.resolve()) and path.stat().st_size==raw['file']['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==raw['file']['sha256']
            remote_run=trained['arguments']['output'];assert remote_run.startswith(REMOTE+'/runs/'+stage+'_'+date+'/')
            jobs.append(dict(name=name,dataset=dataset,variant=variant,seed=42,gpu=gpu,run=remote_run,
                normal_local=str(path),normal_file=raw['file'],normal_restore=remote_run+'/best_official_arrays_restore_outlets_20261011.npz',
                selected_epoch=trained['best']['epoch'],full_metrics=trained['full_metrics']))
    assert len(jobs)==4 and {j['gpu'] for j in jobs}=={0,1,2,3}
    required=4*1024**3;assert shutil.disk_usage(ARCHIVE.parent).free>required
    restores={j['normal_restore']:j['normal_file'] for j in jobs}
    neural=('evaluate_selected_expert_outlets.py','launch_selected_expert_outlets.py')
    code=f'''import hashlib,shutil
from pathlib import Path
root=Path({REMOTE!r});assert not Path({ROOT!r}).exists()
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {reuse!r}.items() if not n.startswith('results/'))
assert all(not Path(n).exists() for n in {restores!r})
assert shutil.disk_usage(root).free>sum(f['bytes'] for f in {restores!r}.values())+2_500_000_000
print('ACTUAL_FOUR_CARD_OUTLET_DIAG_CAPACITY_AND_CLOSED_SOURCES_READY')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    LOCAL.mkdir(parents=True)
    registry=dict(jobs=jobs,selected_at=datetime.now().astimezone().isoformat(timespec='seconds'),
        physical_gpus=[0,1,2,3],optimizer_updates=0,new_weights=0,new49=False,
        limits='P RGB100 and R MS fixed normal-best; diagnostic only, no loss/framework edits, K201 protected. Full official Q/G, thirteen stages per model; raw-only local archive ACK; no retained new feature-bank files.')
    registry_path=LOCAL/'jobs.json';registry_path.write_text(json.dumps(registry,indent=2)+'\n',encoding='utf-8')
    for job in jobs:command(['scp',*OPTIONS,job['normal_local'],'2026:'+job['normal_restore']])
    for n in neural:command(['scp',*OPTIONS,str(PROJECT/n),'2026:'+REMOTE+'/'+n])
    remote_registry=REMOTE+'/runs/'+TAG+'_jobs.json';command(['scp',*OPTIONS,str(registry_path),'2026:'+remote_registry])
    code=f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r})
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {dict((n,sources[n]) for n in neural)!r}.items())
assert all(Path(n).stat().st_size==f['bytes'] and hashlib.sha256(Path(n).read_bytes()).hexdigest()==f['sha256'] for n,f in {restores!r}.items())
print('ACTUAL_REVIEWED_OUTLET_SOURCES_AND_SELECTED_NORMAL_RESTORES_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    state=dict(status='ACTUAL_FOUR_SELECTED_P_R_OUTLET_DIAGNOSTICS_STARTED',started=datetime.now().astimezone().isoformat(timespec='seconds'),receiver_pid=os.getpid(),registry=str(registry_path),controls=4,physical_gpus=[0,1,2,3],optimizer_updates=0,new_weights=0,new49=False)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    argv=[PYTHON,'-u','launch_selected_expert_outlets.py','--jobs',remote_registry,'--output',ROOT]
    process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026','cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
    raw,cleared,finished={},set(),False
    for line in process.stdout:
        if not line.startswith(PREFIX):print(line,end='',flush=True);continue
        message=json.loads(line[len(PREFIX):]);event=message['event']
        if event=='RAW_READY':
            name=message['job'];assert name in {j['name'] for j in jobs} and name not in raw and message['installed_gt_cases']==13
            assert shutil.disk_usage(ARCHIVE.parent).free>message['file']['bytes']+1024**3
            path=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVE)
            raw[name]=dict(file=message['file'],local=path,remote=message['path'])
            process.stdin.write(json.dumps(dict(event='ARCHIVED',job=name,condition='normal_outlets',file=message['file']))+'\n');process.stdin.flush()
        elif event=='RAW_CLEARED':
            name=message['job'];assert name in raw and name not in cleared;cleared.add(name)
            print('OUTLET_RAW_LOCAL_VERIFIED_REMOTE_CLEARED',name,flush=True)
        else:
            assert event=='CONTROLLER_COMPLETE' and not finished and message==dict(event='CONTROLLER_COMPLETE',controls=4,conditions=4,cases=52,optimizer_updates=0)
            finished=True
    process.stdin.close();assert process.wait()==0 and finished and set(raw)==cleared and len(raw)==4
    code=f'''import hashlib,json,tarfile
from pathlib import Path
root=Path({ROOT!r});package=Path('/tmp/selected_expert_outlets_closed_20261011.tar.gz');assert not package.exists()
files={{}}
with tarfile.open(package,'w:gz') as tar:
 for p in sorted(root.rglob('*')):
  if p.is_file() and p.suffix in ('.json','.csv','.log','.txt'):
   n=p.relative_to(root).as_posix();data=p.read_bytes();files[n]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());tar.add(p,arcname=n,recursive=False)
print(json.dumps(dict(files=files,package=dict(bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()))))'''
    inventory=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    package=pf/'selected_expert_outlets_closed_20261011.tar.gz';assert not package.exists()
    command(['scp',*OPTIONS,'2026:/tmp/'+package.name,str(package)])
    assert package.stat().st_size==inventory['package']['bytes'] and hashlib.sha256(package.read_bytes()).hexdigest()==inventory['package']['sha256']
    with tarfile.open(package,'r:gz') as tar:
        assert set(tar.getnames())==set(inventory['files']);tar.extractall(LOCAL,filter='data')
    assert all((LOCAL/n).stat().st_size==f['bytes'] and hashlib.sha256((LOCAL/n).read_bytes()).hexdigest()==f['sha256'] for n,f in inventory['files'].items())
    code=f'''import hashlib
from pathlib import Path
for n,f in {restores!r}.items():
 p=Path(n);assert p.name=='best_official_arrays_restore_outlets_20261011.npz' and p.resolve().is_relative_to(Path({REMOTE!r}).resolve())
 assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];p.unlink()
Path('/tmp/selected_expert_outlets_closed_20261011.tar.gz').unlink()
print('ONLY_OWN_OUTLET_NORMAL_RESTORES_AND_PACKAGE_CLEARED')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True);package.unlink()
    state.update(status='ACTUAL_FOUR_SELECTED_P_R_FULL_NORMAL_OUTLET_GT_RAW_TEXT_COMPLETE',finished=datetime.now().astimezone().isoformat(timespec='seconds'),exit_code=0,raw=raw,text_manifest=inventory['files'],stage_cases=52,normal_restores_cleared=True,unique_raw_lost=0)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,'-X','utf8','-B','-S','analyze_selected_expert_outlets.py'],cwd=PROJECT,check=True)
    state['cpu_status']=load(LOCAL/'outlet_analysis/result.json')['status']
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(state['status'],state['cpu_status'],flush=True)


if __name__=='__main__':main()
