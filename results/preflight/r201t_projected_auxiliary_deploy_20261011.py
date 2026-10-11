"""One new MS mean-outlet receiver; preserve live RGB S and failed MS prefix."""
from datetime import datetime
import hashlib,json,os,shlex,shutil,subprocess,sys
from pathlib import Path
PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
TAG='r201t_projected_auxiliary_20261011';ROOT=REMOTE+'/runs/'+TAG
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')/TAG


def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    failed=load(pf/'r201s_MSVR_native_failure_text_intake_actual_20261011.json')
    assert failed['status']=='ACTUAL_S_MSVR_FAILED_NATIVE_FULL_TEXT_INTAKE_NO_FORMAL' and failed['inventory']['old_controller_absent']
    assert failed['inventory']['ordinary_exit']['exit_code']==1 and failed['inventory']['axis_exit']['exit_code']==0
    assert load(pf/'r201s_RGBNT100_native_actual_20261011.json')['actual_updates']==6
    rescue=load(pf/'r201s_MSVR_native_failure_rescue_20261011.json')
    assert rescue['scope']=='SOURCE_ONLY'
    review=load(pf/'r201t_projected_auxiliary_source_review_20261011.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources,reuse=review['sources_sha256'],review['directly_reused_sources_sha256']
    expected={'routed_projected_auxiliary_mean.py','run_r201t_projected_auxiliary_mean.py','launch_r201t_projected_auxiliary_mean.py','analyze_r201t_dataset_normal.py',
        'results/preflight/r201t_projected_auxiliary_deploy_20261011.py','results/preflight/r201t_dataset_normal_complete_intake_20261011.py','results/preflight/r201t_projected_auxiliary_plan_20261011.json'}
    assert set(sources)==expected and all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    proof=pf/'r201t_projected_auxiliary_actual_session_20261011.json';assert not proof.exists() and not ARCHIVE.exists()
    assert shutil.disk_usage(ARCHIVE.parent).free>2*1024**3
    protected={n:sha for n,sha in load(pf/'r201p_auxiliary_balance_actual_session_20261010.json')['protected_weights'].items()
        if n.endswith('/MSVR310_demo_s42/best.pth') or n.endswith('/RGBNT100_demo_s42/best.pth') or n.endswith('/RGBNT201_r201k_axis_shared_s42/best.pth')}
    assert len(protected)==3
    code=f'''import hashlib,json,shutil
from pathlib import Path
r=Path({REMOTE!r});assert not Path({ROOT!r}).exists()
assert not Path('/proc/'+str({failed['inventory']['old_controller_pid']!r})).exists()
assert shutil.disk_usage(r).free>1_800_000_000
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {reuse!r}.items() if not n.startswith('results/'))
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {protected!r}.items())
print(json.dumps(dict(project_free=shutil.disk_usage(r).free,protected_weights={protected!r},old_failed_MS_controller_absent=True,capacity_basis='Two MS best~0.81GB+normalraw~20MB+text/nativepackage+0.8GB headroom')))'''
    ready=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    (pf/'r201t_capacity_anchors_actual_20261011.json').write_text(json.dumps(ready,indent=2)+'\n',encoding='utf-8')
    for n in sources:
        if not n.startswith('results/'):command(['scp',*OPTIONS,str(PROJECT/n),'2026:'+REMOTE+'/'+n])
    code=f'''import hashlib
from pathlib import Path
r=Path({REMOTE!r});assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {(sources|reuse)!r}.items() if not n.startswith('results/'))
print('ACTUAL_REVIEWED_T_NEW_SOURCES_AND_UNCHANGED_LIVE_S_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    state=dict(status='ACTUAL_T_MS_RECEIVER_STARTED_NATIVE_PENDING',started=datetime.now().astimezone().isoformat(timespec='seconds'),receiver_pid=os.getpid(),physical_gpus=[0,1],maximum_NN=2,
        native_updates_closed=0,successful_updates_closed=0,RGB_S_original_receiver_unchanged=True,protected_weights=protected,sources_sha256=sources,goal='ACTIVE_UNMET')
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    out=ROOT+'/MSVR310'
    argv=[PYTHON,'-u','launch_r201t_projected_auxiliary_mean.py','--dataset','MSVR310','--data-root','/data/gaob/Re-ID/dataset',
        '--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt','--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',out]
    process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026','cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
    archives,cleared,native,finished={},set(),False,False
    for line in process.stdout:
        if not line.startswith('R201T_STREAM '):print(line,end='',flush=True);continue
        message=json.loads(line[len('R201T_STREAM '):]);event=message['event']
        if event=='NATIVE_PASS':
            assert not native and message['controls']==2 and message['actual_updates']==6 and message['partial_loss_effective_zero']
            assert message['direct_PM_PF_auxiliary_gradient_positive'] and message['projected_auxiliary_call_contract']
            assert message['final_unit_shift_epsilon']==.10 and message['controlled_final_shift_bound'] and message['PI_outlet']=='shared_relation_local'
            assert message['M_F_auxiliary_weight']==.01 and message['contribution_regression_weight']==0. and message['final_full_CE_weight']==.05
            native=True
            native_record=dict(status='ACTUAL_T_MS_NATIVE6_ACCEPTED_FORMAL_LAUNCH_PENDING',at=datetime.now().astimezone().isoformat(timespec='seconds'),**message)
            (pf/'r201t_MSVR310_native_actual_20261011.json').write_text(json.dumps(native_record,indent=2)+'\n',encoding='utf-8')
            state['native_updates_closed']=6;proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
            print('ACTUAL_T_NATIVE6_MEAN_DIRECT_PROJECTION_GRADIENT_PASS',flush=True)
        elif event=='NORMAL_READY':
            name=message['name'];assert native and name not in archives and name.startswith('MSVR310_r201t_') and name.endswith('_s42')
            assert shutil.disk_usage(ARCHIVE.parent).free>message['file']['bytes']+1024**3
            local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVE)
            archives[name]=dict(file=message['file'],local=local)
            process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED',name=name,file=message['file']))+'\n');process.stdin.flush()
        elif event=='NORMAL_CLEARED':
            name=message['name'];assert name in archives and name not in cleared;cleared.add(name)
            print('T_NORMAL_LOCAL_SHA_ACK_REMOTE_CLEARED',name,flush=True)
        else:
            assert event=='CONTROLLER_COMPLETE' and not finished and message['controls']==2 and message['successful_updates']==2000;finished=True
    process.stdin.close();assert process.wait()==0 and native and finished and set(archives)==cleared and len(archives)==2
    value=dict(status='ACTUAL_T_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW',finished=datetime.now().astimezone().isoformat(timespec='seconds'),exit_code=0,dataset='MSVR310',
        remote_root=out,archives=archives,native_updates=6,successful_updates=2000,additional_epochs=100,sources_sha256=sources,
        limits='New mean+two obsolete pool_score frozen1024 recovery; failed S native preserved/no stale acceptance or RGB source edits. Same MS parentR CE/reg/aux, framework/totalparams/5120/cap/P8K8/seed/BN. No201NN/new49 or gain claim before CPU readout.')
    (pf/'r201t_projected_auxiliary_MSVR310_actual_session_20261011.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,'-X','utf8','-B','-S',str(pf/'r201t_dataset_normal_complete_intake_20261011.py'),'--dataset','MSVR310'],check=True)
    subprocess.run([sys.executable,'-X','utf8','-B','-S',str(PROJECT/'analyze_r201t_dataset_normal.py'),'--dataset','MSVR310'],check=True)
    state.update(status='ACTUAL_T_MS_TWO_FULL50_GT_RAW_CPU_COMPLETE',finished=value['finished'],exit_code=0,dataset=value,native_updates=6,successful_updates=2000,additional_epochs=100,missing_evaluated=False,new49=False)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');print(state['status'],flush=True)


if __name__=='__main__':main()
