"""S paired native3 then full50 on all four2026 cards; sole archive receiver."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib,json,os,shlex,shutil,subprocess,sys,threading
from pathlib import Path
PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
TAG='r201w_reduced_auxiliary_20261011';ROOT=REMOTE+'/runs/'+TAG
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')/TAG
UPDATES={'MSVR310':2000,'RGBNT100':13126}


def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    assert load(pf/'r201v_paper6_original_owner_closed_actual_20261011.json')['exit_code']==0
    assert load(pf/'r201v_partial_relation_paper6_actual_session_20261011.json')['status']=='ACTUAL_V_ALL_DECLARED_FIXED_BEST_PAPER6_GT_RAW_CPU_COMPLETE_K201_RETAINED'
    assert load(pf/'r201t_original_owner_normal_closed_actual_20261011.json')['exit_code']==0
    s=load(pf/'r201s_original_owner_RGB_complete_MS_native_failure_closed_actual_20261011.json')
    assert s['exit_code']==1 and s['RGB_normal_GT_raw_CPU_complete'] and not s['MS_formal_started']
    assert load(pf/'r201u_obsolete_remote_best_local_archive_actual_20261011.json')['unique_weights_lost']==0
    review=load(pf/'r201w_reduced_auxiliary_source_review_20261011.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources,reuse=review['sources_sha256'],review['directly_reused_sources_sha256']
    expected={'run_r201w_reduced_auxiliary.py', 'launch_r201w_reduced_auxiliary.py', 'results/preflight/r201w_reduced_auxiliary_deploy_20261011.py', 'analyze_r201w_dataset_normal.py', 'results/preflight/r201w_dataset_normal_complete_intake_20261011.py', 'results/preflight/r201w_reduced_auxiliary_plan_20261011.json'}
    assert set(sources)==expected and all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    proof=pf/'r201w_reduced_auxiliary_actual_session_20261011.json';assert not proof.exists() and not ARCHIVE.exists()
    assert shutil.disk_usage(ARCHIVE.parent).free>2*1024**3
    protected={n:sha for n,sha in load(pf/'r201p_auxiliary_balance_actual_session_20261010.json')['protected_weights'].items()
        if n.endswith('/MSVR310_demo_s42/best.pth') or n.endswith('/RGBNT100_demo_s42/best.pth') or n.endswith('/RGBNT201_r201k_axis_shared_s42/best.pth')}
    assert len(protected)==3
    code=f'''import hashlib,json,shutil
from pathlib import Path
r=Path({REMOTE!r});assert not Path({ROOT!r}).exists()
assert shutil.disk_usage(r).free>3_100_000_000
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {reuse!r}.items() if not n.startswith('results/'))
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {protected!r}.items())
print(json.dumps(dict(project_free=shutil.disk_usage(r).free,protected_weights={protected!r},capacity_basis='Fourbest~1.62GB+twoRGB/twoMSnormalarrays~0.57GB+text/nativepkg+0.8GBheadroom; no smoke/last weights')))'''
    ready=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    (pf/'r201w_capacity_anchors_actual_20261011.json').write_text(json.dumps(ready,indent=2)+'\n',encoding='utf-8')
    print('ACTUAL_W_CAPACITY_AND_THREE_REQUIRED_WEIGHTS_READY',ready['project_free'],flush=True)
    for n in sources:
        if not n.startswith('results/'):command(['scp',*OPTIONS,str(PROJECT/n),'2026:'+REMOTE+'/'+n])
    code=f'''import hashlib
from pathlib import Path
r=Path({REMOTE!r});assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {(sources|reuse)!r}.items() if not n.startswith('results/'))
print('ACTUAL_REVIEWED_S_AND_UNCHANGED_P_R_SOURCES_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    state=dict(status='ACTUAL_W_FOUR_CARD_NATIVE_THEN_NORMAL_FULL50_STARTED',started=datetime.now().astimezone().isoformat(timespec='seconds'),receiver_pid=os.getpid(),
        physical_gpus=[0,1,2,3],maximum_NN=4,formal_datasets_closed=[],new_formal_updates_closed=0,native_updates_closed=0,
        retained_RGBNT201='K42E30 untouched/no201NN',protected_weights=protected,sources_sha256=sources,goal='ACTIVE_UNMET')
    lock=threading.Lock();outcomes={};proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    def worker(dataset):
        updates=UPDATES[dataset];out=ROOT+'/'+dataset;main_CE={'MSVR310':.05,'RGBNT100':.25}[dataset];reg={'MSVR310':0.,'RGBNT100':.05}[dataset]
        receipt=pf/('r201w_reduced_auxiliary_'+dataset+'_actual_session_20261011.json');assert not receipt.exists()
        argv=[PYTHON,'-u','launch_r201w_reduced_auxiliary.py','--dataset',dataset,'--data-root','/data/gaob/Re-ID/dataset',
            '--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt','--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',out]
        process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026','cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        archives,cleared,native,finished={},set(),False,False
        print('S_ORIGINAL_DATASET_CONTROLLER_STARTED',dataset,flush=True)
        for line in process.stdout:
            if not line.startswith('R201W_STREAM '):print(line,end='',flush=True);continue
            message=json.loads(line[len('R201W_STREAM '):]);event=message['event']
            if event=='NATIVE_PASS':
                assert not native and message['controls']==2 and message['actual_updates']==6 and message['partial_loss_effective_zero']
                assert message['direct_PM_PF_auxiliary_gradient_positive'] and message['projected_auxiliary_call_contract']
                assert message['final_unit_shift_epsilon']==.10 and message['controlled_final_shift_bound'] and message['PI_outlet']=='shared_relation_local'
                assert message['M_F_auxiliary_weight']==.001 and message['contribution_regression_weight']==reg and message['final_full_CE_weight']==main_CE
                native=True
                native_record=dict(status='ACTUAL_W_DATASET_NATIVE6_ACCEPTED_FORMAL_LAUNCH_PENDING',at=datetime.now().astimezone().isoformat(timespec='seconds'),dataset=dataset,**message)
                (pf/('r201w_'+dataset+'_native_actual_20261011.json')).write_text(json.dumps(native_record,indent=2)+'\n',encoding='utf-8')
                with lock:
                    state['native_updates_closed']+=6;proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
                print('ACTUAL_W_NATIVE6_DIRECT_PROJECTION_GRADIENT_PASS',dataset,flush=True)
            elif event=='NORMAL_READY':
                name=message['name'];assert native and name not in archives and name.startswith(dataset+'_r201w_') and name.endswith('_s42')
                assert shutil.disk_usage(ARCHIVE.parent).free>message['file']['bytes']+1024**3
                local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVE)
                archives[name]=dict(file=message['file'],local=local)
                process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED',name=name,file=message['file']))+'\n');process.stdin.flush()
            elif event=='NORMAL_CLEARED':
                name=message['name'];assert name in archives and name not in cleared;cleared.add(name)
                print('S_NORMAL_LOCAL_SHA_ACK_REMOTE_CLEARED',dataset,name,flush=True)
            else:
                assert event=='CONTROLLER_COMPLETE' and not finished and message['controls']==2 and message['successful_updates']==updates;finished=True
        process.stdin.close();assert process.wait()==0 and native and finished and set(archives)==cleared and len(archives)==2
        value=dict(status='ACTUAL_W_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW',finished=datetime.now().astimezone().isoformat(timespec='seconds'),exit_code=0,dataset=dataset,
            remote_root=out,archives=archives,native_updates=6,successful_updates=updates,additional_epochs=100,sources_sha256=sources,
            limits='OnlyMFaux .001 relative TMS/SRGB; same existing outlets/frozen scorers/mainCE/reg/5120/params/order/BN/cap50budget. Fullnormal first; K201protected/no201NN/new49/missing reselect/failedseed search. Compare completedsamebudget parent controls andoriginalDeMo, allnegatives retained.')
        receipt.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        subprocess.run([sys.executable,'-X','utf8','-B','-S',str(pf/'r201w_dataset_normal_complete_intake_20261011.py'),'--dataset',dataset],check=True)
        subprocess.run([sys.executable,'-X','utf8','-B','-S',str(PROJECT/'analyze_r201w_dataset_normal.py'),'--dataset',dataset],check=True)
        with lock:
            outcomes[dataset]=value;state.update(status='ACTUAL_W_NORMAL_DATASET_CLOSED_OTHER_ORIGINAL_LANES_CONTINUE',formal_datasets_closed=list(outcomes),new_formal_updates_closed=sum(v['successful_updates'] for v in outcomes.values()))
            proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        print('ACTUAL_W_TWO_FULL50_GT_RAW_CPU_CLOSED',dataset,flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(worker,UPDATES))
    assert set(outcomes)==set(UPDATES)
    state.update(status='ACTUAL_W_TWO_WEAK_NORMAL_FOUR_FULL50_GT_RAW_CPU_COMPLETE',finished=datetime.now().astimezone().isoformat(timespec='seconds'),exit_code=0,datasets=outcomes,
        successful_updates=sum(UPDATES.values()),native_updates=12,additional_epochs=200,missing_evaluated=False,new49=False)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');print(state['status'],flush=True)


if __name__=='__main__':main()
