"""W reduced auxiliary weights: exact T MS/S RGB paired native3 and full50."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib,json,os,shlex,shutil,subprocess,sys,threading
from pathlib import Path
PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
TAG='r201z_MS_modality_adapt_20261012';ROOT=REMOTE+'/runs/'+TAG
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')/TAG
UPDATES={'MSVR310':1410}


def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    closed=load(pf/'r201y_original_owner_normal_closed_actual_20261012.json')
    assert closed['exit_code']==0 and closed['terminal_consumed'] and closed['original_process_absent'] and closed['normal_GT_raw_CPU_complete'] and closed['formal_updates']==1410
    assert load(pf/'r201w_obsolete_remote_best_local_archive_actual_20261012.json')['unique_weights_lost']==0
    review=load(pf/'r201z_MS_modality_adapt_source_review_20261012.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources,reuse=review['sources_sha256'],review['directly_reused_sources_sha256']
    expected={'launch_r201z_MS_modality_adapt.py', 'results/preflight/r201z_MS_modality_adapt_normal_complete_intake_20261012.py', 'analyze_r201z_MS_modality_adapt_normal.py', 'adapt_existing_modality_fusion.py', 'run_r201z_MS_modality_adapt.py', 'results/preflight/r201z_MS_modality_adapt_deploy_20261012.py', 'results/preflight/r201z_MS_modality_adapt_plan_20261012.json'}
    assert set(sources)==expected and all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    proof=pf/'r201z_MS_modality_adapt_actual_session_20261012.json';assert not proof.exists() and not ARCHIVE.exists()
    assert shutil.disk_usage(ARCHIVE.parent).free>2*1024**3
    protected={n:sha for n,sha in load(pf/'r201p_auxiliary_balance_actual_session_20261010.json')['protected_weights'].items()
        if n.endswith('/MSVR310_demo_s42/best.pth') or n.endswith('/RGBNT100_demo_s42/best.pth') or n.endswith('/RGBNT201_r201k_axis_shared_s42/best.pth')}
    assert len(protected)==3
    code=f'''import hashlib,json,shutil
from pathlib import Path
r=Path({REMOTE!r});assert not Path({ROOT!r}).exists()
assert shutil.disk_usage(r).free>1_800_000_000
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {reuse!r}.items() if not n.startswith('results/'))
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {protected!r}.items())
print(json.dumps(dict(project_free=shutil.disk_usage(r).free,protected_weights={protected!r},capacity_basis='TwoMSbest~0.81GB+two normal arrays<0.2GB+text/nativepkg+0.7GBheadroom; X RGB two-card lane separate, initial quartet budget verified before both launches')))'''
    ready=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    (pf/'r201z_capacity_anchors_actual_20261012.json').write_text(json.dumps(ready,indent=2)+'\n',encoding='utf-8')
    print('ACTUAL_Z_CAPACITY_AND_THREE_REQUIRED_WEIGHTS_READY',ready['project_free'],flush=True)
    for n in sources:
        if not n.startswith('results/'):command(['scp',*OPTIONS,str(PROJECT/n),'2026:'+REMOTE+'/'+n])
    code=f'''import hashlib
from pathlib import Path
r=Path({REMOTE!r});assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {(sources|reuse)!r}.items() if not n.startswith('results/'))
print('ACTUAL_REVIEWED_W_AND_UNCHANGED_T_S_SOURCES_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    state=dict(status='ACTUAL_Z_REVIEWED_READY_RECEIVER_CONTROLLER_LAUNCH_PENDING_NOT_FULL50',started=datetime.now().astimezone().isoformat(timespec='seconds'),receiver_pid=os.getpid(),
        physical_gpus=[0,1],maximum_NN=2,formal_datasets_closed=[],new_formal_updates_closed=0,native_updates_closed=0,
        retained_RGBNT201='K42E30 untouched/no201NN',protected_weights=protected,sources_sha256=sources,goal='ACTIVE_UNMET')
    lock=threading.Lock();outcomes={};proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    def worker(dataset):
        updates=UPDATES[dataset];out=ROOT+'/'+dataset;main_CE={'MSVR310':.05,'RGBNT100':.25}[dataset];reg={'MSVR310':0.,'RGBNT100':.05}[dataset]
        receipt=pf/('r201z_MS_modality_adapt_'+dataset+'_actual_session_20261012.json');assert not receipt.exists()
        argv=[PYTHON,'-u','launch_r201z_MS_modality_adapt.py','--dataset',dataset,'--data-root','/data/gaob/Re-ID/dataset',
            '--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt','--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',out]
        process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026','cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        archives,cleared,native,finished={},set(),False,False
        print('Z_ORIGINAL_DATASET_CONTROLLER_STARTED',dataset,flush=True)
        for line in process.stdout:
            if not line.startswith('R201Z_STREAM '):print(line,end='',flush=True);continue
            message=json.loads(line[len('R201Z_STREAM '):]);event=message['event']
            if event=='NATIVE_PASS':
                assert not native and message['controls']==2 and message['actual_updates']==6 and message['partial_loss_effective_zero']
                assert message['direct_PM_PF_auxiliary_gradient_positive'] and message['projected_auxiliary_call_contract']
                assert message['final_unit_shift_epsilon']==.10 and message['controlled_final_shift_bound'] and message['PI_outlet']=='shared_relation_local'
                assert message['M_F_auxiliary_weight']==.001 and message['contribution_regression_weight']==reg and message['final_full_CE_weight']==main_CE
                native=True
                native_record=dict(status='ACTUAL_Z_DATASET_NATIVE6_ACCEPTED_FORMAL_LAUNCH_PENDING',at=datetime.now().astimezone().isoformat(timespec='seconds'),dataset=dataset,**message)
                (pf/('r201z_'+dataset+'_native_actual_20261012.json')).write_text(json.dumps(native_record,indent=2)+'\n',encoding='utf-8')
                with lock:
                    state['native_updates_closed']+=6;proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
                print('ACTUAL_Z_NATIVE6_DIRECT_PROJECTION_GRADIENT_PASS',dataset,flush=True)
            elif event=='NORMAL_READY':
                name=message['name'];assert native and name not in archives and name.startswith(dataset+'_r201z_') and name.endswith('_s42')
                assert shutil.disk_usage(ARCHIVE.parent).free>message['file']['bytes']+1024**3
                local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVE)
                archives[name]=dict(file=message['file'],local=local)
                process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED',name=name,file=message['file']))+'\n');process.stdin.flush()
            elif event=='NORMAL_CLEARED':
                name=message['name'];assert name in archives and name not in cleared;cleared.add(name)
                print('Z_NORMAL_LOCAL_SHA_ACK_REMOTE_CLEARED',dataset,name,flush=True)
            else:
                assert event=='CONTROLLER_COMPLETE' and not finished and message['controls']==2 and message['successful_updates']==updates;finished=True
        process.stdin.close();assert process.wait()==0 and native and finished and set(archives)==cleared and len(archives)==2
        value=dict(status='ACTUAL_Z_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW',finished=datetime.now().astimezone().isoformat(timespec='seconds'),exit_code=0,dataset=dataset,
            remote_root=out,archives=archives,native_updates=6,successful_updates=updates,additional_epochs=100,sources_sha256=sources,
            limits='Z versus completed Y: only existing generalFusion HDM/ATMoE9958912 parameter gradients enabled; all five original modules stay eval, four other identity modules and all generalFusion buffers retained. Same P16K4/705updates per arm/additional50/newAdam/mean512/MFaux.001/mainCE.05/reg0/hinge.6/cap.10/PI/FFT/experts/router/gates/5120/BN2/partial0/two backwards. Total100821638 unchanged; active3122822->13081734, both Z controls matched but not active-budget matched to Y. 00 is adapting base; cap.10 references current base, not the original teacher. OriginalDeMo42 teacher50 plus expert50; normalbenchmarkmAP-best earliesttie/samecheckpoint allsix/fullofficial1032train591Q1055G/noholdout/GT/rawCPU. Native3 each before50; GF two-parameter early gradient diagnostics are observational, no global cause claim. No201NN/new49/missing selection/failedseedsearch; X GPU2/3 unchanged; K201 protected.')
        receipt.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        subprocess.run([sys.executable,'-X','utf8','-B','-S',str(pf/'r201z_MS_modality_adapt_normal_complete_intake_20261012.py'),'--dataset',dataset],check=True)
        subprocess.run([sys.executable,'-X','utf8','-B','-S',str(PROJECT/'analyze_r201z_MS_modality_adapt_normal.py'),'--dataset',dataset],check=True)
        with lock:
            outcomes[dataset]=value;state.update(status='ACTUAL_Z_NORMAL_DATASET_CLOSED_OTHER_ORIGINAL_LANES_CONTINUE',formal_datasets_closed=list(outcomes),new_formal_updates_closed=sum(v['successful_updates'] for v in outcomes.values()))
            proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        print('ACTUAL_Z_TWO_FULL50_GT_RAW_CPU_CLOSED',dataset,flush=True)
    worker('MSVR310')
    assert set(outcomes)==set(UPDATES)
    state.update(status='ACTUAL_Z_MS_NORMAL_TWO_FULL50_GT_RAW_CPU_COMPLETE',finished=datetime.now().astimezone().isoformat(timespec='seconds'),exit_code=0,datasets=outcomes,
        successful_updates=sum(UPDATES.values()),native_updates=6,additional_epochs=100,missing_evaluated=False,new49=False)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');print(state['status'],flush=True)


if __name__=='__main__':main()
