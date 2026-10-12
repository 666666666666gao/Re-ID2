"""AC train-only identity grouping: fresh paired native3 then full50."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib,json,os,shlex,shutil,subprocess,sys,threading
from pathlib import Path
PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
TAG='r201ad_RGB_precision_matched_20261012';ROOT=REMOTE+'/runs/'+TAG
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')/TAG
UPDATES={'RGBNT100':26252}


def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    for file in ('r201x_original_owner_normal_closed_actual_20261012.json','r201aa_original_owner_normal_closed_actual_20261012.json','r201ab_RGB_training_neighbors_original_owner_closed_actual_20261012.json'):
        closed=load(pf/file)
        assert closed['exit_code']==0 and closed['terminal_consumed'] and closed['original_process_absent']
    for file in ('r201ac_RGB_M_probe_original_owner_closed_actual_20261012.json','msvr_stronger_private_reference_original_owner_closed_actual_20261012.json'):
        closed=load(pf/file);assert closed['exit_code']==0 and closed['terminal_consumed'] and closed['original_process_absent'] and not closed['formal_training_admitted']
    replay=load(pf/'r201ac_RGB_neighbors_schedule_CPU_actual_20261012.json')
    assert replay['total_batches']==6563 and replay['each_epoch_record_multiset_exact'] and replay['new_random_draws']==0
    review=load(pf/'r201ad_RGB_precision_matched_source_review_20261012.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources,reuse=review['sources_sha256'],review['directly_reused_sources_sha256']
    expected={'rgb_auxiliary_modality_score_fp32.py','run_r201ad_RGB_random.py','run_r201ad_RGB_hard.py','launch_r201ad_RGB_precision_matched.py','analyze_r201ad_RGB_precision_matched_normal.py','results/preflight/r201ad_RGB_precision_matched_deploy_20261012.py','results/preflight/r201ad_RGB_normal_complete_intake_20261012.py','results/preflight/r201ad_RGB_precision_matched_plan_20261012.json'}
    assert set(sources)==expected and all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    proof=pf/'r201ad_RGB_precision_matched_actual_session_20261012.json';assert not proof.exists() and not ARCHIVE.exists()
    assert shutil.disk_usage(ARCHIVE.parent).free>3*1024**3
    protected={n:sha for n,sha in load(pf/'r201p_auxiliary_balance_actual_session_20261010.json')['protected_weights'].items()
        if n.endswith('/MSVR310_demo_s42/best.pth') or n.endswith('/RGBNT100_demo_s42/best.pth') or n.endswith('/RGBNT201_r201k_axis_shared_s42/best.pth')}
    protected['runs/full_official_baselines_20261004/training/MSVR310_demo_shared_s42/best.pth']='7744f4f7f796e17e778b2d23e5c025c2197029610634cac26c88ba9f73121fb3'
    assert len(protected)==4
    code=f'''import hashlib,json,shutil
from pathlib import Path
r=Path({REMOTE!r});assert not Path({ROOT!r}).exists()
assert shutil.disk_usage(r).free>3_300_000_000
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {reuse!r}.items() if not n.startswith('results/'))
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {protected!r}.items())
print(json.dumps(dict(project_free=shutil.disk_usage(r).free,protected_weights={protected!r},capacity_basis='FourRGBbest~1.62GB+four normal arrays up to1.1GB+text/nativepkg+headroom; source prelaunch requires3.3GBfree; no smoke/last weights')))'''
    ready=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    (pf/'r201ad_capacity_anchors_actual_20261012.json').write_text(json.dumps(ready,indent=2)+'\n',encoding='utf-8')
    print('ACTUAL_AD_CAPACITY_AND_FOUR_REQUIRED_WEIGHTS_READY',ready['project_free'],flush=True)
    for n in sources:
        if not n.startswith('results/'):command(['scp',*OPTIONS,str(PROJECT/n),'2026:'+REMOTE+'/'+n])
    code=f'''import hashlib
from pathlib import Path
r=Path({REMOTE!r});assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {(sources|reuse)!r}.items() if not n.startswith('results/'))
print('ACTUAL_REVIEWED_AD_AND_UNCHANGED_PARENT_SOURCES_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    state=dict(status='ACTUAL_AD_RGB_FOUR_CARD_NATIVE_THEN_NORMAL_FULL50_STARTED',started=datetime.now().astimezone().isoformat(timespec='seconds'),receiver_pid=os.getpid(),
        physical_gpus=[0,1,2,3],maximum_NN=4,formal_datasets_closed=[],new_formal_updates_closed=0,native_updates_closed=0,
        retained_RGBNT201='K42E30 untouched/no201NN',protected_weights=protected,sources_sha256=sources,goal='ACTIVE_UNMET')
    lock=threading.Lock();outcomes={};proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    def worker(dataset):
        updates=UPDATES[dataset];out=ROOT+'/'+dataset;main_CE={'MSVR310':.05,'RGBNT100':.25}[dataset];reg={'MSVR310':0.,'RGBNT100':.05}[dataset]
        receipt=pf/('r201ad_RGB_precision_matched_'+dataset+'_actual_session_20261012.json');assert not receipt.exists()
        argv=[PYTHON,'-u','launch_r201ad_RGB_precision_matched.py','--dataset',dataset,'--data-root','/data/gaob/Re-ID/dataset',
            '--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt','--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',out]
        process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026','cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        archives,cleared,native,finished={},set(),False,False
        print('AD_ORIGINAL_DATASET_CONTROLLER_STARTED',dataset,flush=True)
        for line in process.stdout:
            if not line.startswith('R201AD_STREAM '):print(line,end='',flush=True);continue
            message=json.loads(line[len('R201AD_STREAM '):]);event=message['event']
            if event=='NATIVE_PASS':
                assert not native and message['controls']==4 and message['actual_updates']==12 and message['partial_loss_effective_zero']
                assert message['direct_PM_PF_auxiliary_gradient_positive'] and message['projected_auxiliary_call_contract']
                assert message['final_unit_shift_epsilon']==.10 and message['controlled_final_shift_bound'] and message['PI_outlet']=='shared_relation_local'
                assert message['M_F_auxiliary_weight']==.001 and message['contribution_regression_weight']==reg and message['final_full_CE_weight']==main_CE
                assert message['F_auxiliary_scorer_FP32'] and message['M_auxiliary_scorer_FP32']
                native=True
                native_record=dict(status='ACTUAL_AD_DATASET_NATIVE12_ACCEPTED_FORMAL_LAUNCH_PENDING',at=datetime.now().astimezone().isoformat(timespec='seconds'),dataset=dataset,**message)
                (pf/('r201ad_'+dataset+'_native_actual_20261012.json')).write_text(json.dumps(native_record,indent=2)+'\n',encoding='utf-8')
                with lock:
                    state['native_updates_closed']+=12;proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
                print('ACTUAL_AD_NATIVE12_DIRECT_PROJECTION_GRADIENT_PASS',dataset,flush=True)
            elif event=='NORMAL_READY':
                name=message['name'];assert native and name not in archives and name.startswith(dataset+'_r201ad_') and name.endswith('_s42')
                assert shutil.disk_usage(ARCHIVE.parent).free>message['file']['bytes']+1024**3
                local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVE)
                archives[name]=dict(file=message['file'],local=local)
                process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED',name=name,file=message['file']))+'\n');process.stdin.flush()
            elif event=='NORMAL_CLEARED':
                name=message['name'];assert name in archives and name not in cleared;cleared.add(name)
                print('AD_NORMAL_LOCAL_SHA_ACK_REMOTE_CLEARED',dataset,name,flush=True)
            else:
                assert event=='CONTROLLER_COMPLETE' and not finished and message['controls']==4 and message['successful_updates']==updates;finished=True
        process.stdin.close();assert process.wait()==0 and native and finished and set(archives)==cleared and len(archives)==4
        value=dict(status='ACTUAL_AD_FOUR_FULL50_NORMAL_GT_AND_LOCAL_RAW',finished=datetime.now().astimezone().isoformat(timespec='seconds'),exit_code=0,dataset=dataset,
            remote_root=out,archives=archives,native_updates=12,successful_updates=updates,additional_epochs=200,sources_sha256=sources,
            limits='Only train-only identity grouping differs within newly matched random/hard AD controls; all M/Fauxscorers FP32. OldX MAMP is historical. Original sampler RNG first, deterministic top3 swaps, same per-epoch record multiset/P8K8/6563/partial sets. All X model/loss/FP32/BN2/cap/teacher42/freshheads42/newAdam/additional50 held. Allnegatives retained, no missing专项/201NN/49/failedseed search or guaranteed gain.')
        receipt.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        subprocess.run([sys.executable,'-X','utf8','-B','-S',str(pf/'r201ad_RGB_normal_complete_intake_20261012.py'),'--dataset',dataset],check=True)
        subprocess.run([sys.executable,'-X','utf8','-B','-S',str(PROJECT/'analyze_r201ad_RGB_precision_matched_normal.py'),'--dataset',dataset],check=True)
        with lock:
            outcomes[dataset]=value;state.update(status='ACTUAL_AD_RGB_NORMAL_DATASET_GT_RAW_CPU_CLOSED',formal_datasets_closed=list(outcomes),new_formal_updates_closed=sum(v['successful_updates'] for v in outcomes.values()))
            proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        print('ACTUAL_AD_FOUR_FULL50_GT_RAW_CPU_CLOSED',dataset,flush=True)
    worker('RGBNT100')
    assert set(outcomes)==set(UPDATES)
    state.update(status='ACTUAL_AD_RGB_NORMAL_FOUR_FULL50_GT_RAW_CPU_COMPLETE',finished=datetime.now().astimezone().isoformat(timespec='seconds'),exit_code=0,datasets=outcomes,
        successful_updates=sum(UPDATES.values()),native_updates=12,additional_epochs=200,missing_evaluated=False,new49=False)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');print(state['status'],flush=True)


if __name__=='__main__':main()
