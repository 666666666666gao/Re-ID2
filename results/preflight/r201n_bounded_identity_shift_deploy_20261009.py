"""Run reviewed N bounded-update controls; preserve the selected RGBNT201 weights."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified

REMOTE='/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/r201n_bounded_identity_shift_20261009'
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201n_bounded_identity_shift_20261009')
UPDATES={'MSVR310':2000,'RGBNT100':13126}


def main():
    pf=PROJECT/'results/preflight'
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    parent_normal=load(pf/'r201m_primary_margin06_actual_session_20261008.json')
    assert parent_normal['status']=='ACTUAL_M_TWO_WEAK_NORMAL_DATASETS_FOUR_FULL50_GT_RAW_COMPLETE' and parent_normal['exit_code']==0
    assert parent_normal['successful_updates']==15126 and parent_normal['native_updates']==12
    parent_paper=load(pf/'r201m_paper_missing6_actual_session_20261009.json')
    assert parent_paper['status']=='ACTUAL_M_ALL_DECLARED_FIXED_BEST_PAPER6_GT_RAW_CPU_COMPLETE_K201_RETAINED'
    assert parent_paper['new_full_conditions']==24 and parent_paper['new_full_state_cases']==96 and not parent_paper['new_49_evaluation']
    parent_gradient=load(pf/'r201m_selected_task_gradients_actual_20261009.json')
    assert parent_gradient['status']=='ACTUAL_FOUR_SELECTED_M_TRAIN_ONLY_TASK_GRADIENT_DIAGNOSTICS_COMPLETE'
    assert parent_gradient['neural_forwards']==16 and parent_gradient['optimizer_updates']==0
    protected=dict(parent_normal['protected_weights'])
    for item in parent_gradient['rows']:
        key=item['job']['run']+'/best.pth'
        assert key.startswith(REMOTE+'/')
        protected[key[len(REMOTE)+1:]]=item['result']['protected_inputs'][key]
    assert len(protected)==20
    replay=load(pf/'r201l_uniform_K8_sampler_cpu_replay_20261007.json')
    assert replay['status']=='ACTUAL_EXISTING_SAMPLER_CPU_FULL50_OLD_AND_K8_REPLAY_COMPLETE'
    assert all(replay['datasets'][d]['existing_K_50_epoch_orders_exact'] and 2*replay['datasets'][d]['8']['total_steps']==steps for d,steps in UPDATES.items())
    plan=load(pf/'r201n_bounded_identity_shift_plan_20261009.json')
    assert plan['datasets_in_order']==list(UPDATES) and plan['planned_formal_updates']==sum(UPDATES.values())==15126
    review=load(pf/'r201n_bounded_identity_shift_source_review_20261009.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    expected={'bounded_identity_descriptor.py','run_r201n_bounded_identity_shift.py','launch_r201n_bounded_identity_shift.py','analyze_r201n_dataset_normal.py','results/preflight/r201n_bounded_identity_shift_deploy_20261009.py','results/preflight/r201n_dataset_normal_complete_intake_20261009.py'}
    sources,reused=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert set(sources)==expected and all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reused).items())
    proof=pf/'r201n_bounded_identity_shift_actual_session_20261009.json'
    assert not proof.exists() and not ARCHIVE.exists()
    assert shutil.disk_usage(ARCHIVE.parent).free>900_000_000
    code=f'''import hashlib,json,shutil
from pathlib import Path
root=Path({REMOTE!r});output=Path({ROOT!r})
assert not output.exists() and shutil.disk_usage(root).free>4_400_000_000
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {reused!r}.items() if not name.startswith('results/'))
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {protected!r}.items())
print(json.dumps(dict(status='ACTUAL_CLOSED_M_STORAGE_AND_N_REUSED_SOURCES_READY',free=shutil.disk_usage(root).free,protected_anchors={protected!r})))'''
    ready=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    assert ready['status']=='ACTUAL_CLOSED_M_STORAGE_AND_N_REUSED_SOURCES_READY'
    assert protected==ready['protected_anchors']
    (pf/'r201n_capacity_anchors_actual_20261009.json').write_text(json.dumps(ready,indent=2)+'\n',encoding='utf-8')
    print('ACTUAL_N_CAPACITY_AND_TWENTY_PROTECTED_WEIGHTS_READY',ready['free'],flush=True)
    for name in sources:
        if not name.startswith('results/'):
            command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+REMOTE+'/'+name])
    deployed={name:sha for name,sha in sources.items() if not name.startswith('results/')}
    code=f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r});assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {deployed!r}.items())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {protected!r}.items())
print('ACTUAL_REVIEWED_N_SOURCES_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),end='')
    outcomes={}
    state=dict(status='ACTUAL_N_REVIEWED_CAPACITY_AND_SOURCES_READY',started=datetime.now().isoformat(timespec='seconds'),receiver_pid=os.getpid(),native_updates_closed=0,formal_datasets_closed=[],new_formal_updates_closed=0,all_three_datasets_new_training=False,
        retained_RGBNT201='K_axis_shared_s42 E30, primary margin.3; no new201 training',protected_weights=protected)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    for dataset,updates in UPDATES.items():
        out=ROOT+'/'+dataset
        receipt=pf/('r201n_bounded_identity_shift_'+dataset+'_actual_session_20261009.json')
        assert not receipt.exists()
        argv=[PYTHON,'-u','launch_r201n_bounded_identity_shift.py','--dataset',dataset,
            '--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
            '--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',out]
        state.update(status='ACTUAL_N_DATASET_NATIVE_THEN_FULL50_RUNNING',current_dataset=dataset,started_current_dataset=datetime.now().isoformat(timespec='seconds'))
        proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026',
            'cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        archives,cleared,native,finished={},set(),False,False
        print('N_ORIGINAL_DATASET_CONTROLLER_STARTED',dataset,flush=True)
        for line in process.stdout:
            if not line.startswith('R201N_STREAM '):
                print(line,end='',flush=True)
                continue
            message=json.loads(line[len('R201N_STREAM '):])
            event=message['event']
            if event=='NATIVE_PASS':
                assert not native and message['controls']==2 and message['actual_updates']==6 and message['partial_loss_effective_zero'] and message['PI_outlet']=='shared_relation_local' and message['final_unit_shift_epsilon']==.10 and message['controlled_final_shift_bound']
                native=True
                (pf/('r201n_bounded_identity_shift_'+dataset+'_native_actual_20261009.json')).write_text(json.dumps(dict(
                    status='ACTUAL_TWO_N_NATIVE3_PRIMARY_MARGIN06_UNIFORM_P8K8_RELATION_PI_PASS_FORMAL_NEXT',observed_at=datetime.now().isoformat(timespec='seconds'),
                    dataset=dataset,remote_root=out,actual_native_updates=6,primary_margin=.6,auxiliary_soft_unchanged=True,PI_outlet='shared_relation_local',PI_native_input_isolation=True,controlled10_01_pre_bound_parent_equal=True,final_unit_shift_epsilon=.10,controlled_final_shift_bound=True,
                    partial_loss_effective_zero=True,normal_full50_complete=False),indent=2)+'\n',encoding='utf-8')
                state.update(status='ACTUAL_N_DATASET_NATIVE6_PASSED_FULL50_RUNNING',native_updates_closed=state['native_updates_closed']+6)
                proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
                print('ACTUAL_N_NATIVE6_PASS',dataset,flush=True)
            elif event=='NORMAL_READY':
                name=message['name']
                assert native and name.startswith(dataset+'_r201n_') and name not in archives
                assert shutil.disk_usage(ARCHIVE.parent).free>message['file']['bytes']
                local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVE)
                archives[name]=dict(file=message['file'],local=local)
                process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED',name=name,file=message['file']))+'\n')
                process.stdin.flush()
            elif event=='NORMAL_CLEARED':
                name=message['name']
                assert name in archives and name not in cleared
                cleared.add(name)
                print('N_NORMAL_LOCAL_SHA_ACK_REMOTE_CLEARED',dataset,name,flush=True)
            else:
                assert event=='CONTROLLER_COMPLETE' and not finished and message['controls']==2 and message['successful_updates']==updates
                finished=True
        process.stdin.close()
        assert process.wait()==0 and native and finished and set(archives)==cleared and len(archives)==2
        value=dict(status='ACTUAL_N_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW',finished=datetime.now().isoformat(timespec='seconds'),
            exit_code=0,dataset=dataset,remote_root=out,archives=archives,native_updates=6,successful_updates=updates,
            additional_epochs=100,sources_sha256=sources,limits='Weakdatasets only final unit5120 shift bound0.10 relative to M, same M hinge.6+CE.25/P8K8 orders/updates/K graph/auxsoft/partial0; paired M controls matched. Preserve K201 margin.3 best. No +1/missing/multiseed success assertion.')
        receipt.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        outcomes[dataset]=value
        print('ACTUAL_N_TWO_FULL50_CLOSED',dataset,flush=True)
        subprocess.run([sys.executable,'-X','utf8','-B','-S',str(pf/'r201n_dataset_normal_complete_intake_20261009.py'),'--dataset',dataset],check=True)
        subprocess.run([sys.executable,'-X','utf8','-B','-S',str(PROJECT/'analyze_r201n_dataset_normal.py'),'--dataset',dataset],check=True)
        state.update(status='ACTUAL_N_DATASET_FULL50_GT_RAW_TEXT_CPU_CLOSED',formal_datasets_closed=list(outcomes),new_formal_updates_closed=sum(v['successful_updates'] for v in outcomes.values()))
        proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    result=dict(status='ACTUAL_N_TWO_WEAK_NORMAL_DATASETS_FOUR_FULL50_GT_RAW_COMPLETE',finished=datetime.now().isoformat(timespec='seconds'),
        exit_code=0,datasets=outcomes,native_updates=12,successful_updates=15126,additional_epochs=200,
        sources_sha256=sources,all_three_datasets_new_training=False,missing_evaluated=False,retained_RGBNT201=state['retained_RGBNT201'],protected_weights=protected,
        limits='Weakdatasets only bounded final unit5120 update0.10 on unchanged M primarymargin.6, K201 best margin.3 retained under user-authorized dataset-specific training parameters. FourNEW50/15126updates, same M orders/auxsoft/graph/5120/teacher42. No new trainable parameters; inactive bound preserves exact candidate; physical2/3 only. Normal first, only paper-six masks later, no new49. Known loss control, no +1/missing/stability/causality success claim from closure.')
    proof.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('ACTUAL_N_TWO_WEAK_NORMAL_CONTROLLER_CLOSED',flush=True)


if __name__=='__main__':main()
