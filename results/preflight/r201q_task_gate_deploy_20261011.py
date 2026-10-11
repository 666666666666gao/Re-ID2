"""Q MSVR native6/full50 on GPU0/1; keep original P normal and paper queue GPU2/3 unchanged."""
from datetime import datetime
import hashlib,json,os,shlex,shutil,subprocess,sys
from pathlib import Path
PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/r201q_task_gate_20261011'
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201q_task_gate_20261011')
UPDATES={'MSVR310':2000}

def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    parent_normal=load(pf/'r201p_auxiliary_balance_MSVR310_actual_session_20261010.json')
    assert parent_normal['status']=='ACTUAL_P_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW' and parent_normal['exit_code']==0
    assert parent_normal['successful_updates']==2000 and parent_normal['native_updates']==6
    parent_cpu=load(PROJECT/'results/r201p_auxiliary_balance_20261010/MSVR310/normal_analysis/result.json')
    assert parent_cpu['status']=='ACTUAL_P_DATASET_TWO_FULL50_NORMAL_CPU_READOUT' and parent_cpu['official_GT_six_CMC50_perquery_groups']
    gradients=load(pf/'r201p_MSVR310_selected_gradients_actual_20261011.json')
    assert gradients['status']=='ACTUAL_TWO_SELECTED_P_TRAIN_ONLY_TASK_GRADIENT_DIAGNOSTICS_COMPLETE'
    assert gradients['optimizer_updates']==gradients['checkpoint_writes']==0 and gradients['batches']==8
    old_weights=load(pf/'r201o_same_state_contribution_actual_session_20261010.json')['protected_weights']
    protected={n:s for n,s in old_weights.items() if n.endswith('/MSVR310_demo_s42/best.pth') or n.endswith('/RGBNT201_r201k_axis_shared_s42/best.pth')}
    assert len(protected)==2
    protected.update({Path(n).relative_to(Path(REMOTE)).as_posix():s for r in gradients['rows'] for n,s in r['result']['protected_inputs'].items() if n.endswith('/best.pth')})
    assert len(protected)==4
    replay=load(pf/'r201l_uniform_K8_sampler_cpu_replay_20261007.json')
    assert replay['status']=='ACTUAL_EXISTING_SAMPLER_CPU_FULL50_OLD_AND_K8_REPLAY_COMPLETE'
    assert replay['datasets']['MSVR310']['existing_K_50_epoch_orders_exact'] and 2*replay['datasets']['MSVR310']['8']['total_steps']==2000
    plan=load(pf/'r201q_task_gate_plan_20261011.json');assert plan['datasets']==['MSVR310']
    review=load(pf/'r201q_task_gate_source_review_20261011.json');assert review['status']=='PASS' and not review['blocking_findings']
    expected={'run_r201q_task_gate.py','launch_r201q_task_gate.py','analyze_r201q_dataset_normal.py','results/preflight/r201q_task_gate_deploy_20261011.py','results/preflight/r201q_dataset_normal_complete_intake_20261011.py','results/preflight/r201q_task_gate_plan_20261011.json'}
    sources,reused=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert set(sources)==expected and all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==s for n,s in (sources|reused).items())
    proof=pf/'r201q_task_gate_actual_session_20261011.json';assert not proof.exists() and not ARCHIVE.exists()
    assert shutil.disk_usage(ARCHIVE.parent).free>900_000_000
    code=f"""import hashlib,json,shutil
from pathlib import Path
root=Path({REMOTE!r});output=Path({ROOT!r});assert not output.exists() and shutil.disk_usage(root).free>4_400_000_000
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==s for n,s in {reused!r}.items() if not n.startswith('results/'))
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==s for n,s in {protected!r}.items())
print(json.dumps(dict(status='ACTUAL_Q_P_CLOSED_MSVR_STORAGE_AND_REUSED_SOURCES_READY',free=shutil.disk_usage(root).free,protected_anchors={protected!r})))"""
    ready=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    assert protected==ready['protected_anchors'];(pf/'r201q_capacity_anchors_actual_20261011.json').write_text(json.dumps(ready,indent=2)+'\n',encoding='utf-8')
    print('ACTUAL_Q_CAPACITY_AND_REQUIRED_WEIGHTS_READY',ready['free'],flush=True)
    for name in sources:
        if not name.startswith('results/'):command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+REMOTE+'/'+name])
    deployed={n:s for n,s in sources.items() if not n.startswith('results/')}
    code=f"""import hashlib
from pathlib import Path
root=Path({REMOTE!r});assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==s for n,s in {deployed!r}.items())
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==s for n,s in {reused!r}.items() if not n.startswith('results/'))
print('ACTUAL_REVIEWED_Q_AND_UNCHANGED_P_SOURCES_EXACT')"""
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),end='')
    outcomes={}
    state=dict(status='ACTUAL_Q_REVIEWED_CAPACITY_AND_SOURCES_READY',started=datetime.now().isoformat(timespec='seconds'),receiver_pid=os.getpid(),native_updates_closed=0,formal_datasets_closed=[],new_formal_updates_closed=0,physical_gpus=[0,1],maximum_total_NN_with_P=4,all_three_datasets_new_training=False,retained_RGBNT201='K42E30 unchanged',protected_weights=protected)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    for dataset,updates in UPDATES.items():
        out=ROOT+'/'+dataset
        receipt=pf/('r201q_task_gate_'+dataset+'_actual_session_20261011.json')
        assert not receipt.exists()
        argv=[PYTHON,'-u','launch_r201q_task_gate.py','--dataset',dataset,
            '--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
            '--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',out]
        state.update(status='ACTUAL_Q_DATASET_NATIVE_THEN_FULL50_RUNNING',current_dataset=dataset,started_current_dataset=datetime.now().isoformat(timespec='seconds'))
        proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026',
            'cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        archives,cleared,native,finished={},set(),False,False
        print('Q_ORIGINAL_DATASET_CONTROLLER_STARTED',dataset,flush=True)
        for line in process.stdout:
            if not line.startswith('R201Q_STREAM '):
                print(line,end='',flush=True)
                continue
            message=json.loads(line[len('R201Q_STREAM '):])
            event=message['event']
            if event=='NATIVE_PASS':
                assert not native and message['controls']==2 and message['actual_updates']==6 and message['partial_loss_effective_zero'] and message['PI_outlet']=='shared_relation_local' and message['final_unit_shift_epsilon']==.10 and message['controlled_final_shift_bound'] and message['M_F_auxiliary_weight']=={'MSVR310':.01,'RGBNT100':.05}[dataset] and message['contribution_regression_weight']==0
                native=True
                (pf/('r201q_task_gate_'+dataset+'_native_actual_20261011.json')).write_text(json.dumps(dict(
                    status='ACTUAL_TWO_Q_NATIVE3_AUXILIARY_WEIGHT_PASS_FORMAL_NEXT',observed_at=datetime.now().isoformat(timespec='seconds'),
                    dataset=dataset,remote_root=out,actual_native_updates=6,contribution_regression_weight=0.,M_F_auxiliary_weight={'MSVR310':.01,'RGBNT100':.05}[dataset],primary_margin=.6,auxiliary_soft_unchanged=True,PI_outlet='shared_relation_local',PI_native_input_isolation=True,controlled10_01_pre_bound_parent_equal=True,final_unit_shift_epsilon=.10,controlled_final_shift_bound=True,
                    partial_loss_effective_zero=True,normal_full50_complete=False),indent=2)+'\n',encoding='utf-8')
                state.update(status='ACTUAL_Q_DATASET_NATIVE6_PASSED_FULL50_RUNNING',native_updates_closed=state['native_updates_closed']+6)
                proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
                print('ACTUAL_Q_NATIVE6_PASS',dataset,flush=True)
            elif event=='NORMAL_READY':
                name=message['name']
                assert native and name.startswith(dataset+'_r201q_') and name not in archives
                assert shutil.disk_usage(ARCHIVE.parent).free>message['file']['bytes']
                local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVE)
                archives[name]=dict(file=message['file'],local=local)
                process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED',name=name,file=message['file']))+'\n')
                process.stdin.flush()
            elif event=='NORMAL_CLEARED':
                name=message['name']
                assert name in archives and name not in cleared
                cleared.add(name)
                print('Q_NORMAL_LOCAL_SHA_ACK_REMOTE_CLEARED',dataset,name,flush=True)
            else:
                assert event=='CONTROLLER_COMPLETE' and not finished and message['controls']==2 and message['successful_updates']==updates
                finished=True
        process.stdin.close()
        assert process.wait()==0 and native and finished and set(archives)==cleared and len(archives)==2
        value=dict(status='ACTUAL_Q_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW',finished=datetime.now().isoformat(timespec='seconds'),
            exit_code=0,dataset=dataset,remote_root=out,archives=archives,native_updates=6,successful_updates=updates,
            additional_epochs=100,sources_sha256=sources,limits='Only contribution regression.05->0 relative to P MSVR; same task gates, MFaux.01, physical0/1 and liveP2/3 unchanged, unchanged final unit5120 cap0.10, same M hinge.6+CE.25/P8K8 orders/updates/K graph/auxsoft/partial0; paired M controls matched. Preserve K201 margin.3 best. No +1/missing/multiseed success assertion.')
        receipt.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        outcomes[dataset]=value
        print('ACTUAL_Q_TWO_FULL50_CLOSED',dataset,flush=True)
        subprocess.run([sys.executable,'-X','utf8','-B','-S',str(pf/'r201q_dataset_normal_complete_intake_20261011.py'),'--dataset',dataset],check=True)
        subprocess.run([sys.executable,'-X','utf8','-B','-S',str(PROJECT/'analyze_r201q_dataset_normal.py'),'--dataset',dataset],check=True)
        state.update(status='ACTUAL_Q_DATASET_FULL50_GT_RAW_TEXT_CPU_CLOSED',formal_datasets_closed=list(outcomes),new_formal_updates_closed=sum(v['successful_updates'] for v in outcomes.values()))
        proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    result=dict(status='ACTUAL_Q_SINGLE_MSVR_TWO_FULL50_GT_RAW_CPU_COMPLETE',finished=datetime.now().isoformat(timespec='seconds'),
        exit_code=0,datasets=outcomes,native_updates=6,successful_updates=2000,additional_epochs=100,
        sources_sha256=sources,all_three_datasets_new_training=False,missing_evaluated=False,retained_RGBNT201=state['retained_RGBNT201'],protected_weights=protected,
        limits='Only contribution regression.05->0 relative to P MSVR; same task gates, MFaux.01, physical0/1 and liveP2/3 unchanged; cap.10 and M primarymargin.6 unchanged. K201 best margin.3 retained under user-authorized dataset-specific parameters. TwoNEW50/2000updates, same P orders/auxiliary loss definitions/graph/5120/teacher42; MSVR.01/RGB100.05 auxiliary weights matched controls. No new parameters/forwards; physical0/1 only; existingP2/3 unchanged. Normal first, paper-six masks later, no new49. No +1/missing/stability/causality success claim from closure.')
    proof.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('ACTUAL_Q_SINGLE_MSVR_NORMAL_CONTROLLER_CLOSED',flush=True)


if __name__=='__main__':main()
