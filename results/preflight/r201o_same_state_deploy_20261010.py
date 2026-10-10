"""Run reviewed O same-state contribution controls; preserve K RGBNT201."""
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
ROOT=REMOTE+'/runs/r201o_same_state_contribution_20261010'
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201o_same_state_contribution_20261010')
UPDATES={'MSVR310':2000,'RGBNT100':13126}


def main():
    pf=PROJECT/'results/preflight'
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    parent_normal=load(pf/'r201n_bounded_identity_shift_actual_session_20261009.json')
    assert parent_normal['status']=='ACTUAL_N_TWO_WEAK_FOUR_FULL50_GT_RAW_COMPLETE_SPLIT_RECEIVERS'
    assert parent_normal['new_formal_updates_closed']==15126 and parent_normal['native_updates_closed']==12
    parent_paper=load(pf/'r201n_paper_missing6_actual_session_20261009.json')
    assert parent_paper['status']=='ACTUAL_N_ALL_DECLARED_FIXED_BEST_PAPER6_GT_RAW_CPU_COMPLETE_K201_RETAINED'
    assert parent_paper['phases']['full']['conditions']==24 and parent_paper['phases']['full']['state_cases']==96
    closure=load(pf/'r201n_final_closure_verified_20261010.json')
    protected=dict(parent_normal['protected_weights'])
    protected.update({name:record['sha256'] for name,record in closure['remote']['weights'].items()})
    assert len(protected)==24
    replay=load(pf/'r201l_uniform_K8_sampler_cpu_replay_20261007.json')
    assert replay['status']=='ACTUAL_EXISTING_SAMPLER_CPU_FULL50_OLD_AND_K8_REPLAY_COMPLETE'
    assert all(replay['datasets'][d]['existing_K_50_epoch_orders_exact'] and 2*replay['datasets'][d]['8']['total_steps']==steps for d,steps in UPDATES.items())
    plan=load(pf/'r201o_same_state_plan_20261010.json')
    assert plan['datasets']==list(UPDATES) and sum(UPDATES.values())==15126
    review=load(pf/'r201o_same_state_source_review_20261010.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    expected={'run_r201o_same_state_contribution.py','launch_r201o_same_state_contribution.py','analyze_r201o_dataset_normal.py','results/preflight/r201o_same_state_deploy_20261010.py','results/preflight/r201o_dataset_normal_complete_intake_20261010.py','results/preflight/r201o_same_state_plan_20261010.json','results/preflight/r201o_target_algebra_cpu_20261010.py'}
    sources,reused=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert set(sources)==expected and all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reused).items())
    proof=pf/'r201o_same_state_contribution_actual_session_20261010.json'
    assert not proof.exists() and not ARCHIVE.exists()
    assert shutil.disk_usage(ARCHIVE.parent).free>900_000_000
    code=f'''import hashlib,json,shutil
from pathlib import Path
root=Path({REMOTE!r});output=Path({ROOT!r})
assert not output.exists() and shutil.disk_usage(root).free>4_400_000_000
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {reused!r}.items() if not name.startswith('results/'))
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {protected!r}.items())
print(json.dumps(dict(status='ACTUAL_CLOSED_N_STORAGE_AND_O_REUSED_SOURCES_READY',free=shutil.disk_usage(root).free,protected_anchors={protected!r})))'''
    ready=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    assert ready['status']=='ACTUAL_CLOSED_N_STORAGE_AND_O_REUSED_SOURCES_READY'
    assert protected==ready['protected_anchors']
    (pf/'r201o_capacity_anchors_actual_20261010.json').write_text(json.dumps(ready,indent=2)+'\n',encoding='utf-8')
    print('ACTUAL_O_CAPACITY_AND_24_PROTECTED_WEIGHTS_READY',ready['free'],flush=True)
    for name in sources:
        if not name.startswith('results/'):
            command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+REMOTE+'/'+name])
    deployed={name:sha for name,sha in sources.items() if not name.startswith('results/')}
    code=f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r});assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {deployed!r}.items())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {protected!r}.items())
print('ACTUAL_REVIEWED_O_SOURCES_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),end='')
    outcomes={}
    state=dict(status='ACTUAL_O_REVIEWED_CAPACITY_AND_SOURCES_READY',started=datetime.now().isoformat(timespec='seconds'),receiver_pid=os.getpid(),native_updates_closed=0,formal_datasets_closed=[],new_formal_updates_closed=0,all_three_datasets_new_training=False,
        retained_RGBNT201='K_axis_shared_s42 E30, primary margin.3; no new201 training',protected_weights=protected)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    for dataset,updates in UPDATES.items():
        out=ROOT+'/'+dataset
        receipt=pf/('r201o_same_state_contribution_'+dataset+'_actual_session_20261010.json')
        assert not receipt.exists()
        argv=[PYTHON,'-u','launch_r201o_same_state_contribution.py','--dataset',dataset,
            '--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
            '--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',out]
        state.update(status='ACTUAL_O_DATASET_NATIVE_THEN_FULL50_RUNNING',current_dataset=dataset,started_current_dataset=datetime.now().isoformat(timespec='seconds'))
        proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026',
            'cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        archives,cleared,native,finished={},set(),False,False
        print('O_ORIGINAL_DATASET_CONTROLLER_STARTED',dataset,flush=True)
        for line in process.stdout:
            if not line.startswith('R201O_STREAM '):
                print(line,end='',flush=True)
                continue
            message=json.loads(line[len('R201O_STREAM '):])
            event=message['event']
            if event=='NATIVE_PASS':
                assert not native and message['controls']==2 and message['actual_updates']==6 and message['partial_loss_effective_zero'] and message['PI_outlet']=='shared_relation_local' and message['final_unit_shift_epsilon']==.10 and message['controlled_final_shift_bound']
                native=True
                (pf/('r201o_same_state_contribution_'+dataset+'_native_actual_20261010.json')).write_text(json.dumps(dict(
                    status='ACTUAL_TWO_O_NATIVE3_PRIMARY_MARGIN06_UNIFORM_P8K8_RELATION_PI_PASS_FORMAL_NEXT',observed_at=datetime.now().isoformat(timespec='seconds'),
                    dataset=dataset,remote_root=out,actual_native_updates=6,primary_margin=.6,auxiliary_soft_unchanged=True,PI_outlet='shared_relation_local',PI_native_input_isolation=True,controlled10_01_pre_bound_parent_equal=True,final_unit_shift_epsilon=.10,controlled_final_shift_bound=True,
                    partial_loss_effective_zero=True,normal_full50_complete=False),indent=2)+'\n',encoding='utf-8')
                state.update(status='ACTUAL_O_DATASET_NATIVE6_PASSED_FULL50_RUNNING',native_updates_closed=state['native_updates_closed']+6)
                proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
                print('ACTUAL_O_NATIVE6_PASS',dataset,flush=True)
            elif event=='NORMAL_READY':
                name=message['name']
                assert native and name.startswith(dataset+'_r201o_') and name not in archives
                assert shutil.disk_usage(ARCHIVE.parent).free>message['file']['bytes']
                local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVE)
                archives[name]=dict(file=message['file'],local=local)
                process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED',name=name,file=message['file']))+'\n')
                process.stdin.flush()
            elif event=='NORMAL_CLEARED':
                name=message['name']
                assert name in archives and name not in cleared
                cleared.add(name)
                print('O_NORMAL_LOCAL_SHA_ACK_REMOTE_CLEARED',dataset,name,flush=True)
            else:
                assert event=='CONTROLLER_COMPLETE' and not finished and message['controls']==2 and message['successful_updates']==updates
                finished=True
        process.stdin.close()
        assert process.wait()==0 and native and finished and set(archives)==cleared and len(archives)==2
        value=dict(status='ACTUAL_O_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW',finished=datetime.now().isoformat(timespec='seconds'),
            exit_code=0,dataset=dataset,remote_root=out,archives=archives,native_updates=6,successful_updates=updates,
            additional_epochs=100,sources_sha256=sources,limits='Weakdatasets only contribution-gallery reference relative to N, unchanged final unit5120 cap0.10, same M hinge.6+CE.25/P8K8 orders/updates/K graph/auxsoft/partial0; paired M controls matched. Preserve K201 margin.3 best. No +1/missing/multiseed success assertion.')
        receipt.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        outcomes[dataset]=value
        print('ACTUAL_O_TWO_FULL50_CLOSED',dataset,flush=True)
        subprocess.run([sys.executable,'-X','utf8','-B','-S',str(pf/'r201o_dataset_normal_complete_intake_20261010.py'),'--dataset',dataset],check=True)
        subprocess.run([sys.executable,'-X','utf8','-B','-S',str(PROJECT/'analyze_r201o_dataset_normal.py'),'--dataset',dataset],check=True)
        state.update(status='ACTUAL_O_DATASET_FULL50_GT_RAW_TEXT_CPU_CLOSED',formal_datasets_closed=list(outcomes),new_formal_updates_closed=sum(v['successful_updates'] for v in outcomes.values()))
        proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    result=dict(status='ACTUAL_O_TWO_WEAK_NORMAL_DATASETS_FOUR_FULL50_GT_RAW_COMPLETE',finished=datetime.now().isoformat(timespec='seconds'),
        exit_code=0,datasets=outcomes,native_updates=12,successful_updates=15126,additional_epochs=200,
        sources_sha256=sources,all_three_datasets_new_training=False,missing_evaluated=False,retained_RGBNT201=state['retained_RGBNT201'],protected_weights=protected,
        limits='Weakdatasets only same-state contribution gallery relative to N; cap.10 and M primarymargin.6 unchanged. K201 best margin.3 retained under user-authorized dataset-specific parameters. FourNEW50/15126updates, same N orders/auxsoft/graph/5120/teacher42. No new parameters/forwards; physical2/3 only. Normal first, paper-six masks later, no new49. No +1/missing/stability/causality success claim from closure.')
    proof.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('ACTUAL_O_TWO_WEAK_NORMAL_CONTROLLER_CLOSED',flush=True)


if __name__=='__main__':main()
