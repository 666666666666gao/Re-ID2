"""Run reviewed M final-primary-margin0.6 controls on all three official datasets."""
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
ROOT=REMOTE+'/runs/r201m_primary_margin06_20261008'
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201m_primary_margin06_20261008')
UPDATES={'RGBNT201':5294,'RGBNT100':13126,'MSVR310':2000}


def main():
    pf=PROJECT/'results/preflight'
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    closed=load(pf/'r201l_missing49_complete_20261008_actual.json')
    assert closed['original_exit_code']==0 and closed['combined_conditions']==294 and closed['combined_state_cases']==1176
    assert load(pf/'r201k_best5_complete_20261007_actual.json')['status']=='ACTUAL_RGBNT201_EXPERT_N5_NORMAL_AND_BEST5_FIXED49_COMPLETE'
    audit=load(pf/'r201l_full49_integrity_audit_20261008.json')
    assert audit['core_result_integrity']=='PASS' and not audit['blocking_findings']
    archive=load(pf/'closed_L_weights_archive27_after_full49_actual_20261008.json')
    assert archive['unique_weights_lost']==0 and len(archive['records'])==4
    replay=load(pf/'r201l_uniform_K8_sampler_cpu_replay_20261007.json')
    assert replay['status']=='ACTUAL_EXISTING_SAMPLER_CPU_FULL50_OLD_AND_K8_REPLAY_COMPLETE'
    assert all(replay['datasets'][d]['existing_K_50_epoch_orders_exact'] and 2*replay['datasets'][d]['8']['total_steps']==steps for d,steps in UPDATES.items())
    plan=load(pf/'r201m_primary_margin06_plan_20261008.json')
    assert plan['datasets_in_order']==list(UPDATES) and plan['planned_formal_updates']==sum(UPDATES.values())==20420
    review=load(pf/'r201m_primary_margin06_source_review_20261008.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    expected={'run_r201m_primary_margin06.py','launch_r201m_primary_margin06.py','analyze_r201m_dataset_normal.py','results/preflight/r201m_primary_margin06_deploy_20261008.py','results/preflight/r201m_dataset_normal_complete_intake_20261008.py'}
    sources,reused=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert set(sources)==expected and all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reused).items())
    proof=pf/'r201m_primary_margin06_actual_session_20261008.json'
    assert not proof.exists() and not ARCHIVE.exists()
    assert shutil.disk_usage(ARCHIVE.parent).free>900_000_000
    code=f'''import hashlib,json,shutil
from pathlib import Path
root=Path({REMOTE!r});output=Path({ROOT!r})
assert not output.exists() and shutil.disk_usage(root).free>4_400_000_000
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {reused!r}.items() if not name.startswith('results/'))
anchors=list(root.glob('runs/full_official_baselines_20261004/training/*_demo_s42/best.pth'))+list(root.glob('runs/r201i_primary_margin_20261006/*/training/*/best.pth'))
assert len(anchors)==9
print(json.dumps(dict(status='ACTUAL_CLOSED_L_STORAGE_AND_M_REUSED_SOURCES_READY',free=shutil.disk_usage(root).free,protected_anchors={{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in anchors}})))'''
    ready=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    assert ready['status']=='ACTUAL_CLOSED_L_STORAGE_AND_M_REUSED_SOURCES_READY'
    protected=ready['protected_anchors']
    assert protected=={name:sha for name,sha in archive['protected_anchors_and_I42_best'].items() if name.startswith(('runs/full_official_baselines_20261004/','runs/r201i_primary_margin_20261006/'))}
    (pf/'r201m_capacity_anchors_actual_20261008.json').write_text(json.dumps(ready,indent=2)+'\n',encoding='utf-8')
    print('ACTUAL_M_CAPACITY_AND_NINE_ANCHORS_READY',ready['free'],flush=True)
    for name in sources:
        if not name.startswith('results/'):
            command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+REMOTE+'/'+name])
    deployed={name:sha for name,sha in sources.items() if not name.startswith('results/')}
    code=f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r});assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {deployed!r}.items())
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {protected!r}.items())
print('ACTUAL_REVIEWED_M_SOURCES_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),end='')
    outcomes={}
    state=dict(status='ACTUAL_M_REVIEWED_CAPACITY_AND_SOURCES_READY',started=datetime.now().isoformat(timespec='seconds'),receiver_pid=os.getpid(),native_updates_closed=0,formal_datasets_closed=[],new_formal_updates_closed=0,all_three_datasets_new_training=True)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    for dataset,updates in UPDATES.items():
        out=ROOT+'/'+dataset
        receipt=pf/('r201m_primary_margin06_'+dataset+'_actual_session_20261008.json')
        assert not receipt.exists()
        argv=[PYTHON,'-u','launch_r201m_primary_margin06.py','--dataset',dataset,
            '--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
            '--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',out]
        state.update(status='ACTUAL_M_DATASET_NATIVE_THEN_FULL50_RUNNING',current_dataset=dataset,started_current_dataset=datetime.now().isoformat(timespec='seconds'))
        proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026',
            'cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        archives,cleared,native,finished={},set(),False,False
        print('M_ORIGINAL_DATASET_CONTROLLER_STARTED',dataset,flush=True)
        for line in process.stdout:
            if not line.startswith('R201M_STREAM '):
                print(line,end='',flush=True)
                continue
            message=json.loads(line[len('R201M_STREAM '):])
            event=message['event']
            if event=='NATIVE_PASS':
                assert not native and message['controls']==2 and message['actual_updates']==6 and message['partial_loss_effective_zero'] and message['PI_outlet']=='shared_relation_local'
                native=True
                (pf/('r201m_primary_margin06_'+dataset+'_native_actual_20261008.json')).write_text(json.dumps(dict(
                    status='ACTUAL_TWO_M_NATIVE3_PRIMARY_MARGIN06_UNIFORM_P8K8_RELATION_PI_PASS_FORMAL_NEXT',observed_at=datetime.now().isoformat(timespec='seconds'),
                    dataset=dataset,remote_root=out,actual_native_updates=6,primary_margin=.6,auxiliary_soft_unchanged=True,PI_outlet='shared_relation_local',PI_native_input_isolation=True,controlled10_01_same_input_parent_equal=True,
                    partial_loss_effective_zero=True,normal_full50_complete=False),indent=2)+'\n',encoding='utf-8')
                state.update(status='ACTUAL_M_DATASET_NATIVE6_PASSED_FULL50_RUNNING',native_updates_closed=state['native_updates_closed']+6)
                proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
                print('ACTUAL_M_NATIVE6_PASS',dataset,flush=True)
            elif event=='NORMAL_READY':
                name=message['name']
                assert native and name.startswith(dataset+'_r201m_') and name not in archives
                assert shutil.disk_usage(ARCHIVE.parent).free>message['file']['bytes']
                local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVE)
                archives[name]=dict(file=message['file'],local=local)
                process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED',name=name,file=message['file']))+'\n')
                process.stdin.flush()
            elif event=='NORMAL_CLEARED':
                name=message['name']
                assert name in archives and name not in cleared
                cleared.add(name)
                print('M_NORMAL_LOCAL_SHA_ACK_REMOTE_CLEARED',dataset,name,flush=True)
            else:
                assert event=='CONTROLLER_COMPLETE' and not finished and message['controls']==2 and message['successful_updates']==updates
                finished=True
        process.stdin.close()
        assert process.wait()==0 and native and finished and set(archives)==cleared and len(archives)==2
        value=dict(status='ACTUAL_M_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW',finished=datetime.now().isoformat(timespec='seconds'),
            exit_code=0,dataset=dataset,remote_root=out,archives=archives,native_updates=6,successful_updates=updates,
            additional_epochs=100,sources_sha256=sources,limits='Only final primary hinge margin.3->.6, same L P8K8 orders/updates and K graph/auxsoft/partial0; paired M controls matched. No +2/missing/multiseed success assertion.')
        receipt.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        outcomes[dataset]=value
        print('ACTUAL_M_TWO_FULL50_CLOSED',dataset,flush=True)
        subprocess.run([sys.executable,'-X','utf8','-B','-S',str(pf/'r201m_dataset_normal_complete_intake_20261008.py'),'--dataset',dataset],check=True)
        subprocess.run([sys.executable,'-X','utf8','-B','-S',str(PROJECT/'analyze_r201m_dataset_normal.py'),'--dataset',dataset],check=True)
        state.update(status='ACTUAL_M_DATASET_FULL50_GT_RAW_TEXT_CPU_CLOSED',formal_datasets_closed=list(outcomes),new_formal_updates_closed=sum(v['successful_updates'] for v in outcomes.values()))
        proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    result=dict(status='ACTUAL_M_THREE_NORMAL_DATASETS_SIX_FULL50_GT_RAW_COMPLETE',finished=datetime.now().isoformat(timespec='seconds'),
        exit_code=0,datasets=outcomes,native_updates=18,successful_updates=20420,additional_epochs=300,
        sources_sha256=sources,all_three_datasets_new_training=True,missing_evaluated=False,limits='M only final-primary hinge margin.3->.6 relative to uniform L recipe; all3 sixNEW50/20420updates, same L orders/auxsoft/graph/5120/teacher42. Normal first, fixed49 later. Known loss control, no all3+2/missing/stability/causality claim from closure.')
    proof.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('ACTUAL_M_THREE_NORMAL_CONTROLLER_CLOSED',flush=True)


if __name__=='__main__':main()
