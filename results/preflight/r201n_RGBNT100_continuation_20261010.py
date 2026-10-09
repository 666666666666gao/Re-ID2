"""Run only the still-unstarted RGBNT100 pair from the reviewed N plan."""
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


def main():
    pf=PROJECT/'results/preflight'
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    original_review=load(pf/'r201n_bounded_identity_shift_source_review_20261009.json')
    review=load(pf/'r201n_RGBNT100_continuation_source_review_20261010.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    assert original_review['status']=='PASS' and not original_review['blocking_findings']
    sources=review['sources_sha256']|review['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in sources.items())
    recovered=load(pf/'r201n_MSVR_post_transport_collection_actual_20261009.json')
    assert recovered['status']=='ACTUAL_N_ORIGINAL_MSVR_POST_TRANSPORT_GT_RAW_TEXT_CPU_COMPLETE'
    collection=recovered['collection']
    assert collection['successful_updates']==2000 and collection['native_updates']==6 and collection['additional_epochs']==100
    assert collection['original_receiver_exit_code']==1 and not collection['original_controller_complete']
    original_proof=pf/'r201n_bounded_identity_shift_actual_session_20261009.json'
    state=load(original_proof)
    assert state['status']=='ACTUAL_N_MSVR_ORIGINAL_TWO_FULL50_RECOVERED_GT_RAW_CPU_RGBNT100_PENDING'
    assert not state['RGBNT100_started'] and not state['paper6_started']
    protected=dict(state['protected_weights'])
    for job in collection['jobs']:
        protected[job['run'][len(REMOTE)+1:]+'/best.pth']=job['best_sha256']
    assert len(protected)==22
    proof=pf/'r201n_RGBNT100_continuation_actual_20261010.json'
    receipt=pf/'r201n_bounded_identity_shift_RGBNT100_actual_session_20261009.json'
    assert not proof.exists() and not receipt.exists() and not (ARCHIVE/'RGBNT100').exists()
    assert not (PROJECT/'results/r201n_bounded_identity_shift_20261009/RGBNT100').exists()
    assert shutil.disk_usage(ARCHIVE.parent).free>2_000_000_000
    dataset='RGBNT100';updates=13126;out=ROOT+'/'+dataset
    remote_sources={n:sha for n,sha in sources.items() if not n.startswith('results/')}
    code=f'''import hashlib,json,shutil
from pathlib import Path
root=Path({REMOTE!r});out=Path({out!r})
assert not out.exists() and shutil.disk_usage(root).free>2_200_000_000
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {remote_sources!r}.items())
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {protected!r}.items())
prior=root/'runs/r201n_bounded_identity_shift_20261009/MSVR310'
assert not (prior/'controller_result.json').exists()
for variant in ('frequency_shared','axis_shared'):
 name='MSVR310_r201n_'+variant+'_s42';run=prior/'training'/name
 launch=json.loads((prior/'training'/(name+'_launch.json')).read_text())
 assert not Path('/proc/'+str(launch['pid'])).exists()
 assert json.loads((prior/'training'/(name+'_exit.json')).read_text())['exit_code']==0
 data=json.loads((run/'result.json').read_text())
 assert data['status']=='COMPLETE' and data['epochs']==50 and data['optimizer_steps']==1000
print(json.dumps(dict(status='ACTUAL_N_RGBNT100_UNSTARTED_SOURCES_AND_22_WEIGHTS_READY',free=shutil.disk_usage(root).free)))'''
    ready=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    assert ready['status']=='ACTUAL_N_RGBNT100_UNSTARTED_SOURCES_AND_22_WEIGHTS_READY'
    result=dict(status='ACTUAL_REVIEWED_N_RGBNT100_CONTINUATION_STARTING',started=datetime.now().astimezone().isoformat(timespec='seconds'),
        receiver_pid=os.getpid(),dataset=dataset,remote_root=out,capacity=ready,sources_sha256=sources,protected_weights=protected,
        prior_MSVR_original_receiver_exit_code=1,prior_MSVR_controller_complete=False,new_MSVR_training_starts=0)
    proof.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    state.update(status='ACTUAL_N_MSVR_RECOVERED_RGBNT100_NATIVE_THEN_FULL50_RUNNING',RGBNT100_started=True,current_dataset=dataset,
        active_receiver_pid=os.getpid(),continuation_receiver=str(proof.relative_to(PROJECT)),continued_at=result['started'])
    original_proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    argv=[PYTHON,'-u','launch_r201n_bounded_identity_shift.py','--dataset',dataset,
        '--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
        '--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',out]
    process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026',
        'cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,encoding='utf-8')
    archives,cleared,native,finished={},set(),False,False
    print('N_RGBNT100_NEW_NATIVE_THEN_ORIGINAL_PLANNED_FULL50',flush=True)
    for line in process.stdout:
        if not line.startswith('R201N_STREAM '):
            print(line,end='',flush=True)
            continue
        message=json.loads(line[len('R201N_STREAM '):]);event=message['event']
        if event=='NATIVE_PASS':
            assert not native and message['controls']==2 and message['actual_updates']==6 and message['partial_loss_effective_zero']
            assert message['PI_outlet']=='shared_relation_local' and message['final_unit_shift_epsilon']==.10 and message['controlled_final_shift_bound']
            native=True
            native_result=dict(status='ACTUAL_N_RGBNT100_NATIVE6_PASS_FULL50_RUNNING',observed_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                dataset=dataset,remote_root=out,actual_native_updates=6,primary_margin=.6,final_unit_shift_epsilon=.10,
                partial_loss_effective_zero=True,controlled_final_shift_bound=True,normal_full50_complete=False,event=message)
            (pf/'r201n_bounded_identity_shift_RGBNT100_native_actual_20261009.json').write_text(json.dumps(native_result,indent=2)+'\n',encoding='utf-8')
            result.update(status=native_result['status'],actual_native_updates=6)
            proof.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
            state.update(status='ACTUAL_N_MSVR_RECOVERED_RGBNT100_NATIVE6_PASS_FULL50_RUNNING',native_updates_closed=12)
            original_proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
            print('ACTUAL_N_RGBNT100_NATIVE6_PASS',flush=True)
        elif event=='NORMAL_READY':
            name=message['name'];assert native and name.startswith(dataset+'_r201n_') and name not in archives
            assert shutil.disk_usage(ARCHIVE.parent).free>message['file']['bytes']
            local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVE)
            archives[name]=dict(file=message['file'],local=local)
            process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED',name=name,file=message['file']))+'\n');process.stdin.flush()
        elif event=='NORMAL_CLEARED':
            name=message['name'];assert name in archives and name not in cleared
            cleared.add(name);print('N_RGBNT100_LOCAL_SHA_ACK_REMOTE_DUPLICATE_CLEARED',name,flush=True)
        else:
            assert event=='CONTROLLER_COMPLETE' and not finished and message['controls']==2 and message['successful_updates']==updates
            finished=True
    process.stdin.close()
    assert process.wait()==0 and native and finished and set(archives)==cleared and len(archives)==2
    value=dict(status='ACTUAL_N_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW',finished=datetime.now().astimezone().isoformat(timespec='seconds'),
        exit_code=0,dataset=dataset,remote_root=out,archives=archives,native_updates=6,successful_updates=updates,additional_epochs=100,
        sources_sha256=original_review['sources_sha256'],continuation_source_review=str((pf/'r201n_RGBNT100_continuation_source_review_20261010.json').relative_to(PROJECT)),
        limits='Original reviewed N RGBNT100 pair only. No MSVR retraining or forged original MSVR controller; full official normal first, no new49.')
    receipt.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,'-X','utf8','-B','-S',str(pf/'r201n_dataset_normal_complete_intake_20261009.py'),'--dataset',dataset],check=True)
    subprocess.run([sys.executable,'-X','utf8','-B','-S',str(PROJECT/'analyze_r201n_dataset_normal.py'),'--dataset',dataset],check=True)
    result.update(status='ACTUAL_N_RGBNT100_NEW_TWO_FULL50_GT_RAW_TEXT_CPU_COMPLETE',finished=value['finished'],
        exit_code=0,native_updates=6,successful_updates=updates,additional_epochs=100,archives=archives)
    proof.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    state.update(status='ACTUAL_N_TWO_WEAK_FOUR_FULL50_GT_RAW_COMPLETE_SPLIT_RECEIVERS',current_dataset=None,active_receiver_pid=None,
        formal_datasets_closed=['MSVR310','RGBNT100'],new_formal_updates_closed=15126,native_updates_closed=12,
        additional_epochs=200,RGBNT100_normal_closed=True,MSVR_recovered_receipt=str((pf/'r201n_MSVR_post_transport_collection_actual_20261009.json').relative_to(PROJECT)),
        RGBNT100_receipt=str(receipt.relative_to(PROJECT)),original_receiver_exit_code=1,original_controller_complete=False,
        limits='Actual completed two datasets via original interrupted MSVR receiver and new RGBNT100 receiver; original MSVR controller remains absent. No single-original-controller success claim, no missing or +1 claim.')
    original_proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print('ACTUAL_N_RGBNT100_CONTINUATION_CLOSED_NORMAL_ONLY',flush=True)


if __name__=='__main__':main()
