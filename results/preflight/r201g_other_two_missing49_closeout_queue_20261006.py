"""Wait on the original G full receiver, then collect closed text and CPU results once."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
pf=PROJECT/'results/preflight'
load=lambda path:json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--receiver-pid',type=int,required=True)
    receiver=parser.parse_args().receiver_pid
    review=load(pf/'r201g_other_two_missing49_cpu_closeout_source_review_20261006.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    assert 'results/preflight/r201g_other_two_missing49_closeout_queue_20261006.py' in review['sources_sha256']
    sources=review['sources_sha256']|review['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in sources.items())
    original=load(pf/'r201g_other_two_missing49_stream_full_started_20261006.json')
    assert original['status']=='ACTUAL_ORIGINAL_FULL_LOCAL_RECEIVER_LIVE_NATIVE4_PASS'
    assert receiver==original['local_pid'] and original['original_unified_session']==29653
    native=load(pf/'r201g_other_two_missing49_stream_native_actual_session_20261006.json')
    assert native['status']=='ACTUAL_G_OTHER_TWO_FOUR_NATIVE_INFERENCE_CONTROLS_ALL16_NORMAL_GT_AND_LOCAL_RAW'
    assert native['exit_code']==0 and len(native['raw'])==4
    started=pf/'r201g_other_two_missing49_closeout_queue_started_20261006.json'
    proof=pf/'r201g_other_two_missing49_closeout_queue_actual_session_20261006.json'
    future=[PROJECT/'results/r201g_other_two_missing49_stream_full_completed_20261006',
        pf/'r201g_other_two_missing49_full_intake_20261006.json',
        pf/'r201g_other_two_missing49_stream_full_actual_session_20261006.json',
        pf/'r201g_other_two_missing49_native_intake_20261006.json',
        PROJECT/'results/r201g_other_two_missing49_stream_native_completed_20261006']
    assert not started.exists() and not proof.exists() and not any(path.exists() for path in future)
    entry=dict(status='NATIVE_CLOSED_TEXT_THEN_WAIT_ORIGINAL_G_OTHER_FULL_RECEIVER',
        started_at=datetime.now().isoformat(timespec='seconds'),local_queue_pid=os.getpid(),
        original_receiver_pid=receiver,original_unified_session=29653,new_observer_or_timer=0,
        new_training=0,new_neural_calls=0,sources_sha256=review['sources_sha256'],
        policy='Native closed text once, then one OS wait on the original full receiver. Only full exit0/196raw/784GT permits once-only full text/CPU. No remote polling, NN, restart, rerun or stage retries.')
    wait=f"$ErrorActionPreference='Stop'; $p=Get-CimInstance Win32_Process -Filter 'ProcessId={receiver}'; if ($null -eq $p -or $p.CommandLine -notlike '*r201g_other_two_missing49_stream_deploy_20261006.py*--mode full*') {{ throw 'Original G full receiver identity is not live' }}; Write-Output 'ACTUAL_ORIGINAL_G_OTHER_RECEIVER_LIVE_WAIT'; Wait-Process -Id {receiver}"
    started.write_text(json.dumps(entry,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(entry),flush=True)
    subprocess.run([sys.executable,'-X','utf8','-B','-S','results/preflight/r201g_other_two_missing49_stream_intake_20261006.py','--mode','native'],cwd=PROJECT,check=True)
    subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',wait],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    actual=load(pf/'r201g_other_two_missing49_stream_full_actual_session_20261006.json')
    assert actual['status']=='ACTUAL_G_OTHER_TWO_FOUR_FULL49_INFERENCE_CONTROLS_ALL784_GT_AND_LOCAL_RAW'
    assert actual['exit_code']==0 and len(actual['raw'])==196 and actual['restored_copies_cleared']
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in sources.items())
    print('ACTUAL_G_OTHER_FULL_EXIT0_BEGIN_ONCE_ONLY_TEXT_CPU_CLOSEOUT',flush=True)
    subprocess.run([sys.executable,'-X','utf8','-B','-S','results/preflight/r201g_other_two_missing49_stream_intake_20261006.py','--mode','full'],cwd=PROJECT,check=True)
    intake=load(pf/'r201g_other_two_missing49_full_intake_20261006.json')
    reduced=load(PROJECT/'results/r201g_other_two_missing49_stream_full_completed_20261006/three_dataset_missing_analysis/result.json')
    assert intake['status']=='ACTUAL_CLOSED_MISSING_INFERENCE_PRIMARY_TEXT_LOCAL_VERIFIED' and intake['mode']=='full'
    assert reduced['status']=='ACTUAL_G_THREE_FULL_OFFICIAL_FIXED_BEST_ALL49_CPU_READOUT'
    assert reduced['model_state_metric_rows']==2499 and reduced['comparisons']==1911 and reduced['paired_query_rows']==461874
    assert reduced['checked_condition_query_rows']==2617286 and reduced['all_six_CMC50_identity_camera_scene_verified']
    result=dict(status='ACTUAL_G_OTHER_FULL49_TEXT_AND_THREE_DATASET_CPU_CLOSEOUT_QUEUE_COMPLETE',
        finished_at=datetime.now().isoformat(timespec='seconds'),exit_code=0,original_unified_session=29653,
        raw_local_verified=196,GT_state_cases=784,model_state_metric_rows=2499,comparisons=1911,
        paired_query_rows=461874,unique_query_records=3142,new_neural_calls=0,new_optimizer_updates=0,
        limits='Fixed normal benchmark-selected seed42; original50 plus additional50. No new +2/all3 improvement, calibrated contribution, retrain ablations or seeds claim; no NN/CPU stage rerun.')
    proof.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
