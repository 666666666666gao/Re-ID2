"""Wait on the original local receiver, then run reviewed G closeout stages once."""
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
    review=load(pf/'r201g_postnormal_missing49_queue_source_review_20261006.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources=review['sources_sha256']|review['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in sources.items())
    started=pf/'r201g_postnormal_missing49_queue_started_20261006.json'
    proof=pf/'r201g_postnormal_missing49_queue_actual_session_20261006.json'
    assert not started.exists() and not proof.exists()
    future=[pf/'r201g_RGBNT100_normal_terminal_text_intake_20261006.json',
        PROJECT/'results/r201g_other_two_normal_20261006/RGBNT100',
        PROJECT/'results/r201g_other_two_normal_20261006/three_dataset_normal_analysis',
        pf/'r201g_201_missing49_stream_native_actual_session_20261006.json',
        pf/'r201g_201_missing49_stream_full_actual_session_20261006.json',
        PROJECT/'results/r201g_201_missing49_stream_native_completed_20261006',
        PROJECT/'results/r201g_201_missing49_stream_full_completed_20261006']
    assert not any(path.exists() for path in future)
    assert load(pf/'r201g_MSVR310_normal_terminal_text_intake_20261006.json')['status']=='ACTUAL_G_DATASET_TWO_FULL50_NORMAL_GT_TEXT_VERIFIED'
    entry=dict(status='WAIT_ON_ORIGINAL_LOCAL_G_MAIN_RECEIVER',started_at=datetime.now().isoformat(timespec='seconds'),
        local_queue_pid=os.getpid(),original_receiver_pid=receiver,original_unified_session=28890,
        original_observer_unified_session=32692,new_observer_or_timer=0,new_training=0,
        new_neural_calls=0,sources_sha256=review['sources_sha256'],
        policy='One OS process wait, no repeated remote/GPU query. Original receiver exit plus its actual COMPLETE/exit0 proof gates every follow-up. No original restart, no H launch.')
    wait=f"$ErrorActionPreference='Stop'; $p=Get-CimInstance Win32_Process -Filter 'ProcessId={receiver}'; if ($null -eq $p -or $p.CommandLine -notlike '*r201g_other_two_normal_deploy_20261006.py*') {{ throw 'Original receiver identity is not live' }}; Write-Output 'ACTUAL_ORIGINAL_G_RECEIVER_LIVE_WAIT'; Wait-Process -Id {receiver}"
    started.write_text(json.dumps(entry,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(entry),flush=True)
    subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',wait],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    actual=load(pf/'r201g_other_two_normal_actual_session_20261006.json')
    assert actual['status']=='ACTUAL_UNCHANGED_G_OTHER_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW'
    assert actual['exit_code']==0 and actual['successful_updates']==14124 and len(actual['archives'])==4
    print('ACTUAL_ORIGINAL_G_COMPLETE_EXIT0_START_REVIEWED_CLOSEOUT',flush=True)
    stages=[
        ['results/preflight/r201g_dataset_normal_complete_intake_20261006.py','--dataset','RGBNT100'],
        ['analyze_r201g_dataset_normal.py','--dataset','RGBNT100'],
        ['analyze_r201g_three_normal.py'],
        ['results/preflight/r201g_201_missing49_stream_deploy_20261006.py','--mode','native'],
        ['results/preflight/r201g_201_missing49_stream_intake_20261006.py','--mode','native'],
        ['results/preflight/r201g_201_missing49_stream_deploy_20261006.py','--mode','full'],
        ['results/preflight/r201g_201_missing49_stream_intake_20261006.py','--mode','full']]
    completed=[]
    for argv in stages:
        assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in sources.items())
        print('ACTUAL_CLOSEOUT_STAGE_START '+json.dumps(argv),flush=True)
        subprocess.run([sys.executable,'-X','utf8','-B','-S',*argv],cwd=PROJECT,check=True)
        completed.append(argv)
    three=load(PROJECT/'results/r201g_other_two_normal_20261006/three_dataset_normal_analysis/result.json')
    missing=load(pf/'r201g_201_missing49_stream_full_actual_session_20261006.json')
    reduced=load(PROJECT/'results/r201g_201_missing49_stream_full_completed_20261006/missing_analysis/result.json')
    assert three['status']=='ACTUAL_UNIFIED_G_THREE_OFFICIAL_NORMAL_DATASETS_CPU_READOUT' and three['model_results']==15
    assert missing['status']=='ACTUAL_G_201_TWO_FULL49_INFERENCE_CONTROLS_ALL392_GT_AND_LOCAL_RAW'
    assert missing['exit_code']==0 and len(missing['raw'])==98
    assert reduced['status']=='ACTUAL_G201_FIXED_NORMAL_BEST_ALL49_FOURSTATE_CPU_READOUT'
    assert reduced['model_state_metric_rows']==833 and reduced['paired_query_rows']==122892
    result=dict(status='ACTUAL_G_THREE_NORMAL_THEN_FIXED201_ALL49_CPU_QUEUE_COMPLETE',
        finished_at=datetime.now().isoformat(timespec='seconds'),exit_code=0,completed_stages=completed,
        normal_datasets=3,normal_models=15,new_G201_missing_GT_cases=392,new_G201_missing_raw_local=98,
        G201_metric_rows=833,G201_paired_query_rows=122892,new_optimizer_updates=0,
        limits='Protocol closeout only; no +2/three missing datasets/multiseed/fair-ablation/H success claim. Any child failure stops this queue; original processes are never restarted.')
    proof.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
