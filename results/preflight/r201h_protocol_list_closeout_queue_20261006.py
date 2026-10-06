"""Wait on the original H receiver, then execute reviewed complete text/CPU intake once."""
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
    review=load(pf/'r201h_protocol_list_closeout_queue_source_review_20261006.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources=review['sources_sha256']|review['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in sources.items())
    original=load(pf/'r201h_protocol_list_current_receiver_20261006.json')
    assert original['status']=='ACTUAL_ORIGINAL_H_FULL_LOCAL_RECEIVER_LIVE'
    assert receiver==original['local_pid'] and original['primary_session']==33893
    native=load(pf/'r201h_protocol_list_native_actual_session_20261006.json')
    assert native['status']=='ACTUAL_H201_FOUR_NATIVE3_PROTOCOL_REFERENCE_PASS' and native['exit_code']==0 and native['native_updates']==12
    started=pf/'r201h_protocol_list_closeout_queue_started_20261006.json'
    proof=pf/'r201h_protocol_list_closeout_queue_actual_session_20261006.json'
    future=[PROJECT/'results/r201h_protocol_list_20261006',pf/'r201h_protocol_list_terminal_text_intake_20261006.json',
        pf/'r201h_protocol_list_closed_text_20261006.tar.gz',pf/'r201h_protocol_list_full_actual_session_20261006.json']
    assert not started.exists() and not proof.exists() and not any(path.exists() for path in future)
    entry=dict(status='WAIT_ON_ORIGINAL_LOCAL_H_FULL_RECEIVER',started_at=datetime.now().isoformat(timespec='seconds'),
        local_queue_pid=os.getpid(),original_receiver_pid=receiver,original_unified_session=33893,
        original_observer_unified_session=40821,new_observer_or_timer=0,new_training=0,new_neural_calls=0,
        sources_sha256=review['sources_sha256'],
        policy='One native OS wait on the actual original local receiver; no remote polling/NN/restart. Only its complete exit0/full10588/raw4 receipt permits once-only text intake and its built-in CPU reducer. Any failure stops without rerunning a stage.')
    wait=f"$ErrorActionPreference='Stop'; $p=Get-CimInstance Win32_Process -Filter 'ProcessId={receiver}'; if ($null -eq $p -or $p.CommandLine -notlike '*r201h_protocol_list_deploy_20261006.py*--mode full*') {{ throw 'Original H receiver identity is not live' }}; Write-Output 'ACTUAL_ORIGINAL_H_RECEIVER_LIVE_WAIT'; Wait-Process -Id {receiver}"
    started.write_text(json.dumps(entry,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(entry),flush=True)
    subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',wait],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    actual=load(pf/'r201h_protocol_list_full_actual_session_20261006.json')
    assert actual['status']=='ACTUAL_H201_FOUR_FULL50_PROTOCOL_REFERENCE_NORMAL_GT_LOCAL_RAW'
    assert actual['exit_code']==0 and actual['native_updates']==12 and actual['additional_epochs']==200 and actual['successful_updates']==10588
    assert len(actual['archives'])==4 and all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in sources.items())
    print('ACTUAL_H_FULL_EXIT0_BEGIN_ONCE_ONLY_TEXT_CPU_CLOSEOUT',flush=True)
    subprocess.run([sys.executable,'-X','utf8','-B','-S','results/preflight/r201h_protocol_list_complete_intake_20261006.py'],cwd=PROJECT,check=True)
    intake=load(pf/'r201h_protocol_list_terminal_text_intake_20261006.json')
    reduced=load(PROJECT/'results/r201h_protocol_list_20261006/normal_analysis/result.json')
    assert intake['status']=='ACTUAL_H201_FOUR_FULL50_NORMAL_GT_TEXT_VERIFIED' and intake['raw_local_verified']==4
    assert reduced['status']=='ACTUAL_H201_FOUR_FULL50_PROTOCOL_REFERENCE_CPU_READOUT'
    assert reduced['model_results']==9 and len(reduced['comparisons'])==14 and reduced['paired_query_rows']==11704
    assert reduced['new_epochs']==200 and reduced['new_batches']==10588 and reduced['six_metrics_CMC50_identity_camera_scene_verified']
    result=dict(status='ACTUAL_H_FOUR_FULL50_TEXT_AND_CPU_CLOSEOUT_QUEUE_COMPLETE',finished_at=datetime.now().isoformat(timespec='seconds'),
        exit_code=0,original_unified_session=33893,normal_models=9,comparisons=14,paired_query_rows=11704,
        additional_epochs=200,formal_updates_verified=10588,native_updates_verified=12,raw_local_verified=4,
        new_neural_calls=0,new_optimizer_updates=0,
        limits='H201 normal only, benchmark-selected seed42, original50 plus additional50; not +2/all3/49/seeds/collaboration proof. No duplicated NN, raw transfer, analyzer or restarted source.')
    proof.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
