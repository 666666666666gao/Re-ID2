"""Launch reviewed H201 controls only after original G three-normal and fixed49 closeout."""
import argparse
from datetime import datetime
import hashlib
import json
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
ROOT=REMOTE+'/runs/r201h_protocol_list_20261006'
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201h_protocol_list_20261006')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('native','full'),required=True)
    mode=parser.parse_args().mode
    pf=PROJECT/'results/preflight'
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    other=load(pf/'r201g_other_two_normal_actual_session_20261006.json')
    assert other['status']=='ACTUAL_UNCHANGED_G_OTHER_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW'
    assert other['exit_code']==0 and other['successful_updates']==14124 and len(other['archives'])==4
    three=load(PROJECT/'results/r201g_other_two_normal_20261006/three_dataset_normal_analysis/result.json')
    assert three['status']=='ACTUAL_UNIFIED_G_THREE_OFFICIAL_NORMAL_DATASETS_CPU_READOUT' and three['model_results']==15
    missing=load(pf/'r201g_201_missing49_stream_full_actual_session_20261006.json')
    assert missing['status']=='ACTUAL_G_201_TWO_FULL49_INFERENCE_CONTROLS_ALL392_GT_AND_LOCAL_RAW'
    assert missing['exit_code']==0 and len(missing['raw'])==98
    missing_cpu=load(PROJECT/'results/r201g_201_missing49_stream_full_completed_20261006/missing_analysis/result.json')
    assert missing_cpu['status']=='ACTUAL_G201_FIXED_NORMAL_BEST_ALL49_FOURSTATE_CPU_READOUT'
    assert missing_cpu['model_state_metric_rows']==833 and missing_cpu['paired_query_rows']==122892
    assert missing_cpu['all_six_CMC50_identity_camera_scene_verified']
    math=load(pf/'r201h_protocol_list_actual_math_20261006.json')
    assert math['status']=='ACTUAL_CPU_PROTOCOL_AP_ALGEBRA_AND_GRADIENT_PASS' and math['new_neural_calls']==0
    review=load(pf/'r201h_protocol_list_launch_source_review_20261006.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources,reused=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in (sources|reused).items())
    proof=pf/('r201h_protocol_list_'+mode+'_actual_session_20261006.json')
    assert not proof.exists() and not ARCHIVE.exists()
    if mode=='full':
        native=load(pf/'r201h_protocol_list_native_actual_session_20261006.json')
        assert native['status']=='ACTUAL_H201_FOUR_NATIVE3_PROTOCOL_REFERENCE_PASS'
        assert native['exit_code']==0 and native['native_updates']==12 and native['sources_sha256']==sources
    code=f'''import hashlib,json,shutil
from pathlib import Path
project=Path({REMOTE!r});root=Path({ROOT!r})
assert all(hashlib.sha256((project/name).read_bytes()).hexdigest()==sha for name,sha in {reused!r}.items() if not name.startswith('results/'))
for dataset,updates in (('MSVR310',1410),('RGBNT100',12714)):
 closed=json.loads((project/'runs/r201g_other_two_normal_20261006'/dataset/'controller_result.json').read_text())
 assert closed['status']=='COMPLETE' and closed['successful_updates']==updates and closed['normal_archives_local_verified']==2
closed=json.loads((project/'runs/r201g_201_missing49_stream_20261006/controller_full_result.json').read_text())
assert closed['status']=='COMPLETE' and closed['missing_GT_state_cases']==392
assert closed['raw_conditions']==98 and closed['all_raw_local_verified']
assert shutil.disk_usage(project).free>2_200_000_000
if {mode!r}=='native':assert not root.exists()
else:
 accepted=json.loads((root/'native_acceptance.json').read_text())
 assert accepted['status']=='PASS' and accepted['controls']==4 and accepted['actual_updates']==12
 assert not (root/'training').exists()
print('ACTUAL_G_THREE_NORMAL_AND_FIXED49_CLOSED_H_NOT_STARTED')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),end='')
    if mode=='native':
        command(['scp',*OPTIONS,str(PROJECT/'launch_r201h_protocol_list.py'),'2026:'+REMOTE+'/launch_r201h_protocol_list.py'])
    code=f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {sources!r}.items() if not name.startswith('results/'))
print('REVIEWED_H_LAUNCH_SOURCES_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),end='')
    argv=[PYTHON,'-u','launch_r201h_protocol_list.py','--mode',mode,'--data-root','/data/gaob/Re-ID/dataset',
        '--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
        '--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',ROOT]
    process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026',
        'cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,encoding='utf-8')
    archives,cleared,accepted,complete={},set(),mode=='full',False
    for line in process.stdout:
        if not line.startswith('R201H_STREAM '):
            print(line,end='',flush=True)
            continue
        message=json.loads(line[len('R201H_STREAM '):])
        event=message['event']
        if event=='NATIVE_PASS':
            assert mode=='native' and not accepted and message['controls']==4 and message['actual_updates']==12
            accepted=True
            print('ACTUAL_H_NATIVE12_PASS_NO_FORMAL_STARTED',flush=True)
        elif event=='NORMAL_READY':
            assert mode=='full' and accepted and message['name'] not in archives
            assert shutil.disk_usage(ARCHIVE.parent).free>message['file']['bytes']
            local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVE)
            archives[message['name']]=dict(file=message['file'],local=local)
            process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED',name=message['name'],file=message['file']))+'\n')
            process.stdin.flush()
        elif event=='NORMAL_CLEARED':
            name=message['name']
            assert name in archives and name not in cleared
            cleared.add(name)
            print('ACTUAL_H_NORMAL_LOCAL_SHA_ACK_REMOTE_CLEARED',name,flush=True)
        else:
            assert event=='CONTROLLER_COMPLETE' and not complete and message['mode']==mode and message['controls']==4
            if mode=='native':assert message['native_updates']==12 and message['new_formal_updates']==0
            else:assert message['successful_updates']==10588 and message['additional_epochs']==200
            complete=True
    process.stdin.close()
    assert process.wait()==0 and accepted and complete
    assert len(archives)==len(cleared)==(0 if mode=='native' else 4)
    result=dict(status='ACTUAL_H201_FOUR_NATIVE3_PROTOCOL_REFERENCE_PASS' if mode=='native' else 'ACTUAL_H201_FOUR_FULL50_PROTOCOL_REFERENCE_NORMAL_GT_LOCAL_RAW',
        finished_at=datetime.now().isoformat(timespec='seconds'),exit_code=0,mode=mode,remote_root=ROOT,
        sources_sha256=sources,native_updates=12,successful_updates=0 if mode=='native' else 10588,
        additional_epochs=0 if mode=='native' else 200,archives=archives,
        limits='Four controlled AP-reference arms only; no three-dataset/missing/multiseed/+2 success claim.')
    proof.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('status','exit_code','successful_updates','native_updates')}),flush=True)


if __name__=='__main__':
    main()
