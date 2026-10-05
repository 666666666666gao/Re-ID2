"""Reviewed4x50 launcher and archive receiver; independent per-card NN progress."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified


REMOTE='/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/rgbnt201_identity_outlet_r201c_20261005'
NATIVE=REMOTE+'/runs/rgbnt201_identity_outlet_native_20261005'
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/rgbnt201_identity_outlet_r201c_20261005')
SOURCES=('evaluate_rgbnt201_identity_states49.py','audit_rgbnt201_identity_states49.py',
    'launch_rgbnt201_identity_outlet50.py','results/preflight/rgbnt201_identity_outlet50_deploy_20261005.py')


def remote(code):
    return json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))


def main():
    pf=PROJECT/'results/preflight'
    proof=pf/'rgbnt201_identity_outlet50_actual_session_20261005.json'
    assert not proof.exists() and not ARCHIVE.exists()
    review=json.loads((pf/'rgbnt201_identity_outlet50_source_review_20261005.json').read_text(encoding='utf-8'))
    assert review['status']=='PASS' and not review['blocking_findings']
    sources={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in SOURCES}
    assert sources==review['sources_sha256']
    native=json.loads((pf/'rgbnt201_identity_outlet_actual_native_session_20261005.json').read_text(encoding='utf-8'))
    assert native['status']=='COMPLETE_IDENTITY_OUTLET_NATIVE_LOCAL_RAW_VERIFIED_REMOTE_CLEARED'
    assert native['exit_code']==0 and native['actual_updates']==12 and native['new_weight_files']==0
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==digest for name,digest in native['sources_sha256'].items())
    parents=review['directly_reused_sources_sha256']
    remote(f'''import hashlib,json
from pathlib import Path
root=Path({REMOTE!r})
assert not Path({ROOT!r}).exists()
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in {parents!r}.items())
native=json.loads((Path({NATIVE!r})/'controller_result.json').read_text())
assert native['status']=='COMPLETE_IDENTITY_OUTLET_NATIVE_ONLY' and native['actual_updates']==12
print(json.dumps({{'native_complete':True,'parents_exact':True}}))''')
    for name in SOURCES:
        if not name.startswith('results/'):
            command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+REMOTE+'/'+name])
    deployed={name:digest for name,digest in sources.items() if not name.startswith('results/')}
    remote(f'''import hashlib,json
from pathlib import Path
root=Path({REMOTE!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in {deployed!r}.items())
print(json.dumps({{'sources_exact':True}}))''')
    argv=[PYTHON,'-u','launch_rgbnt201_identity_outlet50.py',
        '--data-root','/data/gaob/Re-ID/dataset',
        '--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
        '--anchor-run-dir',REMOTE+'/runs/full_official_baselines_20261004/training/RGBNT201_demo_s42',
        '--native-root',NATIVE,'--output',ROOT]
    print('RGBNT201_R201C_FOUR50_START',flush=True)
    process=subprocess.Popen(['ssh',*OPTIONS,'2026','cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding='utf-8')
    prefix='OFFICIAL_STREAM '
    frozen,frozen_cleared,normal,normal_cleared={},set(),{},set()
    complete=False
    for line in process.stdout:
        if not line.startswith(prefix):
            print(line,end='',flush=True)
            continue
        message=json.loads(line[len(prefix):])
        event=message['event']
        if event=='FROZEN_READY':
            job=message['job']
            assert job not in frozen and len(message['files'])==49
            frozen[job]={path:dict(file=info,local=copy_verified(path,info,Path(ROOT),ARCHIVE)) for path,info in message['files'].items()}
            process.stdin.write(json.dumps(dict(event='FROZEN_ARCHIVED',job=job,files=message['files']))+'\n')
            process.stdin.flush()
        elif event=='FROZEN_CLEARED':
            job=message['job']
            assert job in frozen and job not in frozen_cleared
            frozen_cleared.add(job)
            print('R201C_ALL49_FOURSTATE_RAW_LOCAL_VERIFIED',job,flush=True)
        elif event=='NORMAL_READY':
            job=message['job']
            assert job in frozen_cleared and job not in normal
            normal[job]=dict(file=message['file'],local=copy_verified(message['path'],message['file'],Path(ROOT),ARCHIVE))
            process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED',job=job,file=message['file']))+'\n')
            process.stdin.flush()
        elif event=='NORMAL_CLEARED':
            job=message['job']
            assert job in normal and job not in normal_cleared
            normal_cleared.add(job)
        else:
            assert event=='CONTROLLER_COMPLETE' and not complete
            assert message['runs']==4 and message['frozen_bundles']==196 and message['state_cases']==784
            complete=True
    process.stdin.close()
    assert process.wait()==0 and complete
    assert len(frozen)==len(frozen_cleared)==len(normal)==len(normal_cleared)==4
    proof.write_bytes((json.dumps(dict(status='COMPLETE_R201C_FOUR50_ALL784STATE_GT_AND_LOCAL_RAW',
        exit_code=0,remote_root=ROOT,local_root=str(ARCHIVE),sources_sha256=sources,
        frozen=frozen,normal=normal,finished=datetime.now().isoformat(timespec='seconds')),
        indent=2)+'\n').encode('utf-8'))
    print('RGBNT201_R201C_FOUR50_ARCHIVE_SESSION_COMPLETE',flush=True)


if __name__=='__main__':
    main()
