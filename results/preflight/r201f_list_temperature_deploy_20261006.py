"""Fresh-reviewed next factor; begin only after the existing R201D closure."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0, str(PROJECT))
from collect_results import OPTIONS, command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified

REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT = REMOTE + '/runs/r201f_list_temperature_20261006'
ARCHIVE = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201f_list_temperature_20261006')


def main():
    pf = PROJECT / 'results/preflight'
    old = json.loads((pf / 'r201e_list_objective_fp32check_actual_session_20261006.json').read_text(encoding='utf-8'))
    analysis = json.loads((PROJECT / 'results/r201e_list_objective_fp32check_20261006/normal_analysis/result.json').read_text(encoding='utf-8'))
    assert old['exit_code'] == 0 and old['successful_updates'] == 5294
    assert analysis['status'] == 'ACTUAL_R201E_TWO_FULL50_NORMAL_CPU_ANALYSIS_COMPLETE'
    probe = json.loads((pf / 'r201e_training_signal_probe_actual_20261006.json').read_text(encoding='utf-8'))
    assert probe['optimizer_updates'] == 0 and probe['neural_forwards'] == 32
    probe_root = PROJECT / 'results/r201e_training_signal_probe_20261006'
    probe_pids = [json.loads((probe_root / name).read_text(encoding='utf-8'))['pid']
                  for name in ('controller_launch.json', 'frequency_shared_launch.json', 'axis_shared_launch.json')]
    proof = pf / 'r201f_list_temperature_actual_session_20261006.json'
    assert not proof.exists() and not ARCHIVE.exists()
    assert shutil.disk_usage(ARCHIVE.parent).free > 128*1024**2
    review = json.loads((pf / 'r201f_list_temperature_source_review_20261006.json').read_text(encoding='utf-8'))
    assert review['status'] == 'PASS' and not review['blocking_findings']
    sources, reused = review['sources_sha256'], review['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() == sha for name, sha in {**sources, **reused}.items())
    code = f'''import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path({REMOTE!r}); output=Path({ROOT!r})
assert not output.exists() and shutil.disk_usage(root).free>1_100_000_000
old=json.loads((root/'runs/r201e_list_objective_fp32check_20261006/controller_result.json').read_text())
assert old['status']=='COMPLETE' and old['successful_updates']==5294
for pid in {tuple([3672643,3673725,3673726]+probe_pids)!r}:
 assert subprocess.run(['ps','-p',str(pid)],stdout=subprocess.DEVNULL).returncode==1
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {reused!r}.items() if not name.startswith('results/'))
print('R201F_PREVIOUS_E_TWO50_AND_ZERO_UPDATE_PROBE_CLOSED_SOURCES_STORAGE_VERIFIED')'''
    print(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON)+' -'], input=code), end='')
    for name in sources:
        if not name.startswith('results/'):
            command(['scp', *OPTIONS, str(PROJECT/name), '2026:'+REMOTE+'/'+name])
    deployed = {name: sha for name, sha in sources.items() if not name.startswith('results/')}
    code = f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {deployed!r}.items())
print('R201F_NEW_SOURCES_EXACT')'''
    print(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON)+' -'], input=code), end='')
    argv = [PYTHON, '-u', 'launch_r201f_list_temperature.py', '--data-root', '/data/gaob/Re-ID/dataset',
        '--pretrained', '/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
        '--anchor-root', REMOTE+'/runs/full_official_baselines_20261004/training', '--output', ROOT]
    process = subprocess.Popen(['ssh', *OPTIONS, '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', '2026',
        'cd '+shlex.quote(REMOTE)+' && '+shlex.join(argv)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8')
    archives, cleared, native, complete = {}, set(), False, False
    print('R201F_ACTUAL_CONTROLLER_STARTED', flush=True)
    for line in process.stdout:
        if not line.startswith('R201F_STREAM '):
            print(line, end='', flush=True)
            continue
        message = json.loads(line[len('R201F_STREAM '):]); event = message['event']
        if event == 'NATIVE_PASS':
            assert not native and message['controls'] == 2 and message['actual_updates'] == 6
            native = True
            (pf/'r201f_list_temperature_native_actual_20261006.json').write_text(json.dumps(dict(status='ACTUAL_TWO_NATIVE3_LIST_OBJECTIVE_PASS_FORMAL_NEXT',
                observed_at=datetime.now().isoformat(timespec='seconds'), remote_root=ROOT, actual_native_updates=6, new_formal_results=0), indent=2)+'\n', encoding='utf-8')
            print('R201F_ACTUAL_NATIVE6_PASS', flush=True)
        elif event == 'NORMAL_READY':
            name = message['name']; assert native and name not in archives
            assert shutil.disk_usage(ARCHIVE.parent).free > message['file']['bytes']
            local = copy_verified(message['path'], message['file'], Path(ROOT), ARCHIVE)
            archives[name] = dict(file=message['file'], local=local)
            process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED', name=name, file=message['file']))+'\n'); process.stdin.flush()
        elif event == 'NORMAL_CLEARED':
            name = message['name']; assert name in archives and name not in cleared
            cleared.add(name); print('R201F_NORMAL_LOCAL_VERIFIED_REMOTE_CLEARED', name, flush=True)
        else:
            assert event == 'CONTROLLER_COMPLETE' and not complete and message['controls'] == 2 and message['successful_updates'] == 5294
            complete = True
    process.stdin.close()
    code = process.wait()
    assert code == 0 and native and complete and len(archives) == len(cleared) == 2
    proof.write_text(json.dumps(dict(status='ACTUAL_R201F_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW', finished=datetime.now().isoformat(timespec='seconds'),
        exit_code=0, remote_root=ROOT, archives=archives, native_updates=6, successful_updates=5294, additional_epochs=100,
        sources_sha256=sources, limits='No missing or three-dataset/multiseed/+2 success claim.'), indent=2)+'\n', encoding='utf-8')
    print('R201F_TWO50_NORMAL_GT_ARCHIVE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
