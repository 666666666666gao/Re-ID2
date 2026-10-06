"""After all three G normal datasets, receive the two RGBNT201 fixed-best missing streams."""
import argparse
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
NORMAL = REMOTE + '/runs/r201g_normal_priority_20261006'
OTHER = REMOTE + '/runs/r201g_other_two_normal_20261006'
ROOT = REMOTE + '/runs/r201g_201_missing49_stream_20261006'
ARCHIVE_ROOT = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('native', 'full'), required=True)
    args = parser.parse_args()
    pf = PROJECT / 'results/preflight'
    normal = json.loads((pf/'r201g_normal_priority_actual_session_20261006.json').read_text(encoding='utf-8'))
    other = json.loads((pf/'r201g_other_two_normal_actual_session_20261006.json').read_text(encoding='utf-8'))
    assert normal['status']=='ACTUAL_R201G_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW' and normal['exit_code']==0
    assert normal['successful_updates']==5294 and len(normal['archives'])==2
    assert other['status']=='ACTUAL_UNCHANGED_G_OTHER_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW' and other['exit_code']==0
    assert other['successful_updates']==14124 and other['additional_epochs']==200 and len(other['archives'])==4
    three=json.loads((PROJECT/'results/r201g_other_two_normal_20261006/three_dataset_normal_analysis/result.json').read_text(encoding='utf-8'))
    assert three['status']=='ACTUAL_UNIFIED_G_THREE_OFFICIAL_NORMAL_DATASETS_CPU_READOUT' and three['model_results']==15 and three['paired_query_rows']==15710
    review=json.loads((pf/'r201g_201_missing49_stream_source_review_20261006.json').read_text(encoding='utf-8'))
    assert review['status']=='PASS' and not review['blocking_findings']
    runtime=review['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==digest for name,digest in (review['sources_sha256']|runtime).items())
    receipt=pf/('r201g_201_missing49_stream_'+args.mode+'_actual_session_20261006.json')
    assert not receipt.exists()
    local_archive=ARCHIVE_ROOT/'r201g_201_missing49_stream_20261006'
    if args.mode=='native':
        assert not local_archive.exists()
    else:
        native=json.loads((pf/'r201g_201_missing49_stream_native_actual_session_20261006.json').read_text(encoding='utf-8'))
        assert native['status']=='ACTUAL_G_201_TWO_NATIVE_INFERENCE_CONTROLS_ALL8_NORMAL_GT_AND_LOCAL_RAW'
        assert native['exit_code']==0 and len(native['raw'])==2 and native['sources_sha256']==review['sources_sha256']
    restores = {}
    for name, value in normal['archives'].items():
        local = Path(value['local'])
        assert local.resolve().is_relative_to(ARCHIVE_ROOT.resolve())
        assert local.stat().st_size == value['file']['bytes'] and hashlib.sha256(local.read_bytes()).hexdigest() == value['file']['sha256']
        restores[NORMAL + '/training/' + name + '/best_official_arrays_restore_missing49.npz'] = value['file']
    remote_code = f'''import hashlib,json
from pathlib import Path
root=Path({REMOTE!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in {runtime!r}.items() if not name.startswith('results/'))
normal=json.loads((Path({NORMAL!r})/'controller_result.json').read_text())
assert normal['status']=='COMPLETE' and normal['successful_updates']==5294 and normal['normal_archives_local_verified']==2
for dataset,updates in (('MSVR310',1410),('RGBNT100',12714)):
 closed=json.loads((Path({OTHER!r})/dataset/'controller_result.json').read_text())
 assert closed['status']=='COMPLETE' and closed['successful_updates']==updates and closed['normal_archives_local_verified']==2
if {args.mode!r}=='native': assert not Path({ROOT!r}).exists()
paths={restores!r}
for name,file in paths.items():
    path=Path(name)
    if {args.mode!r}=='native': assert not path.exists()
    else: assert path.stat().st_size==file['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==file['sha256']
print('ACTUAL_ALL_THREE_G_NORMAL_CLOSED_NO_NEW_NEURAL_RUN_YET')'''
    print(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=remote_code), end='')
    if args.mode == 'native':
        for name, value in normal['archives'].items():
            destination = NORMAL + '/training/' + name + '/best_official_arrays_restore_missing49.npz'
            command(['scp', *OPTIONS, value['local'], '2026:' + destination])
        for name in review['sources_sha256']:
            if not name.startswith('results/'):
                command(['scp', *OPTIONS, str(PROJECT / name), '2026:' + REMOTE + '/' + name])
    code = f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in {review['sources_sha256']!r}.items() if not name.startswith('results/'))
assert all(Path(name).stat().st_size==file['bytes'] and hashlib.sha256(Path(name).read_bytes()).hexdigest()==file['sha256'] for name,file in {restores!r}.items())
print('REVIEWED_NEW_EVALUATION_SOURCES_AND_SELECTED_RESTORES_EXACT')'''
    print(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=code), end='')
    argv = [PYTHON, '-u', 'launch_r201g_201_missing49_stream.py', '--training-root', NORMAL,
        '--other-normal-root', OTHER, '--output', ROOT, '--mode', args.mode]
    process = subprocess.Popen(['ssh', *OPTIONS, '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
        '2026', 'cd ' + shlex.quote(REMOTE) + ' && ' + shlex.join(argv)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8')
    raw, cleared, complete = {}, set(), False
    for line in process.stdout:
        if not line.startswith('IDENTITY_MISSING_STREAM '):
            print(line, end='', flush=True)
            continue
        message = json.loads(line[len('IDENTITY_MISSING_STREAM '):])
        event = message['event']
        if event == 'RAW_READY':
            key = message['job'] + '/' + message['condition']
            assert message['installed_gt_cases'] == 4 and key not in raw
            assert shutil.disk_usage(ARCHIVE_ROOT).free > message['file']['bytes']
            local = copy_verified(message['path'], message['file'], Path(ROOT), local_archive)
            raw[key] = dict(remote=message['path'], local=local, file=message['file'])
            process.stdin.write(json.dumps(dict(event='ARCHIVED', job=message['job'], condition=message['condition'], file=message['file'])) + '\n')
            process.stdin.flush()
        elif event == 'RAW_CLEARED':
            key = message['job'] + '/' + message['condition']
            assert key in raw and key not in cleared
            cleared.add(key)
            print('IDENTITY_MISSING_LOCAL_VERIFIED_REMOTE_CLEARED', args.mode, key, flush=True)
        else:
            expected = 2 if args.mode == 'native' else 98
            assert event == 'CONTROLLER_COMPLETE' and not complete
            assert message['mode'] == args.mode and message['controls'] == 2 and message['conditions'] == expected
            assert message['cases'] == 4 * expected and message['optimizer_updates'] == 0
            complete = True
    process.stdin.close()
    assert process.wait() == 0 and complete and set(raw) == cleared
    assert len(raw) == (2 if args.mode == 'native' else 98)
    if args.mode == 'full':
        code = f'''import hashlib
from pathlib import Path
for name,file in {restores!r}.items():
    path=Path(name)
    assert path.name=='best_official_arrays_restore_missing49.npz' and path.resolve().is_relative_to(Path({NORMAL!r}).resolve())
    assert path.stat().st_size==file['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==file['sha256']
    path.unlink()
print('ONLY_TWO_EXACT_RESTORED_SELECTED_ARRAY_COPIES_CLEARED_ORIGINAL_LOCAL_FILES_RETAINED')'''
        print(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=code), end='')
    status = 'ACTUAL_G_201_TWO_NATIVE_INFERENCE_CONTROLS_ALL8_NORMAL_GT_AND_LOCAL_RAW' if args.mode == 'native' else 'ACTUAL_G_201_TWO_FULL49_INFERENCE_CONTROLS_ALL392_GT_AND_LOCAL_RAW'
    receipt.write_text(json.dumps(dict(status=status, finished=datetime.now().isoformat(timespec='seconds'), exit_code=0,
        mode=args.mode, raw=raw, remote_root=ROOT, local_root=str(local_archive), new_optimizer_updates=0,
        restored_selected_arrays=len(restores), restored_copies_cleared=args.mode == 'full', sources_sha256=review['sources_sha256'],
        source_review='results/preflight/r201g_201_missing49_stream_source_review_20261006.json',
        limits='Actual frozen inference/all declared GT and raw copies only; no guaranteed +2 advantage, calibrated contribution or multiseed claim.'), indent=2) + '\n', encoding='utf-8')
    print('IDENTITY_MISSING_STREAM_SESSION_CLOSED', args.mode, len(raw), flush=True)


if __name__ == '__main__':
    main()
