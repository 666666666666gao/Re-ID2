"""Restore exact selected arrays, receive bounded raw streams, then clear restored copies."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0, str(PROJECT))
from collect_results import OPTIONS, command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified

REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
NORMAL = REMOTE + '/runs/identity_coordinate_full_normal_three_20261005'
OLD = REMOTE + '/runs/rgbnt201_identity_outlet_r201c_20261005'
ROOT = REMOTE + '/runs/identity_coordinate_missing49_stream_20261005'
ARCHIVE_ROOT = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('native', 'full'), required=True)
    args = parser.parse_args()
    pf = PROJECT / 'results/preflight'
    original = json.loads((pf / 'rgbnt201_identity_outlet50_actual_session_20261005.json').read_text(encoding='utf-8'))
    normal = json.loads((pf / 'identity_coordinate_three_normal_actual_session_20261005.json').read_text(encoding='utf-8'))
    assert original['status'] == 'COMPLETE_R201C_FOUR50_ALL784STATE_GT_AND_LOCAL_RAW' and original['exit_code'] == 0
    assert normal['status'] == 'COMPLETE_THREE_NORMAL_DATASETS_AND_FOUR_NEW50_GT_AND_LOCAL_RAW' and normal['exit_code'] == 0
    assert len(normal['archives']) == 4 and normal['new_successful_updates'] == 14124
    review = json.loads((pf / 'identity_coordinate_missing49_stream_source_review_20261005.json').read_text(encoding='utf-8'))
    assert review['status'] == 'PASS' and not review['blocking_findings']
    for name, digest in review['sources_sha256'].items():
        assert hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == digest
    accepted = json.loads((pf / 'identity_coordinate_three_normal_source_review_20261005.json').read_text(encoding='utf-8'))
    runtime = accepted['sources_sha256'] | accepted['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == digest for name, digest in runtime.items())
    receipt = pf / ('identity_coordinate_missing49_stream_' + args.mode + '_actual_session_20261005.json')
    assert not receipt.exists()
    if args.mode == 'full':
        native = json.loads((pf / 'identity_coordinate_missing49_stream_native_actual_session_20261005.json').read_text(encoding='utf-8'))
        assert native['status'] == 'ACTUAL_FOUR_NATIVE_INFERENCE_CONTROLS_ALL16_NORMAL_GT_AND_LOCAL_RAW'
        assert native['exit_code'] == 0 and len(native['raw']) == 4 and native['sources_sha256'] == review['sources_sha256']
    restores = {}
    for name, value in normal['archives'].items():
        local = Path(value['local'])
        assert local.resolve().is_relative_to(ARCHIVE_ROOT.resolve())
        assert local.stat().st_size == value['file']['bytes'] and hashlib.sha256(local.read_bytes()).hexdigest() == value['file']['sha256']
        restores[NORMAL + '/training/' + name + '/best_official_arrays_restore_missing49.npz'] = value['file']
    local_archive = ARCHIVE_ROOT / 'identity_coordinate_missing49_stream_20261005'
    remote_code = f'''import hashlib,json
from pathlib import Path
root=Path({REMOTE!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in {runtime!r}.items() if not name.startswith('results/'))
normal=json.loads((Path({NORMAL!r})/'controller_result.json').read_text())
old=json.loads((Path({OLD!r})/'controller_result.json').read_text())
assert normal['status']==old['status']=='COMPLETE' and normal['normal_datasets_completed']==3
assert old['closed_state_cases']==784 and old['all_raw_local_verified']
paths={restores!r}
for name,file in paths.items():
    path=Path(name)
    if {args.mode!r}=='native': assert not path.exists()
    else: assert path.stat().st_size==file['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==file['sha256']
print('ACTUAL_THREE_NORMAL_AND_OLD_QUEUE_CLOSED_NO_NEW_NEURAL_RUN_YET')'''
    print(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=remote_code), end='')
    if args.mode == 'native':
        assert not local_archive.exists()
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
    argv = [PYTHON, '-u', 'launch_identity_coordinate_missing49_stream.py', '--normal-root', NORMAL,
        '--old-root', OLD, '--output', ROOT, '--mode', args.mode]
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
            expected = 4 if args.mode == 'native' else 196
            assert event == 'CONTROLLER_COMPLETE' and not complete
            assert message['mode'] == args.mode and message['controls'] == 4 and message['conditions'] == expected
            assert message['cases'] == 4 * expected and message['optimizer_updates'] == 0
            complete = True
    process.stdin.close()
    assert process.wait() == 0 and complete and set(raw) == cleared
    assert len(raw) == (4 if args.mode == 'native' else 196)
    if args.mode == 'full':
        code = f'''import hashlib
from pathlib import Path
for name,file in {restores!r}.items():
    path=Path(name)
    assert path.name=='best_official_arrays_restore_missing49.npz' and path.resolve().is_relative_to(Path({NORMAL!r}).resolve())
    assert path.stat().st_size==file['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==file['sha256']
    path.unlink()
print('ONLY_FOUR_EXACT_RESTORED_SELECTED_ARRAY_COPIES_CLEARED_ORIGINAL_LOCAL_FILES_RETAINED')'''
        print(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=code), end='')
    status = 'ACTUAL_FOUR_NATIVE_INFERENCE_CONTROLS_ALL16_NORMAL_GT_AND_LOCAL_RAW' if args.mode == 'native' else 'ACTUAL_FOUR_FULL49_INFERENCE_CONTROLS_ALL784_GT_AND_LOCAL_RAW'
    receipt.write_text(json.dumps(dict(status=status, finished=datetime.now().isoformat(timespec='seconds'), exit_code=0,
        mode=args.mode, raw=raw, remote_root=ROOT, local_root=str(local_archive), new_optimizer_updates=0,
        restored_selected_arrays=len(restores), restored_copies_cleared=args.mode == 'full', sources_sha256=review['sources_sha256'],
        source_review='results/preflight/identity_coordinate_missing49_stream_source_review_20261005.json',
        limits='Actual frozen inference/all declared GT and raw copies only; no guaranteed +2 advantage, calibrated contribution or multiseed claim.'), indent=2) + '\n', encoding='utf-8')
    print('IDENTITY_MISSING_STREAM_SESSION_CLOSED', args.mode, len(raw), flush=True)


if __name__ == '__main__':
    main()
