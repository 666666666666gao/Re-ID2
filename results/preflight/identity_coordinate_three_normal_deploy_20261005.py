"""Deploy reviewed normal-first sources and receive four selected normal arrays."""
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
ROOT = REMOTE + '/runs/identity_coordinate_full_normal_three_20261005'
ARCHIVE = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/identity_coordinate_full_normal_three_20261005')


def main():
    pf = PROJECT / 'results/preflight'
    review = json.loads((pf / 'identity_coordinate_three_normal_source_review_20261005.json').read_text(encoding='utf-8'))
    assert review['status'] == 'PASS' and not review['blocking_findings']
    sources = review['sources_sha256']
    assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == digest for name,digest in sources.items())
    assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == digest
        for name,digest in review['directly_reused_sources_sha256'].items())
    hold = json.loads((pf / 'three_dataset_priority_missing_deferred_20261005.json').read_text(encoding='utf-8'))
    assert hold['controller_pid'] == 1808059 and hold['neural_restarts'] == 0
    proof = pf / 'identity_coordinate_three_normal_actual_session_20261005.json'
    assert not proof.exists() and not ARCHIVE.exists()
    remote_code = f'''import hashlib,json,subprocess
from pathlib import Path
root=Path({REMOTE!r}); output=Path({ROOT!r})
assert not output.exists()
assert subprocess.check_output(['ps','-p','1808059','-o','stat='],text=True).strip().startswith('T')
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    for name,digest in {review['directly_reused_sources_sha256']!r}.items())
print('UNCHANGED_REMOTE_PARENTS_AND_OWN_PRIORITY_HOLD_VERIFIED')'''
    print(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=remote_code), end='')
    for name in sources:
        if name.startswith('results/'):
            continue
        command(['scp', *OPTIONS, str(PROJECT / name), '2026:' + REMOTE + '/' + name])
    deployed = {name:digest for name,digest in sources.items() if not name.startswith('results/')}
    remote_code = f'''import hashlib
from pathlib import Path
root=Path({REMOTE!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in {deployed!r}.items())
print('NEW_NORMAL_FIRST_RUNTIME_SOURCES_EXACT')'''
    print(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON) + ' -'], input=remote_code), end='')
    argv = [PYTHON, '-u', 'launch_identity_coordinate_three_normal.py',
        '--data-root', '/data/gaob/Re-ID/dataset', '--pretrained', '/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
        '--anchor-root', REMOTE + '/runs/full_official_baselines_20261004/training',
        '--old-controller-root', REMOTE + '/runs/rgbnt201_identity_outlet_r201c_20261005', '--output', ROOT]
    process = subprocess.Popen(['ssh', *OPTIONS, '2026', 'cd ' + shlex.quote(REMOTE) + ' && ' + shlex.join(argv)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8')
    archived, cleared, complete, native = {}, set(), False, False
    print('THREE_DATASET_NORMAL_FIRST_CONTROLLER_STARTED', flush=True)
    for line in process.stdout:
        if not line.startswith('THREE_NORMAL_STREAM '):
            print(line, end='', flush=True)
            continue
        message = json.loads(line[len('THREE_NORMAL_STREAM '):])
        event = message['event']
        if event == 'NATIVE_PASS':
            assert not native and message['controls'] == 4 and message['actual_updates'] == 12
            native = True
            print('THREE_NORMAL_NATIVE12_ACTUAL_PASS', flush=True)
        elif event == 'NORMAL_READY':
            name = message['name']
            assert native and name not in archived
            archived[name] = dict(file=message['file'], local=copy_verified(message['path'], message['file'], Path(ROOT), ARCHIVE))
            process.stdin.write(json.dumps(dict(event='NORMAL_ARCHIVED', name=name, file=message['file'])) + '\n')
            process.stdin.flush()
        elif event == 'NORMAL_CLEARED':
            name = message['name']
            assert name in archived and name not in cleared
            cleared.add(name)
            print('SELECTED_NORMAL_ARRAY_LOCAL_VERIFIED_REMOTE_CLEARED', name, flush=True)
        else:
            assert event == 'CONTROLLER_COMPLETE' and not complete
            assert message['normal_models'] == 4 and message['normal_datasets'] == 3
            assert message['missing_controller_resumed']
            complete = True
    process.stdin.close()
    assert process.wait() == 0 and complete and native and len(archived) == len(cleared) == 4
    proof.write_bytes((json.dumps(dict(status='COMPLETE_THREE_NORMAL_DATASETS_AND_FOUR_NEW50_GT_AND_LOCAL_RAW',
        finished=datetime.now().isoformat(timespec='seconds'), exit_code=0, remote_root=ROOT,
        archives=archived, native_updates=12, new_successful_updates=14124, additional_epochs=200,
        old_missing_controller_resumed=1808059, old_receiver=17662, sources_sha256=sources), indent=2) + '\n').encode('utf-8'))
    print('THREE_NORMAL_FULL50_GT_ARCHIVE_SESSION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
