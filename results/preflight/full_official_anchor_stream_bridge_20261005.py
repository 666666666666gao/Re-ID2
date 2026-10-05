"""Local size/SHA acknowledgement for the reviewed remote stream controller."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import OPTIONS, command

PREFIX = 'OFFICIAL_STREAM '
REMOTE_PROJECT = '/data/gaob/Re-ID/DeMo-DualAxis'
REMOTE_PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'


def copy_verified(remote, info, root, local):
    relative = Path(remote).relative_to(root)
    destination = local / relative
    assert not destination.exists()
    destination.parent.mkdir(parents=True, exist_ok=True)
    command(['scp', *OPTIONS, '2026:' + remote, str(destination)])
    assert destination.stat().st_size == info['bytes']
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == info['sha256']
    return str(destination)


def archive_session(argv, root, local, proof_path, mode):
    assert mode in ('gate', 'train', 'preflight')
    assert not proof_path.exists()
    process = subprocess.Popen(['ssh', *OPTIONS, '2026', 'cd ' + shlex.quote(REMOTE_PROJECT) + ' && ' + shlex.join(argv)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8')
    receipts, cleared, frozen, frozen_cleared, runs = {}, set(), {}, set(), set()
    complete = False
    for line in process.stdout:
        if not line.startswith(PREFIX):
            print(line, end='', flush=True)
            continue
        message = json.loads(line[len(PREFIX):])
        event = message['event']
        if event == 'RAW_READY':
            assert mode in ('gate', 'train')
            job = message['job'] if mode == 'train' else 'M6_neural_stream_gate'
            key = job + '/' + message['condition']
            assert key not in receipts and message['installed_gt_cases'] == 6 and message['full11_exact']
            destination = copy_verified(message['path'], message['file'], root, local)
            receipts[key] = dict(file=message['file'], remote=message['path'], local=destination)
            ack = dict(event='ARCHIVED', condition=message['condition'], file=message['file'])
            if mode == 'train':
                ack['job'] = job
            process.stdin.write(json.dumps(ack) + '\n'); process.stdin.flush()
        elif event == 'RAW_CLEARED':
            assert mode in ('gate', 'train')
            job = message['job'] if mode == 'train' else 'M6_neural_stream_gate'
            key = job + '/' + message['condition']
            assert key in receipts and key not in cleared
            cleared.add(key)
            print('LOCAL_RAW_VERIFIED_REMOTE_CLEARED', key, flush=True)
        elif event == 'FROZEN_READY':
            assert mode == 'train' and message['job'] not in frozen and len(message['files']) == 49
            frozen[message['job']] = {path: dict(file=info, local=copy_verified(path, info, root, local)) for path, info in message['files'].items()}
            process.stdin.write(json.dumps(dict(event='FROZEN_ARCHIVED', job=message['job'], files=message['files'])) + '\n')
            process.stdin.flush()
        elif event == 'FROZEN_CLEARED':
            assert mode == 'train' and message['job'] in frozen and message['job'] not in frozen_cleared
            frozen_cleared.add(message['job'])
        elif event == 'RUN_COMPLETE':
            assert mode == 'train' and message['job'] not in runs and message['job'] in frozen_cleared
            runs.add(message['job'])
            print('FULL50_ALL49_RUN_CLOSED', message['job'], flush=True)
        else:
            expected = {'gate': 'DIAGNOSIS_COMPLETE', 'train': 'CONTROLLER_COMPLETE', 'preflight': 'PREFLIGHT_COMPLETE'}[mode]
            assert event == expected and not complete
            complete = True
    process.stdin.close()
    assert process.wait() == 0 and complete and set(receipts) == cleared
    if mode == 'gate':
        assert len(receipts) == 49 and not frozen and not runs
    elif mode == 'train':
        assert len(receipts) == 196 and len(frozen) == len(frozen_cleared) == len(runs) == 5
    else:
        assert not receipts and not frozen and not runs
    result = dict(status='STREAM_SESSION_COMPLETE_ALL_DECLARED_ARCHIVES_VERIFIED', mode=mode,
                  raw=receipts, frozen=frozen, completed_runs=sorted(runs), remote_root=str(root), local_root=str(local),
                  verified_at=datetime.now().isoformat(timespec='seconds'), exit_code=0)
    proof_path.write_bytes((json.dumps(result, indent=2) + '\n').encode('utf-8'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preflight', 'train'), required=True)
    for key in ('root', 'local-root', 'proof', 'anchor-run-dir'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    archive_session([REMOTE_PYTHON, '-u', 'launch_full_official_frozen_anchor_trial.py', '--mode', args.mode,
                     '--data-root', '/data/gaob/Re-ID/dataset', '--pretrained', '/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
                     '--anchor-run-dir', args.anchor_run_dir, '--output', args.root],
                    Path(args.root), Path(args.local_root), Path(args.proof), args.mode)
