"""M8's three-run receiver; M7's active five-run receiver remains unchanged."""
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import OPTIONS
from results.preflight.full_official_anchor_stream_bridge_20261005 import PREFIX, REMOTE_PROJECT, copy_verified


def archive_session(argv, root, local, proof_path):
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
        job = message.get('job')
        if event == 'RAW_READY':
            key = job + '/' + message['condition']
            assert key not in receipts and message['installed_gt_cases'] == 6 and message['full11_exact']
            destination = copy_verified(message['path'], message['file'], root, local)
            receipts[key] = dict(file=message['file'], remote=message['path'], local=destination)
            process.stdin.write(json.dumps(dict(event='ARCHIVED', job=job, condition=message['condition'], file=message['file'])) + '\n')
            process.stdin.flush()
        elif event == 'RAW_CLEARED':
            key = job + '/' + message['condition']
            assert key in receipts and key not in cleared
            cleared.add(key)
            print('LOCAL_RAW_VERIFIED_REMOTE_CLEARED', key, flush=True)
        elif event == 'FROZEN_READY':
            assert job not in frozen and len(message['files']) == 49
            frozen[job] = {path: dict(file=info, local=copy_verified(path, info, root, local)) for path, info in message['files'].items()}
            process.stdin.write(json.dumps(dict(event='FROZEN_ARCHIVED', job=job, files=message['files'])) + '\n')
            process.stdin.flush()
        elif event == 'FROZEN_CLEARED':
            assert job in frozen and job not in frozen_cleared
            frozen_cleared.add(job)
        elif event == 'RUN_COMPLETE':
            assert job not in runs and job in frozen_cleared
            assert message['completed_epochs'] == 50 and message['frozen_conditions'] == 49 and message['state_cases'] == 294
            runs.add(job)
            print('FULL50_ALL49_RUN_CLOSED', job, flush=True)
        else:
            assert event == 'CONTROLLER_COMPLETE' and not complete
            assert message['runs'] == 3 and message['frozen_cases'] == 147 and message['state_cases'] == 882
            complete = True
    process.stdin.close()
    assert process.wait() == 0 and complete and set(receipts) == cleared
    assert len(receipts) == 147 and len(frozen) == len(frozen_cleared) == len(runs) == 3
    assert all(sum(k.startswith(job + '/') for k in receipts) == 49 for job in runs)
    result = dict(status='STREAM_SESSION_COMPLETE_ALL_DECLARED_ARCHIVES_VERIFIED', mode='M8_train3', raw=receipts,
        frozen=frozen, completed_runs=sorted(runs), remote_root=str(root), local_root=str(local),
        verified_at=datetime.now().isoformat(timespec='seconds'), exit_code=0)
    proof_path.write_bytes((json.dumps(result, indent=2) + '\n').encode('utf-8'))
    return result
