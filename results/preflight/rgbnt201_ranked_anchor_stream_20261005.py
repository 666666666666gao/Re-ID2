"""Receive the single R201B best49 archive before its bounded remote cleanup."""
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import OPTIONS
from results.preflight.full_official_anchor_stream_bridge_20261005 import PREFIX, REMOTE_PROJECT, copy_verified


def archive_session(argv, root, local, proof):
    assert not proof.exists()
    process = subprocess.Popen(['ssh', *OPTIONS, '2026', 'cd ' + shlex.quote(REMOTE_PROJECT) + ' && ' + shlex.join(argv)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8')
    frozen, cleared, runs = {}, set(), set()
    complete = False
    for line in process.stdout:
        if not line.startswith(PREFIX):
            print(line, end='', flush=True)
            continue
        message = json.loads(line[len(PREFIX):])
        event, job = message['event'], message.get('job')
        if event == 'FROZEN_READY':
            assert not frozen and len(message['files']) == 49
            frozen[job] = {path: dict(file=info, local=copy_verified(path, info, root, local))
                           for path, info in message['files'].items()}
            process.stdin.write(json.dumps(dict(event='FROZEN_ARCHIVED', job=job, files=message['files'])) + '\n')
            process.stdin.flush()
        elif event == 'FROZEN_CLEARED':
            assert job in frozen and not cleared
            cleared.add(job)
        elif event == 'RUN_COMPLETE':
            assert job in cleared and not runs
            assert message['completed_epochs'] == 50 and message['frozen_conditions'] == 49
            runs.add(job)
        else:
            assert event == 'CONTROLLER_COMPLETE' and not complete
            assert message['runs'] == 1 and message['frozen_cases'] == 49
            complete = True
    process.stdin.close()
    assert process.wait() == 0 and complete and len(frozen) == len(cleared) == len(runs) == 1
    value = dict(status='R201B_STREAM_COMPLETE_ONE_RUN49_LOCAL_VERIFIED', frozen=frozen,
        completed_runs=sorted(runs), remote_root=str(root), local_root=str(local),
        verified_at=datetime.now().isoformat(timespec='seconds'), exit_code=0)
    proof.write_bytes((json.dumps(value, indent=2) + '\n').encode('utf-8'))
    return value
