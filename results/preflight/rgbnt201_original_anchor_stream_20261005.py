"""Receive only the two R201A frozen49 archives; preserve the live M8 receiver."""
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
        event = message['event']
        if event == 'FROZEN_READY':
            job = message['job']
            assert job not in frozen and len(message['files']) == 49
            frozen[job] = {path: dict(file=info, local=copy_verified(path, info, root, local))
                           for path, info in message['files'].items()}
            process.stdin.write(json.dumps(dict(event='FROZEN_ARCHIVED', job=job, files=message['files'])) + '\n')
            process.stdin.flush()
        elif event == 'FROZEN_CLEARED':
            job = message['job']
            assert job in frozen and job not in cleared
            cleared.add(job)
        elif event == 'RUN_COMPLETE':
            job = message['job']
            assert job in cleared and job not in runs
            assert message['completed_epochs'] == 50 and message['frozen_conditions'] == 49
            runs.add(job)
            print('R201A_FULL50_ALL49_CLOSED', job, flush=True)
        else:
            assert event == 'CONTROLLER_COMPLETE' and not complete
            assert message['runs'] == 2 and message['frozen_cases'] == 98
            complete = True
    process.stdin.close()
    assert process.wait() == 0 and complete and len(frozen) == len(cleared) == len(runs) == 2
    result = dict(status='R201A_STREAM_COMPLETE_TWO_RUNS98_LOCAL_VERIFIED', frozen=frozen,
        completed_runs=sorted(runs), remote_root=str(root), local_root=str(local),
        verified_at=datetime.now().isoformat(timespec='seconds'), exit_code=0)
    proof.write_bytes((json.dumps(result, indent=2) + '\n').encode('utf-8'))
    return result
