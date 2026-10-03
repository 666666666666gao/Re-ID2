from datetime import datetime
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, copy_file, remote_python

receipt = json.loads((PROJECT / 'results/preflight/shared_states_2026_launch.json').read_text())
root, pid = receipt['output'], receipt['pid']
deadline = time.time() + 1200
while True:
    code = f'''import json,subprocess
from pathlib import Path
root=Path({root!r});stages={{}}
for path in root.glob('**/*_exit.json'):
 stages[str(path.relative_to(root))]=json.loads(path.read_text())
for path in root.glob('**/status.json'):
 stages[str(path.relative_to(root))]=json.loads(path.read_text())
for path in (root/'preflight/tensor/result.json',root/'preflight_result.json',root/'controller_result.json'):
 if path.exists():stages[str(path.relative_to(root))]=json.loads(path.read_text())
live=Path('/proc/{pid}').exists()
if live: live=Path('/proc/{pid}/stat').read_text().split()[2]!='Z'
log=Path({receipt['log']!r})
print(json.dumps(dict(pid={pid},live=live,stages=stages,controller_tail=log.read_text()[-6000:] if log.exists() else '',gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True))))
'''
    snapshot = json.loads(remote_python('2026', code))
    snapshot['observed_at'] = datetime.now().isoformat(timespec='seconds')
    (PROJECT / 'results/preflight/shared_states_latest_snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    rows = snapshot['stages']
    failure = [name for name,value in rows.items() if name.endswith('_exit.json') and value['exit_code'] != 0]
    progress = {name: value.get('epoch', value.get('status')) for name,value in rows.items()}
    print('ACTUAL_CONTROLLER', json.dumps(dict(observed_at=snapshot['observed_at'],live=snapshot['live'],progress=progress,failures=failure,gpus=snapshot['gpus'])), flush=True)
    if failure or not snapshot['live'] or 'controller_result.json' in rows:
        (PROJECT / 'results/preflight/shared_states_observer_terminal.json').write_text(json.dumps(snapshot, indent=2) + '\n')
        break
    if time.time() >= deadline:
        print('OBSERVATION_WINDOW_ENDED_SAME_CONTROLLER_MAY_STILL_BE_RUNNING_NO_RESTART', flush=True)
        break
    time.sleep(240)
