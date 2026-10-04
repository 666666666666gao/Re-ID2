from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, remote_python

p = PROJECT / 'results/preflight'
launch = json.loads((p / 'cross_identity_coordinate_2026_launch.json').read_text(encoding='utf-8'))
started = p / 'cross_identity_coordinate_observer_started.json'
assert not started.exists()
started.write_text(json.dumps(dict(pid=os.getpid(), started=datetime.now().isoformat(timespec='seconds'),
    controller_pid=launch['pid'], poll_seconds=240, neural_launches=0), indent=2) + '\n', encoding='utf-8')
identity = None
while True:
    code = f'''import json
from pathlib import Path
pid={launch['pid']};root=Path({launch['output']!r});proc=Path('/proc')/str(pid)
live=proc.exists() and b'launch_cross_identity_coordinates.py' in (proc/'cmdline').read_bytes()
identity=(proc/'stat').read_text().split()[21] if live else None
smokes={{}};full={{}};exits={{}};active=[]
for variant in ('axis_shared','frequency_shared','twins_shared'):
 folder=root/variant
 for stage in ('smoke','full'):
  target=folder/stage/('smoke.json' if stage=='smoke' else 'result.json')
  if target.exists():
   v=json.loads(target.read_text());summary={{k:v[k] for k in ('status','variant','normal_feature_max_error','optimizer_updates','new_weights')}}
   if stage=='full':summary['metric_count']=v['metric_count']
   (smokes if stage=='smoke' else full)[variant]=summary
  launched=folder/(stage+'_launch.json');exit_path=folder/(stage+'_exit.json')
  if exit_path.exists():exits[variant+'/'+stage]=json.loads(exit_path.read_text())
  elif launched.exists():
   row=json.loads(launched.read_text());child=Path('/proc')/str(row['pid'])
   if child.exists() and b'diagnose_cross_identity_coordinates.py' in (child/'cmdline').read_bytes():active.append(dict(variant=variant,stage=stage,pid=row['pid'],gpu=row['gpu']))
result=json.loads((root/'controller_result.json').read_text()) if (root/'controller_result.json').exists() else None
log=Path({launch['log']!r});tail=log.read_text()[-3000:].splitlines()[-8:] if log.exists() else []
print(json.dumps(dict(pid=pid,live=live,identity=identity,smokes=smokes,full=full,exits=exits,active=active,result=result,log_tail=tail)))
'''
    value = json.loads(remote_python('2026', code))
    value['observed_at'] = datetime.now().isoformat(timespec='seconds')
    assert len(value['active']) <= 2
    assert len({row['gpu'] for row in value['active']}) == len(value['active'])
    assert all(row['gpu'] in (2, 3) for row in value['active'])
    if value['live']:
        if identity is None:
            identity = value['identity']
        assert value['identity'] == identity
    (p / 'cross_identity_coordinate_latest_snapshot.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    first = p / 'cross_identity_coordinate_first_snapshot.json'
    if not first.exists():
        first.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print('CROSS_COORDINATE_OBSERVED', json.dumps({k: value[k] for k in ('observed_at','live','smokes','full','active','log_tail')}), flush=True)
    if not value['live']:
        terminal = p / 'cross_identity_coordinate_observer_terminal.json'
        assert not terminal.exists()
        terminal.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
        break
    time.sleep(240)
