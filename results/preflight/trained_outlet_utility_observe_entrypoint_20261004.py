from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,remote_python

p=PROJECT/'results/preflight'
launch=json.loads((p/'trained_outlet_utility_2026_launch.json').read_text(encoding='utf-8'))
started=p/'trained_outlet_utility_observer_started.json'
assert not started.exists()
started.write_text(json.dumps(dict(pid=os.getpid(),controller_pid=launch['pid'],
 started=datetime.now().isoformat(timespec='seconds'),poll_seconds=240),indent=2)+'\n',encoding='utf-8')
identity=None
while True:
    code=f'''import json
from pathlib import Path
r=Path({launch['output']!r});pid={launch['pid']};proc=Path('/proc')/str(pid)
live=proc.exists() and proc.joinpath('stat').read_text().split()[2] not in ('Z','X')
identity=None
if live:
 command=proc.joinpath('cmdline').read_bytes().split(b'\\0')[:-1]
 assert [v.decode() for v in command]=={launch['command']!r}
 identity=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]
exits={{p.relative_to(r).as_posix():json.loads(p.read_text()) for p in r.glob('*/*/*_exit.json')}}
smokes={{p.parent.relative_to(r).as_posix():json.loads(p.read_text()) for p in r.glob('*/*/smoke/smoke.json')}}
completed={{p.parent.relative_to(r).as_posix():dict(status=json.loads(p.read_text())['status'],cases=json.loads(p.read_text())['metric_count']) for p in r.glob('*/*/full/result.json')}}
active=[]
for f in r.glob('*/*/*_launch.json'):
 row=json.loads(f.read_text());d=Path('/proc')/str(row['pid'])
 if d.exists() and d.joinpath('stat').read_text().split()[2] not in ('Z','X'):
  active.append(dict(path=f.relative_to(r).as_posix(),pid=row['pid'],gpu=row['gpu']))
end=json.loads((r/'controller_result.json').read_text()) if (r/'controller_result.json').exists() else None
log=Path({launch['log']!r}).read_text().splitlines()[-8:]
print(json.dumps(dict(pid=pid,live=live,identity=identity,exits=exits,smokes=smokes,completed=completed,active=active,result=end,log_tail=log)))
'''
    row=json.loads(remote_python('2026',code))
    row['observed_at']=datetime.now().isoformat(timespec='seconds')
    if row['live']:
        if identity is None:
            identity=row['identity']
        assert row['identity']==identity
    assert len(row['active'])<=2 and all(v['gpu'] in (2,3) for v in row['active'])
    assert len({v['gpu'] for v in row['active']})==len(row['active'])
    (p/'trained_outlet_utility_latest_snapshot.json').write_text(json.dumps(row,indent=2)+'\n',encoding='utf-8')
    print('TRAINED_OUTLET_OBSERVED',json.dumps(dict(observed_at=row['observed_at'],live=row['live'],
      smokes=len(row['smokes']),complete=len(row['completed']),active=row['active'],log_tail=row['log_tail'])),flush=True)
    if not row['live']:
        (p/'trained_outlet_utility_observer_terminal.json').write_text(json.dumps(row,indent=2)+'\n',encoding='utf-8')
        assert row['result']['status']=='COMPLETE' and len(row['result']['runs'])==6
        assert all(v['exit_code']==0 for v in row['exits'].values())
        break
    time.sleep(240)
