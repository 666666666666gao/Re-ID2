from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,remote_python

p=PROJECT/'results/preflight';launch=json.loads((p/'frequency_relation_m2_2026_launch.json').read_text(encoding='utf-8'))
start=p/'frequency_relation_m2_observer_started.json';assert not start.exists()
start.write_text(json.dumps(dict(pid=os.getpid(),controller_pid=launch['pid'],started=datetime.now().isoformat(timespec='seconds'),poll_seconds=240),indent=2)+'\n',encoding='utf-8')
identity=None
while True:
 code=f'''import json
from pathlib import Path
r=Path({launch['output']!r});pid={launch['pid']};proc=Path('/proc')/str(pid)
live=proc.exists() and proc.joinpath('stat').read_text().split()[2] not in ('Z','X')
identity=None
if live:
 assert [v.decode() for v in proc.joinpath('cmdline').read_bytes().split(b'\\0')[:-1]]=={launch['command']!r}
 identity=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]
tensor=json.loads((r/'preflight/tensor/result.json').read_text()) if (r/'preflight/tensor/result.json').exists() else None
smokes={{f.parent.name:json.loads(f.read_text()) for f in (r/'preflight').glob('*/smoke.json')}}
training={{}}
for f in (r/'original_mean/development').glob('*/status.json'):
 v=json.loads(f.read_text());training[f.parent.name]=dict(status=v['status'],epoch=v.get('epoch',v.get('epochs')),steps=v['steps'],best=v['best'])
exits={{f.relative_to(r).as_posix():json.loads(f.read_text()) for f in r.rglob('*_exit.json')}}
active=[]
for f in r.rglob('*_launch.json'):
 v=json.loads(f.read_text());c=Path('/proc')/str(v['pid'])
 if c.exists() and c.joinpath('stat').read_text().split()[2] not in ('Z','X'):
  active.append(dict(path=f.relative_to(r).as_posix(),pid=v['pid'],gpu=v['gpu']))
result=json.loads((r/'controller_result.json').read_text()) if (r/'controller_result.json').exists() else None
print(json.dumps(dict(pid=pid,live=live,identity=identity,tensor=tensor,smokes=smokes,training=training,exits=exits,active=active,result=result,log_tail=Path({launch['log']!r}).read_text().splitlines()[-12:])))
'''
 row=json.loads(remote_python('2026',code));row['observed_at']=datetime.now().isoformat(timespec='seconds')
 if row['live']:
  if identity is None:identity=row['identity']
  assert row['identity']==identity
 assert len(row['active'])<=2 and all(v['gpu'] in (2,3) for v in row['active'])
 assert len({v['gpu'] for v in row['active']})==len(row['active'])
 (p/'frequency_relation_m2_latest_snapshot.json').write_text(json.dumps(row,indent=2)+'\n',encoding='utf-8')
 print('M2_OBSERVED',json.dumps(dict(observed_at=row['observed_at'],live=row['live'],tensor=row['tensor'] is not None,
  smokes=len(row['smokes']),training=row['training'],active=row['active'],log_tail=row['log_tail'])),flush=True)
 if not row['live']:
  (p/'frequency_relation_m2_observer_terminal.json').write_text(json.dumps(row,indent=2)+'\n',encoding='utf-8')
  assert row['result']['status']=='COMPLETE' and len(row['result']['runs'])==3
  assert all(v['exit_code']==0 for v in row['exits'].values())
  break
 time.sleep(240)
