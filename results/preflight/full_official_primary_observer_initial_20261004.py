"""Track authoritative launch PIDs every240s; DataLoader workers are not NN jobs."""
from datetime import datetime
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, remote_python

proof = json.loads((PROJECT / 'results/preflight/full_official_baselines_launch_20261004.json').read_text(encoding='utf-8'))
pid = proof['launch']['pid']
root = proof['launch']['output']
out = PROJECT / 'results/preflight/full_official_baselines_primary_observer_20261004.jsonl'
assert not out.exists()
code = f'''import json,os,time
from pathlib import Path
root=Path({root!r});runs=[];failures=[]
for phase in ('preflight','training','frozen49','audit'):
 for f in (root/phase).glob('*_exit.json'):
  row=json.loads(f.read_text())
  if row['exit_code']!=0:failures.append(dict(phase=phase,name=row['name'],exit_code=row['exit_code'],tail=(f.parent/(row['name']+'.log')).read_text()[-6000:]))
for dataset in ('MSVR310','RGBNT201','RGBNT100'):
 for variant in ('demo','demo_shared'):
  name=dataset+'_'+variant+'_s42';run=root/'training'/name
  path=run/'status.json';row=dict(name=name,status='PENDING')
  if path.exists():
   s=json.loads(path.read_text());b=s.get('best',{{}})
   row.update(status=s['status'],epoch=s.get('epochs',s.get('epoch',0)),best_epoch=b.get('epoch'),mAP=b.get('mAP'),Rank1=b.get('Rank-1'))
   if 'latest' in s:row['latest_epoch_seconds']=s['latest']['seconds'];row['coverage']=s['latest']['cumulative_unique_train_records']
   if 'optimizer_steps' in s:row['optimizer_steps']=s['optimizer_steps'];row['amp_skipped_steps']=s['amp_skipped_steps']
  audit=root/'frozen49'/name/'independent_cpu_audit.json'
  row['audit_complete']=audit.exists()
  runs.append(row)
active=[]
for phase in ('preflight','training','frozen49'):
 for f in (root/phase).glob('*_launch.json'):
  launch=json.loads(f.read_text())
  exitfile=f.with_name(f.name.replace('_launch.json','_exit.json'))
  if not exitfile.exists() and Path('/proc/'+str(launch['pid'])).exists():
   assert launch['gpu'] in (2,3)
   active.append(dict(pid=launch['pid'],gpu=str(launch['gpu']),phase=phase,name=f.name[:-12]))
assert len(active)<=2 and len({{r['gpu'] for r in active}})==len(active)
pre=root/'preflight_result.json';terminal=root/'controller_result.json'
print(json.dumps(dict(controller_pid={pid},controller_alive=Path('/proc/{pid}').exists(),preflight_pass=pre.exists(),
 controller_complete=terminal.exists(),runs=runs,failures=failures,active_neural_processes=active,observed_at=time.time(),temperature_power_control=False)))
'''
while True:
    snapshot = json.loads(remote_python('2026', code))
    snapshot['local_observed_at'] = datetime.now().isoformat(timespec='seconds')
    with out.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(snapshot) + '\n')
    print('FULL_OFFICIAL_OBSERVATION', json.dumps(snapshot), flush=True)
    if snapshot['failures'] or not snapshot['controller_alive'] or snapshot['controller_complete']:
        target = PROJECT / 'results/preflight/full_official_baselines_primary_observer_terminal_20261004.json'
        assert not target.exists()
        target.write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')
        break
    time.sleep(240)
