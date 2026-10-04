"""Observe the six actual full-data jobs every240s using canonical launch PIDs."""
from datetime import datetime
import json
import sys
import time

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, remote_python

launch = json.loads((PROJECT / 'results/preflight/full_official_control_m3b_launch_20261004.json').read_text(encoding='utf-8'))['launch']
pid, root = launch['pid'], launch['output']
out = PROJECT / 'results/preflight/full_official_control_primary_observer_20261004.jsonl'
assert not out.exists()
code = f'''import json,time
from pathlib import Path
root=Path({root!r});runs=[];failures=[];active=[]
for phase in ('contract','preflight','training','frozen49','audit','diagnosis','diagnosis_audit'):
 for file in (root/phase).glob('*_exit.json'):
  row=json.loads(file.read_text())
  if row['exit_code']!=0:
   failures.append(dict(phase=phase,name=row['name'],exit_code=row['exit_code'],tail=(file.parent/(row['name']+'.log')).read_text()[-6000:]))
for mode in ('measurement_only','independent_control'):
 for variant in ('axis_shared','frequency_shared','twins_shared'):
  name='MSVR310_'+mode+'_'+variant+'_s42';run=root/'training'/name
  row=dict(name=name,status='PENDING')
  if (run/'status.json').exists():
   status=json.loads((run/'status.json').read_text());best=status.get('best',{{}})
   row.update(status=status['status'],epoch=status.get('epochs',status.get('epoch',0)),best_epoch=best.get('epoch'),mAP=best.get('mAP'),Rank1=best.get('Rank-1'))
   if 'latest' in status:
    row['latest_epoch_seconds']=status['latest']['seconds'];row['coverage']=status['latest']['cumulative_unique_train_records']
   if 'optimizer_steps' in status:
    row['optimizer_steps']=status['optimizer_steps'];row['amp_skipped_steps']=status['amp_skipped_steps']
  audit49=root/'frozen49'/name/'independent_cpu_audit.json'
  audit294=root/'diagnosis'/name/'independent_cpu_audit.json'
  row['frozen49_cpu_pass']=audit49.exists() and json.loads(audit49.read_text())['status']=='PASS'
  row['states294_cpu_pass']=audit294.exists() and json.loads(audit294.read_text())['status']=='PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT'
  runs.append(row)
for phase in ('contract','preflight','training','frozen49','diagnosis'):
 for file in (root/phase).glob('*_launch.json'):
  launched=json.loads(file.read_text());exited=file.with_name(file.name.replace('_launch.json','_exit.json'))
  if not exited.exists() and Path('/proc/'+str(launched['pid'])).exists():
   assert launched['gpu'] in (2,3)
   active.append(dict(pid=launched['pid'],gpu=launched['gpu'],phase=phase,name=file.name.removesuffix('_launch.json')))
controller_alive=Path('/proc/{pid}').exists();complete=(root/'controller_result.json').exists()
contract=root/'contract/gradient/result.json'
smokes=list((root/'preflight').glob('*/smoke.json'))
preflight=root/'preflight_result.json'
snapshot=dict(controller_pid={pid},controller_alive=controller_alive,controller_complete=complete,
 contract_status=json.loads(contract.read_text())['status'] if contract.exists() else 'PENDING',
 completed_smokes=len(smokes),preflight_pass=preflight.exists(),runs=runs,failures=failures,
 active_neural_processes=active,snapshot_atomic=False,concurrency_enforced_by_serial_gpu_workers=True,
 observed_at=time.time(),temperature_power_control=False)
if not controller_alive and not complete:
 snapshot['controller_log_tail']=Path({(root+'_controller.log')!r}).read_text()[-6000:]
print(json.dumps(snapshot))
'''
while True:
    snapshot = json.loads(remote_python('2026', code))
    snapshot['local_observed_at'] = datetime.now().isoformat(timespec='seconds')
    with out.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(snapshot) + '\n')
    print('FULL_OFFICIAL_CONTROL_OBSERVATION', json.dumps(snapshot), flush=True)
    if snapshot['failures'] or not snapshot['controller_alive'] or snapshot['controller_complete']:
        target = PROJECT / 'results/preflight/full_official_control_primary_observer_terminal_20261004.json'
        assert not target.exists()
        target.write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')
        break
    time.sleep(240)
