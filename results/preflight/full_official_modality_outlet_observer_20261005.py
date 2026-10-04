"""Observe the one M4 controller at240 seconds; never relaunch its jobs."""
from datetime import datetime
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, remote_python

pf = PROJECT / 'results/preflight'
launch = json.loads((pf / 'full_official_modality_outlet_launch_20261005.json').read_text(encoding='utf-8'))['launch']
pid, source = launch['pid'], launch['output']
destination = pf / 'full_official_modality_outlet_primary_observer_20261005.jsonl'
terminal = pf / 'full_official_modality_outlet_primary_observer_terminal_20261005.json'
assert not destination.exists() and not terminal.exists()
code = f'''import json
from pathlib import Path
root=Path({source!r})
def alive(pid):
 p=Path('/proc')/str(pid)/'stat'
 return p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z'
result=dict(controller_pid={pid},controller_alive=alive({pid}),controller_complete=False,runs=[],active_neural_processes=[],failures=[])
preflight=root/'preflight_result.json'
result['preflight_pass']=preflight.exists() and json.loads(preflight.read_text())['status']=='PASS'
for variant,gpu in (('axis_shared',2),('frequency_shared',3),('twins_shared',3)):
 name='MSVR310_measurement_only_'+variant+'_s42'
 row=dict(variant=variant,gpu=gpu,contract_pass=False,smoke_pass=False,training_epoch=0,training_complete=False,frozen_complete=False,state_cpu_pass=False)
 contract=root/'contract'/name/'result.json'
 if contract.exists():row['contract_pass']=json.loads(contract.read_text())['status']=='PASS_FULL_OFFICIAL_PM_OUTLET_CONTRACT'
 smoke=root/'preflight'/name/'smoke.json'
 if smoke.exists():
  data=json.loads(smoke.read_text());row.update(smoke_pass=data['status']=='SMOKE_PASS',smoke_updates=data['steps'],smoke_amp_skips=data['amp_skipped_steps'])
 status=root/'training'/name/'status.json'
 if status.exists():
  data=json.loads(status.read_text());row['training_complete']=data['status']=='COMPLETE'
  row['training_epoch']=data['epochs'] if row['training_complete'] else data['epoch']
  if row['training_complete']:row.update(optimizer_updates=data['optimizer_steps'],amp_skips=data['amp_skipped_steps'])
 frozen=root/'frozen49'/name/'result.json'
 if frozen.exists():row['frozen_complete']=json.loads(frozen.read_text())['status']=='COMPLETE'
 audit=root/'diagnosis'/name/'independent_cpu_audit.json'
 if audit.exists():
  data=json.loads(audit.read_text());row.update(state_cpu_pass=data['status']=='PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT' and data['cases']==294,metric_cases=data['cases'])
 for phase in ('contract','preflight','training','frozen49','audit','diagnosis','diagnosis_audit'):
  entry=root/phase/(name+'_launch.json');exitfile=root/phase/(name+'_exit.json')
  if entry.exists() and phase not in ('audit','diagnosis_audit'):
   child=json.loads(entry.read_text())
   if alive(child['pid']):result['active_neural_processes'].append(dict(pid=child['pid'],gpu=gpu,phase=phase,variant=variant))
  if exitfile.exists():
   value=json.loads(exitfile.read_text())
   if value['exit_code']!=0:result['failures'].append(dict(variant=variant,phase=phase,exit_code=value['exit_code']))
 result['runs'].append(row)
done=root/'controller_result.json'
if done.exists():
 data=json.loads(done.read_text());result.update(controller_complete=data['status']=='COMPLETE',metric_cases=data['frozen_state_metric_cases'],paired_identity_and_partial_sampling_exact=data['paired_identity_and_partial_sampling_exact'])
result['snapshot_atomic']=False;result['temperature_power_control']=False
print(json.dumps(result))
'''
while True:
    record = json.loads(remote_python('2026', code))
    record['local_observed_at'] = datetime.now().isoformat(timespec='seconds')
    with destination.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record) + '\n')
    print('FULL_OFFICIAL_M4_OBSERVATION', json.dumps(record), flush=True)
    if record['controller_complete']:
        assert record['metric_cases'] == 882 and record['paired_identity_and_partial_sampling_exact']
        assert record['preflight_pass'] and all(row['state_cpu_pass'] and row['frozen_complete'] for row in record['runs'])
        assert not record['active_neural_processes'] and not record['failures']
        terminal.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        break
    assert record['controller_alive'] and not record['failures'], record
    time.sleep(240)
