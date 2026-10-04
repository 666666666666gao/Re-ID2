"""Observe the one launched frozen readout controller at240-second intervals."""
from datetime import datetime
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, remote_python

proof = json.loads((PROJECT / 'results/preflight/full_official_control_readout_launch_20261004.json').read_text(encoding='utf-8'))
pid, source = proof['launch']['pid'], proof['launch']['output']
destination = PROJECT / 'results/preflight/full_official_control_readout_primary_observer_20261004.jsonl'
terminal = PROJECT / 'results/preflight/full_official_control_readout_primary_observer_terminal_20261004.json'
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
for mode,gpu in (('measurement_only',2),('independent_control',3)):
 row=dict(mode=mode,gpu=gpu,smoke_pass=False,frozen_complete=False,cpu_pass=False,raw_cases_written=0)
 smoke=root/'preflight'/mode/'smoke.json'
 if smoke.exists():row['smoke_pass']=json.loads(smoke.read_text())['status']=='PASS'
 frozen=root/'frozen'/mode
 row['raw_cases_written']=len(list(frozen.glob('q_*_g_*/raw.npz')))
 report=frozen/'result.json'
 if report.exists():
  data=json.loads(report.read_text());row.update(frozen_complete=data['status']=='COMPLETE',metric_cases=data['metric_cases'],selected_epoch=data['selected_epoch'])
 audit=frozen/'independent_cpu_audit.json'
 if audit.exists():
  data=json.loads(audit.read_text());row['cpu_pass']=data['status']=='PASS_FULL_OFFICIAL_READOUT_INSTALLED_GT' and data['cases']==588
 for phase in ('preflight','frozen','audit'):
  launch=root/phase/(mode+'_launch.json');exitfile=root/phase/(mode+'_exit.json')
  if launch.exists() and phase!='audit':
   child=json.loads(launch.read_text())
   if alive(child['pid']):result['active_neural_processes'].append(dict(pid=child['pid'],gpu=gpu,phase=phase,mode=mode))
  if exitfile.exists():
   value=json.loads(exitfile.read_text())
   if value['exit_code']!=0:result['failures'].append(dict(mode=mode,phase=phase,exit_code=value['exit_code']))
 result['runs'].append(row)
done=root/'controller_result.json'
if done.exists():
 data=json.loads(done.read_text());result.update(controller_complete=data['status']=='COMPLETE',metric_cases=data['metric_cases'],repeated_condition_query_rows=data['repeated_condition_query_rows'])
result['snapshot_atomic']=False;result['temperature_power_control']=False
print(json.dumps(result))
'''
while True:
    record = json.loads(remote_python('2026', code))
    record['local_observed_at'] = datetime.now().isoformat(timespec='seconds')
    with destination.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record) + '\n')
    print('FULL_OFFICIAL_READOUT_OBSERVATION', json.dumps(record), flush=True)
    if record['controller_complete']:
        assert record['metric_cases'] == 1176 and record['repeated_condition_query_rows'] == 695016
        assert record['preflight_pass'] and all(r['cpu_pass'] and r['frozen_complete'] for r in record['runs'])
        assert not record['active_neural_processes'] and not record['failures']
        terminal.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        break
    assert record['controller_alive'] and not record['failures'], record
    time.sleep(240)
