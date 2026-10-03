from datetime import datetime
import json
import subprocess
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, remote_python

launch = json.loads((PROJECT / 'results/preflight/axis_collaboration_v4_deployment_launch.json').read_text(encoding='utf-8'))
project = HOSTS['2025'][0]
code = f'''import hashlib,json,subprocess
from pathlib import Path
root=Path({project!r})
expected={launch['source_sha256']!r}
actual={{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in expected}}
rows=[]
for folder in sorted((root/'runs/axis_collaboration_v4_development').glob('*_s42')):
 row={{'name':folder.name}}
 for key,path in [('status',folder/'status.json'),('result',folder/'result.json'),('exit',folder.parent/(folder.name+'_exit.json'))]:
  if path.exists():row[key]=json.loads(path.read_text())
 rows.append(row)
pid=subprocess.run(['ps','-p',{str(launch['pid'])!r},'-o','pid,stat,etime,args'],capture_output=True,text=True)
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
print(json.dumps({{'source_sha256':actual,'source_mismatches':{{name:{{'expected':expected[name],'actual':actual[name]}} for name in expected if expected[name]!=actual[name]}},'controller_process':{{'returncode':pid.returncode,'stdout':pid.stdout,'stderr':pid.stderr}},'gpu':{{'returncode':gpu.returncode,'stdout':gpu.stdout,'stderr':gpu.stderr}},'rows':rows,'controller_log_tail':(root/'runs/axis_collaboration_v4_controller.log').read_text()[-3000:]}}))
'''
target = PROJECT / 'results/preflight/axis_collaboration_v4_observer_failure_probe.json'
assert not target.exists()
try:
    result = json.loads(remote_python('2025', code))
except subprocess.CalledProcessError as error:
    result = {'status': 'ACTUAL_REMOTE_PROBE_FAILED', 'returncode': error.returncode, 'stdout': error.stdout, 'stderr': error.stderr}
result['observed_at'] = datetime.now().isoformat(timespec='seconds')
result['original_readonly_observer_uv_pid_absent'] = True
result['original_last_successful_snapshot'] = '2026-10-03T07:14:41'
target.write_bytes(json.dumps(result, indent=2).encode('utf-8'))
print(json.dumps({k: v for k, v in result.items() if k not in ('source_sha256', 'rows')}, ensure_ascii=False), flush=True)
if 'rows' in result:
    print(json.dumps([{'name': row['name'], 'epoch': row.get('result', row.get('status', {})).get('epochs', row.get('status', {}).get('epoch')), 'exit': row.get('exit')} for row in result['rows']]), flush=True)
