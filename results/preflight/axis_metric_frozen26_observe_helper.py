import base64
from datetime import datetime
import hashlib
import json
import sys
import time

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,remote_python
launch=json.loads((PROJECT/'results/preflight/axis_metric_frozen_launch.json').read_text(encoding='utf-8'))
target=PROJECT/'results/axis_collaboration_v7_metric_frozen_trial'


def collect():
    closed=[path.parent.relative_to(target).as_posix() for path in target.glob('*/*/intake.json')]
    code=f'''import base64,hashlib,json,subprocess
from pathlib import Path
root=Path({launch['output']!r});project=root.parent.parent
assert all(hashlib.sha256((project/name).read_bytes()).hexdigest()==sha for name,sha in {launch['source_sha256']!r}.items())
record=dict(controller_live=bool(subprocess.run(['ps','-p',{str(launch['pid'])!r},'-o','pid='],capture_output=True,text=True).stdout.strip()),rows=[],files={{}})
for dataset in ('MSVR310',):
 name=dataset+'_axis_metric_fullref_s42';folder=root/name
 for stage in ('four_state_smoke','four_state_full','missing_smoke','missing_full'):
  row=dict(name=name,stage=stage)
  for key,path in [('launch',folder/(stage+'_launch.json')),('exit',folder/(stage+'_exit.json')),('smoke',folder/stage/'smoke.json'),('diagnostic',folder/stage/'diagnostic.json'),('result',folder/stage/'result.json')]:
   if path.exists():row[key]=json.loads(path.read_text())
  log=folder/(stage+'.log')
  if log.exists():row['log_tail']=log.read_text()[-1800:]
  record['rows'].append(row)
  if 'exit' in row and name+'/'+stage not in {closed!r}:
   paths=[path for path in (folder/stage).glob('*') if path.is_file() and path.suffix in ('.json','.csv','.log')]
   paths.extend(folder/(stage+suffix) for suffix in ('.log','_launch.json','_exit.json'))
   for path in paths:
    if path.exists():
     data=path.read_bytes();record['files'][path.relative_to(root).as_posix()]=dict(data=base64.b64encode(data).decode(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 if (folder/'controller_result.json').exists():
  record.setdefault('dataset_controllers',{{}})[name]=json.loads((folder/'controller_result.json').read_text())
  record.setdefault('frozen_inputs',{{}})[name]=json.loads((folder/'frozen_inputs.json').read_text())
if (root/'controller_result.json').exists():record['controller_result']=json.loads((root/'controller_result.json').read_text())
record['controller_log_tail']=Path({launch['log']!r}).read_text()[-3000:]
print(json.dumps(record))
'''
    record=json.loads(remote_python('2026',code));record['observed_at']=datetime.now().isoformat(timespec='seconds')
    files=record.pop('files')
    for relative,proof in files.items():
        data=base64.b64decode(proof['data'])
        assert len(data)==proof['bytes'] and hashlib.sha256(data).hexdigest()==proof['sha256']
        path=target/relative;assert not path.exists(),relative
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    for row in record['rows']:
        relative=row['name']+'/'+row['stage']
        if 'exit' not in row or relative in closed:continue
        exact=[relative+suffix for suffix in ('.log','_launch.json','_exit.json')]
        manifest={name:{key:proof[key] for key in ('bytes','sha256')} for name,proof in files.items() if name.startswith(relative+'/') or name in exact}
        path=target/relative/'intake.json';assert not path.exists()
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(json.dumps(dict(observed_at=record['observed_at'],files=manifest,exit_code=row['exit']['exit_code']),indent=2).encode('utf-8'))
    for key,filename in [('dataset_controllers','controller_result.json'),('frozen_inputs','frozen_inputs.json')]:
        for name,value in record.get(key,{}).items():
            path=target/name/filename;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(json.dumps(value,indent=2).encode('utf-8'))
    if 'controller_result' in record:
        (target/'controller_result.json').write_bytes(json.dumps(record['controller_result'],indent=2).encode('utf-8'))
    (PROJECT/'results/preflight/axis_metric_frozen_latest_snapshot.json').write_bytes(json.dumps(record,indent=2).encode('utf-8'))
    failures=[row for row in record['rows'] if row.get('exit',{}).get('exit_code',0)!=0]
    print('METRIC_INTERFACE_FROZEN_OBSERVATION',json.dumps(dict(time=record['observed_at'],controller_live=record['controller_live'],failed=[row['name']+'/'+row['stage'] for row in failures],rows=[dict(name=row['name'],stage=row['stage'],exit=row.get('exit',{}).get('exit_code')) for row in record['rows']])),flush=True)
    if failures:print('METRIC_INTERFACE_FROZEN_PRIMARY_FAILURE',json.dumps({row['name']+'/'+row['stage']:row.get('log_tail','') for row in failures}),flush=True)
    assert not failures
    complete=record.get('controller_result',{}).get('status')=='COMPLETE'
    assert record['controller_live'] or complete,record['controller_log_tail']
    return complete


while not collect():
    time.sleep(240)
print('METRIC_INTERFACE_FROZEN_ALL13_MISSING_AND4_FOUR_STATES_COLLECTED',flush=True)
