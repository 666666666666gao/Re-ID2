import base64
from datetime import datetime
import hashlib
import json
import sys
import time

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,remote_python

launch=json.loads((PROJECT/'results/preflight/axis_contribution_gradient_probe_launch.json').read_text(encoding='utf-8'))
target=PROJECT/'results/axis_contribution_gradient_diagnostic_20261003'
while True:
    collected=[name for name in ('V5','V6') if (target/(name+'_intake.json')).exists()]
    code=f'''import base64,hashlib,json,subprocess
from pathlib import Path
root=Path({launch['output']!r});project=root.parent.parent
assert all(hashlib.sha256((project/name).read_bytes()).hexdigest()==sha for name,sha in {launch['source_sha256']!r}.items())
record={{'controller_live':bool(subprocess.run(['ps','-p',{str(launch['pid'])!r},'-o','pid='],capture_output=True,text=True).stdout.strip()),'cases':{{}},'files':{{}}}}
for name in ('V5','V6'):
 row={{}}
 for key,suffix in [('launch','_launch.json'),('exit','_exit.json'),('result','.json')]:
  path=root/(name+suffix)
  if path.exists():row[key]=json.loads(path.read_text())
 path=root/(name+'.log')
 if path.exists():row['log_tail']=path.read_text()[-2400:]
 record['cases'][name]=row
 if 'exit' in row and name not in {collected!r}:
  for suffix in ('.json','.log','_launch.json','_exit.json'):
   path=root/(name+suffix)
   if path.exists():
    data=path.read_bytes();record['files'][path.name]={{'data':base64.b64encode(data).decode(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}
path=root/'controller_result.json'
if path.exists():record['controller_result']=json.loads(path.read_text())
record['controller_log_tail']=Path({launch['log']!r}).read_text()[-3000:]
print(json.dumps(record))
'''
    record=json.loads(remote_python('2026',code));record['observed_at']=datetime.now().isoformat(timespec='seconds')
    files=record.pop('files');target.mkdir(parents=True,exist_ok=True)
    for name,proof in files.items():
        data=base64.b64decode(proof['data']);assert len(data)==proof['bytes'] and hashlib.sha256(data).hexdigest()==proof['sha256']
        path=target/name;assert not path.exists();path.write_bytes(data)
    for name,row in record['cases'].items():
        if 'exit' not in row or name in collected:continue
        manifest={filename:{'sha256':proof['sha256'],'bytes':proof['bytes']} for filename,proof in files.items() if filename in [name+suffix for suffix in ('.json','.log','_launch.json','_exit.json')]}
        path=target/(name+'_intake.json');assert not path.exists()
        path.write_bytes(json.dumps(dict(observed_at=record['observed_at'],exit_code=row['exit']['exit_code'],files=manifest),indent=2).encode('utf-8'))
    if 'controller_result' in record:(target/'controller_result.json').write_bytes(json.dumps(record['controller_result'],indent=2).encode('utf-8'))
    (PROJECT/'results/preflight/axis_contribution_gradient_probe_latest_snapshot.json').write_bytes(json.dumps(record,indent=2).encode('utf-8'))
    failures={name:row['log_tail'] for name,row in record['cases'].items() if row.get('exit',{}).get('exit_code',0)!=0}
    print('GRADIENT_PROBE_OBSERVATION',json.dumps(dict(time=record['observed_at'],controller_live=record['controller_live'],exits={name:row.get('exit',{}).get('exit_code') for name,row in record['cases'].items()},failures=failures)),flush=True)
    assert not failures
    complete=record.get('controller_result',{}).get('status')=='COMPLETE'
    assert record['controller_live'] or complete,record['controller_log_tail']
    if complete:break
    time.sleep(240)
print('BOTH_ACTUAL_CONTRIBUTION_GRADIENT_PROBES_COLLECTED',flush=True)
