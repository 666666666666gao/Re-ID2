import base64
from datetime import datetime
import hashlib
import json
import sys
import time

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,remote_python

launch=json.loads((PROJECT/'results/preflight/axis_p0_utility_launch.json').read_text(encoding='utf-8'))
target=PROJECT/'results/axis_p0_retrieval_utility_20261003'
names=('V5_MSVR310','V5_RGBNT100','V5_RGBNT201','V6_MSVR310')
while True:
    collected=[name for name in names if (target/(name+'_intake.json')).exists()]
    code=f'''import base64,hashlib,json,subprocess
from pathlib import Path
root=Path({launch['output']!r});project=root.parent.parent
assert all(hashlib.sha256((project/name).read_bytes()).hexdigest()==sha for name,sha in {launch['source_sha256']!r}.items())
record={{'controller_live':bool(subprocess.run(['ps','-p',{str(launch['pid'])!r},'-o','pid='],capture_output=True,text=True).stdout.strip()),'cases':{{}},'files':{{}}}}
for name in {names!r}:
 row={{}}
 for stage in ('smoke','utility','gradients'):
  item={{}}
  for key in ('launch','exit'):
   path=root/(name+'_'+stage+'_'+key+'.json')
   if path.exists():item[key]=json.loads(path.read_text())
  path=root/(name+'_'+stage+'.log')
  if path.exists():item['log_tail']=path.read_text()[-2400:]
  row[stage]=item
 record['cases'][name]=row
 if all('exit' in item for item in row.values()) and name not in {collected!r}:
  paths=list(root.glob(name+'_*'))
  paths=[leaf for path in paths for leaf in (path.rglob('*') if path.is_dir() else [path])]
  for path in paths:
   if path.is_file() and path.suffix in ('.json','.csv','.log'):
    data=path.read_bytes();record['files'][path.relative_to(root).as_posix()]={{'data':base64.b64encode(data).decode(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}
path=root/'controller_result.json'
if path.exists():record['controller_result']=json.loads(path.read_text())
record['controller_log_tail']=Path({launch['log']!r}).read_text()[-3000:]
print(json.dumps(record))
'''
    record=json.loads(remote_python('2026',code));record['observed_at']=datetime.now().isoformat(timespec='seconds')
    files=record.pop('files');target.mkdir(parents=True,exist_ok=True)
    for name,proof in files.items():
        data=base64.b64decode(proof['data']);assert len(data)==proof['bytes'] and hashlib.sha256(data).hexdigest()==proof['sha256']
        path=target/name;assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    for name,row in record['cases'].items():
        if name in collected or not all('exit' in item for item in row.values()):continue
        manifest={filename:{'sha256':proof['sha256'],'bytes':proof['bytes']} for filename,proof in files.items() if filename.startswith(name+'_')}
        (target/(name+'_intake.json')).write_bytes(json.dumps(dict(observed_at=record['observed_at'],exit_codes={stage:item['exit']['exit_code'] for stage,item in row.items()},files=manifest),indent=2).encode('utf-8'))
    if 'controller_result' in record:(target/'controller_result.json').write_bytes(json.dumps(record['controller_result'],indent=2).encode('utf-8'))
    (PROJECT/'results/preflight/axis_p0_utility_latest_snapshot.json').write_bytes(json.dumps(record,indent=2).encode('utf-8'))
    failures={name+'_'+stage:item['log_tail'] for name,row in record['cases'].items() for stage,item in row.items() if item.get('exit',{}).get('exit_code',0)!=0}
    print('AXIS_P0_OBSERVATION',json.dumps(dict(time=record['observed_at'],controller_live=record['controller_live'],exits={name:{stage:item.get('exit',{}).get('exit_code') for stage,item in row.items()} for name,row in record['cases'].items()},failures=failures)),flush=True)
    assert not failures
    complete=record.get('controller_result',{}).get('status')=='COMPLETE'
    assert record['controller_live'] or complete,record['controller_log_tail']
    if complete:break
    time.sleep(240)
print('ALL_AXIS_P0_UTILITY_AND_GRADIENT_DIAGNOSTICS_COLLECTED',flush=True)
