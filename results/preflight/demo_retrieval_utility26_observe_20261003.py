import base64
from datetime import datetime
import hashlib
import json
import sys
import time

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, remote_python
HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')

launch=json.loads((PROJECT/'results/preflight/retrieval_utility_launch.json').read_text(encoding='utf-8'))
output=PROJECT/'results/axis_collaboration_v10_retrieval_utility_trial'

def collect():
    collected=[path.parent.relative_to(output).as_posix() for path in output.glob('*/*/intake.json')]
    code=f'''import base64,hashlib,json,subprocess
from pathlib import Path
root=Path({launch['output']!r})
project=root.parent.parent
assert all(hashlib.sha256((project/name).read_bytes()).hexdigest()==sha for name,sha in {launch['source_sha256']!r}.items())
rows=[('preflight','tensor',1),('preflight','smoke',1),('development','MSVR310_axis_retrieval_utility_fullref_s42',1)]
record={{'controller_live':bool(subprocess.run(['ps','-p',{str(launch['pid'])!r},'-o','pid='],capture_output=True,text=True).stdout.strip()),'rows':[],'files':{{}}}}
for stage,name,gpu in rows:
 campaign=root/stage;folder=campaign/name
 row={{'stage':stage,'name':name,'gpu':gpu}}
 for key,path in [('launch',campaign/(name+'_launch.json')),('exit',campaign/(name+'_exit.json')),('status',folder/'status.json'),('result',folder/'result.json'),('smoke',folder/'smoke.json'),('tensor',campaign/(name+'.json'))]:
  if path.exists():row[key]=json.loads(path.read_text())
 log=campaign/(name+'.log')
 if log.exists():row['log_tail']=log.read_text()[-2400:]
 record['rows'].append(row)
 if 'exit' in row and stage+'/'+name not in {collected!r}:
  selected=[path for path in folder.glob('*') if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log')]
  selected+=[campaign/(name+suffix) for suffix in ('.json','.log','_launch.json','_exit.json')]
  for path in selected:
   if path.exists():
    data=path.read_bytes();relative=path.relative_to(root).as_posix()
    record['files'][relative]={{'data':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}}
for key,path in [('controller_result',root/'controller_result.json'),('preflight_result',root/'preflight_result.json')]:
 if path.exists():record[key]=json.loads(path.read_text())
record['controller_log_tail']=Path({launch['log']!r}).read_text()[-4000:]
print(json.dumps(record))
'''
    record=json.loads(remote_python('2026',code))
    record['observed_at']=datetime.now().isoformat(timespec='seconds')
    files=record.pop('files')
    for relative,proof in files.items():
        data=base64.b64decode(proof['data'])
        assert len(data)==proof['bytes'] and hashlib.sha256(data).hexdigest()==proof['sha256']
        path=output/relative
        assert not path.exists(),relative
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    for row in record['rows']:
        relative=row['stage']+'/'+row['name']
        if 'exit' not in row or relative in collected:continue
        exact=[relative+suffix for suffix in ('.json','.log','_launch.json','_exit.json')]
        manifest={name:{'sha256':proof['sha256'],'bytes':proof['bytes']} for name,proof in files.items() if name.startswith(relative+'/') or name in exact}
        path=output/relative/'intake.json';path.parent.mkdir(parents=True,exist_ok=True)
        assert not path.exists()
        path.write_bytes(json.dumps(dict(observed_at=record['observed_at'],files=manifest,exit_code=row['exit']['exit_code']),indent=2).encode('utf-8'))
    for key,relative in [('controller_result','controller_result.json'),('preflight_result','preflight/controller_result.json')]:
        if key in record:
            path=output/relative;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(json.dumps(record[key],indent=2).encode('utf-8'))
    (PROJECT/'results/preflight/retrieval_utility_latest_snapshot.json').write_bytes(json.dumps(record,indent=2).encode('utf-8'))
    failures=[row for row in record['rows'] if row.get('exit',{}).get('exit_code',0)!=0]
    print('RETRIEVAL_UTILITY_OBSERVATION',json.dumps(dict(time=record['observed_at'],controller_live=record['controller_live'],failed=[row['name'] for row in failures],rows=[dict(stage=row['stage'],name=row['name'],epoch=row.get('status',{}).get('epoch'),exit=row.get('exit',{}).get('exit_code')) for row in record['rows']])),flush=True)
    if failures:
        print('RETRIEVAL_UTILITY_PRIMARY_FAILURE_LOGS',json.dumps({row['name']:row.get('log_tail','') for row in failures}),flush=True)
    assert not failures
    terminal=record.get('controller_result',{}).get('status')=='COMPLETE'
    assert record['controller_live'] or terminal,record['controller_log_tail']
    return record,terminal

while True:
    record,terminal=collect()
    if terminal:break
    remaining=[]
    for row in record['rows']:
        if row['stage']=='development' and 'launch' in row and 'exit' not in row and 'latest' in row.get('status',{}):
            remaining.append((50-row['status']['epoch'])*row['status']['latest']['seconds'])
    delay=max(240,min(remaining)-180) if remaining else 240
    print('NEXT_METRIC_OBSERVATION_SECONDS',round(delay),flush=True)
    time.sleep(delay)
print('RETRIEVAL_UTILITY_ONE_CLEAN50_TERMINAL_COLLECTED',flush=True)
