import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, remote_python

launch=json.loads((PROJECT/'results/preflight/axis_collaboration_v4_deployment_launch.json').read_text(encoding='utf-8'))
root=HOSTS['2025'][0]
schedule={0:[('MSVR310','axis_scaled_fullref'),('RGBNT100','frequency_scaled_fullref'),('MSVR310','axis_raw_fullref')],
 1:[('RGBNT100','axis_scaled_fullref'),('MSVR310','frequency_scaled_fullref')],
 2:[('RGBNT100','plain_scaled_fullref'),('MSVR310','plain_scaled_fullref')],
 3:[('RGBNT201','axis_scaled_fullref'),('RGBNT201','plain_scaled_fullref'),('RGBNT201','frequency_scaled_fullref'),('MSVR310','axis_scaled_base')]}

def collect():
 collected=[]
 for phase in ('preflight','development'):
  destination=PROJECT/('results/axis_collaboration_v4_'+phase)
  for path in destination.glob('*/intake.json'):
   collected.append(phase+'/'+path.parent.name)
 code=f'''import base64,hashlib,json,subprocess
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in {launch['source_sha256']!r}.items())
schedule={schedule!r}
record={{'controller_live':bool(subprocess.run(['ps','-p',{str(launch['pid'])!r},'-o','pid='],capture_output=True,text=True).stdout.strip()),'rows':[],'files':{{}}}}
for phase in ('preflight','development'):
 campaign=root/('runs/axis_collaboration_v4_'+phase)
 names=([dataset+'_tensor' for dataset in ('MSVR310','RGBNT201','RGBNT100')]+[dataset+'_'+variant+'_smoke' for jobs in schedule.values() for dataset,variant in jobs]) if phase=='preflight' else [dataset+'_'+variant+'_s42' for jobs in schedule.values() for dataset,variant in jobs]
 for name in names:
  folder=campaign/name
  row={{'phase':phase,'name':name}}
  for key,path in [('launch',campaign/(name+'_launch.json')),('exit',campaign/(name+'_exit.json')),('status',folder/'status.json'),('result',folder/'result.json'),('smoke',folder/'smoke.json'),('tensor',folder/'tensor_contract.json')]:
   if path.exists():row[key]=json.loads(path.read_text())
  record['rows'].append(row)
  if 'exit' in row and phase+'/'+name not in {collected!r}:
   selected=[path for path in folder.glob('*') if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log')]
   selected+=[campaign/(name+suffix) for suffix in ('.log','_launch.json','_exit.json')]
   for path in selected:
    if path.exists():
     data=path.read_bytes()
     record['files'][phase+'/'+path.relative_to(campaign).as_posix()]={{'data':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}}
 path=campaign/'controller_result.json'
 if path.exists():record[phase+'_controller_result']=json.loads(path.read_text())
log=root/'runs/axis_collaboration_v4_controller.log'
record['controller_log_tail']=log.read_text()[-8000:]
record['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
print(json.dumps(record))
'''
 record=json.loads(remote_python('2025',code))
 record['observed_at']=datetime.now().isoformat(timespec='seconds')
 files=record.pop('files')
 for relative,value in files.items():
  phase,path=relative.split('/',1)
  target=PROJECT/('results/axis_collaboration_v4_'+phase)/path
  data=base64.b64decode(value['data'])
  assert len(data)==value['bytes'] and hashlib.sha256(data).hexdigest()==value['sha256']
  assert not target.exists(),relative
  target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(data)
 for row in record['rows']:
  if 'exit' not in row or row['phase']+'/'+row['name'] in collected:continue
  prefix=row['phase']+'/'
  manifest={name[len(prefix):]:{'sha256':value['sha256'],'bytes':value['bytes']} for name,value in files.items() if name.startswith(prefix) and (name[len(prefix):].startswith(row['name']+'/') or name[len(prefix):] in [row['name']+suffix for suffix in ('.log','_launch.json','_exit.json')])}
  destination=PROJECT/('results/axis_collaboration_v4_'+row['phase'])/row['name']/'intake.json'
  destination.parent.mkdir(parents=True,exist_ok=True)
  destination.write_bytes(json.dumps({'observed_at':record['observed_at'],'exit_code':row['exit']['exit_code'],'files':manifest},indent=2).encode('utf-8'))
 for phase in ('preflight','development'):
  if phase+'_controller_result' in record:
   target=PROJECT/('results/axis_collaboration_v4_'+phase)/'controller_result.json'
   target.parent.mkdir(parents=True,exist_ok=True)
   target.write_bytes(json.dumps(record[phase+'_controller_result'],indent=2).encode('utf-8'))
 (PROJECT/'results/preflight/axis_collaboration_v4_latest_snapshot.json').write_bytes(json.dumps(record,indent=2).encode('utf-8'))
 failed=[row['phase']+'/'+row['name'] for row in record['rows'] if row.get('exit',{}).get('exit_code',0)!=0]
 training=[{'name':row['name'],'gpu':row['launch']['gpu'],'epoch':row.get('result',{}).get('epochs',row.get('status',{}).get('epoch',0))} for row in record['rows'] if row['phase']=='development' and 'launch' in row and 'exit' not in row]
 counts={phase:sum(row['phase']==phase and row.get('exit',{}).get('exit_code')==0 for row in record['rows']) for phase in ('preflight','development')}
 print('V4_ACTUAL_OBSERVATION',json.dumps({'time':record['observed_at'],'controller_live':record['controller_live'],'complete':counts,'failed':failed,'training':training,'gpu':record['gpu']}),flush=True)
 assert not failed,failed
 assert record['controller_live'] or record.get('development_controller_result',{}).get('status')=='COMPLETE',record['controller_log_tail']
 return record.get('development_controller_result',{}).get('status')=='COMPLETE'

while True:
 if collect():break
 time.sleep(240)
print('V4_ALL_ELEVEN_TERMINAL_COLLECTED',flush=True)
