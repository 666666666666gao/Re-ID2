from datetime import datetime
import hashlib
import json
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

previous=json.loads((PROJECT/'results/preflight/axis_collaboration_v4_extra_cross26_launch.json').read_text(encoding='utf-8'))
review=json.loads((PROJECT/'results/preflight/axis_collaboration_v4_missing_development_review.json').read_text(encoding='utf-8'))
assert review['status']=='PASS' and not review['blockers']
assert review['helper_sha256']==hashlib.sha256(open(__file__,'rb').read()).hexdigest()
source=PROJECT/'missing_development.py'
assert review['module_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
root,python=HOSTS['2026']
inputs=root+'/runs/axis_collaboration_v4_missing_development26_input'
output=root+'/runs/axis_collaboration_v4_missing_development26'
jobs={2:[('MSVR310','axis_scaled_fullref'),('RGBNT201','axis_scaled_fullref'),('RGBNT100','axis_scaled_fullref')],
      3:[('MSVR310','demo'),('RGBNT201','demo'),('RGBNT100','demo')]}
guard=f'''import hashlib,json,subprocess
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in {previous['source_sha256']!r}.items())
assert not Path({output!r}).exists()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
used={{int(line.split(',')[0]):int(line.split(',')[1]) for line in gpu.strip().splitlines()}}
assert used[2]<500 and used[3]<500
Path({inputs!r}).mkdir(exist_ok=False)
print(json.dumps({{'gpu':gpu}}))
'''
availability=json.loads(remote_python('2026',guard))
proofs={}
for dataset,host in (('MSVR310','2025'),('RGBNT201','2025'),('RGBNT100','2026')):
 name=dataset+'_demo_s42'
 original=HOSTS[host][0]+'/runs/dynamic_amp_comparison/'+name
 code=f'''import hashlib,json
from pathlib import Path
root=Path({original!r})
terminal=json.loads((root/'result.json').read_text())
assert terminal['status']=='COMPLETE' and terminal['epochs']==50 and terminal['arguments']['variant']=='demo'
assert json.loads((root/'exit.json').read_text())['exit_code']==0
print(json.dumps({{filename:{{'sha256':hashlib.sha256((root/filename).read_bytes()).hexdigest(),'bytes':(root/filename).stat().st_size}} for filename in ('best.pth','best_dev_arrays.npz','result.json','exit.json')}}))
'''
 proof=json.loads(remote_python(host,code))
 proofs[name]=dict(source_host=host,source_path=original,files=proof)
 assert remote_python('2026',f'from pathlib import Path;Path({(inputs+"/"+name)!r}).mkdir(exist_ok=False);print("CREATED")').strip()=='CREATED'
 if host=='2026':
  code=f'''import shutil
from pathlib import Path
source=Path({original!r});destination=Path({(inputs+'/'+name)!r})
for filename in ('best.pth','best_dev_arrays.npz','result.json','exit.json'):shutil.copyfile(source/filename,destination/filename)
print('COPIED')
'''
  assert remote_python('2026',code).strip()=='COPIED'
 else:
  for filename in ('best.pth','best_dev_arrays.npz','result.json','exit.json'):
   command(['scp','-3',*OPTIONS,host+':'+original+'/'+filename,'2026:'+inputs+'/'+name+'/'+filename])
for dataset in ('MSVR310','RGBNT201','RGBNT100'):
 name=dataset+'_axis_scaled_fullref_s42'
 original=root+'/runs/axis_collaboration_v4_cross26_input/'+name
 code=f'''import hashlib,json
from pathlib import Path
root=Path({original!r})
terminal=json.loads((root/'result.json').read_text())
assert terminal['status']=='COMPLETE' and terminal['epochs']==50 and terminal['arguments']['variant']=='axis_scaled_fullref'
exit=root.parent/(root.name+'_exit.json')
assert json.loads(exit.read_text())['exit_code']==0
print(json.dumps({{path.name:{{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}} for path in (root/'best.pth',root/'best_dev_arrays.npz',root/'result.json',exit)}}))
'''
 proofs[name]=dict(source_host='2026',source_path=original,files=json.loads(remote_python('2026',code)))
command(['scp',*OPTIONS,str(source),'2026:'+root+'/missing_development.py'])
source_sha={**previous['source_sha256'],'missing_development.py':review['module_sha256']}
controller_code=f'''from concurrent.futures import ThreadPoolExecutor
import json,os,subprocess,sys,time
from pathlib import Path
from launch_axis_scaled import idle
from launch_runs import write_json
root=Path({output!r})
proofs={proofs!r}
def slot(gpu):
 rows=[]
 for dataset,variant in {jobs!r}[gpu]:
  name=dataset+'_'+variant+'_s42'
  folder=root/name
  folder.mkdir(exist_ok=False)
  checks=[]
  for smoke in (True,False):
   idle(gpu)
   stage='smoke' if smoke else 'full'
   command=[sys.executable,'-u','missing_development.py','--run-dir',({inputs!r}+'/'+name if variant=='demo' else proofs[name]['source_path']),'--output',str(folder/stage),'--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt']
   if smoke:command.append('--smoke')
   with (folder/(stage+'.log')).open('x') as log:
    child=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
    write_json(folder/(stage+'_launch.json'),dict(pid=child.pid,gpu=gpu,command=command,started=time.time()))
    code=child.wait()
   row=dict(stage=stage,exit_code=code,finished=time.time())
   write_json(folder/(stage+'_exit.json'),row)
   checks.append(row)
   assert code==0,(name,stage)
   record=json.loads((folder/stage/('smoke.json' if smoke else 'result.json')).read_text())
   assert record['status']==('PASS' if smoke else 'COMPLETE') and record['optimizer_updates']==0 and record['normal_feature_max_error']==0
   if not smoke:assert len(record['measurements'])==13
  result=dict(name=name,gpu=gpu,checks=checks,optimizer_updates=0)
  write_json(folder/'controller_result.json',dict(status='PASS',**result))
  rows.append(result)
 return rows
with ThreadPoolExecutor(max_workers=2) as pool:
 rows=[row for group in pool.map(slot,{list(jobs)!r}) for row in group]
assert len(rows)==6
write_json(root/'controller_result.json',dict(status='COMPLETE',runs=rows,optimizer_updates=0,conditions=78,scope='identity-heldout development only'))
'''
entry=f"import sys,runpy;sys.path.insert(0,{root!r});runpy.run_path({(output+'/controller.py')!r},run_name='__main__')"
launch_code=f'''import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in {source_sha!r}.items())
for run,record in {proofs!r}.items():
 base=Path({inputs!r})/run if record['source_path'].endswith('_demo_s42') else Path(record['source_path'])
 for filename,proof in record['files'].items():
  path=base.parent/filename if filename.endswith('_exit.json') else base/filename
  assert path.stat().st_size==proof['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==proof['sha256']
output=Path({output!r})
output.mkdir(exist_ok=False)
(output/'controller.py').write_text({controller_code!r})
with (output/'controller.log').open('x') as log:
 child=subprocess.Popen([{python!r},'-u','-c',{entry!r}],cwd=root,start_new_session=True,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
print(json.dumps({{'pid':child.pid,'host':'2026','started':time.time(),'status':'FROZEN_DEVELOPMENT_MISSING_STARTED_SMOKE_AND_78_CONDITIONS_PENDING','jobs':{jobs!r},'output':str(output),'source_sha256':{source_sha!r},'input_files':{proofs!r}}}))
'''
launch=json.loads(remote_python('2026',launch_code))
launch.update(observed_at=datetime.now().isoformat(timespec='seconds'),availability26=availability,estimated_seconds=1800)
receipt=PROJECT/'results/preflight/axis_collaboration_v4_missing_development26_launch.json'
assert not receipt.exists()
receipt.write_bytes(json.dumps(launch,indent=2).encode('utf-8'))
print('V4_FROZEN_MISSING_DEVELOPMENT_STARTED',json.dumps({key:launch[key] for key in ('pid','host','status','observed_at','jobs')}),flush=True)
