from datetime import datetime
import hashlib
import json
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS,command,remote_python
HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
prior=json.loads((PROJECT/'results/preflight/axis_collaboration_v4_missing_development27_launch.json').read_text(encoding='utf-8'))
failure=json.loads((PROJECT/'results/preflight/axis_collaboration_v4_missing_development27_failure_snapshot.json').read_text(encoding='utf-8'))
assert not failure['controller_live'] and failure['failures']==['MSVR310_axis_scaled_fullref_s42/full','RGBNT100_demo_s42/full']
review=json.loads((PROJECT/'results/preflight/axis_collaboration_v4_missing_development26_repaired_review.json').read_text(encoding='utf-8'))
assert review['status']=='PASS' and not review['blockers']
assert review['helper_sha256']==hashlib.sha256(open(__file__,'rb').read()).hexdigest()
module=PROJECT/'missing_development.py'
assert review['module_sha256']==hashlib.sha256(module.read_bytes()).hexdigest()
reuse_root=HOSTS['2027'][0]
root,python=HOSTS['2026']
run_paths={name:record['source_path'] for name,record in prior['input_files'].items() if record['source_host']=='2026'}
assert set(run_paths)=={'MSVR310_axis_scaled_fullref_s42','RGBNT201_axis_scaled_fullref_s42','RGBNT100_demo_s42','RGBNT100_axis_scaled_fullref_s42'}
output=root+'/runs/axis_collaboration_v4_missing_development26_repaired'
reused=['MSVR310_demo_s42','RGBNT201_demo_s42']
jobs={0:[('MSVR310','axis_scaled_fullref'),('RGBNT201','axis_scaled_fullref')],1:[('RGBNT100','demo'),('RGBNT100','axis_scaled_fullref')]}
old_source=prior['source_sha256']
source_sha={**old_source,'missing_development.py':review['module_sha256']}
reuse_guard=f'''import hashlib,json,subprocess
from pathlib import Path
root=Path({reuse_root!r})
assert not subprocess.run(['ps','-p',{str(prior['pid'])!r},'-o','pid='],capture_output=True,text=True).stdout.strip()
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in {old_source!r}.items())
old=Path({prior['output']!r})
for name in {reused!r}:
 result=json.loads((old/name/'full/result.json').read_text())
 assert json.loads((old/name/'full_exit.json').read_text())['exit_code']==0
 assert result['status']=='COMPLETE' and len(result['measurements'])==13 and result['optimizer_updates']==0 and result['normal_feature_max_error']==0
for name in ('MSVR310_axis_scaled_fullref_s42','RGBNT100_demo_s42'):
 assert json.loads((old/name/'full_exit.json').read_text())['exit_code']==1
for name in ('RGBNT201_axis_scaled_fullref_s42','RGBNT100_axis_scaled_fullref_s42'):
 assert not (old/name).exists()
print(json.dumps(dict(reused_complete={reused!r},old_controller_dead=True)))
'''
reuse=json.loads(remote_python('2027',reuse_guard))
guard=f'''import hashlib,json,subprocess
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in { {name:sha for name,sha in old_source.items() if name!='missing_development.py'}!r}.items())
assert not (root/'missing_development.py').exists()
for name,path in {run_paths!r}.items():
 base=Path(path)
 terminal=json.loads((base/'result.json').read_text())
 assert terminal['status']=='COMPLETE' and terminal['epochs']==50
 for filename,proof in {prior['input_files']!r}[name]['files'].items():
  file=base.parent/filename if filename.endswith('_exit.json') else base/filename
  assert file.stat().st_size==proof['bytes'] and hashlib.sha256(file.read_bytes()).hexdigest()==proof['sha256']
 exit=base/'exit.json' if name.endswith('_demo_s42') else base.parent/(base.name+'_exit.json')
 assert json.loads(exit.read_text())['exit_code']==0
assert not Path({output!r}).exists()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
used={{int(line.split(',')[0]):int(line.split(',')[1]) for line in gpu.strip().splitlines()}}
assert used[0]<500 and used[1]<500
print(json.dumps(dict(gpu=gpu,existing_input_proof_count=16,warm_environment='original validated2026 Python3.10.14/Torch2.5.1cu121, no environment mutation')))
'''
availability=json.loads(remote_python('2026',guard))
command(['scp',*OPTIONS,str(module),'2026:'+root+'/missing_development.py'])
controller=f'''from concurrent.futures import ThreadPoolExecutor
import json,os,subprocess,sys,time
from pathlib import Path
from launch_axis_scaled import idle
from launch_runs import write_json
root=Path({output!r})
def slot(gpu):
 rows=[]
 for dataset,variant in {jobs!r}[gpu]:
  name=dataset+'_'+variant+'_s42';folder=root/name;folder.mkdir(exist_ok=False);checks=[]
  for smoke in (True,False):
   idle(gpu);stage='smoke' if smoke else 'full'
   argv=[sys.executable,'-u','missing_development.py','--run-dir',{run_paths!r}[name],'--output',str(folder/stage),'--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt']
   if smoke:argv.append('--smoke')
   with (folder/(stage+'.log')).open('x') as log:
    child=subprocess.Popen(argv,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
    write_json(folder/(stage+'_launch.json'),dict(pid=child.pid,gpu=gpu,command=argv,started=time.time()))
    code=child.wait()
   row=dict(stage=stage,exit_code=code,finished=time.time());write_json(folder/(stage+'_exit.json'),row);checks.append(row)
   assert code==0,(name,stage)
   result=json.loads((folder/stage/('smoke.json' if smoke else 'result.json')).read_text())
   assert result['status']==('PASS' if smoke else 'COMPLETE') and result['optimizer_updates']==0 and result['normal_feature_max_error']==0
   if not smoke:assert len(result['measurements'])==13 and result['normal_distance_source'].startswith('saved best_dev_arrays.npz/distances')
  row=dict(name=name,gpu=gpu,checks=checks,optimizer_updates=0);write_json(folder/'controller_result.json',dict(status='PASS',**row));rows.append(row)
 return rows
with ThreadPoolExecutor(max_workers=2) as pool:rows=[row for group in pool.map(slot,{list(jobs)!r}) for row in group]
assert len(rows)==4
write_json(root/'controller_result.json',dict(status='COMPLETE',runs=rows,reused_complete={reused!r},reused_output={prior['output']!r},reused_host='2027',optimizer_updates=0,conditions=78,new_conditions=52,reused_conditions=26,scope='Frozen identity-heldout development only; clean saved distances after exact feature parity; no training/test/checkpoint reselection'))
'''
entry=f"import sys,runpy;sys.path.insert(0,{root!r});runpy.run_path({(output+'/controller.py')!r},run_name='__main__')"
launch_code=f'''import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in {source_sha!r}.items())
for name,path in {run_paths!r}.items():
 record={prior['input_files']!r}[name]
 base=Path(path)
 for filename,proof in record['files'].items():
  path=base.parent/filename if filename.endswith('_exit.json') else base/filename
  assert path.stat().st_size==proof['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==proof['sha256']
output=Path({output!r});output.mkdir(exist_ok=False);(output/'controller.py').write_text({controller!r})
with (output/'controller.log').open('x') as log:
 child=subprocess.Popen([{python!r},'-u','-c',{entry!r}],cwd=root,start_new_session=True,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
print(json.dumps(dict(pid=child.pid,host='2026',started=time.time(),status='FROZEN_REPAIRED_FOUR_STARTED_52_CONDITIONS_PENDING',jobs={jobs!r},output=str(output),source_sha256={source_sha!r},input_files={ {name:prior['input_files'][name] for name in run_paths}!r},reused_complete={reused!r},reused_output={prior['output']!r})))
'''
launch=json.loads(remote_python('2026',launch_code))
launch.update(observed_at=datetime.now().isoformat(timespec='seconds'),availability26=availability,reused_remote_validation=reuse,reused_host='2027',estimated_seconds=900,original_failure_snapshot_sha256=hashlib.sha256((PROJECT/'results/preflight/axis_collaboration_v4_missing_development27_failure_snapshot.json').read_bytes()).hexdigest())
path=PROJECT/'results/preflight/axis_collaboration_v4_missing_development26_repaired_launch.json'
assert not path.exists()
path.write_bytes(json.dumps(launch,indent=2).encode('utf-8'))
print('V4_MISSING26_REPAIRED_STARTED',json.dumps({key:launch[key] for key in ('pid','host','status','observed_at','jobs','reused_complete')}),flush=True)
