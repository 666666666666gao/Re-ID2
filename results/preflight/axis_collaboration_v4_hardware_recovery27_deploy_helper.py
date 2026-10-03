from datetime import datetime
import hashlib
import json
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, remote_python
HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')

incident = json.loads((PROJECT/'results/preflight/axis_collaboration_v4_driver_evidence.json').read_text(encoding='utf-8'))
assert any('Xid' in line and '79,' in line and 'GPU has fallen off the bus' in line for line in incident['kernel']['selected_lines'])
prior = json.loads((PROJECT/'results/preflight/axis_collaboration_v4_extra_cross26_launch.json').read_text(encoding='utf-8'))
review_path = PROJECT/'results/preflight/axis_collaboration_v4_hardware_recovery27_review.json'
review = json.loads(review_path.read_text(encoding='utf-8'))
assert review['status']=='PASS' and not review['blockers']
assert review['helper_sha256']==hashlib.sha256(open(__file__,'rb').read()).hexdigest()
root, python = HOSTS['2027']
output = root+'/runs/axis_collaboration_v4_hardware_recovery27'
jobs = [('RGBNT100','frequency_scaled_fullref',0),('MSVR310','axis_raw_fullref',1)]
guard26 = f'''import hashlib,json,subprocess
from pathlib import Path
root=Path({root!r})
doc=json.loads((root/'setup/env27_doc_validation.json').read_text())
assert doc['status']=='PASS' and doc['spec_sha256']=='4993690212b8f331ceac59c1cf8fe15751a69be89ffdcb2c046cdb7989cae5a0'
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in {prior['source_sha256']!r}.items())
assert not Path({output!r}).exists()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
used={{int(line.split(',')[0]):int(line.split(',')[1]) for line in gpu.strip().splitlines()}}
assert all(used[gpu]<500 for _,_,gpu in {jobs!r})
print(json.dumps({{'source_sha256':{prior['source_sha256']!r},'gpu':gpu}}))
'''
availability=json.loads(remote_python('2027',guard26))
pause25 = f'''import hashlib,json,os,signal,subprocess
from pathlib import Path
root=Path({HOSTS['2025'][0]!r})
process=Path('/proc/1408150')
assert process.exists()
command=(process/'cmdline').read_bytes().replace(b'\\x00',b' ').decode()
assert 'launch_axis_scaled.py --data-root /data2/gb/Re-ID/dataset' in command
kernel=subprocess.check_output(['dmesg','--ctime'],text=True)
assert any('Xid' in line and '79,' in line and 'GPU has fallen off the bus' in line for line in kernel.splitlines())
campaign=root/'runs/axis_collaboration_v4_development'
frequency=campaign/'RGBNT100_frequency_scaled_fullref_s42'
assert json.loads((frequency/'status.json').read_text())['epoch']==43
assert not (campaign/'RGBNT100_frequency_scaled_fullref_s42_exit.json').exists()
assert not (campaign/'MSVR310_axis_raw_fullref_s42').exists()
base=json.loads((campaign/'MSVR310_axis_scaled_base_s42/result.json').read_text())
assert base['status']=='COMPLETE' and base['epochs']==50
assert not (campaign/'MSVR310_axis_scaled_base_s42_exit.json').exists()
os.kill(1408150,signal.SIGSTOP)
print(json.dumps({{'pid':1408150,'signal':'SIGSTOP','reason':'confirmed NVIDIA Xid79; prevent original pending job from starting during hardware recovery','command':command,'base_completed50_but_cleanup_exit_missing':True}}))
'''
paused=json.loads(remote_python('2025',pause25))
controller_code = f'''from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from launch_axis_scaled import idle
from launch_runs import write_json
root=Path({output!r})
def run(job):
 dataset,variant,gpu=job
 name=dataset+'_'+variant+'_s42'
 rows=[]
 for mode in ('smoke','train'):
  idle(gpu)
  stage='preflight' if mode=='smoke' else 'development'
  campaign=root/stage
  command=[sys.executable,'-u','run_experiment.py','--dataset',dataset,'--variant',variant,'--seed','42','--mode',mode,'--contribution-weight','.05','--data-root','/data/gb/Re-ID/dataset','--pretrained','/data/gb/Re-ID/pretrained/ViT-B-16.pt','--output',str(campaign/name)]
  with (campaign/(name+'.log')).open('x') as log:
   child=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
   write_json(campaign/(name+'_launch.json'),dict(pid=child.pid,gpu=gpu,started=time.time(),command=command,hardware_recovery_from='2025 NVIDIA Xid79',restart_from_public_CLIP=True))
   code=child.wait()
  row=dict(name=name,mode=mode,exit_code=code,finished=time.time())
  write_json(campaign/(name+'_exit.json'),row)
  rows.append(row)
  assert code==0,(name,mode)
  if mode=='smoke':
   check=json.loads((campaign/name/'smoke.json').read_text())
   assert check['status']=='SMOKE_PASS' and check['steps']==3 and check['strict_reload_equal']
  else:
   check=json.loads((campaign/name/'result.json').read_text())
   assert check['status']=='COMPLETE' and check['epochs']==50
 return dict(name=name,gpu=gpu,checks=rows)
with ThreadPoolExecutor(max_workers=2) as pool:
 rows=list(pool.map(run,{jobs!r}))
write_json(root/'controller_result.json',dict(status='COMPLETE',runs=rows,original25_interrupted_attempt_separate=True))
'''
entry = f"import sys,runpy;sys.path.insert(0,{root!r});runpy.run_path({(output+'/controller.py')!r},run_name='__main__')"
launch_code=f'''import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in {prior['source_sha256']!r}.items())
output=Path({output!r})
output.mkdir(exist_ok=False)
(output/'preflight').mkdir(exist_ok=False)
(output/'development').mkdir(exist_ok=False)
(output/'controller.py').write_text({controller_code!r})
with (output/'controller.log').open('x') as log:
 child=subprocess.Popen([{python!r},'-u','-c',{entry!r}],cwd=root,start_new_session=True,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
print(json.dumps({{'pid':child.pid,'host':'2027','started':time.time(),'status':'HARDWARE_RECOVERY_STARTED_SMOKE_AND_50_EPOCH_RUNS_PENDING','jobs':{jobs!r},'output':str(output),'source_sha256':{prior['source_sha256']!r}}}))
'''
launch=json.loads(remote_python('2027',launch_code))
launch.update(observed_at=datetime.now().isoformat(timespec='seconds'),original25_controller=paused,availability27=availability,incident_sha256=hashlib.sha256((PROJECT/'results/preflight/axis_collaboration_v4_driver_evidence.json').read_bytes()).hexdigest(),estimated_seconds={'RGBNT100':5400,'MSVR310':900},original_interrupted_steps_not_counted_as_new_updates=True)
receipt=PROJECT/'results/preflight/axis_collaboration_v4_hardware_recovery27_launch.json'
assert not receipt.exists()
receipt.write_bytes(json.dumps(launch,indent=2).encode('utf-8'))
print('V4_HARDWARE_RECOVERY_STARTED',json.dumps({key:launch[key] for key in ('pid','host','status','observed_at','jobs')}),flush=True)
