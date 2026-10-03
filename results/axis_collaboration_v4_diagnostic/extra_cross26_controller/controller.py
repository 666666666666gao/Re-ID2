from concurrent.futures import ThreadPoolExecutor
import os,subprocess,sys
from pathlib import Path
from launch_axis_scaled import idle
from launch_runs import write_json
root=Path('/data/gaob/Re-ID/DeMo-DualAxis/runs/axis_collaboration_v4_extra_cross26_diagnostic')
def run(job):
 name,gpu=job
 output=root/name
 output.mkdir(exist_ok=False)
 checks=[]
 for smoke in (True,False):
  idle(gpu)
  stage='smoke' if smoke else 'full'
  command=[sys.executable,'-u','diagnose_axis_collaboration.py','--run-dir','/data/gaob/Re-ID/DeMo-DualAxis/runs/axis_collaboration_v4_cross26_input'+'/'+name,'--output',str(output/stage),'--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt']
  if smoke:command.append('--smoke')
  with (output/(stage+'.log')).open('x') as log:
   child=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
   write_json(output/(stage+'_launch.json'),dict(pid=child.pid,gpu=gpu,command=command))
   code=child.wait()
  row=dict(name=stage,exit_code=code)
  write_json(output/(stage+'_exit.json'),row)
  checks.append(row)
  assert code==0,stage
 write_json(output/'controller_result.json',dict(status='PASS',checks=checks,optimizer_updates=0))
 return dict(name=name,gpu=gpu,checks=checks)
with ThreadPoolExecutor(max_workers=2) as pool:
 rows=list(pool.map(run,[('RGBNT201_axis_scaled_fullref_s42', 0), ('RGBNT100_axis_scaled_fullref_s42', 1)]))
write_json(root/'controller_result.json',dict(status='PASS',runs=rows,optimizer_updates=0))
