import json,os,subprocess,sys
from pathlib import Path
from launch_axis_scaled import idle
from launch_runs import write_json
output=Path('/data/gaob/Re-ID/DeMo-DualAxis/runs/axis_collaboration_v4_msvr_cross26_diagnostic')
run=Path('/data/gaob/Re-ID/DeMo-DualAxis/runs/axis_collaboration_v4_cross26_input/MSVR310_axis_scaled_fullref_s42')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
checks=[]
for smoke in (True,False):
 idle(3)
 name='smoke' if smoke else 'full'
 command=[sys.executable,'-u','diagnose_axis_collaboration.py','--run-dir',str(run),'--output',str(output/name),'--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt']
 if smoke:command.append('--smoke')
 with (output/(name+'.log')).open('x') as log:
  child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
  write_json(output/(name+'_launch.json'),dict(pid=child.pid,gpu=3,command=command))
  code=child.wait()
 row=dict(name=name,exit_code=code)
 write_json(output/(name+'_exit.json'),row)
 checks.append(row)
 assert code==0,name
write_json(output/'controller_result.json',dict(status='PASS',checks=checks,optimizer_updates=0))
