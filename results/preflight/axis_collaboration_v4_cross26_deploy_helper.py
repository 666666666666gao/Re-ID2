from datetime import datetime
import hashlib
import json
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

review_path=PROJECT/'results/preflight/axis_collaboration_v4_four_state_review.json'
review=json.loads(review_path.read_text(encoding='utf-8'))
assert review['status'].startswith('PASS') and not review['blockers']
assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==digest for name,digest in review['source_sha256'].items())
files=['run_experiment.py','scaled_axis_collaboration.py','diagnose_axis_collaboration.py','launch_scaled_diagnostic.py','launch_axis_scaled.py']
dependencies=['axis_collaboration.py','residual_dual_axis.py','dual_axis.py','experiment_data.py','full_evaluation.py','modeling/make_model.py','utils/reid_evaluation.py','splits.json','configs/MSVR310/DeMo.yml']
digest={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in files+dependencies}
root,python=HOSTS['2026']
name='MSVR310_axis_scaled_fullref_s42'
input_root=root+'/runs/axis_collaboration_v4_cross26_input'
output_root=root+'/runs/axis_collaboration_v4_msvr_cross26_diagnostic'
guard=f'''import hashlib,json,subprocess
from pathlib import Path
root=Path({root!r})
terminal=json.loads((root/'runs/missing_controller_result.json').read_text())
assert len(terminal)==14 and all(row['exit_code']==0 for row in terminal)
assert not subprocess.run(['ps','-p','2152152','-o','pid='],capture_output=True,text=True).stdout.strip()
assert hashlib.sha256((root/'run_experiment.py').read_bytes()).hexdigest()=='b395d5cc13d8c8ea372391c15cf5e28da1b209bd315362521053afbcd288d830'
for filename in {dependencies!r}:
 assert hashlib.sha256((root/filename).read_bytes()).hexdigest()=={digest!r}[filename],filename
target=Path({input_root!r})
target.mkdir(exist_ok=False)
(target/{name!r}).mkdir()
assert not Path({output_root!r}).exists()
print('CROSS26_PREDECESSOR_AND_SOURCE_PASS')
'''
assert remote_python('2026',guard).strip()=='CROSS26_PREDECESSOR_AND_SOURCE_PASS'
for filename in files:
 command(['scp',*OPTIONS,str(PROJECT/filename),'2026:'+root+'/'+filename])
command(['scp',*OPTIONS,str(review_path),'2026:'+root+'/results/preflight/'+review_path.name])
source=HOSTS['2025'][0]+'/runs/axis_collaboration_v4_development/'+name
proof_code=f'''import hashlib,json
from pathlib import Path
root=Path({source!r})
result=json.loads((root/'result.json').read_text())
assert result['status']=='COMPLETE' and result['epochs']==50
exit=root.parent/(root.name+'_exit.json')
assert json.loads(exit.read_text())['exit_code']==0
files=[root/'best.pth',root/'best_dev_arrays.npz',root/'result.json',exit]
print(json.dumps({{path.name:{{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}} for path in files}}))
'''
proof=json.loads(remote_python('2025',proof_code))
for filename in ('best.pth','best_dev_arrays.npz','result.json'):
 command(['scp','-3',*OPTIONS,'2025:'+source+'/'+filename,'2026:'+input_root+'/'+name+'/'+filename])
command(['scp','-3',*OPTIONS,'2025:'+source.rsplit('/',1)[0]+'/'+name+'_exit.json','2026:'+input_root+'/'+name+'_exit.json'])
controller_code=f'''import json,os,subprocess,sys
from pathlib import Path
from launch_axis_scaled import idle
from launch_runs import write_json
output=Path({output_root!r})
run=Path({(input_root+'/'+name)!r})
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
'''
entry=f"import sys,runpy;sys.path.insert(0,{root!r});runpy.run_path({(output_root+'/controller.py')!r},run_name='__main__')"
code=f'''import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path({root!r})
expected={digest!r}
assert {{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in expected}}==expected
inputs=Path({input_root!r})
for name,record in {proof!r}.items():
 path=inputs/name if name.endswith('_exit.json') else inputs/{name!r}/name
 assert path.stat().st_size==record['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
output=Path({output_root!r})
output.mkdir(exist_ok=False)
script=output/'controller.py'
script.write_text({controller_code!r})
log=output/'controller.log'
with log.open('x') as stream:
 child=subprocess.Popen([{python!r},'-u','-c',{entry!r}],cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
 env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
print(json.dumps({{'pid':child.pid,'host':'2026','gpu':3,'started':time.time(),'status':'FROZEN_CONTROLLER_STARTED_SMOKE_AND_FULL_PENDING','output':str(output),'source_sha256':expected,'input_files':{proof!r}}}))
'''
launch=json.loads(remote_python('2026',code))
launch['observed_at']=datetime.now().isoformat(timespec='seconds')
launch['review_file']=str(review_path.relative_to(PROJECT)).replace('\\','/')
receipt=PROJECT/'results/preflight/axis_collaboration_v4_msvr_cross26_launch.json'
assert not receipt.exists()
receipt.write_bytes(json.dumps(launch,indent=2).encode('utf-8'))
print('V4_FROZEN_CROSS26_DIAGNOSTIC_STARTED',json.dumps({'pid':launch['pid'],'gpu':3,'observed_at':launch['observed_at'],'status':launch['status']}),flush=True)
