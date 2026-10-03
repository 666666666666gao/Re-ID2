from datetime import datetime
import hashlib
import json
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

previous = json.loads((PROJECT / 'results/preflight/axis_collaboration_v4_msvr_cross26_launch.json').read_text(encoding='utf-8'))
review = json.loads((PROJECT / 'results/preflight/axis_collaboration_v4_four_state_review.json').read_text(encoding='utf-8'))
assert review['status'] == 'PASS' and not review['blockers']
assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == digest for name, digest in review['source_sha256'].items())
root, python = HOSTS['2026']
inputs = root + '/runs/axis_collaboration_v4_cross26_input'
output = root + '/runs/axis_collaboration_v4_extra_cross26_diagnostic'
jobs = [('RGBNT201_axis_scaled_fullref_s42', 0), ('RGBNT100_axis_scaled_fullref_s42', 1)]
guard = f'''import hashlib,json,subprocess
from pathlib import Path
root=Path({root!r})
old=Path({previous['output']!r})
assert json.loads((old/'controller_result.json').read_text())=={{'status':'PASS','checks':[{{'name':'smoke','exit_code':0}},{{'name':'full','exit_code':0}}],'optimizer_updates':0}}
assert not subprocess.run(['ps','-p',{str(previous['pid'])!r},'-o','pid='],capture_output=True,text=True).stdout.strip()
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in {previous['source_sha256']!r}.items())
assert not Path({output!r}).exists()
for name,gpu in {jobs!r}:
 (Path({inputs!r})/name).mkdir(exist_ok=False)
print('EXTRA_FROZEN_INPUT_ROOTS_READY')
'''
assert remote_python('2026', guard).strip() == 'EXTRA_FROZEN_INPUT_ROOTS_READY'
proofs = {}
for name, gpu in jobs:
    source = HOSTS['2025'][0] + '/runs/axis_collaboration_v4_development/' + name
    proof_code = f'''import hashlib,json
from pathlib import Path
root=Path({source!r})
result=json.loads((root/'result.json').read_text())
assert result['status']=='COMPLETE' and result['epochs']==50 and result['arguments']['variant']=='axis_scaled_fullref'
exit=root.parent/(root.name+'_exit.json')
assert json.loads(exit.read_text())['exit_code']==0
print(json.dumps({{path.name:{{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}} for path in [root/'best.pth',root/'best_dev_arrays.npz',root/'result.json',exit]}}))
'''
    proofs[name] = json.loads(remote_python('2025', proof_code))
    for filename in ('best.pth', 'best_dev_arrays.npz', 'result.json'):
        command(['scp', '-3', *OPTIONS, '2025:' + source + '/' + filename, '2026:' + inputs + '/' + name + '/' + filename])
    command(['scp', '-3', *OPTIONS, '2025:' + source.rsplit('/', 1)[0] + '/' + name + '_exit.json', '2026:' + inputs + '/' + name + '_exit.json'])

controller_code = f'''from concurrent.futures import ThreadPoolExecutor
import os,subprocess,sys
from pathlib import Path
from launch_axis_scaled import idle
from launch_runs import write_json
root=Path({output!r})
def run(job):
 name,gpu=job
 output=root/name
 output.mkdir(exist_ok=False)
 checks=[]
 for smoke in (True,False):
  idle(gpu)
  stage='smoke' if smoke else 'full'
  command=[sys.executable,'-u','diagnose_axis_collaboration.py','--run-dir',{inputs!r}+'/'+name,'--output',str(output/stage),'--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt']
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
 rows=list(pool.map(run,{jobs!r}))
write_json(root/'controller_result.json',dict(status='PASS',runs=rows,optimizer_updates=0))
'''
entry = f"import sys,runpy;sys.path.insert(0,{root!r});runpy.run_path({(output + '/controller.py')!r},run_name='__main__')"
launch_code = f'''import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in {previous['source_sha256']!r}.items())
inputs=Path({inputs!r})
for run,files in {proofs!r}.items():
 for name,proof in files.items():
  path=inputs/name if name.endswith('_exit.json') else inputs/run/name
  assert path.stat().st_size==proof['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==proof['sha256'],name
output=Path({output!r})
output.mkdir(exist_ok=False)
(output/'controller.py').write_text({controller_code!r})
with (output/'controller.log').open('x') as log:
 child=subprocess.Popen([{python!r},'-u','-c',{entry!r}],cwd=root,start_new_session=True,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
print(json.dumps({{'pid':child.pid,'host':'2026','started':time.time(),'status':'FROZEN_CONTROLLER_STARTED_SMOKE_AND_FULL_PENDING','jobs':{jobs!r},'output':str(output),'source_sha256':{previous['source_sha256']!r},'input_files':{proofs!r}}}))
'''
launch = json.loads(remote_python('2026', launch_code))
launch['observed_at'] = datetime.now().isoformat(timespec='seconds')
receipt = PROJECT / 'results/preflight/axis_collaboration_v4_extra_cross26_launch.json'
assert not receipt.exists()
receipt.write_bytes(json.dumps(launch, indent=2).encode('utf-8'))
print('V4_EXTRA_FROZEN_DIAGNOSTIC_STARTED', json.dumps({k: launch[k] for k in ('pid', 'jobs', 'observed_at', 'status')}), flush=True)
