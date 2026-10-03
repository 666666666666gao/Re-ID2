from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

def read(name):
    return json.loads((PROJECT / name).read_text(encoding='utf-8'))

model_review = read('results/preflight/axis_collaboration_v5_mass_model_review.json')
workflow_review = read('results/preflight/axis_collaboration_v5_mass_workflow_review.json')
assert model_review['status'] == workflow_review['status'] == 'PASS'
assert not model_review['blockers'] and not workflow_review['blockers']
assert workflow_review['helper_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
names = ('mass_axis_collaboration.py', 'run_mass_experiment.py', 'verify_axis_mass.py')
sources = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in names}
assert all(model_review['checked_source_sha256'][name] == sources[name] for name in names[:2])
assert workflow_review['checked_source_sha256'] == sources
missing = read('results/preflight/axis_collaboration_v4_missing_development26_complete_analysis.json')
assert missing['status'] == 'PASS_ACTUAL_78_FROZEN_DEVELOPMENT_CONDITIONS' and missing['conditions'] == 78
old = read('results/preflight/axis_collaboration_v4_missing_development26_repaired_launch.json')
root, python = HOSTS['2026']
output = root + '/runs/axis_collaboration_v5_mass'
jobs = {0: 'RGBNT100', 1: 'RGBNT201', 3: 'MSVR310'}
guard = f'''import hashlib,json,subprocess
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {old['source_sha256']!r}.items())
assert not subprocess.run(['ps','-p',{str(old['pid'])!r},'-o','pid='],capture_output=True,text=True).stdout.strip()
terminal=json.loads(Path({(old['output'] + '/controller_result.json')!r}).read_text())
assert terminal['status']=='COMPLETE' and terminal['conditions']==78 and terminal['optimizer_updates']==0
assert not Path({output!r}).exists()
assert all(not (root/name).exists() for name in {names!r})
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
used={{int(line.split(',')[0]):int(line.split(',')[1]) for line in gpu.strip().splitlines()}}
assert all(used[index]<500 for index in {tuple(jobs)!r})
print(json.dumps(dict(gpu=gpu,warm_environment='existing2026 Python3.10.14/Torch2.5.1cu121; no environment or original source changes')))
'''
availability = json.loads(remote_python('2026', guard))
for name in names:
    command(['scp', *OPTIONS, str(PROJECT / name), '2026:' + root + '/' + name])
source_sha = {**old['source_sha256'], **sources}
controller = f'''from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,{root!r})
from launch_axis_scaled import idle
from launch_runs import write_json
project=Path({root!r})
root=Path({output!r})
assert all(hashlib.sha256((project/name).read_bytes()).hexdigest()==sha for name,sha in {source_sha!r}.items())
preflight=root/'preflight';preflight.mkdir(exist_ok=False)
development=root/'development'

def execute(argv,folder,name,gpu):
 idle(gpu)
 with (folder/(name+'.log')).open('x') as log:
  child=subprocess.Popen(argv,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
  write_json(folder/(name+'_launch.json'),dict(pid=child.pid,gpu=gpu,command=argv,started=time.time()))
  code=child.wait()
 write_json(folder/(name+'_exit.json'),dict(exit_code=code,finished=time.time(),name=name,gpu=gpu))
 assert code==0,name
 return dict(name=name,gpu=gpu,exit_code=code)

checks=[]
for dataset in ('MSVR310','RGBNT201','RGBNT100'):
 name=dataset+'_tensor'
 argv=[sys.executable,'-u','verify_axis_mass.py','--dataset',dataset,'--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt','--output',str(preflight/(name+'.json'))]
 checks.append(execute(argv,preflight,name,3))
 assert json.loads((preflight/(name+'.json')).read_text())['status']=='PASS'
 name=dataset+'_axis_mass_fullref_smoke'
 argv=[sys.executable,'-u','run_mass_experiment.py','--dataset',dataset,'--variant','axis_mass_fullref','--seed','42','--mode','smoke','--contribution-weight','.05','--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt','--output',str(preflight/name)]
 checks.append(execute(argv,preflight,name,3))
 smoke=json.loads((preflight/name/'smoke.json').read_text())
 assert smoke['status']=='SMOKE_PASS' and smoke['steps']==3 and smoke['strict_reload_equal'] and all(smoke['gradients'].values())
assert len(checks)==6
write_json(preflight/'controller_result.json',dict(status='PASS',checks=checks,optimizer_updates=9))
development.mkdir(exist_ok=False)

def slot(gpu,dataset):
 name=dataset+'_axis_mass_fullref_s42'
 argv=[sys.executable,'-u','run_mass_experiment.py','--dataset',dataset,'--variant','axis_mass_fullref','--seed','42','--mode','train','--contribution-weight','.05','--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt','--output',str(development/name)]
 row=execute(argv,development,name,gpu)
 result=json.loads((development/name/'result.json').read_text())
 assert result['status']=='COMPLETE' and result['epochs']==50
 return row

with ThreadPoolExecutor(max_workers=3) as workers:
 futures=[workers.submit(slot,gpu,dataset) for gpu,dataset in {jobs!r}.items()]
 rows=[future.result() for future in futures]
write_json(root/'controller_result.json',dict(status='COMPLETE',runs=rows,scope='three fresh50 identity-heldout development comparisons; official test unused'))
'''
launch = f'''import hashlib,json,subprocess
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {source_sha!r}.items())
output=Path({output!r});output.mkdir(exist_ok=False)
script=output/'controller.py';script.write_text({controller!r},encoding='utf-8')
with (output/'controller.log').open('x') as log:
 child=subprocess.Popen([{python!r},'-u',str(script)],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,host='2026',jobs={jobs!r},source_sha256={source_sha!r},output=str(output),started=__import__('time').time(),status='SIX_PREFLIGHT_GATES_STARTED_THEN_THREE_FRESH50',budget_epochs=150)))
'''
receipt = json.loads(remote_python('2026', launch))
receipt['observed_at'] = datetime.now().isoformat(timespec='seconds')
receipt['availability'] = availability
target = PROJECT / 'results/preflight/axis_collaboration_v5_mass_launch.json'
assert not target.exists()
target.write_bytes(json.dumps(receipt, ensure_ascii=False, indent=2).encode('utf-8'))
print('V5_MASS_PREFLIGHT_STARTED', json.dumps({key: receipt[key] for key in ('pid','host','jobs','status')}), flush=True)
