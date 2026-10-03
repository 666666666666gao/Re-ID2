from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

plan_path = PROJECT / 'results/preflight/shared_identity_plan.json'
plan = json.loads(plan_path.read_text(encoding='utf-8'))
review = json.loads((PROJECT / 'results/preflight/shared_identity_review.json').read_text(encoding='utf-8'))
assert review['status'] == 'PASS' and not review['blockers']
assert review['plan_sha256'] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
assert review['helper_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
assert review['checked_source_sha256'] == plan['sources']
assert plan['status'] == 'PREPARED_SOURCE_REVIEW_REQUIRED_NO_NEURAL_RUN'
assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == sha for name, sha in plan['sources'].items())
root, python = HOSTS['2026']
parent = json.loads((PROJECT / 'results/preflight/availability_base_2026_launch.json').read_text())
sources = parent['source_sha256'].copy()
for name in ('layers/softmax_loss.py', 'layers/make_loss.py', 'layers/triplet_loss.py',
             'solver/make_optimizer.py', 'solver/scheduler_factory.py'):
    sources[name] = hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == sha for name, sha in sources.items())
output = root + '/runs/shared_identity_v11_trial_20261003'
guard = f'''import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {sources!r}.items())
assert not Path({output!r}).exists() and not Path({output!r}+'.log').exists()
assert all(not (root/name).exists() for name in {list(plan['sources'])!r})
used={{int(line.split(',')[0]):int(line.split(',')[1]) for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()}}
assert all(used[index]<500 for index in range(4))
assert shutil.disk_usage(root).free>10000000000
rows=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
assert not [row for row in rows if {python!r} in row and ' -c ' not in row]
print(json.dumps(dict(gpus=used,disk_free_bytes=shutil.disk_usage(root).free)))
'''
capacity = json.loads(remote_python('2026', guard))
for name in plan['sources']:
    command(['scp', *OPTIONS, str(PROJECT / name), '2026:' + root + '/' + name])
sources.update(plan['sources'])
data, pretrained = '/data/gaob/Re-ID/dataset', '/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt'
start = f'''import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {sources!r}.items())
assert not Path({output!r}).exists() and not Path({output!r}+'.log').exists()
used={{int(line.split(',')[0]):int(line.split(',')[1]) for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()}}
assert all(used[index]<500 for index in range(4))
argv=[{python!r},'-u','launch_shared_identity_trial.py','--data-root',{data!r},'--pretrained',{pretrained!r},'--output',{output!r}]
with Path({output!r}+'.log').open('x') as handle:
 child=subprocess.Popen(argv,cwd=root,start_new_session=True,stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
print(json.dumps(dict(host='2026',pid=child.pid,output={output!r},log={output!r}+'.log',command=argv,started=time.time(),source_sha256={sources!r})))
'''
receipt = json.loads(remote_python('2026', start))
receipt.update(capacity=capacity, observed_at=datetime.now().isoformat(timespec='seconds'))
target = PROJECT / 'results/preflight/shared_identity_2026_launch.json'
assert not target.exists()
target.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
print('SHARED_IDENTITY_CONTROLLER_STARTED', json.dumps(dict(pid=receipt['pid'],output=output)), flush=True)
