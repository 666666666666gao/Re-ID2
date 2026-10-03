from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS,command,remote_python
HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
review=json.loads((PROJECT/'results/preflight/availability_base_review.json').read_text())
assert review['status']=='PASS' and not review['blockers']
assert review['helper_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
plan_path=PROJECT/'results/preflight/availability_base_plan.json'
assert review['plan_sha256']==hashlib.sha256(plan_path.read_bytes()).hexdigest()
plan=json.loads(plan_path.read_text());assert plan['status']=='PREPARED_SOURCE_REVIEW_REQUIRED_NO_NEURAL_RUN'
files=('availability_base_intervention.py','verify_availability_base_intervention.py','missing_available_base_development.py','launch_available_base_frozen.py')
new_sources={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in files}
assert review['checked_source_sha256']==new_sources
receipts=[]
for host in ('2026','2027'):
    if host=='2027':
        previous=receipts[0]
        code=f'''import json
from pathlib import Path
root=Path({previous['output']!r});stages=[]
for name in ('RGBNT201_axis_mass_fullref_s42','MSVR310_axis_mass_fullref_s42'):
 path=root/name/'controller_result.json'
 if path.exists():
  assert json.loads(path.read_text())['status']=='PASS'
  record=json.loads((root/name/'full_exit.json').read_text());assert record['exit_code']==0,(name,record)
  stages.append(name)
print(json.dumps(dict(released=len(stages)==2)))
'''
        while not json.loads(remote_python('2026',code))['released']:
            print('WAIT_FOR_TWO_2026_FROZEN_JOBS_TO_RELEASE_GLOBAL4_BUDGET',flush=True)
            time.sleep(240)
    root,python=HOSTS[host]
    parent=json.loads((PROJECT/('results/preflight/anytoany49_'+host+'_launch.json')).read_text())
    sources=parent['source_sha256'].copy()
    # New intervention depends directly on the actual base fusion implementation.
    for name in ('modeling/moe/AttnMOE.py','modeling/meta_arch.py','launch_axis_scaled.py','launch_runs.py'):
        sources[name]=hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in sources.items())
    jobs=[dict(job,run=root+'/'+job['run']) for job in plan['jobs'] if job['host']==host]
    output=root+'/runs/availability_base_frozen_'+host+'_20261003'
    jobs_path=root+'/results/availability_base_jobs_'+host+'_20261003.json'
    guard=f'''import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path({root!r});jobs={jobs!r}
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {sources!r}.items())
assert not Path({output!r}).exists() and not Path({jobs_path!r}).exists()
assert all(not (root/name).exists() for name in {files!r})
used={{int(line.split(',')[0]):int(line.split(',')[1]) for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()}}
assert all(used[job['gpu']]<500 for job in jobs) and shutil.disk_usage(root).free>1000000000
for job in jobs:
 run=Path(job['run']);terminal=json.loads((run/'result.json').read_text())
 assert terminal['status']=='COMPLETE' and terminal['epochs']==50 and terminal['arguments']['dataset']==job['dataset'] and terminal['arguments']['variant']==job['variant']
 exit_file=run/'exit.json' if job['variant']=='demo' else run.parent/(run.name+'_exit.json')
 assert json.loads(exit_file.read_text())['exit_code']==0
 job['frozen_inputs']={{path.name:dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in (run/'best.pth',run/'best_dev_arrays.npz',run/'result.json',exit_file)}}
print(json.dumps(dict(jobs=jobs,gpus={{str(job['gpu']):used[job['gpu']] for job in jobs}},disk_free_bytes=shutil.disk_usage(root).free)))
'''
    availability=json.loads(remote_python(host,guard))
    for name in files:command(['scp',*OPTIONS,str(PROJECT/name),host+':'+root+'/'+name])
    all_sources={**sources,**new_sources};jobs=availability['jobs']
    data='/data/gaob/Re-ID/dataset' if host=='2026' else '/data/gb/Re-ID/dataset'
    pretrained='/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt' if host=='2026' else '/data/gb/Re-ID/pretrained/ViT-B-16.pt'
    start=f'''import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path({root!r});jobs={jobs!r}
assert not Path({output!r}).exists() and not Path({jobs_path!r}).exists()
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {all_sources!r}.items())
used={{int(line.split(',')[0]):int(line.split(',')[1]) for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()}}
assert all(used[job['gpu']]<500 for job in jobs)
Path({jobs_path!r}).write_text(json.dumps(jobs,indent=2))
argv=[{python!r},'-u','launch_available_base_frozen.py','--jobs-json',{jobs_path!r},'--output',{output!r},'--data-root',{data!r},'--pretrained',{pretrained!r}]
log=Path({output!r}+'.log')
with log.open('x') as handle:
 child=subprocess.Popen(argv,cwd=root,start_new_session=True,stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
print(json.dumps(dict(host={host!r},pid=child.pid,output={output!r},log=str(log),source_sha256={all_sources!r},jobs=jobs,jobs_path={jobs_path!r},jobs_sha256=hashlib.sha256(Path({jobs_path!r}).read_bytes()).hexdigest(),started=time.time(),command=argv)))
'''
    receipt=json.loads(remote_python(host,start));receipt.update(availability=availability,observed_at=datetime.now().isoformat(timespec='seconds'))
    target=PROJECT/('results/preflight/availability_base_'+host+'_launch.json');assert not target.exists()
    target.write_bytes(json.dumps(receipt,indent=2).encode());receipts.append(receipt)
    print('AVAILABILITY_BASE_FROZEN_CONTROLLER_STARTED',json.dumps(dict(host=host,pid=receipt['pid'],gpus=[job['gpu'] for job in jobs])),flush=True)
