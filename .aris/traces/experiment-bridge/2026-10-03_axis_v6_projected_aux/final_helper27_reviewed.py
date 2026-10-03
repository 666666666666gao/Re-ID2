from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS,command,remote_python
HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
env=json.loads((PROJECT/'.aris/compute/ssh2027.json').read_text(encoding='utf-8'));assert env['state']=='READY'

review=json.loads((PROJECT/'results/preflight/axis_collaboration_v6_projected_review.json').read_text(encoding='utf-8'))
assert review['status']=='PASS' and not review['blockers']
assert review['helper_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
files=('projected_mass_axis_collaboration.py','run_projected_mass_experiment.py','verify_projected_axis_mass.py','launch_projected_mass_trial.py')
new_sources={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in files}
assert review['checked_source_sha256']==new_sources
original={name:sha for name,sha in json.loads((PROJECT/'results/preflight/axis_collaboration_v5_mass_launch.json').read_text(encoding='utf-8'))['source_sha256'].items() if name not in ('run_mass_experiment.py','verify_axis_mass.py','missing_development.py')}
assert len(original)==15
assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in original.items())
plan=json.loads((PROJECT/'results/preflight/axis_collaboration_v6_projected_plan.json').read_text(encoding='utf-8'))
assert plan['status']=='IMPLEMENTED_NOT_REVIEWED_NOT_DEPLOYED' and plan['budget_epochs']==50
root,python=HOSTS['2027'];output=root+'/runs/axis_collaboration_v6_projected_mass_trial'
guard=f'''import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {original!r}.items())
assert all(not (root/name).exists() for name in {files!r}) and not Path({output!r}).exists()
used=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert used<500
free=shutil.disk_usage(root).free;assert free>3200000000
print(json.dumps(dict(gpu0_memory_used_MiB=used,disk_free_bytes=free,disk_budget='actualMSVR~1.197GB smoke+399MB best+1.197GB last=2.793GB, plus400MB margin; no pending own2027 write jobs, no new archives/packages',warm_environment='READY2027 exact93package prefix/CUDA/GT/doc-witness and15FP32model profiles already actualPASS; no environment mutation')))
'''
availability=json.loads(remote_python('2027',guard))
for name in files:command(['scp',*OPTIONS,str(PROJECT/name),'2027:'+root+'/'+name])
sources={**original,**new_sources}
start=f'''import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {sources!r}.items())
used=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert used<500
assert shutil.disk_usage(root).free>3200000000
output=Path({output!r});assert not output.exists()
log=Path(str(output)+'.log');assert not log.exists()
argv=[{python!r},'-u','launch_projected_mass_trial.py','--data-root','/data/gb/Re-ID/dataset','--pretrained','/data/gb/Re-ID/pretrained/ViT-B-16.pt','--output',str(output)]
with log.open('x') as handle:
 child=subprocess.Popen(argv,cwd=root,start_new_session=True,stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
print(json.dumps(dict(pid=child.pid,host='2027',gpu=0,output=str(output),log=str(log),source_sha256={sources!r},started=time.time(),command=argv,status='TWO_REAL_PREFLIGHT_GATES_STARTED_THEN_ONE_FRESH50_MSVR',budget_epochs=50,planned_engineering_updates=3,official_test_uses=0)))
'''
receipt=json.loads(remote_python('2027',start));receipt['availability']=availability;receipt['observed_at']=datetime.now().isoformat(timespec='seconds')
target=PROJECT/'results/preflight/axis_collaboration_v6_projected_launch.json';assert not target.exists()
target.write_bytes(json.dumps(receipt,indent=2).encode('utf-8'))
print('V6_PROJECTED27_STARTED',json.dumps({key:receipt[key] for key in ('pid','host','gpu','status')}),flush=True)
