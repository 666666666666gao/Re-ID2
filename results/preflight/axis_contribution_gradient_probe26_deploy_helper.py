from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS,command,remote_python

review=json.loads((PROJECT/'results/preflight/axis_contribution_gradient_probe_review.json').read_text(encoding='utf-8'))
assert review['status']=='PASS' and not review['blockers']
assert review['helper_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
files=('probe_contribution_gradients.py','launch_contribution_gradient_probe.py')
new_sources={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in files}
assert review['checked_source_sha256']==new_sources
sources=json.loads((PROJECT/'results/preflight/axis_collaboration_v6_projected_frozen_launch.json').read_text(encoding='utf-8'))['source_sha256']
assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in sources.items())
root,python=HOSTS['2026'];output=root+'/runs/axis_contribution_gradient_diagnostic_20261003'
v5=root+'/runs/axis_collaboration_v5_mass/development/MSVR310_axis_mass_fullref_s42'
v6=root+'/runs/axis_collaboration_v6_projected_mass_trial_ampfix/development/MSVR310_axis_mass_projected_fullref_s42'
guard=f'''import hashlib,json,subprocess
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {sources!r}.items())
assert all(not (root/name).exists() for name in {files!r}) and not Path({output!r}).exists()
for run in {[(v5,'axis_mass_fullref'),(v6,'axis_mass_projected_fullref')]!r}:
 path=Path(run[0]);result=json.loads((path/'result.json').read_text())
 assert result['status']=='COMPLETE' and result['epochs']==50 and result['arguments']['variant']==run[1]
 assert json.loads((path.parent/(path.name+'_exit.json')).read_text())['exit_code']==0
used={{gpu:int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()) for gpu in (0,1)}}
assert all(value<500 for value in used.values())
print(json.dumps(dict(memory_used_MiB=used)))
'''
availability=json.loads(remote_python('2026',guard))
for name in files:command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+root+'/'+name])
all_sources={**sources,**new_sources}
start=f'''import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {all_sources!r}.items())
assert all(int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<500 for gpu in (0,1))
output=Path({output!r});assert not output.exists()
log=Path(str(output)+'.log');assert not log.exists()
argv=[{python!r},'-u','launch_contribution_gradient_probe.py','--v5-run',{v5!r},'--v6-run',{v6!r},'--output',str(output)]
with log.open('x') as handle:
 child=subprocess.Popen(argv,cwd=root,start_new_session=True,stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
print(json.dumps(dict(pid=child.pid,host='2026',gpus=[0,1],output=str(output),log=str(log),source_sha256={all_sources!r},started=time.time(),command=argv,status='TWO_TRAINING_MODE_GRADIENT_DIAGNOSTICS_STARTED',optimizer_updates=0,official_test_uses=0)))
'''
receipt=json.loads(remote_python('2026',start));receipt.update(availability=availability,observed_at=datetime.now().isoformat(timespec='seconds'))
target=PROJECT/'results/preflight/axis_contribution_gradient_probe_launch.json';assert not target.exists()
target.write_bytes(json.dumps(receipt,indent=2).encode('utf-8'))
print('CONTRIBUTION_PROBE26_STARTED',json.dumps(dict(pid=receipt['pid'],gpus=receipt['gpus'])),flush=True)
