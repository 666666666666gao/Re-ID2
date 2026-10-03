from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS,command,remote_python
HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
review=json.loads((PROJECT/'results/preflight/axis_collaboration_v5_profile_review.json').read_text(encoding='utf-8'))
assert review['status']=='PASS' and not review['blockers']
assert review['helper_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
files=('profile_reid_models.py','mass_axis_collaboration.py')
sources={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in files}
assert review['checked_source_sha256']==sources
env=json.loads((PROJECT/'.aris/compute/ssh2027.json').read_text(encoding='utf-8'))
assert env['state']=='READY'
launch=json.loads((PROJECT/'results/preflight/axis_collaboration_v5_mass_launch.json').read_text(encoding='utf-8'))
original={name:sha for name,sha in launch['source_sha256'].items() if name not in ('mass_axis_collaboration.py','run_mass_experiment.py','verify_axis_mass.py','missing_development.py')}
assert len(original)==14
root,python=HOSTS['2027'];output=root+'/profile_output/axis_v4_v5_inference_cost_20261003'
guard=f'''import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {original!r}.items())
assert all(not (root/name).exists() for name in {files!r})
assert not Path({output!r}).exists()
used=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
assert used<500
print(json.dumps(dict(gpu0_memory_used_MiB=used,disk_free_bytes=shutil.disk_usage(root).free,scope='known READY2027 warm reuse; GPU0 only; no old source/environment changes')))
'''
availability=json.loads(remote_python('2027',guard))
for name in files:command(['scp',*OPTIONS,str(PROJECT/name),'2027:'+root+'/'+name])
source_sha={**original,**sources}
wrapper=f'''import json,os,subprocess,sys,time
from pathlib import Path
output=Path({output!r})
argv=[sys.executable,'-u','profile_reid_models.py','--data-root','/data/gb/Re-ID/dataset','--pretrained','/data/gb/Re-ID/pretrained/ViT-B-16.pt','--output',str(output)]
with Path(str(output)+'.log').open('x') as log:
 code=subprocess.run(argv,cwd={root!r},env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT).returncode
Path(str(output)+'_exit.json').write_text(json.dumps(dict(exit_code=code,finished=time.time())),encoding='utf-8')
assert code==0
'''
start=f'''import hashlib,json,subprocess,time
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {source_sha!r}.items())
used=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
assert used<500
output=Path({output!r});assert not output.exists() and not Path(str(output)+'_exit.json').exists()
output.parent.mkdir(exist_ok=True)
controller=subprocess.Popen([{python!r},'-u','-c',{wrapper!r}],cwd=root,start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
print(json.dumps(dict(pid=controller.pid,host='2027',gpu=0,output=str(output),source_sha256={source_sha!r},started=time.time(),status='FIFTEEN_BATCH8_FP32_COST_CONDITIONS_STARTED',optimizer_updates=0,official_test_uses=0)))
'''
receipt=json.loads(remote_python('2027',start));receipt['availability']=availability;receipt['observed_at']=datetime.now().isoformat(timespec='seconds')
path=PROJECT/'results/preflight/axis_collaboration_v5_profile_launch.json'
assert not path.exists();path.write_bytes(json.dumps(receipt,indent=2).encode('utf-8'))
print('V5_PROFILE27_STARTED',json.dumps({key:receipt[key] for key in ('pid','host','gpu','status')}),flush=True)
