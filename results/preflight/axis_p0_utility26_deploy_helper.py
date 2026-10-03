from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS,command,remote_python

review=json.loads((PROJECT/'results/preflight/axis_p0_utility_review.json').read_text(encoding='utf-8'))
assert review['status']=='PASS' and not review['blockers']
assert review['helper_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
files=('probe_axis_retrieval_utility.py','probe_axis_task_gradients.py','launch_axis_utility_probes.py')
new_sources={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in files}
assert review['checked_source_sha256']==new_sources
sources=json.loads((PROJECT/'results/preflight/axis_collaboration_v6_projected_frozen_launch.json').read_text(encoding='utf-8'))['source_sha256']
sources.update({name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in
    ('configs/RGBNT100/DeMo.yml','configs/RGBNT201/DeMo.yml','modeling/meta_arch.py','layers/make_loss.py')})
assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in sources.items())
root,python=HOSTS['2026'];output=root+'/runs/axis_p0_retrieval_utility_20261003'
v5=root+'/runs/axis_collaboration_v5_mass/development'
v6=root+'/runs/axis_collaboration_v6_projected_mass_trial_ampfix/development/MSVR310_axis_mass_projected_fullref_s42'
guard=f'''import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {sources!r}.items())
assert all(not (root/name).exists() for name in {files!r}) and not Path({output!r}).exists()
for run in {[v5+'/'+dataset+'_axis_mass_fullref_s42' for dataset in ('MSVR310','RGBNT100','RGBNT201')]+[v6]!r}:
 path=Path(run);result=json.loads((path/'result.json').read_text())
 assert result['status']=='COMPLETE' and result['epochs']==50
 assert json.loads((path.parent/(path.name+'_exit.json')).read_text())['exit_code']==0
used={{gpu:int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()) for gpu in (0,1,3)}}
assert all(value<500 for value in used.values())
assert shutil.disk_usage(root).free>1000000000
print(json.dumps(dict(memory_used_MiB=used,disk_free_bytes=shutil.disk_usage(root).free)))
'''
availability=json.loads(remote_python('2026',guard))
for name in files:command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+root+'/'+name])
all_sources={**sources,**new_sources}
start=f'''import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {all_sources!r}.items())
assert all(int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<500 for gpu in (0,1,3))
assert shutil.disk_usage(root).free>1000000000
output=Path({output!r});assert not output.exists()
log=Path(str(output)+'.log');assert not log.exists()
argv=[{python!r},'-u','launch_axis_utility_probes.py','--v5-root',{v5!r},'--v6-run',{v6!r},'--output',str(output)]
with log.open('x') as handle:
 child=subprocess.Popen(argv,cwd=root,start_new_session=True,stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
print(json.dumps(dict(pid=child.pid,host='2026',gpus=[0,1,3],output=str(output),log=str(log),source_sha256={all_sources!r},started=time.time(),command=argv,status='P0_THREE_CARDS_FOUR_CHECKPOINTS_STARTED',optimizer_updates=0,official_test_uses=0)))
'''
receipt=json.loads(remote_python('2026',start));receipt.update(availability=availability,observed_at=datetime.now().isoformat(timespec='seconds'))
target=PROJECT/'results/preflight/axis_p0_utility_launch.json';assert not target.exists()
target.write_bytes(json.dumps(receipt,indent=2).encode('utf-8'))
print('AXIS_P0_UTILITY_STARTED',json.dumps(dict(pid=receipt['pid'],gpus=receipt['gpus'])),flush=True)
