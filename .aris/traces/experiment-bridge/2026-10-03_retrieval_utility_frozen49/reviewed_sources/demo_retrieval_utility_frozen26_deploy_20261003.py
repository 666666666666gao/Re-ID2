from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS,command,remote_python

review=json.loads((PROJECT/'results/preflight/retrieval_utility_frozen_review.json').read_text(encoding='utf-8'))
assert review['status']=='PASS' and not review['blockers']
assert review['helper_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
files=('missing_retrieval_utility_development.py','diagnose_retrieval_utility_axis.py','launch_retrieval_utility_frozen_evaluation.py')
new_sources={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in files}
assert review['checked_source_sha256']==new_sources
launch=json.loads((PROJECT/'results/preflight/retrieval_utility_launch.json').read_text(encoding='utf-8'))
sources=launch['source_sha256'].copy()
sources['missing_evaluation.py']=hashlib.sha256((PROJECT/'missing_evaluation.py').read_bytes()).hexdigest()
assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in sources.items())
root,python=HOSTS['2026']
training=launch['output']+'/development'
output=root+'/runs/axis_collaboration_v10_retrieval_utility_frozen_trial'
first=training+'/MSVR310_axis_retrieval_utility_fullref_s42'
guard=f'''import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {sources!r}.items())
assert all(not (root/name).exists() for name in {files!r})
assert not Path({output!r}).exists()
assert shutil.disk_usage(root).free>200000000
base=Path({first!r});exit_file=base.parent/(base.name+'_exit.json')
assert json.loads(exit_file.read_text())['exit_code']==0
result=json.loads((base/'result.json').read_text())
assert result['status']=='COMPLETE' and result['epochs']==50 and result['arguments']['variant']=='axis_retrieval_utility_fullref'
inputs=(base/'best.pth',base/'best_dev_arrays.npz',base/'result.json',exit_file)
proofs={{str(path):dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in inputs}}
used=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
assert used<500
print(json.dumps(dict(gpu1_memory_used_MiB=used,first_inputs=proofs,warm_environment='existing2026 validated V5 environment; no environment edits, new metric-interface model sources bound by training receipt')))
'''
availability=json.loads(remote_python('2026',guard))
for name in files:command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+root+'/'+name])
all_sources={**sources,**new_sources}
start=f'''import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {all_sources!r}.items())
for name,proof in {availability['first_inputs']!r}.items():
 path=Path(name);assert path.stat().st_size==proof['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==proof['sha256']
used=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert used<500
assert shutil.disk_usage(root).free>200000000
output=Path({output!r});assert not output.exists()
log=Path(str(output)+'.log');assert not log.exists()
argv=[{python!r},'-u','launch_retrieval_utility_frozen_evaluation.py','--training-root',{training!r},'--output',str(output),'--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt']
with log.open('x') as handle:
 child=subprocess.Popen(argv,cwd=root,start_new_session=True,stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
print(json.dumps(dict(pid=child.pid,host='2026',gpu=1,output=str(output),log=str(log),training_root={training!r},source_sha256={all_sources!r},started=time.time(),command=argv,status='MSVR_FROZEN49_AND4_DIAGNOSTICS_STARTED',planned_missing_conditions=49,planned_four_state_conditions=4,optimizer_updates=0,official_test_uses=0)))
'''
receipt=json.loads(remote_python('2026',start));receipt['availability']=availability;receipt['observed_at']=datetime.now().isoformat(timespec='seconds')
target=PROJECT/'results/preflight/retrieval_utility_frozen_launch.json';assert not target.exists()
target.write_bytes(json.dumps(receipt,indent=2).encode('utf-8'))
print('RETRIEVAL_UTILITY_FROZEN26_STARTED',json.dumps({key:receipt[key] for key in ('pid','host','gpu','status')}),flush=True)
