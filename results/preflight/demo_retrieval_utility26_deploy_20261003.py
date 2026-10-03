from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS,command,remote_python

review=json.loads((PROJECT/'results/preflight/retrieval_utility_review.json').read_text(encoding='utf-8'))
assert review['status']=='PASS' and not review['blockers']
assert review['helper_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
assert json.loads((PROJECT/'results/preflight/axis_p0_utility_analysis.json').read_text())['status']=='PASS_ACTUAL_FOUR_CASE_P0_EVIDENCE_AUDIT'
files=('retrieval_utility_axis.py','run_retrieval_utility_experiment.py','verify_retrieval_utility.py','launch_retrieval_utility_trial.py')
new_sources={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in files}
assert review['checked_source_sha256']==new_sources
sources=json.loads((PROJECT/'results/preflight/relation_frequency_frozen_launch.json').read_text(encoding='utf-8'))['source_sha256']
assert json.loads((PROJECT/'results/preflight/relation_frequency_msvr_development_analysis.json').read_text())['status']=='PASS_EVIDENCE_AUDIT'
assert json.loads((PROJECT/'results/preflight/relation_frequency_msvr_frozen_analysis.json').read_text())['status']=='PASS_ACTUAL_FROZEN_EVIDENCE_AUDIT'
assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in sources.items())
root,python=HOSTS['2026'];output=root+'/runs/axis_collaboration_v10_retrieval_utility_trial'
guard=f'''import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {sources!r}.items())
assert all(not (root/name).exists() for name in {files!r}) and not Path({output!r}).exists()
assert json.loads((root/'runs/axis_collaboration_v9_relation_frequency_trial/controller_result.json').read_text())['status']=='COMPLETE'
assert json.loads((root/'runs/axis_collaboration_v9_relation_frequency_frozen_trial/controller_result.json').read_text())['status']=='COMPLETE'
used=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert used<500
free=shutil.disk_usage(root).free;assert free>3200000000
print(json.dumps(dict(gpu1_memory_used_MiB=used,disk_free_bytes=free,storage='Full original smoke/last Adam checkpoints and best, no deletion or serialization changes')))
'''
availability=json.loads(remote_python('2026',guard))
for name in files:command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+root+'/'+name])
all_sources={**sources,**new_sources}
start=f'''import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {all_sources!r}.items())
assert int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<500
assert shutil.disk_usage(root).free>3200000000
output=Path({output!r});assert not output.exists()
log=Path(str(output)+'.log');assert not log.exists()
argv=[{python!r},'-u','launch_retrieval_utility_trial.py','--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt','--output',str(output)]
with log.open('x') as handle:
 child=subprocess.Popen(argv,cwd=root,start_new_session=True,stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
print(json.dumps(dict(pid=child.pid,host='2026',gpu=1,output=str(output),log=str(log),source_sha256={all_sources!r},started=time.time(),command=argv,status='P1_RETRIEVAL_UTILITY_TENSOR_AMP_THEN_MSVR_FRESH50_STARTED',budget_epochs=50,planned_engineering_updates=3,official_test_uses=0)))
'''
receipt=json.loads(remote_python('2026',start));receipt.update(availability=availability,observed_at=datetime.now().isoformat(timespec='seconds'))
target=PROJECT/'results/preflight/retrieval_utility_launch.json';assert not target.exists()
target.write_bytes(json.dumps(receipt,indent=2).encode('utf-8'))
print('RETRIEVAL_UTILITY_STARTED',json.dumps(dict(pid=receipt['pid'],gpu=receipt['gpu'])),flush=True)
