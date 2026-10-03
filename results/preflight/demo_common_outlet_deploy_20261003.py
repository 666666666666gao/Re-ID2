from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

plan_path=PROJECT/'results/preflight/common_outlet_scale_plan.json'
plan=json.loads(plan_path.read_text())
review=json.loads((PROJECT/'results/preflight/common_outlet_scale_review.json').read_text())
assert review['status']=='PASS' and not review['blockers']
assert review['plan_sha256']==hashlib.sha256(plan_path.read_bytes()).hexdigest()
assert review['helper_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
assert review['checked_source_sha256']==plan['sources']
assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in plan['sources'].items())
parent=json.loads((PROJECT/'results/preflight/common_coordinate_2026_launch.json').read_text())
root,python=HOSTS['2026']
campaign=parent['output']
output=root+'/runs/common_outlet_scale_v12_frozen_20261003'
sources=plan['previous_sources']
guard=f'''import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path({root!r});campaign=Path({campaign!r})
assert json.loads((campaign/'controller_result.json').read_text())['status']=='COMPLETE'
process=Path('/proc/{parent['pid']}')
assert not process.exists() or (process/'stat').read_text().split()[2]=='Z'
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {sources!r}.items())
assert not Path({output!r}).exists() and not Path({output!r}+'.log').exists()
assert all(not (root/name).exists() for name in {list(plan['sources'])!r})
used={{int(line.split(',')[0]):int(line.split(',')[1]) for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()}}
assert all(used[index]<500 for index in (2,3)) and shutil.disk_usage(root).free>5000000000
print(json.dumps(dict(gpus=used,disk_free_bytes=shutil.disk_usage(root).free,selected_gpus=[2,3])))
'''
capacity=json.loads(remote_python('2026',guard))
for name in plan['sources']:
    command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+root+'/'+name])
sources=dict(sources,**plan['sources'])
argv=[python,'-u','launch_common_outlet_scale.py','--campaign',campaign,'--output',output,
      '--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt']
start=f'''import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path({root!r})
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {sources!r}.items())
used={{int(line.split(',')[0]):int(line.split(',')[1]) for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()}}
assert all(used[index]<500 for index in (2,3))
assert not Path({output!r}).exists() and not Path({output!r}+'.log').exists()
with Path({output!r}+'.log').open('x') as handle:
 child=subprocess.Popen({argv!r},cwd=root,start_new_session=True,stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
print(json.dumps(dict(host='2026',pid=child.pid,output={output!r},log={output!r}+'.log',campaign={campaign!r},command={argv!r},started=time.time(),source_sha256={sources!r})))
'''
receipt=json.loads(remote_python('2026',start))
receipt.update(capacity=capacity,observed_at=datetime.now().isoformat(timespec='seconds'),selected_gpus=[2,3])
target=PROJECT/'results/preflight/common_outlet_scale_2026_launch.json'
assert not target.exists()
target.write_text(json.dumps(receipt,indent=2)+'\n')
print('FROZEN_COMMON_OUTLET_STARTED',json.dumps(dict(pid=receipt['pid'],output=output,selected_gpus=[2,3])),flush=True)
