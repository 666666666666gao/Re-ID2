from datetime import datetime
import json
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,remote_python

plan=json.loads((PROJECT/'results/preflight/completed_trial_weight_cleanup_plan.json').read_text())
assert plan['status']=='EXACT_NONBEST_CLEANUP_PREPARED' and len(plan['candidates'])==6
code=f'''import hashlib,json,shutil,subprocess,time
from pathlib import Path
plan={plan!r};root=Path(plan['root']).resolve()
receipt=root/'results/completed_trial_nonbest_weight_cleanup_20261003.json';assert not receipt.exists()
for row in plan['retained']:
 path=Path(row['path']);assert path.resolve().is_relative_to(root/'runs') and path.name=='best.pth'
 assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
process_list=subprocess.check_output(['ps','-eo','pid=,args='],text=True)
for row in plan['candidates']:
 path=Path(row['path']);folder=root/'runs'/row['campaign']
 assert path.resolve().is_relative_to(folder) and path.name in ('smoke.pth','last.pth')
 assert json.loads((folder/'controller_result.json').read_text())['status']=='COMPLETE'
 assert not any(str(folder) in line for line in process_list.splitlines() if 'python' in line and ' -c ' not in line)
 assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
for row in plan['candidates']:Path(row['path']).unlink()
assert all(not Path(row['path']).exists() for row in plan['candidates'])
assert all(Path(row['path']).stat().st_size==row['bytes'] and hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'] for row in plan['retained'])
record=dict(status='COMPLETE_AUTHORIZED_NONBEST_WEIGHT_CLEANUP',deleted=plan['candidates'],retained_best=plan['retained'],deleted_bytes=plan['total_bytes'],free_before_bytes=plan['free_before_bytes'],free_after_bytes=shutil.disk_usage(root).free,finished=time.time(),scope='Three completed own MSVR trials, smoke/last only; datasets, public CLIP, all best checkpoints, metrics and source unchanged')
receipt.write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''
record=json.loads(remote_python('2026',code))
record['observed_at']=datetime.now().isoformat(timespec='seconds')
target=PROJECT/'results/preflight/completed_trial_weight_cleanup_receipt.json';assert not target.exists()
target.write_bytes(json.dumps(record,indent=2).encode('utf-8'))
print('ACTUAL_NONBEST_WEIGHTS_DELETED',json.dumps(dict(files=len(record['deleted']),bytes=record['deleted_bytes'],free_after=record['free_after_bytes'])),flush=True)
