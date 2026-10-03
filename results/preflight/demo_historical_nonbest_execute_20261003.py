from datetime import datetime
import json
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,remote_python
plan=json.loads((PROJECT/'results/preflight/historical_nonbest_weight_cleanup_plan.json').read_text())
assert plan['status']=='EXACT_HISTORICAL_NONBEST_INVENTORY'
records={}
for host,inventory in plan['hosts'].items():
 code=f'''import hashlib,json,shutil,subprocess,time
from pathlib import Path
inventory={inventory!r};root=Path(inventory['root']).resolve()
receipt=root/'results/historical_nonbest_weight_cleanup_20261003.json';assert not receipt.exists()
for row in inventory['retained']:
 path=Path(row['path']);assert path.resolve().is_relative_to(root/'runs') and path.name=='best.pth'
 assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
process_list=subprocess.check_output(['ps','-eo','pid=,args='],text=True)
for row in inventory['candidates']:
 path=Path(row['path']);run=Path(row['run'])
 assert path.resolve().is_relative_to(run.resolve()) and run.resolve().is_relative_to(root/'runs') and path.name=='last.pth'
 assert hashlib.sha256((run/'result.json').read_bytes()).hexdigest()==row['result_sha256']
 assert not any(str(run) in line for line in process_list.splitlines() if 'python' in line and ' -c ' not in line)
 assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
for row in inventory['candidates']:Path(row['path']).unlink()
assert all(not Path(row['path']).exists() for row in inventory['candidates'])
assert all(Path(row['path']).stat().st_size==row['bytes'] and hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'] for row in inventory['retained'])
record=dict(status='COMPLETE_AUTHORIZED_HISTORICAL_NONBEST_CLEANUP',deleted=inventory['candidates'],retained_best=inventory['retained'],deleted_bytes=inventory['total_bytes'],free_after_bytes=shutil.disk_usage(root).free,finished=time.time())
receipt.write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''
 record=json.loads(remote_python(host,code));records[host]=record
 print('HISTORICAL_NONBEST_ACTUALLY_DELETED',json.dumps(dict(host=host,files=len(record['deleted']),bytes=record['deleted_bytes'],retained_best=len(record['retained_best']))),flush=True)
target=PROJECT/'results/preflight/historical_nonbest_weight_cleanup_receipt.json';assert not target.exists()
target.write_bytes(json.dumps(dict(status='COMPLETE_AUTHORIZED_HISTORICAL_NONBEST_CLEANUP',observed_at=datetime.now().isoformat(timespec='seconds'),hosts=records,total_deleted_bytes=sum(record['deleted_bytes'] for record in records.values())),indent=2).encode('utf-8'))
