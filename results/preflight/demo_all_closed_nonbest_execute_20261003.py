from datetime import datetime
import base64
import json
import shlex
import zlib
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,remote_python
HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
plan=json.loads((PROJECT/'results/preflight/all_closed_nonbest_weight_cleanup_plan.json').read_text())
assert plan['status']=='EXACT_ALL_CLOSED_NONBEST_CLEANUP_PREPARED'
records={}
for host,inventory in plan['hosts'].items():
 code=f'''import hashlib,json,shutil,subprocess,time
from pathlib import Path
inventory={inventory!r};root=Path(inventory['root'])
receipt=root/'results/all_closed_nonbest_weight_cleanup_20261003.json'
assert receipt.parent.is_dir() and not receipt.exists()
for row in inventory['retained_best']:
 path=Path(row['path']);assert path.name=='best.pth' and path.resolve().is_relative_to((root/'runs').resolve())
 assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
processes=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines()
for row in inventory['candidates']:
 path=Path(row['path']);campaign=Path(row['campaign'])
 assert path.resolve().is_relative_to((root/'runs').resolve()) and path.name in ('last.pth','smoke.pth')
 assert not any((str(campaign) in line or str(campaign.resolve()) in line) and 'python' in line and ' -c ' not in line for line in processes)
 metadata=Path(row['metadata']);assert hashlib.sha256(metadata.read_bytes()).hexdigest()==row['metadata_sha256']
 data=json.loads(metadata.read_text())
 assert (path.name=='last.pth' and data['status']=='COMPLETE' and data['epochs']==50) or (path.name=='smoke.pth' and data['status']=='SMOKE_PASS')
 assert path.stat().st_size==row['bytes'] and path.stat().st_mtime_ns==row['mtime_ns']
for row in inventory['candidates']:Path(row['path']).unlink()
assert all(not Path(row['path']).exists() for row in inventory['candidates'])
assert all(Path(row['path']).stat().st_size==row['bytes'] and hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'] for row in inventory['retained_best'])
record=dict(status='COMPLETE_AUTHORIZED_ALL_CLOSED_NONBEST_CLEANUP',deleted=inventory['candidates'],retained_best=inventory['retained_best'],deleted_bytes=inventory['deleted_bytes'],free_after_bytes=shutil.disk_usage(root).free,finished=time.time())
receipt.write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''
 payload=base64.b64encode(zlib.compress(code.encode('utf-8'))).decode('ascii')
 records[host]=json.loads(remote_python(host,"import base64,zlib\nexec(zlib.decompress(base64.b64decode("+repr(payload)+")).decode('utf-8'))"))
 print('ALL_CLOSED_NONBEST_ACTUALLY_DELETED',json.dumps(dict(host=host,files=len(records[host]['deleted']),bytes=records[host]['deleted_bytes'])),flush=True)
target=PROJECT/'results/preflight/all_closed_nonbest_weight_cleanup_receipt.json';assert not target.exists()
target.write_bytes(json.dumps(dict(status='COMPLETE_AUTHORIZED_ALL_CLOSED_NONBEST_CLEANUP',observed_at=datetime.now().isoformat(timespec='seconds'),hosts=records,total_deleted_bytes=sum(record['deleted_bytes'] for record in records.values())),indent=2).encode('utf-8'))
