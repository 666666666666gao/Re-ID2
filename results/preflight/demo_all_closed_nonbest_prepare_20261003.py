from datetime import datetime
import base64
import json
import shlex
import zlib
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,remote_python
HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
classification=json.loads((PROJECT/'results/preflight/remaining_nonbest_classification.json').read_text())
new_last=HOSTS['2026'][0]+'/runs/axis_collaboration_v10_retrieval_utility_trial/development/MSVR310_axis_retrieval_utility_fullref_s42/last.pth'
assert new_last not in [row['path'] for row in classification['hosts']['2026']]
classification['hosts']['2026'].append(dict(path=new_last))
def prepare(item):
 host,rows=item
 root,_=HOSTS[host]
 code=f'''import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path({root!r});candidates=[];keepers={{}}
processes=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines()
for row in {rows!r}:
 path=Path(row['path']);assert path.resolve().is_relative_to((root/'runs').resolve())
 campaign=root/'runs'/path.relative_to(root/'runs').parts[0]
 assert not any((str(campaign) in line or str(campaign.resolve()) in line) and 'python' in line and ' -c ' not in line for line in processes)
 metadata=path.parent/('result.json' if path.name=='last.pth' else 'smoke.json')
 data=json.loads(metadata.read_text())
 if path.name=='last.pth':
  assert data['status']=='COMPLETE' and data['epochs']==50
  best=path.parent/'best.pth';assert best.is_file()
  keepers[str(best)]=dict(path=str(best),bytes=best.stat().st_size,sha256=hashlib.sha256(best.read_bytes()).hexdigest(),best=data['best'])
 else:assert path.name=='smoke.pth' and data['status']=='SMOKE_PASS'
 candidates.append(dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,metadata=str(metadata),metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),campaign=str(campaign),kind=path.name))
print(json.dumps(dict(root=str(root),candidates=candidates,retained_best=list(keepers.values()),deleted_bytes=sum(row['bytes'] for row in candidates),free_before_bytes=shutil.disk_usage(root).free)))
'''
 payload=base64.b64encode(zlib.compress(code.encode('utf-8'))).decode('ascii')
 record=json.loads(remote_python(host,"import base64,zlib\nexec(zlib.decompress(base64.b64decode("+repr(payload)+")).decode('utf-8'))"))
 print('ALL_CLOSED_NONBEST_EXACT_PLAN',json.dumps(dict(host=host,files=len(record['candidates']),bytes=record['deleted_bytes'],best=len(record['retained_best']))),flush=True)
 return host,record

with ThreadPoolExecutor(max_workers=3) as pool:
 records=dict(pool.map(prepare,classification['hosts'].items()))
target=PROJECT/'results/preflight/all_closed_nonbest_weight_cleanup_plan.json';assert not target.exists()
target.write_bytes(json.dumps(dict(status='EXACT_ALL_CLOSED_NONBEST_CLEANUP_PREPARED',observed_at=datetime.now().isoformat(timespec='seconds'),authorization='User requested keeping only best experiment weights; completed50 last and passed closed smoke artifacts under ourproject runs only; best preserved',hosts=records),indent=2).encode('utf-8'))
