from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,remote_python
HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
owned={host:{} for host in HOSTS}
for path in (PROJECT/'results').rglob('result.json'):
 data=json.loads(path.read_text(encoding='utf-8'))
 if data.get('status')!='COMPLETE' or data.get('epochs')!=50:continue
 args=data['arguments']
 if args['mode']!='train':continue
 for host,(root,_) in HOSTS.items():
  if args['output'].startswith(root+'/runs/'):
   proof=dict(result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),local_result=path.relative_to(PROJECT).as_posix(),variant=args['variant'],best=data['best'])
   previous=owned[host].get(args['output'])
   assert previous is None or previous['result_sha256']==proof['result_sha256']
   owned[host][args['output']]=proof
records={}
for host,known in owned.items():
 root,_=HOSTS[host]
 code=f'''import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path({root!r}).resolve();known={known!r};candidates=[];keepers=[];absent=[]
process_list=subprocess.check_output(['ps','-eo','pid=,args='],text=True)
for folder,proof in known.items():
 run=Path(folder);assert run.resolve().is_relative_to(root/'runs')
 assert hashlib.sha256((run/'result.json').read_bytes()).hexdigest()==proof['result_sha256']
 result=json.loads((run/'result.json').read_text());assert result['status']=='COMPLETE' and result['epochs']==50 and result['best']==proof['best']
 assert not any(str(run) in line for line in process_list.splitlines() if 'python' in line and ' -c ' not in line)
 best=run/'best.pth';assert best.is_file()
 keepers.append(dict(path=str(best),bytes=best.stat().st_size,sha256=hashlib.sha256(best.read_bytes()).hexdigest(),best=result['best']))
 path=run/'last.pth'
 if path.is_file():candidates.append(dict(path=str(path),run=str(run),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),result_sha256=proof['result_sha256']))
 else:absent.append(str(path))
print(json.dumps(dict(root=str(root),candidates=candidates,retained=keepers,already_absent=absent,total_bytes=sum(row['bytes'] for row in candidates),free_before_bytes=shutil.disk_usage(root).free)))
'''
 record=json.loads(remote_python(host,code));records[host]=record
 print('HISTORICAL_NONBEST_INVENTORY',json.dumps(dict(host=host,files=len(record['candidates']),bytes=record['total_bytes'],retained_best=len(record['retained']),already_absent=len(record['already_absent']))),flush=True)
target=PROJECT/'results/preflight/historical_nonbest_weight_cleanup_plan.json';assert not target.exists()
target.write_bytes(json.dumps(dict(status='EXACT_HISTORICAL_NONBEST_INVENTORY',observed_at=datetime.now().isoformat(timespec='seconds'),authorization='User requested only best weights; local actual COMPLETE50 result artifacts prove ownership, matched exact remote result, absent active run, best preserved',hosts=records),indent=2).encode('utf-8'))
