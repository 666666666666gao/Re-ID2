import ast
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,remote_python
HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
plan=json.loads((PROJECT/'results/preflight/historical_nonbest_weight_cleanup_plan.json').read_text())
records={}
for host in ('2025','2026'):
 root=plan['hosts'][host]['root']
 records[host]=json.loads(remote_python(host,f'import json\nfrom pathlib import Path\nprint((Path({root!r})/"results/historical_nonbest_weight_cleanup_20261003.json").read_text())'))
 assert records[host]['status']=='COMPLETE_AUTHORIZED_HISTORICAL_NONBEST_CLEANUP'
tree=ast.parse(Path('C:/Users/gb/.codex_tmp/demo_historical_nonbest_execute_20261003.py').read_text(encoding='utf-8'))
loop=next(node for node in tree.body if isinstance(node,ast.For))
assignment=loop.body[0];assert isinstance(assignment,ast.Assign) and assignment.targets[0].id=='code'
code=eval(compile(ast.Expression(body=assignment.value),'<original27cleanup>','eval'),{'inventory':plan['hosts']['2027']})
records['2027']=json.loads(remote_python('2027',code))
assert records['2027']['status']=='COMPLETE_AUTHORIZED_HISTORICAL_NONBEST_CLEANUP'
target=PROJECT/'results/preflight/historical_nonbest_weight_cleanup_receipt.json';assert not target.exists()
target.write_bytes(json.dumps(dict(status='COMPLETE_AUTHORIZED_HISTORICAL_NONBEST_CLEANUP',observed_at=datetime.now().isoformat(timespec='seconds'),hosts=records,
 total_deleted_bytes=sum(record['deleted_bytes'] for record in records.values()),interruption='First executor omitted2027 HOSTS mapping, stopped after actual2025/2026 receipts; explicit27 continuation only, no repeated deletions'),indent=2).encode('utf-8'))
print('HISTORICAL_NONBEST_ALL_HOSTS_COMPLETE',json.dumps(dict(files=sum(len(record['deleted']) for record in records.values()),bytes=sum(record['deleted_bytes'] for record in records.values()))),flush=True)
