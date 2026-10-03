from datetime import datetime
import json
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,remote_python
HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
plan=json.loads((PROJECT/'results/preflight/historical_nonbest_weight_cleanup_plan.json').read_text())
diagnostic=json.loads((PROJECT/'results/preflight/historical_nonbest27_state_diagnostic.json').read_text())
assert not diagnostic['receipt_parent_exists'] and not diagnostic['receipt_exists'] and diagnostic['keepers_unchanged']
assert all(not row['exists'] for row in diagnostic['candidates'])
inventory=plan['hosts']['2027']
code=f'''import hashlib,json,shutil,time
from pathlib import Path
inventory={inventory!r};root=Path(inventory['root']).resolve()
assert all(not Path(row['path']).exists() for row in inventory['candidates'])
assert all(Path(row['path']).stat().st_size==row['bytes'] and hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'] for row in inventory['retained'])
parent=root/'results';assert not parent.exists();parent.mkdir()
record=dict(status='COMPLETE_AUTHORIZED_HISTORICAL_NONBEST_CLEANUP',deleted=inventory['candidates'],retained_best=inventory['retained'],deleted_bytes=inventory['total_bytes'],free_after_bytes=shutil.disk_usage(root).free,observed=time.time(),receipt_reconstructed=True,original_command_exit0=False,reason='Deletion occurred after full original guards; final receipt write failed because results directory absent. Independent readonly probes confirm both last files absent and both best hashes unchanged. No second unlink executed.')
receipt=parent/'historical_nonbest_weight_cleanup_20261003.json';assert not receipt.exists();receipt.write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''
records={'2027':json.loads(remote_python('2027',code))}
for host in ('2025','2026'):
 root=plan['hosts'][host]['root']
 records[host]=json.loads(remote_python(host,f'from pathlib import Path\nprint((Path({root!r})/"results/historical_nonbest_weight_cleanup_20261003.json").read_text())'))
 assert records[host]['status']=='COMPLETE_AUTHORIZED_HISTORICAL_NONBEST_CLEANUP'
target=PROJECT/'results/preflight/historical_nonbest_weight_cleanup_receipt.json';assert not target.exists()
target.write_bytes(json.dumps(dict(status='COMPLETE_AUTHORIZED_HISTORICAL_NONBEST_CLEANUP',observed_at=datetime.now().isoformat(timespec='seconds'),hosts=records,
 total_deleted_bytes=sum(record['deleted_bytes'] for record in records.values()),interruption='Original executor omitted2027 mapping. Explicit27 continuation deleted its two candidates, then receipt parent was absent; independent read-only probes preserved. This recovery writes receipt only.'),indent=2).encode('utf-8'))
print('HISTORICAL_NONBEST_RECOVERY_COMPLETE',json.dumps(dict(files=sum(len(record['deleted']) for record in records.values()),bytes=sum(record['deleted_bytes'] for record in records.values()))),flush=True)
