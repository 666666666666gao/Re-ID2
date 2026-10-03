import json
import sys
sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,remote_python
HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
inventory=json.loads((PROJECT/'results/preflight/historical_nonbest_weight_cleanup_plan.json').read_text())['hosts']['2027']
code=f'''import hashlib,json
from pathlib import Path
inventory={inventory!r};root=Path(inventory['root'])
print(json.dumps(dict(receipt_parent_exists=(root/'results').is_dir(),receipt_exists=(root/'results/historical_nonbest_weight_cleanup_20261003.json').exists(),candidates=[dict(path=row['path'],exists=Path(row['path']).exists()) for row in inventory['candidates']],keepers_unchanged=all(Path(row['path']).stat().st_size==row['bytes'] and hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256'] for row in inventory['retained']))))
'''
record=json.loads(remote_python('2027',code))
target=PROJECT/'results/preflight/historical_nonbest27_state_diagnostic.json';assert not target.exists();target.write_bytes(json.dumps(record,indent=2).encode('utf-8'))
print(json.dumps(record))
