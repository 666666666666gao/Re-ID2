import json
import sys
sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,remote_python
root,_=HOSTS['2026']
code=f'''import json
from pathlib import Path
root=Path({root!r})/'runs';records=[]
for path in root.glob('**/*_demo_s42/result.json'):
 value=json.loads(path.read_text())
 if value.get('status')=='COMPLETE' and value.get('epochs')==50 and (path.parent/'best.pth').is_file():
  records.append(dict(run=str(path.parent),dataset=value['arguments']['dataset'],best=value['best'],best_bytes=(path.parent/'best.pth').stat().st_size))
print(json.dumps(records))
'''
rows=json.loads(remote_python('2026',code))
(PROJECT/'results/preflight/availability_base_input_inventory.json').write_bytes(json.dumps(rows,indent=2).encode())
print(json.dumps(rows),flush=True)
