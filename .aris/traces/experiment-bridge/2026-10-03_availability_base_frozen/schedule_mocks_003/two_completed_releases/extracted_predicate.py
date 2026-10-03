import json
from pathlib import Path
root=Path('C:\\Users\\gb\\projects\\demo_dual_axis_20261002\\.aris\\traces\\experiment-bridge\\2026-10-03_availability_base_frozen\\schedule_mocks_003\\two_completed_releases');stages=[]
for name in ('RGBNT201_axis_mass_fullref_s42','MSVR310_axis_mass_fullref_s42'):
 for stage in ('smoke','full'):
  exit_path=root/name/(stage+'_exit.json')
  if exit_path.exists():
   record=json.loads(exit_path.read_text());assert record['exit_code']==0,(name,stage,record)
 path=root/name/'controller_result.json'
 if path.exists():
  assert json.loads(path.read_text())['status']=='PASS'
  record=json.loads((root/name/'full_exit.json').read_text());assert record['exit_code']==0,(name,record)
  stages.append(name)
print(json.dumps(dict(released=len(stages)==2)))
