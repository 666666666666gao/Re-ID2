import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, copy_file, remote_python

receipt=json.loads((PROJECT/'results/preflight/shared_identity_2026_launch.json').read_text())
root=receipt['output']
code=f'''import hashlib,json
from pathlib import Path
root=Path({root!r});rows=[]
process=Path('/proc/{receipt['pid']}')
assert not process.exists() or (process/'stat').read_text().split()[2]=='Z','collect only after writer controller terminates'
for path in sorted(root.rglob('*')):
 if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log'):
  rows.append(dict(relative=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
print(json.dumps(rows))
'''
rows=json.loads(remote_python('2026',code))
local=PROJECT/'results/shared_identity_v11_trial_20261003'
for row in rows:
    destination=local/row['relative']
    copy_file('2026',root+'/'+row['relative'],destination)
    assert destination.stat().st_size==row['bytes'] and hashlib.sha256(destination.read_bytes()).hexdigest()==row['sha256']
(local/'intake.json').write_text(json.dumps(dict(source=root,files=rows),indent=2)+'\n',encoding='utf-8')
print('ACTUAL_TEXT_INTAKE',json.dumps(dict(files=len(rows),bytes=sum(row['bytes'] for row in rows))),flush=True)
