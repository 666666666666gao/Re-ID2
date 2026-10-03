import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, copy_file, remote_python

receipt=json.loads((PROJECT/'results/preflight/shared_identity_2026_launch.json').read_text())
root=receipt['output']
code=f'''import hashlib,json,tarfile
from pathlib import Path
root=Path({root!r});rows=[]
process=Path('/proc/{receipt['pid']}')
assert not process.exists() or (process/'stat').read_text().split()[2]=='Z','collect only after writer controller terminates'
for path in sorted(root.rglob('*')):
 if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log'):
  rows.append(dict(relative=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
archive=root.parent/(root.name+'_texts.tar.gz')
assert not archive.exists()
with tarfile.open(archive,'w:gz') as bundle:
 for row in rows:bundle.add(root/row['relative'],arcname=row['relative'])
print(json.dumps(dict(files=rows,archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
'''
inventory=json.loads(remote_python('2026',code))
rows=inventory['files']
local=PROJECT/'results/shared_identity_v11_trial_20261003'
archive=Path('C:/Users/gb/.codex_tmp/shared_identity_v11_terminal_texts_20261003.tar.gz')
assert not archive.exists()
copy_file('2026',inventory['archive'],archive)
assert archive.stat().st_size==inventory['archive_bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==inventory['archive_sha256']
with tarfile.open(archive) as bundle:
    bundle.extractall(local,filter='data')
for row in rows:
    destination=local/row['relative']
    assert destination.stat().st_size==row['bytes'] and hashlib.sha256(destination.read_bytes()).hexdigest()==row['sha256']
(local/'intake.json').write_text(json.dumps(dict(source=root,files=rows),indent=2)+'\n',encoding='utf-8')
print('ACTUAL_TEXT_INTAKE',json.dumps(dict(files=len(rows),bytes=sum(row['bytes'] for row in rows))),flush=True)
