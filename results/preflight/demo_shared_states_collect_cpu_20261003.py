import hashlib
import json
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, copy_file, remote_python

launch = json.loads((PROJECT / 'results/preflight/shared_states_2026_launch.json').read_text())
root = launch['output']
files = ('independent_cpu_audit.json', 'independent_cpu_audit.log', 'independent_cpu_audit_exit.json')
code = f'''import hashlib,json
from pathlib import Path
root=Path({root!r})
assert json.loads((root/'independent_cpu_audit_exit.json').read_text())['exit_code']==0
assert json.loads((root/'independent_cpu_audit.json').read_text())['status']=='PASS'
assert not list(root.rglob('*.pth')) and not list(root.rglob('*.pt'))
print(json.dumps(dict(files={{name:dict(bytes=(root/name).stat().st_size,sha256=hashlib.sha256((root/name).read_bytes()).hexdigest()) for name in {files!r}}},diagnostic_weights_created=0,raw_distance_archives={{str(path.relative_to(root)):path.stat().st_size for path in root.rglob('raw_distances.npz')}})))
'''
inventory = json.loads(remote_python('2026', code))
dest = PROJECT / 'results/shared_identity_v11_frozen_states_20261003'
for name, evidence in inventory['files'].items():
    path = dest / name
    assert not path.exists()
    copy_file('2026', root + '/' + name, path)
    assert path.stat().st_size == evidence['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == evidence['sha256']
(dest / 'cpu_intake.json').write_text(json.dumps(inventory, indent=2) + '\n')
print('CPU_AUDIT_TEXT_INTAKE', json.dumps(dict(files=len(files),diagnostic_weights_created=0,raw_distance_bytes=sum(inventory['raw_distance_archives'].values()))), flush=True)
