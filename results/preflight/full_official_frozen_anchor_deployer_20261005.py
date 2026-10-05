"""Deploy only the reviewed sources; no NN or controller is started here."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, OPTIONS, command, remote_python

pf = PROJECT / 'results/preflight'
training = json.loads((pf / 'full_official_frozen_identity_anchor_training_review_20261005.json').read_text(encoding='utf-8'))
execution = json.loads((pf / 'full_official_frozen_anchor_execution_review_20261005.json').read_text(encoding='utf-8'))
plan = json.loads((pf / 'full_official_frozen_anchor_runtime_plan_20261005.json').read_text(encoding='utf-8'))
for review in (training, execution):
    assert review['status'] == 'PASS' and not review['blocking_findings']
    for file, digest in review['sources_sha256'].items():
        assert hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() == digest
        ast.parse((PROJECT / file).read_text(encoding='utf-8'))
assert all(hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() == digest for file, digest in plan['protected_sources_sha256'].items())
proof = pf / 'full_official_frozen_anchor_deployment_20261005.json'
assert not proof.exists()
remote_root = '/data/gaob/Re-ID/DeMo-DualAxis'
files = {**training['sources_sha256'], **execution['sources_sha256']}
remote_files = {file: digest for file, digest in files.items() if not file.startswith('results/')}
for file in remote_files:
    command(['scp', *OPTIONS, str(PROJECT / file), '2026:' + remote_root + '/' + file])
observed = json.loads(remote_python('2026', f'''import ast,hashlib,json,shutil,subprocess
from pathlib import Path
root=Path({remote_root!r});files={remote_files!r};protected={plan['protected_sources_sha256']!r}
for file,digest in files.items():
 path=root/file
 assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
 ast.parse(path.read_text())
for file,digest in protected.items():assert hashlib.sha256((root/file).read_bytes()).hexdigest()==digest
resources=subprocess.check_output(['nvidia-smi','-i','2,3','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
rows=[[int(v.strip()) for v in line.split(',')] for line in resources.strip().splitlines()]
assert [v[0] for v in rows]==[2,3] and all(v[1]<500 for v in rows)
assert not (root/'runs/full_official_frozen_identity_anchor_m7_20261005').exists()
print(json.dumps(dict(remote_sources_exact=True,protected_original_sources_unchanged=True,gpus=rows,free_bytes=shutil.disk_usage(root).free,neural_execution=0)))
'''))
observed.update(status='REVIEWED_FROZEN_ANCHOR_SOURCES_DEPLOYED_NO_NEURAL_START', observed_at=datetime.now().isoformat(timespec='seconds'),
                sources_sha256=files, source_reviews=['full_official_frozen_identity_anchor_training_review_20261005.json',
                                                      'full_official_frozen_anchor_execution_review_20261005.json'],
                limits='Deployment/AST/resource observation only. No contracts, smoke, full50, neural stream or scientific result completed.')
proof.write_bytes((json.dumps(observed, indent=2) + '\n').encode('utf-8'))
print('FROZEN_ANCHOR_SOURCES_DEPLOYED_NO_NEURAL_START', observed['free_bytes'], flush=True)
