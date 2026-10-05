"""Deploy only reviewed M8 preflight sources after M7 releases its final GPU3 neural job."""
import hashlib
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, OPTIONS, command, remote_python
from results.preflight.full_official_anchor_stream_bridge_20261005 import archive_session


PF = PROJECT / 'results/preflight'
REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT = REMOTE + '/runs/full_official_frozen_public_identity_m8_preflight_20261005'
M7 = REMOTE + '/runs/full_official_frozen_identity_anchor_m7_20261005'
ANCHOR = REMOTE + '/runs/full_official_baselines_20261004/training/MSVR310_demo_shared_s42'


if __name__ == '__main__':
    plan = json.loads((PF / 'full_official_frozen_public_identity_m8_preparation_20261005.json').read_text(encoding='utf-8'))
    reviewed = json.loads((PF / 'full_official_frozen_public_identity_m8_preflight_execution_review_20261005.json').read_text(encoding='utf-8'))
    assert reviewed['status'] == 'PASS'
    files = plan['preflight_execution_sources_sha256']
    assert reviewed['sources_sha256'] == files
    for file, digest in files.items():
        assert hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() == digest
    proof = PF / 'full_official_frozen_public_identity_m8_native_session_20261005.json'
    assert not proof.exists()
    parents = plan['directly_reused_source_sha256']
    ready = json.loads(remote_python('2026', f'''import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,{REMOTE!r})
from gpu_thermal_execute import check_limits
project=Path({REMOTE!r});m7=Path({M7!r})
assert not Path({ROOT!r}).exists()
assert all(hashlib.sha256((project/f).read_bytes()).hexdigest()==s for f,s in {parents!r}.items())
previous=json.loads((m7/'diagnosis/MSVR310_anchor_twins_shared_fixed_s42_exit.json').read_text())
launch=json.loads((m7/'diagnosis/MSVR310_anchor_twins_shared_fixed_s42_launch.json').read_text())
assert previous['exit_code']==0 and previous['gpu']==3 and not Path('/proc/'+str(launch['pid'])).exists()
capacity=check_limits(3);assert capacity['memory_used_mib']<500
print(json.dumps(dict(previous_neural_exit=previous,capacity=capacity)))'''))
    remote_files = {f:s for f,s in files.items() if not f.startswith('results/')}
    for file in remote_files:
        command(['scp', *OPTIONS, str(PROJECT / file), '2026:' + REMOTE + '/' + file])
    deployed = json.loads(remote_python('2026', f'''import ast,hashlib,json
from pathlib import Path
root=Path({REMOTE!r});files={remote_files!r}
for f,s in files.items():
 assert hashlib.sha256((root/f).read_bytes()).hexdigest()==s
 ast.parse((root/f).read_text(),filename=f)
print(json.dumps(files))'''))
    assert deployed == remote_files
    print('M8_REVIEWED_NATIVE_PREFLIGHT_SOURCE_READY_GPU3', json.dumps(ready), flush=True)
    archive_session([PYTHON, '-u', 'launch_frozen_public_identity_preflight.py',
        '--data-root', '/data/gaob/Re-ID/dataset', '--pretrained', '/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
        '--anchor-run-dir', ANCHOR, '--m7-root', M7, '--output', ROOT], Path(ROOT),
        Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/full_official_frozen_public_identity_m8_preflight_20261005'),
        proof, 'preflight')
