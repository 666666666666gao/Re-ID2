"""Re-run every official M6 condition through actual NN streaming and compare the old GT audit."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, OPTIONS, command, remote_python
from full_official_anchor_stream_bridge_20261005 import archive_session, REMOTE_PYTHON

pf = PROJECT / 'results/preflight'
review = json.loads((pf / 'full_official_frozen_anchor_execution_review_20261005.json').read_text(encoding='utf-8'))
assert review['status'] == 'PASS' and not review['blocking_findings']
for file, digest in review['sources_sha256'].items():
    assert hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() == digest
archive = json.loads((pf / 'full_official_public_outlet_identity_distances_local_archive_20261005.json').read_text(encoding='utf-8'))
assert archive['status'] == 'ALL343_PUBLIC_IDENTITY_DISTANCE_FILES_LOCAL_SHA256_VERIFIED_SERVER_COPIES_CLEARED'
name = 'MSVR310_public_identity_axis_shared_s42'
source = archive['source']
previous = '/data/gaob/Re-ID/DeMo-DualAxis/runs/full_official_condition_cpu_equivalence_20261005/frozen49'
root = '/data/gaob/Re-ID/DeMo-DualAxis/runs/full_official_neural_stream_equivalence_20261005'
local = Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/full_official_neural_stream_equivalence_20261005')
proof_path = pf / 'full_official_neural_stream_gate_20261005.json'
assert not proof_path.exists()
state = json.loads(remote_python('2026', f'''import json
from pathlib import Path
p=Path({previous!r});r=Path({root!r})
assert not r.exists() and not list(p.glob('q_*_g_*.npz'))
print(json.dumps(dict(previous_complete=json.loads((p/'result.json').read_text())['status']=='COMPLETE')))
'''))
assert state['previous_complete']
files = {relative: info for relative, info in archive['files'][name].items() if relative.startswith('frozen49/')}
assert len(files) == 49
for relative, info in files.items():
    path = Path(archive['local_root']) / relative
    assert path.stat().st_size == info['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256']
    command(['scp', *OPTIONS, str(path), '2026:' + previous + '/' + path.name])
remote_python('2026', f'''import hashlib,json
from pathlib import Path
p=Path({previous!r});files={files!r}
assert len(list(p.glob('q_*_g_*.npz')))==49
for relative,info in files.items():
 path=p/Path(relative).name
 assert path.stat().st_size==info['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
print('RESTORED49_FROZEN_BANK_EXACT')
''')
argv = ['env', 'CUDA_VISIBLE_DEVICES=2', 'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
        REMOTE_PYTHON, '-u', 'diagnose_full_official_public_identity_stream.py', '--run-dir', source + '/training/' + name,
        '--previous-frozen', previous, '--output', root]
session = archive_session(argv, Path(root), local, pf / 'full_official_neural_stream_session_20261005.json', 'gate')
result = json.loads(remote_python('2026', f'''import hashlib,json
from pathlib import Path
root=Path({root!r});old=Path({source!r})/'diagnosis'/{name!r};previous=Path({previous!r})
actual=json.loads((root/'independent_cpu_audit.json').read_text())
expected=json.loads((old/'independent_cpu_audit.json').read_text())
assert actual['status']==expected['status']=='PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT'
assert {{k:v for k,v in actual.items() if k!='limits'}}=={{k:v for k,v in expected.items() if k!='limits'}}
reported=json.loads((root/'result.json').read_text());original=json.loads((old/'result.json').read_text())
for key in ('measurements','contributions','model_arguments','selected_epoch','normal_feature_max_error','state_tensor_versions_unchanged'):
 assert reported[key]==original[key]
stream=json.loads((root/'stream_archive.json').read_text())
assert stream['status']=='ALL49_CONDITIONS_AUDITED_ARCHIVED_AND_SERVER_RAW_CLEARED' and len(stream['conditions'])==49
assert not list(root.glob('q_*_g_*/raw.npz'))
files={files!r}
for relative,info in files.items():
 path=previous/Path(relative).name
 assert path.stat().st_size==info['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
 assert path.resolve().is_relative_to(previous.resolve());path.unlink()
assert not list(previous.glob('q_*_g_*.npz'))
print(json.dumps(dict(status='PASS_FULL49_NEURAL_STREAM_EQUALS_COMPLETED_M6_GT_AUDIT',conditions=49,cases=294,
 neural_inference=True,optimizer_updates=0,full_split_counts=dict(train=1032,query=591,gallery=1055),
 initial_all49_bank_retained_for_exact_full11_comparison=True,all49_gt_audits_equal=True,
 all49_calibration_and_metrics_equal=True,max_live_enhanced_raw_files=1,peak_live_enhanced_raw_bytes=stream['peak_live_enhanced_raw_bytes'],
 all49_raw_locally_size_sha_verified=True,all_restored_frozen_and_new_enhanced_remote_raw_cleared=True)))
'''))
result.update(verified_at=datetime.now().isoformat(timespec='seconds'), local_raw_archive=str(local),
              remote_root=root, source_review='results/preflight/full_official_frozen_anchor_execution_review_20261005.json',
              limits='Actual complete MSVR M6 NN inference/all49 equality gate. No stage2 training, no RGBNT100 performance or RGBNT100 packed storage measurement.')
proof_path.write_bytes((json.dumps(result, indent=2) + '\n').encode('utf-8'))
print('ALL49_NEURAL_STREAM_GATE_PASS', result['peak_live_enhanced_raw_bytes'], flush=True)
