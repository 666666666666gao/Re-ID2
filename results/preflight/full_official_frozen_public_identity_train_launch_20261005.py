"""Deploy M8 once after completed M7 comparison and real three-variant native gates."""
import hashlib
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, OPTIONS, command, remote_python
from results.preflight.full_official_frozen_public_identity_stream_bridge_20261005 import archive_session


PF = PROJECT / 'results/preflight'
REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT = REMOTE + '/runs/full_official_frozen_public_identity_m8_20261005'
M7 = REMOTE + '/runs/full_official_frozen_identity_anchor_m7_20261005'
PREFLIGHT = REMOTE + '/runs/full_official_frozen_public_identity_m8_preflight_20261005'
ANCHOR = REMOTE + '/runs/full_official_baselines_20261004/training/MSVR310_demo_shared_s42'


if __name__ == '__main__':
    plan = json.loads((PF / 'full_official_frozen_public_identity_m8_preparation_20261005.json').read_text(encoding='utf-8'))
    review = json.loads((PF / 'full_official_frozen_public_identity_m8_full_execution_review_20261005.json').read_text(encoding='utf-8'))
    files = plan['formal_execution_sources_sha256']
    assert review['status'] == 'PASS' and not review['blocking_findings']
    assert review['sources_sha256'] == files
    for file, digest in files.items():
        assert hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() == digest
    actual = json.loads((PF / 'full_official_frozen_public_identity_m8_actual_preflight_20261005.json').read_text(encoding='utf-8'))
    assert actual['status'] == 'ACTUAL_THREE_NATIVE_AND_NINE_SMOKE_UPDATES_PASS_FROZEN_M8_NOT_FULL50'
    assert actual['done']['actual_native_updates'] == 3 and actual['done']['actual_smoke_updates'] == 9
    assert actual['done']['amp_skipped_steps'] == 0
    analysis = json.loads((PROJECT / 'results/full_official_frozen_identity_anchor_m7_20261005/analysis.json').read_text(encoding='utf-8'))
    assert analysis['status'] == 'FIVE_FULL_OFFICIAL_ANCHOR_RUNS245_FROZEN_AND1176_STATE_CASES_ANALYZED'
    decision = json.loads((PF / 'full_official_frozen_public_identity_m8_training_decision_20261005.json').read_text(encoding='utf-8'))
    assert decision['status'] == 'PROCEED_M8_FROZEN_ACTUAL_PUBLIC_CE_MATCHED_THREE_CONTROL_TRIAL'
    assert decision['goal_complete'] is False
    proof = PF / 'full_official_frozen_public_identity_m8_training_session_20261005.json'
    assert not proof.exists()
    parents = review['directly_reused_sources_sha256']
    ready = json.loads(remote_python('2026', f'''import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,{REMOTE!r})
from gpu_thermal_execute import check_limits
project=Path({REMOTE!r});m7=Path({M7!r});pre=Path({PREFLIGHT!r})
assert not Path({ROOT!r}).exists()
assert all(hashlib.sha256((project/f).read_bytes()).hexdigest()==s for f,s in {parents!r}.items())
done=json.loads((m7/'controller_result.json').read_text())
assert done['status']=='COMPLETE' and len(done['runs'])==5
assert done['frozen_metric_cases']==245 and done['enhanced_state_metric_cases']==1176
assert done['all_frozen_and_state_cpu_audits_passed'] and done['paired_identity_and_partial_sampling_exact']
assert done['anchor_run_dir']=={ANCHOR!r}
preflight=json.loads((pre/'preflight_result.json').read_text())
assert preflight=={actual['done']!r}
assert not list((m7/'diagnosis').glob('*/q_*_g_*/raw.npz')) and not list((m7/'frozen49').glob('*/q_*_g_*.npz'))
resources={{gpu:check_limits(gpu) for gpu in (2,3)}}
assert all(s['memory_used_mib']<500 for s in resources.values())
print(json.dumps(resources))'''))
    remote_files = {file:digest for file,digest in files.items() if not file.startswith('results/')}
    for file in remote_files:
        command(['scp', *OPTIONS, str(PROJECT / file), '2026:' + REMOTE + '/' + file])
    deployed = json.loads(remote_python('2026', f'''from pathlib import Path
import ast,hashlib,json
root=Path({REMOTE!r});files={remote_files!r}
for f,s in files.items():
 assert hashlib.sha256((root/f).read_bytes()).hexdigest()==s
 ast.parse((root/f).read_text(),filename=f)
print(json.dumps(files))'''))
    assert deployed == remote_files
    print('M8_REVIEWED_FULL50_SOURCE_READY_GPU2_3', json.dumps(ready), flush=True)
    archive_session([PYTHON, '-u', 'launch_full_official_frozen_public_identity_trial.py',
        '--data-root', '/data/gaob/Re-ID/dataset', '--pretrained', '/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
        '--anchor-run-dir', ANCHOR, '--preflight-root', PREFLIGHT, '--m7-root', M7, '--output', ROOT],
        Path(ROOT), Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/full_official_frozen_public_identity_m8_20261005'), proof)
