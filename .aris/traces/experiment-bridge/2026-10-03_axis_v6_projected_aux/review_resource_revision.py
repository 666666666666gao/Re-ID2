"""Recheck only the explicit host27/GPU0/resource change; stdlib mocks only."""
import argparse
import ast
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest import mock

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_axis_v6_projected_aux'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_v6_projected27_deploy_20261003.py')
FILES = ('projected_mass_axis_collaboration.py', 'run_projected_mass_experiment.py',
         'verify_projected_axis_mass.py', 'launch_projected_mass_trial.py')
CHECKS = []
initial = json.loads((TRACE / 'initial_checks.json').read_text(encoding='utf-8'))
old_harness = (TRACE / 'review_static_mock.py').read_text(encoding='utf-8')
old_tree = ast.parse(old_harness)


def extract(name):
    node = next(x for x in old_tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    return ast.get_source_segment(old_harness, node)


for name in ('check', 'read', 'tree', 'dump', 'function'):
    exec(compile(extract(name), 'reused_' + name, 'exec'), globals())

sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
helper_sha = hashlib.sha256(HELPER.read_bytes()).hexdigest()
check('three_neural_sources_unchanged_reuse_completed_AST_review',
      all(sources[name] == initial['checked_source_sha256'][name] for name in FILES[:3]),
      'No repeat tensor/model execution or repeat source-equivalence work; all three exact bytes retain their earlier successful review.')

launcher = read(FILES[3])
old_launcher = launcher.replace('idle(0)', 'idle(1)').replace("CUDA_VISIBLE_DEVICES='0'", "CUDA_VISIBLE_DEVICES='1'").replace('gpu=0,command=argv', 'gpu=1,command=argv')
old_digest = hashlib.sha256(old_launcher.replace('\n', '\r\n').encode('utf-8')).hexdigest()
old_digest_lf = hashlib.sha256(old_launcher.encode('utf-8')).hexdigest()
check('launcher_only_three_physical_GPU_literal_changes',
      initial['checked_source_sha256'][FILES[3]] in (old_digest, old_digest_lf),
      'Only idle(1)->idle(0), CUDA_VISIBLE_DEVICES=1->0 and launch receipt GPU1->0 changed; all existing engineering gates remain identical.')
compile(launcher, str(ROOT / FILES[3]), 'exec')
compile(HELPER.read_text(encoding='utf-8'), str(HELPER), 'exec')
check('final_launcher_and_host27_helper_compile', True, 'Compilation only; no imported NN or remote operation.')

full_original = json.loads(read('results/preflight/axis_collaboration_v5_mass_launch.json'))['source_sha256']
check('all_original18_local_sources_still_unchanged', all(
    hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in full_original.items()),
    'Preserves ongoing V5 training and frozen evaluation inputs on the original host; reviewer writes no source files.')
original = {name: digest for name, digest in full_original.items()
            if name not in ('run_mass_experiment.py', 'verify_axis_mass.py', 'missing_development.py')}
profile = json.loads(read('results/preflight/axis_collaboration_v5_profile_launch.json'))
check('host27_protected15_match_existing_profile_sources', len(original) == 15 and all(
    profile['source_sha256'][name] == digest for name, digest in original.items()),
    'The 15 protected dependencies include mass_axis_collaboration.py already uploaded for completed V5 profiling. Old run_mass/verify_mass and irrelevant missing_development are not required, uploaded or overwritten.')
ledger = json.loads(read('.aris/compute/ssh2027.json'))
profile_result = json.loads(read('results/preflight/axis_collaboration_v5_profile_analysis.json'))
check('existing_host27_READY_real_witness_and_profile_receipts',
      ledger['state'] == 'READY' and ledger['documented_invocation']['state'] == 'EXECUTED_VERBATIM_REAL_PASS'
      and ledger['tier3']['real_tensor_contract']['status'] == 'PASS'
      and profile_result['status'] == 'PASS_ACTUAL_ALL15_COST_EVIDENCE' and profile_result['conditions'] == 15,
      'Read actual existing artifacts only: matching environment package ledger, MSVR GT split/CUDA/model witness and all15 completed profile cases.')
plan = json.loads(read('results/preflight/axis_collaboration_v6_projected_plan.json'))
check('plan_final_source_bytes_and_explicit_host27_scope',
      plan['source_sha256'] == sources and '2027 GPU0' in plan['proposed_gpu']
      and plan['budget_epochs'] == 50 and plan['resource_change']['minimum_free_bytes'] == 3200000000,
      'Explicit one-host choice, one MSVR seed42/B64/freshCLIP50; no automatic fallback or scope expansion.')
probe = json.loads(read('results/preflight/axis_collaboration_v6_actual_disk_probe.json'))
checkpoint_bytes = sum(row['bytes'] for row in probe['checkpoints'] if 'MSVR310_' in row['path'])
check('corrected_disk_budget_covers_measured_smoke_best_last', checkpoint_bytes == 2792932497 and 3200000000 - checkpoint_bytes > 400000000,
      {'actual_MSVR_smoke_best_last_bytes': checkpoint_bytes, 'minimum_free_bytes': 3200000000,
       'space_beyond_these_three_files_bytes': 3200000000 - checkpoint_bytes,
       'host26_pending_V5_outputs_preserved': True, 'host27_no_pending_own_job': 'Parent explicit scope; actual deployment rechecks current free space before upload and start.'})

controller_code = compile(ast.fix_missing_locations(ast.Module(body=[function(tree(FILES[3]), 'main')], type_ignores=[])), FILES[3], 'exec')
controller_source = extract('controller_case').replace("== '1'", "== '0'").replace('waits == [1] * expected_calls', 'waits == [0] * expected_calls')
exec(compile(controller_source, 'reused_controller_mock_GPU0.py', 'exec'), globals())
controller_case('final_GPU0_all_pass')


class HostMap(dict):
    def __init__(self, local_remote):
        super().__init__()
        self.local_remote = local_remote

    def __setitem__(self, host, value):
        assert host == '2027'
        assert value == ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
        super().__setitem__(host, (str(self.local_remote), 'MOCK_REMOTE_PYTHON'))


helper_source = extract('helper_case')
helper_source = helper_source.replace("'2026'", "'2027'").replace("'2026:'", "'2027:'")
helper_source = helper_source.replace("local, {'2027': (str(remote), 'MOCK_REMOTE_PYTHON')}, []", 'local, HostMap(remote), []')
helper_source = helper_source.replace("'/data/gaob/Re-ID/dataset'", "'/data/gb/Re-ID/dataset'")
helper_source = helper_source.replace("(local / 'results/preflight').mkdir(parents=True)",
    "(local / 'results/preflight').mkdir(parents=True)\n        (local / '.aris/compute').mkdir(parents=True)\n        mock_ledger = dict(ledger)\n        if label == 'READY_required_before_upload':\n            mock_ledger['state'] = 'NOT_READY'\n        (local / '.aris/compute/ssh2027.json').write_text(json.dumps(mock_ledger), encoding='utf-8')")
helper_source = helper_source.replace("stage = 'guard' if len(scripts) == 1 else 'start'", 
    "stage = 'guard' if len(scripts) == 1 else 'start'\n            assert \"'-i','0'\" in code\n            assert '3200000000' in code")
helper_source = helper_source.replace("'generated_remote_'", "'final_generated_remote_'")
helper_source = helper_source.replace("spawns.append({'argv': argv, 'cwd': str(kwargs['cwd']), 'detached': True})",
    "assert argv[argv.index('--pretrained') + 1] == '/data/gb/Re-ID/pretrained/ViT-B-16.pt'\n                spawns.append({'argv': argv, 'cwd': str(kwargs['cwd']), 'detached': True})")
helper_source = helper_source.replace("'old18_unchanged': True", "'protected15_unchanged': True")
exec(compile(helper_source, 'host27_stdlib_mock.py', 'exec'), globals())
helper_case('success')
helper_case('GPU_busy_before_upload', first_used=1500, expected_uploads=0, expected_starts=0)
helper_case('disk_below_corrected_gate_before_upload', disk_free=3_000_000_000, expected_uploads=0, expected_starts=0)
helper_case('GPU_busy_after_upload', second_used=1500, expected_starts=0)
helper_case('READY_required_before_upload', expected_uploads=0, expected_starts=0)

assert 'torch' not in sys.modules
result = {'status': 'PASS_SOURCE_AND_STDLIB_MOCKS', 'observed_at': datetime.now(timezone.utc).isoformat(),
          'checks_count': len(CHECKS), 'checks': CHECKS, 'checked_source_sha256': sources,
          'helper_sha256': helper_sha, 'helper_path': str(HELPER), 'no_NN_model_import': True,
          'real_SSH_calls': 0, 'real_GPU_calls': 0, 'real_optimizer_updates': 0,
          'scope': 'Only final host/resource/READY/dependency/source changes, plus actual controller GPU0 mock. Reuses unchanged earlier source and failure-gate checks.'}
(TRACE / 'resource_revision_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: result[key] for key in ('status', 'checks_count', 'helper_sha256', 'no_NN_model_import')}))
