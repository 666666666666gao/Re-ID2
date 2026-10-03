"""Narrow terminal-storage and explicit host26 revision; stdlib only."""
import argparse
import ast
import copy
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
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_v6_projected26_compact_deploy_20261003.py')
FILES = ('projected_mass_axis_collaboration.py', 'run_projected_mass_experiment.py',
         'verify_projected_axis_mass.py', 'launch_projected_mass_trial.py')
CHECKS = []
previous = json.loads((TRACE / 'previous_host27_review_PASS41.json').read_text(encoding='utf-8'))
initial = json.loads((TRACE / 'initial_checks.json').read_text(encoding='utf-8'))
old_harness = (TRACE / 'review_static_mock.py').read_text(encoding='utf-8')
old_tree = ast.parse(old_harness)


def extract(name):
    node = next(x for x in old_tree.body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name == name)
    return ast.get_source_segment(old_harness, node)


for name in ('check', 'read', 'tree', 'dump', 'function', 'RunnerCanonical'):
    exec(compile(extract(name), 'reused_' + name, 'exec'), globals())

sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
helper_sha = hashlib.sha256(HELPER.read_bytes()).hexdigest()
check('compact26_model_and_tensor_verifier_bytes_unchanged',
      all(sources[name] == previous['checked_source_sha256'][name] for name in (FILES[0], FILES[2])),
      'Reuse prior forward/parameter/shape/normalization/gradient-path review without importing any neural model.')
check('compact26_controller_bytes_restore_original_reviewed_GPU1_controller',
      sources[FILES[3]] == initial['checked_source_sha256'][FILES[3]],
      'Exact full-file equality to the initially reviewed GPU1 controller and its eleven failure/success mocks; no gate change.')

runner_bytes = (ROOT / FILES[1]).read_bytes()
metadata_lines = [line for line in runner_bytes.splitlines(keepends=True) if b"info['terminal_checkpoint_storage']" in line]
assert len(metadata_lines) == 1
before_bytes = runner_bytes.replace(metadata_lines[0], b'').replace(
    b"torch.save({'model': model.state_dict(), 'scheduler': scheduler.state_dict(),",
    b"torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),")
check('compact26_runner_exact_only_metadata_and_last_optimizer_omission',
      hashlib.sha256(before_bytes).hexdigest() == previous['checked_source_sha256'][FILES[1]],
      'Reversing only the one metadata line and one terminal-dictionary optimizer entry recovers the exact prior reviewed runner bytes.')
runner_tree = tree(FILES[1])
runner_main = function(runner_tree, 'main')
old_main = function(tree('run_mass_experiment.py'), 'main')
current_with = next(x for x in runner_main.body if isinstance(x, ast.With))
old_with = next(x for x in old_main.body if isinstance(x, ast.With))
current_loop = next(x for x in current_with.body if isinstance(x, ast.For))
old_loop = next(x for x in old_with.body if isinstance(x, ast.For))
last_save = current_with.body[-1]
check('compact26_full50_training_evaluation_best_loop_identical',
      dump(current_loop) == dump(old_loop) and 'range(1, 51)' in ast.unparse(current_loop),
      'Entire50-epoch loop AST unchanged: loss/optimizer updates, sampling seeds, FP32 dev evaluation, earliest-max best save and all logs.')
check('compact26_last_serialization_runs_only_after_all50',
      current_with.body.index(last_save) > current_with.body.index(current_loop)
      and isinstance(last_save, ast.Expr) and isinstance(last_save.value, ast.Call)
      and ast.unparse(last_save.value.args[1]) == "out / 'last.pth'",
      'Terminal save is the statement after the completed full50 for-loop, before selected-best reload and COMPLETE outputs.')
payload = last_save.value.args[0]
keys = [x.value for x in payload.keys]
check('compact26_terminal_payload_model_scheduler_scaler_counters_only',
      keys == ['model', 'scheduler', 'scaler', 'epoch', 'steps', 'optimizer_steps']
      and payload.values[keys.index('epoch')].value == 50,
      'Omits only optimizer moments at the already completed run end; model, scheduler, scaler, epoch50 and attempt/effective-step counters remain.')
new_smoke = next(x for x in runner_main.body if isinstance(x, ast.If) and ast.unparse(x.test) == "args.mode == 'smoke'")
old_smoke = next(x for x in old_main.body if isinstance(x, ast.If) and ast.unparse(x.test) == "args.mode == 'smoke'")
check('compact26_native3AMP_full_optimizer_smoke_AST_identical', dump(new_smoke) == dump(old_smoke),
      'Unchanged complete model+optimizer save/restore, strict model load, all-gradients check, three effective steps and equal reloaded features.')
check('compact26_terminal_metadata_discloses_no_optimizer_resume',
      b'no Adam moment duplicate or optimizer resume from last' in metadata_lines[0],
      'Checkpoint is explicitly documented as model/scheduler/scaler/counters only; optimizer-resume support is not claimed.')

# Execute the actual terminal-save and post-save statements with simple sentinels.
# No model, optimizer or torch is imported; a call to optimizer.state_dict fails.
events, saved_payloads, records = [], [], []
metrics = {'mAP': 1.0, 'Rank-1': 2.0, 'Rank-5': 3.0, 'Rank-10': 4.0}


class Model:
    def state_dict(self):
        return {'weight': 'FINAL_EPOCH50_STATE'}

    def load_state_dict(self, value, strict):
        assert value == {'weight': 'SELECTED_BEST_STATE'} and strict is True
        events.append('strict_best_reload')


def save(value, target):
    events.append('terminal_save')
    assert target.name == 'last.pth'
    saved_payloads.append(value)


def evaluate(*args):
    events.append('best_dev_evaluate')
    return dict(metrics)


def write_json(path, value):
    records.append((path.name, value))


def forbid_optimizer_serialization():
    raise AssertionError('optimizer must not be serialized by the compact terminal path')


suffix = runner_main.body[runner_main.body.index(current_with) + 1:]
terminal_code = compile(ast.fix_missing_locations(ast.Module(body=[last_save] + suffix, type_ignores=[])), 'actual_compact_terminal_AST.py', 'exec')
namespace = {'torch': types.SimpleNamespace(save=save, load=lambda *a, **kw: {'weight': 'SELECTED_BEST_STATE'},
                 cuda=types.SimpleNamespace(max_memory_allocated=lambda: 123)),
             'model': Model(), 'optimizer': types.SimpleNamespace(state_dict=forbid_optimizer_serialization),
             'scheduler': types.SimpleNamespace(state_dict=lambda: {'epoch': 50}),
             'scaler': types.SimpleNamespace(state_dict=lambda: {'scale': 512}),
             'out': Path('MOCK_ONLY'), 'steps': 500, 'optimizer_steps': 497, 'info': {'terminal_checkpoint_storage': metadata_lines[0].decode()},
             'best': dict(metrics, epoch=40), 'evaluate': evaluate, 'val_loader': 'MOCK', 'queries': 'MOCK',
             'args': types.SimpleNamespace(dataset='MSVR310'), 'write_json': write_json,
             'time': types.SimpleNamespace(time=lambda: 1.0), 'json': json, 'print': lambda *a, **kw: None}
exec(terminal_code, namespace)
check('compact26_post50_save_reload_and_COMPLETE_stdlib_witness',
      events == ['terminal_save', 'strict_best_reload', 'best_dev_evaluate']
      and len(saved_payloads) == 1 and 'optimizer' not in saved_payloads[0]
      and [name for name, _ in records] == ['result.json', 'status.json']
      and all(row['status'] == 'COMPLETE' and row['epochs'] == 50 for _, row in records),
      'Executed actual post50 AST using sentinels: compact terminal saves, selected best strictly reloads, dev equality gate executes and both completion records remain COMPLETE.')

original = json.loads(read('results/preflight/axis_collaboration_v5_mass_launch.json'))['source_sha256']
check('compact26_original18_sources_unchanged', len(original) == 18 and all(
    hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in original.items()),
    'Original V5 training/frozen input source bytes untouched.')
controller = json.loads(read('results/axis_collaboration_v5_mass/controller_result.json'))
terminals = [json.loads(read('results/axis_collaboration_v5_mass/development/' + row['name'] + '/result.json')) for row in controller['runs']]
check('compact26_actual_V5_three_training_terminals_already_COMPLETE',
      controller['status'] == 'COMPLETE' and len(controller['runs']) == 3
      and all(row['exit_code'] == 0 for row in controller['runs'])
      and all(row['status'] == 'COMPLETE' and row['epochs'] == 50 for row in terminals),
      'Read existing result/exit records only. Their unchanged runner saves last before result COMPLETE, so no pending1.2GB V5 training terminal remains.')
guard27 = json.loads(read('results/preflight/axis_collaboration_v6_guard27_primary.json'))
check('compact26_prior27_guard_failure_retained_no_upload_or_output',
      guard27['exit_code'] == 1 and not any(guard27['details']['new_files'].values())
      and not guard27['details']['output_exists'] and int(guard27['details']['gpu0']) == 1304,
      'Primary historical failure explicitly shows GPU0 occupied and all4 sources/output absent; no retry or preemption inferred.')
probe = json.loads(read('results/preflight/axis_collaboration_v6_actual_disk_probe.json'))
sizes = {Path(row['path']).name: row['bytes'] for row in probe['checkpoints'] if 'MSVR310_' in row['path']}
compact_estimate = sizes['smoke.pth'] + 2 * sizes['best.pth']
pending_states_raw = 4 * 3125 * 5632 * 4
margin = 2600000000 - compact_estimate - pending_states_raw
check('compact26_budget_uses_measured_full_smoke_and_two_model_states',
      compact_estimate == 1995123480 and pending_states_raw == 281600000 and margin > 323000000,
      {'compact_trial_estimate_bytes': compact_estimate, 'pending_V5_RGB100_four_state_raw_feature_bytes': pending_states_raw,
       'remaining_margin_bytes': margin, 'minimum_free_bytes': 2600000000,
       'notes': 'Margin covers terminal metadata, feature archive metadata, V6 dev arrays and text outputs; V5 missing evaluator writes metrics/text, not additional feature archives. Actual disk availability must pass both runtime guards.'})
plan = json.loads(read('results/preflight/axis_collaboration_v6_projected_plan.json'))
check('compact26_plan_source_bytes_storage_limit_and_explicit_history',
      plan['source_sha256'] == sources and '2026 GPU1' in plan['proposed_gpu']
      and plan['budget_epochs'] == 50 and plan['resource_change']['minimum_free_bytes'] == 2600000000
      and len(plan['resource_history']) == 2 and 'optimizer' in plan['storage_only_change'],
      'Single factor method remains auxiliary taps; separate post-training storage choice and26->27 guard failure->26 selection are explicit.')
for path in [ROOT / FILES[1], ROOT / FILES[3], HELPER]:
    compile(path.read_text(encoding='utf-8'), str(path), 'exec')
check('compact26_changed_runner_launcher_helper_compile', True, 'Compilation only, no model imports.')

helper_source = extract('helper_case')
helper_source = helper_source.replace("stage = 'guard' if len(scripts) == 1 else 'start'",
    "stage = 'guard' if len(scripts) == 1 else 'start'\n            assert \"'-i','1'\" in code\n            assert '2600000000' in code")
helper_source = helper_source.replace('current_used = next(used)',
    "current_used = next(used)\n            current_free = disk_free[len(scripts) - 1] if isinstance(disk_free, tuple) else disk_free")
helper_source = helper_source.replace("'free_bytes': disk_free", "'free_bytes': current_free")
helper_source = helper_source.replace('types.SimpleNamespace(free=disk_free)', 'types.SimpleNamespace(free=current_free)')
helper_source = helper_source.replace("'generated_remote_'", "'compact26_generated_remote_'")
exec(compile(helper_source, 'compact26_stdlib_mock.py', 'exec'), globals())
helper_case('success')
helper_case('compact26_GPU1_busy_before_upload', first_used=1500, expected_uploads=0, expected_starts=0)
helper_case('compact26_disk_below2p6_before_upload', disk_free=2500000000, expected_uploads=0, expected_starts=0)
helper_case('compact26_GPU1_busy_after_upload', second_used=1500, expected_starts=0)
helper_case('compact26_disk_below2p6_after_upload', disk_free=(3000000000, 2500000000), expected_starts=0)

assert 'torch' not in sys.modules
result = {'status': 'PASS_SOURCE_AND_STDLIB_MOCKS', 'observed_at': datetime.now(timezone.utc).isoformat(),
          'checks_count': len(CHECKS), 'checks': CHECKS, 'checked_source_sha256': sources,
          'helper_sha256': helper_sha, 'helper_path': str(HELPER), 'no_NN_model_import': True,
          'real_SSH_calls': 0, 'real_GPU_calls': 0, 'real_optimizer_updates': 0,
          'previous_review_archive': 'previous_host27_review_PASS41.json', 'previous_review_checks_count': 41,
          'scope': 'Only terminal serialization/metadata, explicit GPU1/resource/scheduling/deployment changes; original model/verifier checks reused.'}
(TRACE / 'compact26_revision_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: result[key] for key in ('status', 'checks_count', 'helper_sha256', 'no_NN_model_import')}))
