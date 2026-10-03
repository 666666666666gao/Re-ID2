"""Source and standard-library mocks only. Never import torch or run SSH/GPU."""
import ast
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
from unittest import mock

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_axis_v6_projected_aux'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_v6_projected26_deploy_20261003.py')
FILES = ('projected_mass_axis_collaboration.py', 'run_projected_mass_experiment.py',
         'verify_projected_axis_mass.py', 'launch_projected_mass_trial.py')
CHECKS = []


def check(name, condition, detail):
    assert condition, (name, detail)
    CHECKS.append({'check': name, 'status': 'PASS', 'detail': detail})


def read(name):
    return (ROOT / name).read_text(encoding='utf-8')


def tree(name):
    return ast.parse(read(name), filename=name)


def dump(node):
    return ast.dump(node, include_attributes=False)


def function(node, name):
    return next(x for x in node.body if isinstance(x, ast.FunctionDef) and x.name == name)


def assignment_name(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
helper_sha = hashlib.sha256(HELPER.read_bytes()).hexdigest()
for name in FILES:
    compile(read(name), str(ROOT / name), 'exec')
compile(HELPER.read_text(encoding='utf-8'), str(HELPER), 'exec')
check('all_new_sources_and_private_helper_compile', True, list(FILES) + [str(HELPER)])
original = json.loads(read('results/preflight/axis_collaboration_v5_mass_launch.json'))['source_sha256']
check('original_18_source_bytes_match_V5_launch', len(original) == 18 and all(
    hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in original.items()),
    'Read-only comparison against the existing V5 launch manifest; no hash framework added.')

v5class = next(x for x in tree('mass_axis_collaboration.py').body if isinstance(x, ast.ClassDef))
v6class = next(x for x in tree(FILES[0]).body if isinstance(x, ast.ClassDef))
check('V6_inherits_V5_with_no_initializer_or_extra_modules',
      [x.name for x in v6class.body if isinstance(x, ast.FunctionDef)] == ['forward']
      and len(v6class.bases) == 1 and v6class.bases[0].id == v5class.name,
      'Identical inherited parameter initialization, heads, state keys, scales, fuse and controlled_states.')
oldforward, newforward = function(v5class, 'forward'), function(v6class, 'forward')
restored = copy.deepcopy(newforward)
taps = [copy.deepcopy(x) for x in oldforward.body if assignment_name(x) in ('standalone_m', 'standalone_f')]
tap_with = next(x for x in restored.body if isinstance(x, ast.With) and len(x.body) == 2
                and [assignment_name(y) for y in x.body] == ['standalone_m', 'standalone_f'])
at = restored.body.index(tap_with)
restored.body[at:at + 1] = taps
check('entire_forward_AST_equal_except_two_auxiliary_taps', dump(restored) == dump(oldforward),
      'All base, conditions, route, gates, 00/10/01/11, contribution target/loss, output pairing and eval return are unchanged.')
tap_source = ast.unparse(tap_with)
check('projected_taps_use_FP32_normalization_then_512D_heads',
      "torch.autocast('cuda', enabled=False)" in tap_source
      and 'F.normalize(self.modality_projection(self.modality_expert.pool(m0, eligible).float()), dim=-1).mean(1)' in tap_source
      and 'F.normalize(self.frequency_projection(self.frequency_expert.pool(f0, available, self.structured).float()), dim=1)' in tap_source,
      'M preserves seven independent relation vectors through PM then normalizes each and averages; F normalizes PF pooled independent evidence.')


class RunnerCanonical(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == 'ProjectedMassAxisCollaborationDeMo':
            node.id = 'MassAxisCollaborationDeMo'
        return node

    def visit_Constant(self, node):
        if node.value == 'axis_mass_projected_fullref':
            node.value = 'axis_mass_fullref'
        return node

    def visit_ImportFrom(self, node):
        if node.module == 'projected_mass_axis_collaboration':
            node.module = 'mass_axis_collaboration'
            node.names[0].name = 'MassAxisCollaborationDeMo'
        return node

    def visit_Assign(self, node):
        for target in node.targets:
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id == 'info' and isinstance(target.slice, ast.Constant)
                    and target.slice.value in ('method_revision', 'auxiliary_identity_taps',
                                              'unchanged_V4_factors', 'unchanged_V5_factors')):
                return None
        return self.generic_visit(node)


oldrunner, newrunner = tree('run_mass_experiment.py'), tree(FILES[1])
for name in ('configuration', 'write_json', 'evaluate', 'step'):
    check('V5_runner_' + name + '_AST_identical', dump(function(oldrunner, name)) == dump(function(newrunner, name)),
          'Exact full-function AST equality, not a selected-line check.')
check('V6_runner_only_class_variant_and_metadata_changes',
      dump(RunnerCanonical().visit(oldrunner)) == dump(RunnerCanonical().visit(newrunner)),
      'Entire runner equal after normalizing its new class/import, variant string and descriptive run metadata.')

newverify = tree(FILES[2])
oldverify = tree('verify_axis_mass.py')


class VerifyCanonical(RunnerCanonical):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if node.value == 'PROJECTED_MASS_AXIS_TENSOR_CONTRACT_PASS':
            node.value = 'MASS_AXIS_TENSOR_CONTRACT_PASS'
        return super().visit_Constant(node)

    def visit_Assign(self, node):
        if assignment_name(node) in ('auxiliary_loss', 'auxiliary_gradients'):
            return None
        return self.generic_visit(node)

    def visit_Assert(self, node):
        if any(isinstance(x, ast.Name) and x.id == 'auxiliary_gradients' for x in ast.walk(node)):
            return None
        return self.generic_visit(node)

    def visit_Dict(self, node):
        pairs = [(key, value) for key, value in zip(node.keys, node.values)
                 if not (isinstance(key, ast.Constant) and key.value in
                         ('projected_auxiliary_CE_gradient_abs_sum_PM_PF', 'auxiliary_tap_only_changed_same_parameter_heads'))]
        node.keys, node.values = [x[0] for x in pairs], [x[1] for x in pairs]
        return self.generic_visit(node)


check('V5_tensor_contract_retained_with_only_aux_CE_probe_added',
      dump(VerifyCanonical().visit(oldverify)) == dump(VerifyCanonical().visit(newverify)),
      'Retains actual8 dev, six masks, initial V4 state/capacity equality, uniform fusion, routing gradient, closed-expert isolation and semantic B64/zero-update checks.')
verify_source = read(FILES[2])
check('auxiliary_gradient_probe_isolated_from_fused_loss',
      'F.cross_entropy(output[4].float(), labels.cuda()) + F.cross_entropy(output[6].float(), labels.cuda())' in verify_source
      and 'torch.autograd.grad(auxiliary_loss, (model.modality_projection[-1].weight, model.frequency_projection[-1].weight))' in verify_source
      and 'torch.isfinite(value).all() and value.abs().sum() > 0 for value in auxiliary_gradients' in verify_source,
      'CE of the two group heads alone must connect to both final projection weights with finite nonzero derivatives; this does not claim efficacy.')

runner_source = read(FILES[1])
data_source = read('experiment_data.py')
eval_source = read('utils/reid_evaluation.py')
check('GT_identity_dev_only_and_scene_filter',
      "'MSVR310/bounding_box_train'" in data_source and "set(labels).isdisjoint(dev_ids)" in data_source
      and "selector = scenes if dataset == 'MSVR310' else cams" in runner_source
      and 'gallery_ids[order] == query_ids[query_index]' in eval_source,
      'Actual dataset identity/scene ground truth; fixed identity-held-out training split; no model teacher labels or official test access.')
check('fixed50_earliest_dev_mAP_and_strict_best_reload',
      'for epoch in range(1, 51)' in runner_source and "if metrics['mAP'] > best['mAP']" in runner_source
      and "'best.pth', map_location='cuda', weights_only=True), strict=True" in runner_source,
      'Fixed 50 epochs, no early stopping, earliest tie retention and same FP32 evaluation after selected-checkpoint reload.')
check('native_smoke_three_effective_steps_all_gradients_reload',
      "assert updated, 'smoke optimizer step skipped'" in runner_source
      and 'if steps == 3:' in runner_source and 'assert all(gradients.values())' in runner_source
      and "optimizer.load_state_dict(saved['optimizer'])" in runner_source and 'assert torch.equal(before, after)' in runner_source,
      'Source contains three effective native AMP updates, every trainable tensor finite/nonzero and strict model/optimizer load plus equal eval output.')

# Execute only the controller main function with standard-library dependencies.
launch_tree = tree(FILES[3])
controller_code = compile(ast.fix_missing_locations(ast.Module(body=[function(launch_tree, 'main')], type_ignores=[])), FILES[3], 'exec')
import argparse


def controller_case(label, overrides=None, failure_stage=None, expected_calls=3, expected_complete=True):
    overrides = overrides or {}
    with tempfile.TemporaryDirectory(prefix='controller_' + label + '_', dir=TRACE) as folder:
        case_root = Path(folder).resolve()
        assert case_root.is_relative_to(TRACE.resolve())
        output = case_root / 'output'
        calls, waits, gate_at_train = [], [], []

        def write_json(path, value):
            Path(path).write_text(json.dumps(value), encoding='utf-8')

        class Child:
            pid = 123456

            def __init__(self, argv, stdout, stderr, env):
                self.stage = ('tensor', 'smoke', 'development')[len(calls)]
                calls.append({'stage': self.stage, 'argv': argv, 'env_gpu': env['CUDA_VISIBLE_DEVICES']})
                assert env['CUDA_VISIBLE_DEVICES'] == '1'
                assert argv[argv.index('--dataset') + 1] == 'MSVR310'
                assert stdout.mode == 'x'
                stdout.write('MOCK ONLY ' + self.stage + '\n')
                target = Path(argv[argv.index('--output') + 1])
                if self.stage == 'tensor':
                    result = {'status': 'PASS', 'optimizer_updates': 0,
                              'projected_auxiliary_CE_gradient_abs_sum_PM_PF': [1.0, 2.0]}
                elif self.stage == 'smoke':
                    target.mkdir()
                    target = target / 'smoke.json'
                    result = {'status': 'SMOKE_PASS', 'steps': 3, 'gradients': {'example': True}, 'strict_reload_equal': True}
                else:
                    gate = json.loads((output / 'preflight/controller_result.json').read_text())
                    gate_at_train.append(gate['status'] == 'PASS' and len(gate['checks']) == 2)
                    assert '--mode' in argv and argv[argv.index('--mode') + 1] == 'train'
                    assert '--seed' in argv and argv[argv.index('--seed') + 1] == '42'
                    assert argv[argv.index('--pretrained') + 1] == 'PUBLIC_CLIP'
                    target.mkdir()
                    target = target / 'result.json'
                    result = {'status': 'COMPLETE', 'epochs': 50}
                result.update(overrides.get(self.stage, {}))
                write_json(target, result)

            def wait(self):
                return 7 if self.stage == failure_stage else 0

        namespace = {'argparse': argparse, 'json': json, 'os': os, 'Path': Path,
                     'subprocess': types.SimpleNamespace(Popen=Child, STDOUT=-2),
                     'sys': types.SimpleNamespace(executable='MOCK_PYTHON'),
                     'time': types.SimpleNamespace(time=lambda: 100.0), 'write_json': write_json,
                     'idle': lambda gpu: waits.append(gpu)}
        exec(controller_code, namespace)
        failed = False
        with mock.patch.object(sys, 'argv', ['launch_projected_mass_trial.py', '--data-root', 'ACTUAL_TRAIN_ROOT',
                                            '--pretrained', 'PUBLIC_CLIP', '--output', str(output)]):
            try:
                namespace['main']()
            except AssertionError:
                failed = True
        complete = (output / 'controller_result.json').exists()
        exits = sorted(output.rglob('*_exit.json'))
        check('controller_mock_' + label,
              len(calls) == expected_calls and failed == (not expected_complete)
              and complete == expected_complete and len(exits) == expected_calls
              and waits == [1] * expected_calls and (not expected_complete or gate_at_train == [True]),
              {'spawned_stages': [x['stage'] for x in calls], 'exit_receipts_retained': len(exits),
               'gate_read_before_train': gate_at_train, 'controller_complete': complete,
               'mock_only_no_process_created': True})


controller_case('all_pass')
controller_case('tensor_process_failure', failure_stage='tensor', expected_calls=1, expected_complete=False)
controller_case('tensor_status_failure', {'tensor': {'status': 'FAIL'}}, expected_calls=1, expected_complete=False)
controller_case('tensor_optimizer_nonzero', {'tensor': {'optimizer_updates': 1}}, expected_calls=1, expected_complete=False)
controller_case('tensor_aux_gradient_zero', {'tensor': {'projected_auxiliary_CE_gradient_abs_sum_PM_PF': [0., 2.]}}, expected_calls=1, expected_complete=False)
controller_case('smoke_process_failure', failure_stage='smoke', expected_calls=2, expected_complete=False)
controller_case('smoke_steps_wrong', {'smoke': {'steps': 2}}, expected_calls=2, expected_complete=False)
controller_case('smoke_gradient_failure', {'smoke': {'gradients': {'example': False}}}, expected_calls=2, expected_complete=False)
controller_case('smoke_reload_failure', {'smoke': {'strict_reload_equal': False}}, expected_calls=2, expected_complete=False)
controller_case('development_process_failure', failure_stage='development', expected_calls=3, expected_complete=False)
controller_case('development_incomplete', {'development': {'epochs': 49}}, expected_calls=3, expected_complete=False)

idle_code = compile(ast.fix_missing_locations(ast.Module(body=[function(tree('launch_axis_scaled.py'), 'idle')], type_ignores=[])), 'idle_source.py', 'exec')
gpu_values, sleeps = iter(['6200', '0']), []
namespace = {'subprocess': types.SimpleNamespace(check_output=lambda *a, **kw: next(gpu_values)),
             'time': types.SimpleNamespace(sleep=lambda seconds: sleeps.append(seconds))}
exec(idle_code, namespace)
namespace['idle'](1)
check('existing_GPU_idle_poll_240_seconds', sleeps == [240], 'Busy observation waits 240 seconds and resumes only on memory.used<500; mock does not sleep.')

# Import the actual controller under the native script directory lookup while
# stubbing only the heavy imported modules. No torch, data or NN is imported.
import_code = compile(read(FILES[3]), str(ROOT / FILES[3]), 'exec')
fake_launch = types.ModuleType('launch_axis_scaled')
fake_launch.idle = lambda gpu: None
fake_runner = types.ModuleType('run_experiment')
fake_runner.write_json = lambda *args: None
with mock.patch.dict(sys.modules, {'launch_axis_scaled': fake_launch, 'run_experiment': fake_runner}):
    ns = {'__name__': 'review_controller_import', '__file__': str(ROOT / FILES[3])}
    exec(import_code, ns)
check('actual_controller_imports_resolve_with_project_script_path', callable(ns['main']),
      'Real controller module executed with heavy dependencies mocked; remote Popen cwd and script argument separately checked below.')


def helper_case(label, first_used=60, second_used=60, disk_free=10_000_000_000,
                expected_uploads=4, expected_starts=1):
    with tempfile.TemporaryDirectory(prefix='deploy_' + label + '_', dir=TRACE) as folder:
        case_root = Path(folder).resolve()
        assert case_root.is_relative_to(TRACE.resolve())
        local, remote = case_root / 'local', case_root / 'remote'
        local.mkdir(); remote.mkdir(); (remote / 'runs').mkdir()
        (local / 'results/preflight').mkdir(parents=True)
        for name in original:
            for target_root in (local, remote):
                target = target_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())
        for name in FILES:
            (local / name).write_bytes((ROOT / name).read_bytes())
        for name in ('axis_collaboration_v5_mass_launch.json', 'axis_collaboration_v6_projected_plan.json'):
            (local / 'results/preflight' / name).write_bytes((ROOT / 'results/preflight' / name).read_bytes())
        review = {'status': 'PASS', 'blockers': [], 'helper_sha256': helper_sha, 'checked_source_sha256': sources}
        (local / 'results/preflight/axis_collaboration_v6_projected_review.json').write_text(json.dumps(review), encoding='utf-8')
        uploads, spawns, scripts, resources = [], [], [], []
        used = iter([first_used, second_used])
        collector = types.ModuleType('collect_results')
        collector.PROJECT, collector.HOSTS, collector.OPTIONS = local, {'2026': (str(remote), 'MOCK_REMOTE_PYTHON')}, []

        def command(argv):
            assert argv[0] == 'scp' and len(argv) == 3
            path = Path(argv[1])
            assert path.parent == local and path.name in FILES
            assert argv[-1] == '2026:' + str(remote) + '/' + path.name
            uploads.append(path.name)
            (remote / path.name).write_bytes(path.read_bytes())
            return ''

        class Child:
            pid = 234567

            def __init__(self, argv, **kwargs):
                assert argv[:3] == ['MOCK_REMOTE_PYTHON', '-u', 'launch_projected_mass_trial.py']
                assert kwargs['cwd'] == remote and kwargs['start_new_session'] is True
                assert kwargs['stdout'].mode == 'x'
                assert argv[argv.index('--data-root') + 1] == '/data/gaob/Re-ID/dataset'
                spawns.append({'argv': argv, 'cwd': str(kwargs['cwd']), 'detached': True})

        def remote_python(host, code):
            assert host == '2026'
            scripts.append(code)
            stage = 'guard' if len(scripts) == 1 else 'start'
            if label == 'success':
                (TRACE / ('generated_remote_' + stage + '.py')).write_text(code, encoding='utf-8')
            current_used = next(used)
            resources.append({'stage': stage, 'used_MiB': current_used, 'free_bytes': disk_free})
            output = io.StringIO()
            with mock.patch('subprocess.check_output', return_value=str(current_used)), \
                 mock.patch('subprocess.Popen', Child), \
                 mock.patch('shutil.disk_usage', return_value=types.SimpleNamespace(free=disk_free)), \
                 contextlib.redirect_stdout(output):
                exec(compile(code, 'MOCK_remote_' + stage + '.py', 'exec'), {'__name__': 'mock_remote'})
            return output.getvalue()

        collector.command, collector.remote_python = command, remote_python
        failed = False
        output = io.StringIO()
        with mock.patch.dict(sys.modules, {'collect_results': collector}), contextlib.redirect_stdout(output):
            try:
                exec(compile(HELPER.read_text(encoding='utf-8'), str(HELPER), 'exec'),
                     {'__name__': 'mock_helper', '__file__': str(HELPER)})
            except AssertionError:
                failed = True
        check('deployment_helper_mock_' + label,
              len(uploads) == expected_uploads and len(spawns) == expected_starts
              and failed == (expected_starts == 0)
              and all(hashlib.sha256((remote / name).read_bytes()).hexdigest() == digest for name, digest in original.items()),
              {'uploads': uploads, 'starts': len(spawns), 'resources': resources,
               'old18_unchanged': True, 'no_real_remote_operations': True})


helper_case('success')
helper_case('GPU_busy_before_upload', first_used=1500, expected_uploads=0, expected_starts=0)
helper_case('disk_low_before_upload', disk_free=2_000_000_000, expected_uploads=0, expected_starts=0)
helper_case('GPU_busy_after_upload', second_used=1500, expected_starts=0)

smoke = json.loads(read('results/axis_collaboration_v5_mass/preflight/MSVR310_axis_mass_fullref_smoke/smoke.json'))
trainable = smoke['trainable_parameters']
state_only_bytes = smoke['parameters'] * 4
adam_bytes = trainable * 8
disk_evidence = {'V5_MSVR_parameters': smoke['parameters'], 'V5_MSVR_trainable_parameters': trainable,
                 'FP32_parameter_bytes_lower_bound': state_only_bytes,
                 'Adam_two_moment_bytes_lower_bound': adam_bytes,
                 'model_and_optimizer_bytes_lower_bound': state_only_bytes + adam_bytes,
                 'V6_smoke_best_last_parameter_state_bytes_lower_bound': 3 * state_only_bytes + 2 * adam_bytes,
                 'last_serialization_includes_optimizer': "'optimizer': optimizer.state_dict()" in runner_source,
                 'pending_V5_RGB100_last_requires_its_own_model_and_optimizer_budget': True}
(TRACE / 'disk_budget_evidence.json').write_text(json.dumps(disk_evidence, indent=2), encoding='utf-8')
check('checkpoint_budget_primary_code_and_observed_trainable_count',
      trainable == 99669638 and disk_evidence['last_serialization_includes_optimizer'], disk_evidence)

result = {'status': 'PASS_SOURCE_AND_STDLIB_MOCKS', 'observed_at': datetime.now(timezone.utc).isoformat(),
          'checks_count': len(CHECKS), 'checks': CHECKS, 'checked_source_sha256': sources,
          'helper_sha256': helper_sha, 'no_NN_model_import': 'torch' not in sys.modules,
          'real_SSH_calls': 0, 'real_GPU_calls': 0, 'real_optimizer_updates': 0,
          'scope': 'Source equivalence and scheduling/deployment mocks; does not authorize bypassing actual tensor and AMP smoke gates. Disk-budget adjudication is in the reviewer report.'}
assert result['no_NN_model_import']
(TRACE / 'checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: result[key] for key in ('status', 'checks_count', 'helper_sha256', 'no_NN_model_import')}))
