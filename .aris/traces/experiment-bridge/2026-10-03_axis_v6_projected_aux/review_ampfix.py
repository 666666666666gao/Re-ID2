"""Captured AMP failure correction: AST and stdlib operation-order/IO mocks only."""
import argparse
import ast
import base64
import contextlib
import copy
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
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_v6_projected26_ampfix_deploy_20261003.py')
OBSERVER = Path('C:/Users/gb/.codex_tmp/demo_axis_v6_projected_ampfix_observe_20261003.py')
FILES = ('projected_mass_axis_collaboration.py', 'run_projected_mass_experiment.py',
         'verify_projected_axis_mass.py', 'launch_projected_mass_trial.py')
CHECKS = []
old_harness = (TRACE / 'review_static_mock.py').read_text(encoding='utf-8')
old_tree = ast.parse(old_harness)


def extract(name):
    node = next(x for x in old_tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    return ast.get_source_segment(old_harness, node)


for name in ('check', 'read', 'tree', 'dump', 'function'):
    exec(compile(extract(name), 'reused_' + name, 'exec'), globals())

prior_path = ROOT / 'results/preflight/axis_collaboration_v6_projected_review.json'
prior_bytes = prior_path.read_bytes()
prior = json.loads(prior_bytes)
check('original61_report_and_failed4_sources_preserved', prior['checks_count'] == 61
      and (TRACE / 'review61_before_actual_amp.json').read_bytes() == prior_bytes
      and all(hashlib.sha256((TRACE / 'failed_amp_source' / name).read_bytes()).hexdigest() == digest
              for name, digest in prior['checked_source_sha256'].items()),
      'Original61 PASS and failed exact four source files are retained; this revision writes a distinct review artifact.')
sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
helper_sha = hashlib.sha256(HELPER.read_bytes()).hexdigest()
observer_sha = hashlib.sha256(OBSERVER.read_bytes()).hexdigest()
check('only_model_changes_other3_source_bytes_equal_failed_launch',
      all(sources[name] == prior['checked_source_sha256'][name] for name in FILES[1:]),
      'Runner including compact terminal, full optimizer smoke, verifier and controller remain exact bytes.')
new_model = (ROOT / FILES[0]).read_bytes()
old_model = (TRACE / 'failed_amp_source' / FILES[0]).read_bytes()
restored = new_model.replace(b'self.modality_expert.pool(m0.float(), eligible)', b'self.modality_expert.pool(m0, eligible).float()')
restored = restored.replace(b'self.frequency_expert.pool(f0.float(), available, self.structured)', b'self.frequency_expert.pool(f0, available, self.structured).float()')
check('model_exact_two_cast_position_changes_only', restored == old_model,
      'Moving only m0/f0.float() from after to before each auxiliary pool recovers the exact failed model bytes; no parameter, routing, inference or loss change.')
snapshot = json.loads(read('results/preflight/axis_collaboration_v6_projected_latest_snapshot.json'))
failure_log = read('results/axis_collaboration_v6_projected_mass_trial/preflight/MSVR310_tensor.log')
check('actual_primary_half_float_failure_and_no_later_stage_launch',
      'c10::Half != float' in failure_log and 'self.pool_score(evidence)' in failure_log
      and snapshot['rows'][0]['exit']['exit_code'] == 1 and not snapshot['controller_live']
      and all('launch' not in row for row in snapshot['rows'][1:]),
      'Actual failure is inside pooling Linear during the zero-optimizer semantic AMP tensor forward; smoke/fresh50 did not launch.')

new_class = next(x for x in tree(FILES[0]).body if isinstance(x, ast.ClassDef))
new_forward = function(new_class, 'forward')
old_class = next(x for x in ast.parse(old_model).body if isinstance(x, ast.ClassDef))
old_forward = function(old_class, 'forward')
new_block = next(x for x in new_forward.body if isinstance(x, ast.With))
old_block = next(x for x in old_forward.body if isinstance(x, ast.With))
check('both_casts_inside_disabled_autocast_before_pool_and_projection',
      ast.unparse(new_block.items[0].context_expr) == "torch.autocast('cuda', enabled=False)"
      and 'self.modality_expert.pool(m0.float(), eligible)' in ast.unparse(new_block)
      and 'self.frequency_expert.pool(f0.float(), available, self.structured)' in ast.unparse(new_block),
      'Both pool_score Linear calls, subsequent PM/PF, normalization and relation mean execute on FP32 evidence inside the existing disabled-autocast region.')
eval_return = next(x for x in new_forward.body if isinstance(x, ast.If) and ast.unparse(x.test) == 'not self.training')
check('retrieval_eval_returns_before_training_auxiliary_casts',
      new_forward.body.index(eval_return) < new_forward.body.index(new_block)
      and ast.unparse(eval_return.body[0]) == 'return fused'
      and 'detach' not in ast.unparse(new_block),
      'Eval/retrieval is byte-preserved and returns before changed training expressions. Casts do not detach; real autograd remains for the unchanged actual tensor gate.')

# Tagged values verify order and dtype assumptions; no numerical tensor or NN.
events = []
disabled = False


class DtypeMismatch(Exception):
    pass


class Tagged:
    def __init__(self, name, dtype='float32', requires_grad=True):
        self.name, self.dtype, self.requires_grad = name, dtype, requires_grad

    def float(self):
        events.append(self.name + '.float')
        return Tagged(self.name, 'float32', self.requires_grad)

    def __mul__(self, other):
        return Tagged(self.name, self.dtype, self.requires_grad)

    def __getitem__(self, item):
        return self

    def __invert__(self):
        return self

    def squeeze(self, *args):
        return self

    def softmax(self, *args):
        return self

    def sum(self, *args):
        return self

    def mean(self, *args):
        return self

    def flatten(self, *args):
        return self

    def reshape_as(self, *args):
        return self

    def masked_fill(self, *args):
        return self


class Autocast:
    def __init__(self, device, enabled):
        assert device == 'cuda' and enabled is False

    def __enter__(self):
        global disabled
        disabled = True

    def __exit__(self, *args):
        global disabled
        disabled = False


def linear(name):
    def call(value):
        events.append(name + ':' + value.dtype)
        assert disabled
        if value.dtype != 'float32':
            raise DtypeMismatch(name)
        return value
    return call


def normalize(value, dim):
    assert disabled and value.dtype == 'float32' and dim in (-1, 1)
    events.append('normalize:' + value.name)
    return value


expert_tree = tree('axis_collaboration.py')
experts = {}
for class_name, label in [('ModalityAxisExpert', 'M'), ('FrequencyAxisExpert', 'F')]:
    cls = next(x for x in expert_tree.body if isinstance(x, ast.ClassDef) and x.name == class_name)
    pool = function(cls, 'pool')
    assert ast.unparse(pool.body[0]).startswith(('weights = self.pool_score(evidence)', 'logits = self.pool_score(evidence)'))
    ns = {'torch': types.SimpleNamespace(inf=float('inf'))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pool], type_ignores=[])), 'actual_' + class_name + '_pool.py', 'exec'), ns)
    expert = types.SimpleNamespace(pool_score=linear(label + '.pool_score'))
    expert.pool = types.MethodType(ns['pool'], expert)
    experts[label] = expert
self_mock = types.SimpleNamespace(modality_expert=experts['M'], frequency_expert=experts['F'], structured=True,
                                 modality_projection=linear('PM'), frequency_projection=linear('PF'))


def run_aux(block):
    ns = {'torch': types.SimpleNamespace(autocast=Autocast), 'F': types.SimpleNamespace(normalize=normalize),
          'self': self_mock, 'm0': Tagged('m0', 'half'), 'f0': Tagged('f0', 'half'),
          'eligible': Tagged('eligible', 'bool', False), 'available': Tagged('available', 'bool', False)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), 'actual_auxiliary_AST.py', 'exec'), ns)
    return ns


for index, label in enumerate(('M', 'F')):
    one = copy.deepcopy(old_block)
    one.body = [one.body[index]]
    events.clear()
    try:
        run_aux(one)
    except DtypeMismatch as error:
        error_name = str(error)
    else:
        raise AssertionError('Expected old pool to see half before .float()')
    check('old_' + label + '_late_cast_failure_order_reproduced_without_torch',
          error_name == label + '.pool_score' and not any(x.endswith('.float') for x in events),
          {'tagged_mock_events': list(events), 'scope': 'Symbolic dtype/call order only, not a numerical or actual AMP test.'})
events.clear()
ns = run_aux(new_block)
check('new_M_F_actual_pool_AST_receives_FP32_before_linear',
      events.index('m0.float') < events.index('M.pool_score:float32') < events.index('PM:float32')
      and events.index('f0.float') < events.index('F.pool_score:float32') < events.index('PF:float32')
      and ns['standalone_m'].dtype == ns['standalone_f'].dtype == 'float32'
      and ns['standalone_m'].requires_grad and ns['standalone_f'].requires_grad,
      {'tagged_mock_events': list(events), 'scope': 'Executes actual pool/tap AST with tagged values; real PyTorch gradient and AMP verification is still pending.'})

original = json.loads(read('results/preflight/axis_collaboration_v5_mass_launch.json'))['source_sha256']
failed_launch = json.loads(read('results/preflight/axis_collaboration_v6_projected_launch.json'))
helper_tree = ast.parse(HELPER.read_text(encoding='utf-8'))
failed_sources = ast.literal_eval(next(x.value for x in helper_tree.body if isinstance(x, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == 'failed_sources' for t in x.targets)))
failed_pids = ast.literal_eval(next(x.value for x in helper_tree.body if isinstance(x, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == 'failed_pids' for t in x.targets)))
check('helper_exact_failed22_sources_and_both_recorded_PIDs',
      failed_sources == failed_launch['source_sha256'] and len(failed_sources) == 22
      and failed_pids == [failed_launch['pid'], snapshot['rows'][0]['launch']['pid']],
      'Pre-upload guard is anchored to the actual failed22-source receipt and recorded controller/tensor PIDs; no guessed process identifiers.')
check('original18_local_source_hashes_unchanged', all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value for name, value in original.items()),
      'Original V5 source files preserved; only V6 model file is authorized for replacement after the two failed processes are gone.')
plan = json.loads(read('results/preflight/axis_collaboration_v6_projected_plan.json'))
check('plan_failure_correction_new_directory_and_final_four_bytes',
      plan['status'] == 'AMP_FAILURE_CAPTURED_REPAIR_AWAITING_REAL_PREFLIGHT' and plan['budget_epochs'] == 50
      and plan['source_sha256'] == sources and plan['actual_failure']['new_trial'].endswith('_ampfix'),
      'Failure is retained in the machine plan; source and scope refer to the corrected new trial, not a relabeling of the failed run.')
for path in (ROOT / FILES[0], HELPER, OBSERVER):
    compile(path.read_text(encoding='utf-8'), str(path), 'exec')
check('changed_model_helper_observer_compile', True, 'Compilation only; no torch/model import or external command.')

# Reuse the already-reviewed helper mock with the actual failed22-source layout.
helper_case_source = extract('helper_case')
helper_case_source = helper_case_source.replace('expected_uploads=4, expected_starts=1', 'expected_uploads=1, expected_starts=1, alive_pid=None')
helper_case_source = helper_case_source.replace("(local / name).write_bytes((ROOT / name).read_bytes())",
    "(local / name).write_bytes((ROOT / name).read_bytes())\n            (remote / name).write_bytes((TRACE / 'failed_amp_source' / name).read_bytes())")
helper_case_source = helper_case_source.replace("review = {'status': 'PASS'", 
    "old_failure = remote / 'runs/axis_collaboration_v6_projected_mass_trial'\n        old_failure.mkdir()\n        old_marker = old_failure / 'PRIMARY_FAILURE_RETAINED.log'\n        old_marker.write_bytes(b'ORIGINAL_FAILED_RUN')\n        review = {'status': 'PASS'")
helper_case_source = helper_case_source.replace('axis_collaboration_v6_projected_review.json', 'axis_collaboration_v6_projected_ampfix_review.json')
helper_case_source = helper_case_source.replace("stage = 'guard' if len(scripts) == 1 else 'start'",
    "stage = 'guard' if len(scripts) == 1 else 'start'\n            assert \"'-i','1'\" in code\n            assert '2600000000' in code\n            assert 'axis_collaboration_v6_projected_mass_trial_ampfix' in code")
helper_case_source = helper_case_source.replace('current_used = next(used)',
    "current_used = next(used)\n            current_free = disk_free[len(scripts) - 1] if isinstance(disk_free, tuple) else disk_free\n            def ps_call(argv, **kwargs):\n                assert argv[0] == 'ps' and int(argv[2]) in failed_pids\n                return types.SimpleNamespace(stdout=str(alive_pid) if int(argv[2]) == alive_pid else '')")
helper_case_source = helper_case_source.replace("'free_bytes': disk_free", "'free_bytes': current_free")
helper_case_source = helper_case_source.replace('types.SimpleNamespace(free=disk_free)', 'types.SimpleNamespace(free=current_free)')
helper_case_source = helper_case_source.replace("mock.patch('subprocess.Popen', Child),", "mock.patch('subprocess.Popen', Child), mock.patch('subprocess.run', side_effect=ps_call),")
helper_case_source = helper_case_source.replace("'generated_remote_'", "'ampfix_generated_remote_'")
helper_case_source = helper_case_source.replace("spawns.append({'argv': argv, 'cwd': str(kwargs['cwd']), 'detached': True})",
    "assert argv[-1].endswith('axis_collaboration_v6_projected_mass_trial_ampfix')\n                spawns.append({'argv': argv, 'cwd': str(kwargs['cwd']), 'detached': True})")
helper_case_source = helper_case_source.replace("and failed == (expected_starts == 0)",
    "and failed == (expected_starts == 0)\n              and old_marker.read_bytes() == b'ORIGINAL_FAILED_RUN'\n              and all((remote / name).read_bytes() == (TRACE / 'failed_amp_source' / name).read_bytes() for name in FILES[1:])")
exec(compile(helper_case_source, 'ampfix_helper_stdlib_mock.py', 'exec'), globals())
helper_case('success')
helper_case('ampfix_controller_still_alive', alive_pid=3100567, expected_uploads=0, expected_starts=0)
helper_case('ampfix_tensor_still_alive', alive_pid=3100699, expected_uploads=0, expected_starts=0)
helper_case('ampfix_GPU1_busy_before_upload', first_used=1500, expected_uploads=0, expected_starts=0)
helper_case('ampfix_disk_low_before_upload', disk_free=2500000000, expected_uploads=0, expected_starts=0)
helper_case('ampfix_GPU1_busy_after_upload', second_used=1500, expected_starts=0)
helper_case('ampfix_disk_low_after_upload', disk_free=(3000000000, 2500000000), expected_starts=0)

observer_text = OBSERVER.read_text(encoding='utf-8')
old_observer = Path('C:/Users/gb/.codex_tmp/demo_axis_v6_projected_observe_20261003.py').read_text(encoding='utf-8')
restored_observer = observer_text.replace('axis_collaboration_v6_projected_ampfix_launch.json', 'axis_collaboration_v6_projected_launch.json')
restored_observer = restored_observer.replace('axis_collaboration_v6_projected_mass_trial_ampfix', 'axis_collaboration_v6_projected_mass_trial')
restored_observer = restored_observer.replace('axis_collaboration_v6_projected_ampfix_latest_snapshot.json', 'axis_collaboration_v6_projected_latest_snapshot.json')
check('observer_only_new_receipt_output_and_snapshot_paths', restored_observer == old_observer,
      'Observer logic is unchanged and all three output/input namespaces have distinct ampfix names; old failure records remain separate.')
observer_ast = ast.parse(observer_text)
collect_fn = function(observer_ast, 'collect')
collect_code = compile(ast.fix_missing_locations(ast.Module(body=[collect_fn], type_ignores=[])), 'actual_ampfix_observer_collect.py', 'exec')


def observer_case(label, failed=False):
    with tempfile.TemporaryDirectory(prefix='ampobs_', dir='C:/Users/gb/.codex_tmp') as folder:
        temporary = Path(folder).resolve()
        assert temporary.is_relative_to(Path('C:/Users/gb/.codex_tmp').resolve())
        local, remote = temporary / 'local', temporary / 'remote'
        (local / 'results/preflight').mkdir(parents=True)
        remote.mkdir()
        for name, digest in {**original, **sources}.items():
            target = remote / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        root = remote / 'runs/axis_collaboration_v6_projected_mass_trial_ampfix'
        root.mkdir(parents=True)
        output = local / 'results/axis_collaboration_v6_projected_mass_trial_ampfix'
        old_local = local / 'results/axis_collaboration_v6_projected_mass_trial/PRIMARY_FAILURE.log'
        old_local.parent.mkdir(parents=True)
        old_local.write_bytes(b'ORIGINAL_FAILURE_BYTES')
        launch = {'pid': 345678, 'output': str(root), 'log': str(root) + '.log', 'source_sha256': {**original, **sources}}
        Path(launch['log']).write_text('MOCK_CONTROLLER_PRIMARY_LOG', encoding='utf-8')
        rows = [('preflight', 'MSVR310_tensor'), ('preflight', 'MSVR310_axis_mass_projected_fullref_s42_smoke'),
                ('development', 'MSVR310_axis_mass_projected_fullref_s42')]
        for stage, name in rows[:1] if failed else rows:
            campaign = root / stage
            campaign.mkdir(exist_ok=True)
            (campaign / (name + '_launch.json')).write_text(json.dumps({'pid': 100, 'gpu': 1}), encoding='utf-8')
            (campaign / (name + '_exit.json')).write_text(json.dumps({'exit_code': 1 if failed else 0}), encoding='utf-8')
            (campaign / (name + '.log')).write_text('MOCK_FAILED_TENSOR' if failed else 'MOCK_PASS', encoding='utf-8')
            if name == 'MSVR310_tensor':
                if not failed:
                    (campaign / (name + '.json')).write_text(json.dumps({'status': 'PASS'}), encoding='utf-8')
            else:
                leaf = campaign / name
                leaf.mkdir()
                result_name = 'smoke.json' if stage == 'preflight' else 'result.json'
                (leaf / result_name).write_text(json.dumps({'status': 'SMOKE_PASS' if stage == 'preflight' else 'COMPLETE'}), encoding='utf-8')
        if not failed:
            (root / 'controller_result.json').write_text(json.dumps({'status': 'COMPLETE'}), encoding='utf-8')
            (root / 'preflight/controller_result.json').write_text(json.dumps({'status': 'PASS'}), encoding='utf-8')

        def remote_python(host, code):
            assert host == '2026'
            buffer = io.StringIO()
            with mock.patch('subprocess.run', return_value=types.SimpleNamespace(stdout='')), contextlib.redirect_stdout(buffer):
                exec(compile(code, 'actual_observer_remote_collect.py', 'exec'), {'__name__': 'mock_remote'})
            return buffer.getvalue()

        namespace = {'output': output, 'launch': launch, 'PROJECT': local, 'base64': base64, 'json': json,
                     'hashlib': hashlib, 'datetime': datetime, 'remote_python': remote_python, 'print': lambda *a, **kw: None}
        exec(collect_code, namespace)
        caught = False
        try:
            record, terminal = namespace['collect']()
        except AssertionError:
            caught = True
            terminal = False
        if not failed:
            namespace['collect']()
        check('observer_ampfix_' + label,
              caught == failed and terminal == (not failed)
              and old_local.read_bytes() == b'ORIGINAL_FAILURE_BYTES'
              and (output / 'preflight/MSVR310_tensor/intake.json').exists()
              and (output / 'preflight/MSVR310_tensor.log').exists()
              and (local / 'results/preflight/axis_collaboration_v6_projected_ampfix_latest_snapshot.json').exists()
              and not (local / 'results/preflight/axis_collaboration_v6_projected_latest_snapshot.json').exists(),
              'Actual collect() and generated remote read code executed with synthetic files: preserves old namespace; failed logs/intake saved before stop, or completed intake recollects without duplicate file writes.')


observer_case('failure_archived_before_assert', failed=True)
observer_case('complete_idempotent_intake')
check('observer_poll_has_240_second_floor_and_no_retry_launch',
      'delay=max(240,min(remaining)-180) if remaining else 240' in observer_text
      and 'subprocess.Popen' not in observer_text and "command(['scp'" not in observer_text,
      'Observer only reads/collects and uses existing expected-milestone cadence; it does not restart the failed or new trial.')
assert 'torch' not in sys.modules
assert prior_path.read_bytes() == prior_bytes
assert hashlib.sha256((ROOT / FILES[0]).read_bytes()).hexdigest() == sources[FILES[0]]
result = {'status': 'PASS_SOURCE_AND_STDLIB_MOCKS', 'observed_at': datetime.now(timezone.utc).isoformat(),
          'checks_count': len(CHECKS), 'checks': CHECKS, 'checked_source_sha256': sources,
          'helper_path': str(HELPER), 'helper_sha256': helper_sha, 'observer_path': str(OBSERVER),
          'observer_sha256': observer_sha, 'no_NN_model_import': True,
          'real_SSH_calls': 0, 'real_GPU_calls': 0, 'real_optimizer_updates': 0,
          'prior61_report_unchanged': True, 'scope': 'Captured AMP failure correction, single-file redeploy and observer isolation; not actual AMP acceptance.'}
(TRACE / 'ampfix_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: result[key] for key in ('status', 'checks_count', 'helper_sha256', 'observer_sha256', 'no_NN_model_import')}))
