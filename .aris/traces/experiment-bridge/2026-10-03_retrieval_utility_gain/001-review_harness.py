"""Source/AST and standard-library-only review; never imports project ML modules."""
import argparse
import ast
import copy
from contextlib import redirect_stdout
from datetime import datetime
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = Path(__file__).parent
HELPER = Path('C:/Users/gb/.codex_tmp/demo_retrieval_utility26_deploy_20261003.py')
FILES = ('retrieval_utility_axis.py', 'run_retrieval_utility_experiment.py',
         'verify_retrieval_utility.py', 'launch_retrieval_utility_trial.py')
CHECKS = []


def check(name, details):
    CHECKS.append(dict(name=name, status='PASS', details=details))
    print('PASS', name)


def parse(path):
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3, 10))


def func(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def compile_nodes(nodes, name):
    return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), name, 'exec')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


texts = {name: (PROJECT / name).read_text(encoding='utf-8') for name in FILES}
digests = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in FILES}
helper_digest = hashlib.sha256(HELPER.read_bytes()).hexdigest()
trees = {name: parse(PROJECT / name) for name in FILES}
for name, tree in trees.items():
    compile(tree, name, 'exec')
compile(parse(HELPER), str(HELPER), 'exec')
assert sys.version_info[:2] == (3, 10)
check('python310_parse_and_compile', dict(python=sys.version, sources=FILES, helper=str(HELPER)))

snapshots = TRACE / 'reviewed_sources'
snapshots.mkdir(exist_ok=False)
for name in FILES:
    (snapshots / name).write_bytes((PROJECT / name).read_bytes())
(snapshots / HELPER.name).write_bytes(HELPER.read_bytes())
pairs = [('run_relation_frequency_experiment.py', 'run_retrieval_utility_experiment.py'),
         ('launch_relation_frequency_trial.py', 'launch_retrieval_utility_trial.py'),
         ('verify_relation_frequency_interface.py', 'verify_retrieval_utility.py'),
         ('relation_frequency_interface.py', 'retrieval_utility_axis.py')]
diffs = []
for old, new in pairs:
    (snapshots / old).write_bytes((PROJECT / old).read_bytes())
    diffs.extend(difflib.unified_diff((PROJECT / old).read_text().splitlines(True),
                                    texts[new].splitlines(True), fromfile=old, tofile=new))
(TRACE / '001-source.diff').write_text(''.join(diffs), encoding='utf-8')

receipt = json.loads((PROJECT / 'results/preflight/relation_frequency_frozen_launch.json').read_text())
sources = receipt['source_sha256']
assert len(sources) == 51
assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == sha for name, sha in sources.items())
assert not set(sources).intersection(FILES)
assert json.loads((PROJECT / 'results/axis_collaboration_v9_relation_frequency_trial/controller_result.json').read_text())['status'] == 'COMPLETE'
assert json.loads((PROJECT / 'results/axis_collaboration_v9_relation_frequency_frozen_trial/controller_result.json').read_text())['status'] == 'COMPLETE'
check('actual_frozen_B_51_parents_match_and_complete', dict(parent_count=51, new_sources=4))

old_runner = parse(PROJECT / 'run_relation_frequency_experiment.py')
new_runner = trees['run_retrieval_utility_experiment.py']
for name in ('write_json', 'configuration', 'evaluate', 'step'):
    assert ast.dump(func(old_runner, name)) == ast.dump(func(new_runner, name)), name


class NormalizeAddedReporting(ast.NodeTransformer):
    def visit_Constant(self, node):
        if node.value == 'axis_retrieval_utility_fullref':
            node.value = 'axis_relation_frequency_fullref'
        return node

    def visit_Name(self, node):
        if node.id == 'RetrievalUtilityDeMo':
            node.id = 'RelationFrequencyInterfaceDeMo'
        return node

    def visit_Dict(self, node):
        node = self.generic_visit(node)
        pairs = [(k, v) for k, v in zip(node.keys, node.values)
                 if not (isinstance(k, ast.Constant) and k.value == 'retrieval_gain')]
        node.keys = [k for k, v in pairs]
        node.values = [v for k, v in pairs]
        return node

    def visit_Assign(self, node):
        code = ast.unparse(node)
        if code.startswith("info['retrieval_gain']") or code.startswith("result['retrieval_gain_sanity']"):
            return None
        if code.startswith("info['method_revision']") and 'P1 retrieval utility:' in code:
            return None
        return self.generic_visit(node)


for name in ('build', 'main'):
    new = NormalizeAddedReporting().visit(copy.deepcopy(func(new_runner, name)))
    assert ast.dump(new) == ast.dump(func(old_runner, name)), name
check('B_runner_training_AST_identical_except_declared_delta',
      'Only selected model/variant, gain metadata and per-step gain audit differ; config, evaluation, AMP512, optimizer, sampling, 50 epochs, strict earliest dev-best/reload unchanged.')

model_class = next(n for n in trees[FILES[0]].body if isinstance(n, ast.ClassDef))
assert [ast.unparse(n) for n in model_class.bases] == ['RelationFrequencyInterfaceDeMo']
assert {n.name for n in model_class.body if isinstance(n, ast.FunctionDef)} == {'controlled_states', 'forward'}
assert [ast.unparse(n) for n in model_class.body if isinstance(n, ast.Assign)] == ['gain_weight = 0.25', 'gain_margin = 0.02', 'gain_temperature = 0.1']
gain = ast.unparse(func(trees[FILES[0]], 'joint_retrieval_gain'))
for term in ("F.normalize(base.detach(), dim=1)", "states[key].detach().float()", "for key in ('00', '10', '01')",
             "torch.stack(tuple(scores.values()), 1).max(1).values + margin",
             "(-scores['00'] / temperature).sigmoid()", "states['11'].float() - (base - base.detach())",
             "positive = same & ~torch.eye", "negative = ~same"):
    assert term in gain, term
check('loss_and_no_new_parameters_static_contract',
      'lambda .25, margin .02, temperature .1; fixed detached base00 normalized gallery; exclude self positives and same-label negatives; same selected indices and detached max00/10/01 target; full11 subtracts zero-valued direct-base path; unchanged B architecture.')

verifier = texts['verify_retrieval_utility.py']
for term in ("set(old)==set(new)", "torch.equal(before,after)", "old_output[:-1],new_output[:-1]",
             "grad_base.eq(0).all()", "torch.isfinite(grad_delta).all()", "grad_delta.abs().sum()>0",
             "audit['positive_indices']==pos.tolist()", "audit['negative_indices']==neg.tolist()",
             "torch.equal(loss.detach(),expected)", "torch.set_rng_state(cpu_rng)", "torch.cuda.set_rng_state(cuda_rng)"):
    assert term in verifier, term
check('real_CUDA_verifier_contains_required_assertions',
      'Source inspected only: exact state/params/loss weights; eval and all original train output pairs; new scalar; RNG reset; differentiable separate base/residual leaf with exact-zero/finite-nonzero grads; independently recomputed fixed reference. No real CUDA witness in this review.')

controller = compile_nodes([func(trees['launch_retrieval_utility_trial.py'], 'main')], 'controller_mock')
controller_cases = {}
for case in ('success', 'tensor_exit', 'tensor_status', 'smoke_exit', 'smoke_status', 'train_epoch', 'existing_output'):
    output = TRACE / ('mock_controller_' + case)
    if case == 'existing_output':
        output.mkdir()
    calls = []

    def fake_execute(argv, directory, name, gpu):
        assert gpu == 1
        stage = 'tensor' if name == 'tensor' else argv[argv.index('--mode') + 1]
        calls.append(stage)
        if case == stage + '_exit':
            raise AssertionError('MOCK child nonzero exit')
        result_dir = Path(argv[argv.index('--output') + 1])
        result_dir.mkdir()
        if stage == 'tensor':
            value = {'status': 'FAIL' if case == 'tensor_status' else 'PASS_RETRIEVAL_UTILITY_CONTRACT'}
            result_name = 'result.json'
        elif stage == 'smoke':
            value = {'status': 'FAIL' if case == 'smoke_status' else 'SMOKE_PASS'}
            result_name = 'smoke.json'
        else:
            assert argv[argv.index('--dataset') + 1] == 'MSVR310'
            assert argv[argv.index('--seed') + 1] == '42'
            value = {'status': 'COMPLETE', 'epochs': 49 if case == 'train_epoch' else 50,
                     'retrieval_interface': 'source_specific_frequency_identity_residual'}
            result_name = 'result.json'
        write_json(result_dir / result_name, value)
        return {'name': name, 'exit_code': 0}

    namespace = dict(argparse=argparse, json=json, Path=Path, sys=sys,
                     execute=fake_execute, write_json=write_json)
    exec(controller, namespace)
    error = None
    with patch.object(sys, 'argv', ['controller', '--data-root', 'mock_data', '--pretrained', 'mock_public_CLIP', '--output', str(output)]):
        try:
            namespace['main']()
        except (AssertionError, FileExistsError) as exc:
            error = type(exc).__name__
    expected = {'success': ['tensor', 'smoke', 'train'], 'tensor_exit': ['tensor'], 'tensor_status': ['tensor'],
                'smoke_exit': ['tensor', 'smoke'], 'smoke_status': ['tensor', 'smoke'],
                'train_epoch': ['tensor', 'smoke', 'train'], 'existing_output': []}[case]
    assert calls == expected, (case, calls)
    assert bool(error) == (case != 'success'), (case, error)
    assert (output / 'controller_result.json').exists() == (case == 'success')
    controller_cases[case] = dict(stages=calls, exception=error)
check('controller_fail_closed_and_exact_stage_order', controller_cases)

execute_tree = parse(PROJECT / 'launch_axis_scaled.py')
execute_function = compile_nodes([func(execute_tree, 'execute')], 'execute_mock')
execute_cases = {}
for code in (0, 1):
    calls = []
    output = TRACE / ('mock_execute_' + str(code))
    output.mkdir()

    def fake_popen(argv, **kwargs):
        calls.append(dict(argv=argv, cuda_visible_devices=kwargs['env']['CUDA_VISIBLE_DEVICES']))
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '1'
        return SimpleNamespace(pid=12345, wait=lambda: code)

    namespace = dict(idle=lambda gpu: calls.append({'idle_gpu': gpu}), os=os,
                     subprocess=SimpleNamespace(Popen=fake_popen, STDOUT=-2),
                     write_json=write_json, time=SimpleNamespace(time=lambda: 0))
    exec(execute_function, namespace)
    error = None
    try:
        namespace['execute'](['mock_python', 'mock_stage'], output, 'stage', 1)
    except AssertionError:
        error = 'AssertionError'
    assert bool(error) == bool(code)
    assert json.loads((output / 'stage_exit.json').read_text())['exit_code'] == code
    execute_cases[str(code)] = dict(calls=calls, exception=error)
check('inherited_execute_gpu1_and_nonzero_exit_stop', execute_cases)

# Capture generated remote programs without calling SSH or scp, then execute
# their AST with in-memory paths and mock subprocess/disk probes only.
remote_root = '/mock_remote/DeMo-DualAxis'
mock_review = dict(status='PASS', blockers=[], checked_source_sha256=digests, helper_sha256=helper_digest)
base_fs = {str(PROJECT / name).replace('\\', '/'): (PROJECT / name).read_bytes() for name in set(FILES) | set(sources)}
for name in ('axis_p0_utility_analysis.json', 'relation_frequency_frozen_launch.json',
             'relation_frequency_msvr_development_analysis.json', 'relation_frequency_msvr_frozen_analysis.json'):
    base_fs[str(PROJECT / 'results/preflight' / name).replace('\\', '/')] = (PROJECT / 'results/preflight' / name).read_bytes()
base_fs[str(PROJECT / 'results/preflight/retrieval_utility_review.json').replace('\\', '/')] = json.dumps(mock_review).encode()
base_fs[str(HELPER).replace('\\', '/')] = HELPER.read_bytes()
for name in sources:
    base_fs[remote_root + '/' + name] = (PROJECT / name).read_bytes()
for campaign in ('axis_collaboration_v9_relation_frequency_trial', 'axis_collaboration_v9_relation_frequency_frozen_trial'):
    base_fs[remote_root + '/runs/' + campaign + '/controller_result.json'] = b'{"status":"COMPLETE"}'

helper_nodes = [n for n in parse(HELPER).body if not isinstance(n, (ast.Import, ast.ImportFrom))]
helper_nodes = [n for n in helper_nodes if not (isinstance(n, ast.Expr) and ast.unparse(n).startswith('sys.path.insert'))]
helper_code = compile_nodes(helper_nodes, 'helper_mock')
deployment_cases = {}
for case in ('success', 'review_fail', 'local_hash_changed', 'parent_hash_changed', 'parent_incomplete',
             'new_source_exists', 'output_exists', 'gpu_busy', 'disk_low', 'start_gpu_busy', 'start_disk_low'):
    fs = dict(base_fs)
    calls = []
    output_path = remote_root + '/runs/axis_collaboration_v10_retrieval_utility_trial'
    if case == 'review_fail':
        fs[str(PROJECT / 'results/preflight/retrieval_utility_review.json').replace('\\', '/')] = json.dumps({**mock_review, 'status': 'FAIL'}).encode()
    if case == 'local_hash_changed':
        fs[str(PROJECT / FILES[0]).replace('\\', '/')] += b'\n'
    if case == 'parent_hash_changed':
        fs[remote_root + '/' + next(iter(sources))] += b'\n'
    if case == 'parent_incomplete':
        fs[remote_root + '/runs/axis_collaboration_v9_relation_frequency_trial/controller_result.json'] = b'{"status":"RUNNING"}'
    if case == 'new_source_exists':
        fs[remote_root + '/' + FILES[0]] = b'old'
    if case == 'output_exists':
        fs[output_path] = b'old'

    class MockPath:
        def __init__(self, value):
            self.value = str(value).replace('\\', '/')

        def __str__(self):
            return self.value

        def __truediv__(self, suffix):
            return MockPath(self.value + '/' + str(suffix))

        def read_bytes(self):
            return fs[self.value]

        def read_text(self, **kwargs):
            return self.read_bytes().decode('utf-8')

        def exists(self):
            return self.value in fs

        def write_bytes(self, content):
            fs[self.value] = content

        def open(self, mode):
            assert mode == 'x' and self.value not in fs
            fs[self.value] = b''
            calls.append({'exclusive_log': self.value})
            return io.StringIO()

    remote_stage = 0

    def fake_remote(host, source):
        global remote_stage
        assert host == '2026'
        remote_stage += 1
        stage = 'guard' if remote_stage == 1 else 'start'
        calls.append({'remote_program_mock': stage})
        tree = ast.parse(source, feature_version=(3, 10))
        (TRACE / ('001-generated_remote_' + stage + '.py')).write_text(source, encoding='utf-8')
        code = compile_nodes([n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))], stage)

        def fake_query(argv, **kwargs):
            assert argv[:3] == ['nvidia-smi', '-i', '1']
            calls.append({'mock_gpu_query': 1})
            return '800' if case == 'gpu_busy' or (case == 'start_gpu_busy' and stage == 'start') else '100'

        def fake_spawn(argv, **kwargs):
            assert argv[2] == 'launch_retrieval_utility_trial.py'
            assert kwargs['cwd'].value == remote_root
            calls.append({'spawn_mock': argv})
            return SimpleNamespace(pid=777)

        disk = 3100000000 if case == 'disk_low' or (case == 'start_disk_low' and stage == 'start') else 4000000000
        namespace = dict(hashlib=hashlib, json=json, Path=MockPath,
                         shutil=SimpleNamespace(disk_usage=lambda root: SimpleNamespace(free=disk)),
                         subprocess=SimpleNamespace(check_output=fake_query, Popen=fake_spawn, STDOUT=-2),
                         os=os, time=SimpleNamespace(time=lambda: 0))
        stream = io.StringIO()
        with redirect_stdout(stream):
            exec(code, namespace)
        return stream.getvalue()

    def fake_copy(argv):
        assert argv[0] == 'scp' and argv[-1].startswith('2026:' + remote_root + '/')
        name = argv[-1].rsplit('/', 1)[1]
        assert name in FILES
        fs[remote_root + '/' + name] = fs[str(PROJECT / name).replace('\\', '/')]
        calls.append({'copy_mock': name})

    namespace = dict(datetime=datetime, hashlib=hashlib, json=json, Path=MockPath, sys=sys,
                     PROJECT=MockPath(PROJECT), HOSTS={'2026': (remote_root, '/mock_conda/python')},
                     OPTIONS=[], command=fake_copy, remote_python=fake_remote, __file__=str(HELPER))
    error = None
    try:
        with redirect_stdout(io.StringIO()):
            exec(helper_code, namespace)
    except AssertionError:
        error = 'AssertionError'
    assert bool(error) == (case != 'success'), (case, error)
    spawns = [c for c in calls if 'spawn_mock' in c]
    copies = [c for c in calls if 'copy_mock' in c]
    assert len(spawns) == (1 if case == 'success' else 0)
    assert len(copies) == (4 if case in ('success', 'start_gpu_busy', 'start_disk_low') else 0)
    assert all(fs[remote_root + '/' + name] == base_fs[remote_root + '/' + name]
               for name in sources if not (case == 'parent_hash_changed' and name == next(iter(sources))))
    deployment_cases[case] = dict(exception=error, copy_count=len(copies), spawn_count=len(spawns), calls=calls)
check('deployment_guard_and_start_fail_closed', deployment_cases)

assert 'torch' not in sys.modules and 'numpy' not in sys.modules
assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == sha for name, sha in digests.items())
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == helper_digest
assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == sha for name, sha in sources.items())
check('review_bounds_and_source_unchanged_at_end',
      'No torch/numpy imports, NN/CUDA execution, SSH/scp, GPU query, package installation, or implementation edits; all processes/queries above are mocked.')
write_json(TRACE / '001-checks.json', dict(checks=CHECKS, checked_source_sha256=digests,
                                        helper_sha256=helper_digest, parent_source_count=len(sources)))
print('REVIEW_HARNESS_PASS', len(CHECKS))
