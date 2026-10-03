"""Source/AST/stdlib-only review probes. Does not import project/neural modules."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import contextlib
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = PROJECT / '.aris/traces/experiment-bridge/2026-10-03_shared_identity'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_shared_identity_deploy_20261003.py')
PLAN = PROJECT / 'results/preflight/shared_identity_plan.json'
report = {'scope': 'SOURCE_AST_STDLIB_MOCK_ONLY', 'neural_imports': 0,
          'neural_forwards': 0, 'optimizer_updates': 0, 'ssh_calls': 0,
          'gpu_queries': 0, 'checks': []}


def record(name, **evidence):
    report['checks'].append({'name': name, 'status': 'PASS', **evidence})


def tree(name):
    return ast.parse((PROJECT / name).read_text(encoding='utf-8'), filename=name)


def extract(name, names, namespace):
    nodes = [node for node in tree(name).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec'), namespace)
    return namespace


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


plan = json.loads(PLAN.read_text(encoding='utf-8'))
actual = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
          for name in plan['sources']}
assert actual == plan['sources']
report.update(checked_source_sha256=actual,
              plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
              helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest())
dependencies = [
    'axis_collaboration.py', 'dual_axis.py', 'scaled_axis_collaboration.py',
    'mass_axis_collaboration.py', 'modeling/make_model.py', 'modeling/meta_arch.py',
    'modeling/moe/AttnMOE.py', 'modeling/clip/model.py', 'modeling/make_model_clipreid.py',
    'run_mass_experiment.py', 'run_experiment.py', 'experiment_data.py',
    'full_evaluation.py', 'missing_evaluation.py', 'utils/reid_evaluation.py',
    'data/datasets/msvr310.py', 'data/datasets/RGBNT100.py', 'data/datasets/RGBNT201.py',
    'data/datasets/sampler.py', 'layers/make_loss.py', 'layers/softmax_loss.py',
    'layers/triplet_loss.py', 'solver/make_optimizer.py', 'solver/scheduler_factory.py',
    'solver/cosine_lr.py', 'launch_axis_scaled.py', 'launch_runs.py', 'config/defaults.py']
for name in [*actual, *dependencies]:
    tree(name)
ast.parse(HELPER.read_text(encoding='utf-8'))
report['dependency_sha256'] = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
                                for name in [*dependencies, 'configs/MSVR310/DeMo.yml',
                                             'configs/RGBNT201/DeMo.yml', 'configs/RGBNT100/DeMo.yml', 'splits.json']}
record('syntax_and_existing_manifest', parsed_python_files=len(actual) + len(dependencies) + 1,
       all_planned_source_digests_match=True)

step_node = next(node for node in tree('run_shared_identity_experiment.py').body
                 if isinstance(node, ast.FunctionDef) and node.name == 'step')
calls = sorted((node.lineno, ast.unparse(node.func)) for node in ast.walk(step_node)
               if isinstance(node, ast.Call))
backwards = [line for line, func in calls if func.endswith('.backward')]
updates = [line for line, func in calls if func == 'scaler.step']
unscale = [line for line, func in calls if func == 'scaler.unscale_']
partial_call = next(node.lineno for node in ast.walk(step_node)
                    if isinstance(node, ast.Call) and ast.unparse(node.func) == 'model'
                    and any(keyword.arg == 'partial' for keyword in node.keywords))
assert len(backwards) == 2 and len(updates) == 1 and len(unscale) == 1
assert backwards[0] < partial_call < backwards[1] < unscale[0] < updates[0]
assert 'gallery = output[1].detach()' in ast.unparse(step_node)
assert not any(isinstance(node, ast.keyword) and node.arg == 'retain_graph'
               for node in ast.walk(step_node))
record('training_order_ast', backward_lines=backwards, partial_forward_line=partial_call,
       unscale_line=unscale[0], optimizer_step_line=updates[0],
       detached_gallery=True, retain_graph=False)

parts_node = next(node for node in tree('run_shared_identity_experiment.py').body
                  if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'PARTIAL_SETS')
parts = ast.literal_eval(parts_node.value)
relations_node = next(node for node in tree('dual_axis.py').body
                      if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'RELATIONS')
relations = ast.literal_eval(relations_node.value)
assert len(parts) == 6 and set(parts) == set(relations) - {(0, 1, 2)}
eligibility = {str(retained): [all(m in retained for m in subset) for subset in relations]
               for retained in (*parts, (0, 1, 2))}
assert all(any(row) for row in eligibility.values())
shared_source = (PROJECT / 'shared_identity_axis.py').read_text(encoding='utf-8')
assert 'anchor_f = base[:, 5120:].norm(dim=1, keepdim=True).detach()' in shared_source
assert 'COMMON_WEIGHT = .25' in shared_source
record('seven_availability_relations_and_metric_source', eligibility=eligibility,
       common_metric_weight=.25, private_metric_weight=.75,
       frequency_anchor='nonzero learned common identity block; actual tensors unexecuted')


class Arr:
    """Small list adapter for the exact evaluation functions' NumPy surface."""
    def __init__(self, values):
        self.values = values.values if isinstance(values, Arr) else list(values)

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return (Arr(value) if isinstance(value, list) else value for value in self.values)

    def __getitem__(self, key):
        if isinstance(key, Arr):
            if all(isinstance(x, bool) for x in key.values):
                return Arr([value for value, keep in zip(self.values, key.values) if keep])
            return Arr([self.values[i] for i in key.values])
        if isinstance(key, slice):
            return Arr(self.values[key])
        value = self.values[key]
        return Arr(value) if isinstance(value, list) else value

    def operation(self, other, fn):
        right = other.values if isinstance(other, Arr) else [other] * len(self)
        return Arr([fn(left, value) for left, value in zip(self.values, right)])

    def __eq__(self, other):
        return self.operation(other, lambda a, b: a == b)

    def __and__(self, other):
        return self.operation(other, lambda a, b: a and b)

    def __invert__(self):
        return Arr([not value for value in self.values])

    def __add__(self, other):
        return self.operation(other, lambda a, b: a + b)

    def __truediv__(self, other):
        return self.operation(other, lambda a, b: a / b)

    def __mul__(self, other):
        return self.operation(other, lambda a, b: a * b)

    __rmul__ = __mul__

    @property
    def shape(self):
        return (len(self), len(self.values[0])) if self.values and isinstance(self.values[0], (list, Arr)) else (len(self),)

    @property
    def size(self):
        return len(self)

    def astype(self, dtype):
        return Arr(map(dtype, self.values))

    def mean(self):
        return statistics.mean(self.values)

    def tolist(self):
        return self.values


class NP:
    ndarray = Arr
    float64 = float
    asarray = staticmethod(Arr)
    any = staticmethod(any)
    arange = staticmethod(lambda a, b: Arr(range(a, b)))
    flatnonzero = staticmethod(lambda a: Arr(i for i, value in enumerate(a) if value))
    stack = staticmethod(lambda a: Arr([row.tolist() for row in a]))
    minimum = staticmethod(lambda a, b: Arr(min(value, b) for value in a))
    pad = staticmethod(lambda a, width, mode: Arr([*a, *([a[-1]] * width[1])]))

    @staticmethod
    def argsort(values, axis, kind):
        assert axis == 1 and kind == 'stable'
        return Arr([sorted(range(len(row)), key=lambda i: row[i]) for row in values])

    @staticmethod
    def cumsum(values):
        result, total = [], 0
        for value in values:
            total += value
            result.append(total)
        return Arr(result)

    @staticmethod
    def mean(values, axis=None):
        if axis == 0:
            return Arr(statistics.mean(column) for column in zip(*values))
        return statistics.mean(values)


metric_ns = {'np': NP, 'csv': csv, 'write_json': write_json}
extract('utils/reid_evaluation.py', ['evaluate_reid'], metric_ns)
extract('full_evaluation.py', ['full_metrics'], metric_ns)
with tempfile.TemporaryDirectory(prefix='shared-review-metrics-') as tmp:
    gids = Arr([10, 20, 10, 30, 10] + list(range(100, 155)))
    excludes = Arr([0, 0, 1, 0, 2] + [0] * 55)
    distances = Arr([list(range(60))])
    score = metric_ns['full_metrics'](distances, Arr([10]), gids, Arr([0]), excludes,
                                      Arr(['q10']), Arr([7]), Arr([0]), Path(tmp) / 'test')
    assert abs(score['mAP'] - 50) < 1e-12
    assert abs(score['mINP'] - 50) < 1e-12
    assert score['Rank-1'] == 0 and score['Rank-5'] == 100
    assert score['Rank-10'] == score['Rank-20'] == 100
    assert len(score['CMC_1_to_50']) == 50
    rows = list(csv.DictReader((Path(tmp) / 'test.csv').open()))
    assert rows[0]['relevant_gallery'] == '2' and rows[0]['kept_gallery'] == '59'
    # Same-camera/different-scene must be retained in MSVR; caller supplies scenes.
    selector_tree = next(n for n in tree('run_shared_identity_experiment.py').body
                         if isinstance(n, ast.FunctionDef) and n.name == 'metrics_from_arrays')
    assert "arrays['scenes'] if dataset == 'MSVR310' else arrays['cameras']" in ast.unparse(selector_tree)
    record('actual_full_metrics_and_evaluate_reid_stdlib_fixture',
           junk='one same-ID/same-selector item excluded; other IDs retained',
           expected={'mAP': 50, 'mINP': 50, 'Rank-1': 0, 'Rank-5': 100,
                     'Rank-10': 100, 'Rank-20': 100}, cmc_length=50,
           actual={key: score[key] for key in ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')},
           raw_query_rows=rows)


def controller_fixture(failure=None):
    events, lock = [], threading.Lock()
    active = maximum = 0
    finished_smokes = set()
    variants = ('axis_shared', 'frequency_shared', 'twins_shared', 'demo_shared')
    smoke_barrier = threading.Barrier(4)
    train_barrier = threading.Barrier(4)
    with tempfile.TemporaryDirectory(prefix='shared-review-controller-') as tmp:
        out = Path(tmp) / 'wave'

        def execute(command, directory, name, gpu):
            nonlocal active, maximum
            script = command[2]
            if script == 'verify_shared_identity_axis.py':
                stage, variant = 'tensor', None
            elif script == 'run_shared_identity_experiment.py':
                stage = command[command.index('--mode') + 1]
                variant = command[command.index('--variant') + 1]
                assert command[command.index('--dataset') + 1] == 'MSVR310'
                assert command[command.index('--seed') + 1] == '42'
                assert gpu == variants.index(variant)
            else:
                assert script == 'missing_shared_identity_development.py'
                stage, variant = 'frozen49', variants[gpu]
            with lock:
                active += 1
                maximum = max(maximum, active)
                events.append(('start', stage, variant, gpu))
                if stage == 'train':
                    assert finished_smokes == set(variants)
            if stage == 'smoke':
                smoke_barrier.wait(timeout=10)
            if stage == 'train':
                train_barrier.wait(timeout=10)
            target = Path(command[command.index('--output') + 1])
            target.mkdir()
            Path(directory, name + '.log').write_text('mock process output\n')
            code = 1 if failure == (stage, variant) else 0
            write_json(Path(directory, name + '_exit.json'), {'exit_code': code})
            with lock:
                active -= 1
                events.append(('end', stage, variant, gpu, code))
                if stage == 'smoke' and code == 0:
                    finished_smokes.add(variant)
            assert code == 0, str(failure)
            if stage == 'tensor':
                write_json(target / 'result.json', {'status': 'PASS_SHARED_IDENTITY_TENSOR_CONTRACT'})
            elif stage == 'smoke':
                write_json(target / 'smoke.json', {'status': 'SMOKE_PASS', 'steps': 3, 'strict_reload_equal': True})
            elif stage == 'train':
                write_json(target / 'result.json', {'status': 'COMPLETE', 'epochs': 50})
            else:
                write_json(target / 'result.json', {'status': 'COMPLETE',
                           'measurements': dict.fromkeys(map(str, range(49)), {}),
                           'normal_feature_max_error': 0, 'state_tensor_versions_unchanged': True})
            return {'name': name, 'exit_code': 0}

        namespace = {'argparse': argparse, 'ThreadPoolExecutor': ThreadPoolExecutor,
                     'json': json, 'Path': Path, 'sys': SimpleNamespace(executable='MOCK_PYTHON'),
                     'execute': execute, 'write_json': write_json, 'VARIANTS': variants}
        extract('launch_shared_identity_trial.py', ['main'], namespace)
        previous = sys.argv
        sys.argv = ['review', '--data-root', 'MOCK_DATA', '--pretrained', 'MOCK_CLIP', '--output', str(out)]
        error = None
        try:
            namespace['main']()
        except AssertionError as caught:
            error = str(caught)
        finally:
            sys.argv = previous
        terminal_exists = (out / 'controller_result.json').exists()
        if failure:
            assert error is not None and not terminal_exists
            if failure[0] in ('tensor', 'smoke'):
                assert not any(e[:2] == ('start', 'train') for e in events)
            if failure[0] == 'train':
                assert not any(e[:3] == ('start', 'frozen49', failure[1]) for e in events)
        else:
            assert error is None and terminal_exists
            assert len(json.loads((out / 'controller_result.json').read_text())['runs']) == 4
        assert maximum <= 4
        assert len([e for e in events if e[0] == 'start']) == len(set(tuple(e) for e in events if e[0] == 'start'))
        return {'failure': failure, 'max_mock_active': maximum, 'controller_complete': terminal_exists,
                'expected_assertion': error, 'events': events}


controller_cases = [controller_fixture(), controller_fixture(('tensor', None)),
                    controller_fixture(('smoke', 'frequency_shared')),
                    controller_fixture(('train', 'axis_shared')),
                    controller_fixture(('frozen49', 'demo_shared'))]
record('actual_controller_ast_with_mock_processes', cases=controller_cases,
       real_processes=0, real_gpu_jobs=0, automatic_retries=0)


def execute_fixture(code):
    calls, waits = [], []
    outputs = iter(['900', '500', '0'])

    class Subprocess:
        STDOUT = -2

        @staticmethod
        def check_output(argv, text):
            calls.append(('capacity', argv))
            return next(outputs)

        @staticmethod
        def Popen(argv, **kwargs):
            calls.append(('spawn', argv, kwargs['env']['CUDA_VISIBLE_DEVICES']))
            kwargs['stdout'].write('mock stdout/stderr retained\n')
            return SimpleNamespace(pid=12345, wait=lambda: code)

    namespace = {'subprocess': Subprocess, 'time': SimpleNamespace(sleep=waits.append, time=lambda: 123),
                 'os': SimpleNamespace(environ={}), 'write_json': write_json}
    extract('launch_axis_scaled.py', ['idle', 'execute'], namespace)
    with tempfile.TemporaryDirectory(prefix='shared-review-execute-') as tmp:
        error = None
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            try:
                namespace['execute'](['MOCK_PYTHON', 'MOCK_JOB'], Path(tmp), 'probe', 2)
            except AssertionError as caught:
                error = str(caught)
        assert waits == [240, 240]
        assert sum(row[0] == 'spawn' for row in calls) == 1
        assert json.loads((Path(tmp) / 'probe_exit.json').read_text())['exit_code'] == code
        assert (Path(tmp) / 'probe.log').read_text() == 'mock stdout/stderr retained\n'
        assert (error is not None) == (code != 0)
        return {'mock_exit_code': code, 'mock_wait_seconds': waits, 'calls': calls,
                'raw_mock_stdout': captured.getvalue(), 'expected_assertion': error}


record('actual_execute_and_idle_ast_with_mock_subprocess', cases=[execute_fixture(0), execute_fixture(1)],
       real_subprocesses=0, failed_exit_receipt_and_log_retained=True)

# Evaluate the actual helper's generated guard/start strings with all effects stubbed.
helper_tree = ast.parse(HELPER.read_text(encoding='utf-8'))
parent = json.loads((PROJECT / 'results/preflight/availability_base_2026_launch.json').read_text(encoding='utf-8'))
existing = parent['source_sha256'].copy()
for name in ('layers/softmax_loss.py', 'layers/make_loss.py', 'layers/triplet_loss.py',
             'solver/make_optimizer.py', 'solver/scheduler_factory.py'):
    existing[name] = hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == sha for name, sha in existing.items())
record('existing_helper_dependency_verification', existing_sources=len(existing), all_match=True)

generated = {}
for stage in ('guard', 'start'):
    assignment = next(node for node in helper_tree.body if isinstance(node, ast.Assign)
                      and ast.unparse(node.targets[0]) == stage)
    namespace = {'root': '/MOCK_PROJECT', 'python': '/MOCK_ENV/bin/python',
                 'sources': existing if stage == 'guard' else {**existing, **actual},
                 'plan': plan, 'output': '/MOCK_PROJECT/runs/shared_identity_v11_trial_20261003',
                 'data': '/MOCK_DATA', 'pretrained': '/MOCK_CLIP'}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(HELPER), 'exec'), namespace)
    generated[stage] = namespace[stage]
    ast.parse(generated[stage])


def generated_fixture(stage, condition):
    files = {'/MOCK_PROJECT/' + name: (PROJECT / name).read_bytes()
             for name in (existing if stage == 'guard' else {**existing, **actual})}
    disk = 43324407808
    used = [15, 15, 15, 155]
    commands, spawned = [], []
    process_rows = 'PID ARGS\n'
    if condition == 'source_mismatch':
        files['/MOCK_PROJECT/experiment_data.py'] = b'changed mock source'
    elif condition == 'busy_gpu':
        used[2] = 500
    elif condition == 'low_disk':
        disk = 10000000000
    elif condition == 'existing_neural_process':
        process_rows += '123 /MOCK_ENV/bin/python -u run_experiment.py\n'
    elif condition == 'existing_output_log':
        files['/MOCK_PROJECT/runs/shared_identity_v11_trial_20261003.log'] = b'existing'
    elif condition == 'existing_new_source':
        files['/MOCK_PROJECT/shared_identity_axis.py'] = b'pre-existing'

    class MockPath:
        def __init__(self, value):
            self.value = value.value if isinstance(value, MockPath) else str(value)

        def __truediv__(self, other):
            return MockPath(self.value + '/' + other)

        def read_bytes(self):
            return files[self.value]

        def exists(self):
            return self.value in files

        def open(self, mode):
            assert mode == 'x' and not self.exists()
            files[self.value] = b'MOCK LOG'
            return io.StringIO()

    class MockSubprocess:
        STDOUT = -2

        @staticmethod
        def check_output(argv, text):
            commands.append(argv)
            if argv[0] == 'nvidia-smi':
                return '\n'.join(f'{index}, {amount}' for index, amount in enumerate(used))
            assert argv == ['ps', '-eo', 'pid,args']
            return process_rows

        @staticmethod
        def Popen(argv, **kwargs):
            spawned.append(argv)
            assert kwargs['cwd'].value == '/MOCK_PROJECT' and kwargs['start_new_session']
            assert kwargs['env']['OMP_NUM_THREADS'] == '4'
            return SimpleNamespace(pid=24680)

    imports = {'hashlib': hashlib, 'json': json,
               'shutil': SimpleNamespace(disk_usage=lambda p: SimpleNamespace(free=disk)),
               'subprocess': MockSubprocess, 'pathlib': SimpleNamespace(Path=MockPath),
               'os': SimpleNamespace(environ={}), 'time': SimpleNamespace(time=lambda: 987)}
    namespace = {'__builtins__': {**vars(__import__('builtins')),
                  '__import__': lambda name, *args, **kwargs: imports[name]}}
    error = None
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        try:
            exec(compile(generated[stage], 'generated_' + stage, 'exec'), namespace)
        except AssertionError as caught:
            error = type(caught).__name__
    should_pass = condition == 'success'
    assert (error is None) == should_pass
    assert len(spawned) == int(should_pass and stage == 'start')
    return {'stage': stage, 'condition': condition, 'expected_assertion': error,
            'mock_command_calls': commands, 'mock_launches': spawned,
            'raw_mock_stdout': captured.getvalue()}


deployment_cases = [generated_fixture('guard', condition) for condition in
                    ('success', 'source_mismatch', 'busy_gpu', 'low_disk', 'existing_neural_process',
                     'existing_output_log', 'existing_new_source')]
deployment_cases += [generated_fixture('start', condition) for condition in
                     ('success', 'source_mismatch', 'busy_gpu', 'existing_output_log')]
record('actual_generated_deployment_code_with_mock_hardware_and_processes', cases=deployment_cases,
       real_ssh_calls=0, real_gpu_queries=0, real_launches=0)

assert 'Adam' in plan['recipe']['fixed_training'] and 'AdamW' not in plan['recipe']['fixed_training']
assert "OPTIMIZER_NAME: 'Adam'" in (PROJECT / 'configs/MSVR310/DeMo.yml').read_text(encoding='utf-8')
record('corrected_optimizer_plan_matches_existing_config', optimizer='Adam',
       original_plan_error='AdamW was metadata-only; corrected by executor during review without changing implementation')


class FakeBase:
    def __getitem__(self, index):
        assert index[0] == slice(None)
        if index[1] == slice(None, 1536):
            return 'original1536'
        assert index[1] == slice(1536, 5120)
        return 'moe3584'


def make_head(name):
    return lambda feature: name + '(' + feature + ')'


supervision_ns = {}
extract('shared_identity_axis.py', ['base_supervision'], supervision_ns)
supervision_cases = []
for direct in (False, True):
    model = SimpleNamespace(direct=direct)
    for head in ('classifier_moe', 'bottleneck_moe', 'shared_classifier', 'shared_neck',
                 'classifier', 'bottleneck', 'classifier_r', 'classifier_n', 'classifier_t',
                 'bottleneck_r', 'bottleneck_n', 'bottleneck_t'):
        setattr(model, head, make_head(head))
    values = supervision_ns['base_supervision'](model, ['rgb512', 'nir512', 'tir512'],
                                               FakeBase(), 'shared512')
    expected_features = ['moe3584', 'shared512', 'original1536'] if direct else [
                        'moe3584', 'shared512', 'rgb512', 'nir512', 'tir512']
    assert values[1::2] == expected_features
    expected_scores = ['classifier_moe(bottleneck_moe(moe3584))',
                       'shared_classifier(shared_neck(shared512))']
    expected_scores += ['classifier(bottleneck(original1536))'] if direct else [
                       'classifier_r(bottleneck_r(rgb512))',
                       'classifier_n(bottleneck_n(nir512))',
                       'classifier_t(bottleneck_t(tir512))']
    assert values[::2] == expected_scores
    weights = {}
    for class_name, prefix_pairs in (('SharedIdentityDeMo', 1), ('SharedIdentityAxis', 3)):
        class_node = next(node for node in tree('shared_identity_axis.py').body
                          if isinstance(node, ast.ClassDef) and node.name == class_name)
        assignment = next(node for node in ast.walk(class_node)
                          if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'self.loss_weights')
        namespace = {'self': model}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), 'actual_loss_weights', 'exec'), namespace)
        assert len(model.loss_weights) == len(values) // 2 + prefix_pairs
        weights[class_name] = model.loss_weights
    assert weights['SharedIdentityDeMo'] == [1., 1., .25] + ([1.] if direct else [1., 1., 1.])
    assert weights['SharedIdentityAxis'] == [1., .1, .1, 1., .25] + ([1.] if direct else [1., 1., 1.])
    supervision_cases.append({'direct': direct, 'actual_base_outputs': values, 'actual_weights': weights})


def initializer_signature(source):
    parsed = ast.parse(source)
    for class_node in [node for node in parsed.body if isinstance(node, ast.ClassDef)]:
        constructor = next(node for node in class_node.body
                           if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        constructor.body = [node for node in constructor.body
                            if not (isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'self.loss_weights')]
    parsed.body = [node for node in parsed.body if not (isinstance(node, ast.FunctionDef)
                   and node.name == 'base_supervision')]
    return ast.dump(parsed, include_attributes=False)


assert initializer_signature((TRACE / 'shared_identity_axis.py').read_text(encoding='utf-8')) == initializer_signature(shared_source)
record('direct_and_separate_supervision_actual_ast_and_stdlib_calls', cases=supervision_cases,
       model_construction_and_remaining_AST_unchanged=True,
       MSVR_original_pair_weights_unchanged=True,
       RGBNT201_DIRECT_1_supported_by_original_classifier_branch=True)

assert actual == {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in actual}
assert report['plan_sha256'] == hashlib.sha256(PLAN.read_bytes()).hexdigest()
assert report['helper_sha256'] == hashlib.sha256(HELPER.read_bytes()).hexdigest()
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
report['status'] = 'PASS_SOURCE_AST_STDLIB_MOCKS_ONLY'
write_json(TRACE / 'reviewer_stdlib_results.json', report)
print(json.dumps({'status': report['status'], 'checks': len(report['checks']),
                  'neural_imports': 0, 'neural_forwards': 0, 'ssh_calls': 0, 'gpu_queries': 0}))
