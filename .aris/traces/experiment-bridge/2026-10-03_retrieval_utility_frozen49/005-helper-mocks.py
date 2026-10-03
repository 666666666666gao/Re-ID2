"""Fresh source/AST/in-memory orchestration review; no production imports."""
import argparse
import ast
import copy
from datetime import datetime
import difflib
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_retrieval_utility_frozen49'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_retrieval_utility_frozen26_deploy_20261003.py')
FILES = ('diagnose_retrieval_utility_axis.py', 'missing_retrieval_utility_development.py', 'launch_retrieval_utility_frozen_evaluation.py')
CHECKS = []

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check(name, condition, evidence):
    assert condition, name
    CHECKS.append(dict(name=name, status='PASS', evidence=evidence))

def tree(name):
    return ast.parse((ROOT / name).read_text(encoding='utf-8'))

def function(module, name):
    return next((n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name))

def functions_only(module, names):
    return ast.fix_missing_locations(ast.Module(body=[function(module, n) for n in names], type_ignores=[]))

def no_imports(source):
    module = ast.parse(source)
    module.body = [n for n in module.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    module.body = [n for n in module.body if not (isinstance(n, ast.Expr) and ast.unparse(n).startswith('sys.path.insert('))]
    return ast.fix_missing_locations(module)

def dump(node):
    return ast.dump(node, include_attributes=False)
HASHES = {name: sha(ROOT / name) for name in FILES}
HELPER_HASH = sha(HELPER)
TRAIN_RECEIPT = json.loads((ROOT / 'results/preflight/retrieval_utility_launch.json').read_text(encoding='utf-8'))
TRAIN_SOURCES = TRAIN_RECEIPT['source_sha256']

class MemoryFS:

    def __init__(self):
        self.files = {}
        self.dirs = {'/'}
        fs = self

        class MemoryPath:

            def __init__(self, path):
                self.path = str(path).replace('\\', '/').rstrip('/') or '/'

            def __str__(self):
                return self.path

            def __truediv__(self, tail):
                return MemoryPath(self.path + '/' + str(tail))

            @property
            def name(self):
                return self.path.rsplit('/', 1)[-1]

            @property
            def parent(self):
                return MemoryPath(self.path.rsplit('/', 1)[0])

            def exists(self):
                return self.path in fs.files or self.path in fs.dirs

            def read_bytes(self):
                if self.path not in fs.files:
                    raise FileNotFoundError(self.path)
                return fs.files[self.path]

            def read_text(self, encoding='utf-8'):
                return self.read_bytes().decode(encoding)

            def write_bytes(self, data):
                fs.files[self.path] = data
                return len(data)

            def stat(self):
                return SimpleNamespace(st_size=len(self.read_bytes()))

            def mkdir(self, exist_ok=False, parents=False):
                if self.exists() and (not exist_ok):
                    raise FileExistsError(self.path)
                fs.dirs.add(self.path)

            def open(self, mode):
                assert mode == 'x'
                if self.exists():
                    raise FileExistsError(self.path)
                fs.files[self.path] = b''
                return io.StringIO()
        self.Path = MemoryPath

    def put(self, path, value):
        self.files[str(path)] = value if isinstance(value, bytes) else json.dumps(value).encode()
launcher = compile(functions_only(tree(FILES[2]), ('main',)), FILES[2], 'exec')
idle = compile(functions_only(tree('launch_axis_scaled.py'), ('idle',)), 'bound_real_idle', 'exec')
launcher_cases = ()
helper_source = HELPER.read_text(encoding='utf-8')
helper_cases = ('success', 'review_blocked', 'review_has_blocker', 'review_source_stale', 'helper_stale', 'local_training_source_stale', 'local_missing_evaluation_stale', 'remote_missing_evaluation_stale', 'launch_missing', 'remote_training_source_stale', 'remote_new_source_exists', 'remote_output_exists', 'parent_exit_missing', 'parent_exit_failed', 'parent_incomplete', 'parent_epochs49', 'parent_wrong_variant', 'parent_checkpoint_missing', 'initial_gpu_busy', 'initial_disk_at_floor', 'second_gpu_busy', 'second_disk_at_floor', 'staged_source_corrupt', 'scp_failure', 'input_best_between_guards', 'input_arrays_between_guards', 'input_result_between_guards', 'input_exit_between_guards', 'existing_remote_log')
for case in helper_cases:
    fs = MemoryFS()
    calls = []
    stages = []
    spawns = []
    gpu_calls = []
    project = ROOT.as_posix()
    remote = '/data/gaob/Re-ID/DeMo-DualAxis'
    training = TRAIN_RECEIPT['output'] + '/development'
    run = training + '/MSVR310_axis_retrieval_utility_fullref_s42'
    output = remote + '/runs/axis_collaboration_v10_retrieval_utility_frozen_trial'
    exit_file = run + '_exit.json'
    review = dict(status='BLOCKED' if case == 'review_blocked' else 'PASS', blockers=['fixture'] if case == 'review_has_blocker' else [], helper_sha256='stale' if case == 'helper_stale' else HELPER_HASH, checked_source_sha256=dict(HASHES))
    if case == 'review_source_stale':
        review['checked_source_sha256'][FILES[0]] = 'stale'
    fs.put(project + '/results/preflight/retrieval_utility_frozen_review.json', review)
    fs.put(HELPER.as_posix(), HELPER.read_bytes())
    if case != 'launch_missing':
        fs.put(project + '/results/preflight/retrieval_utility_launch.json', TRAIN_RECEIPT)
    for name in set(TRAIN_SOURCES) | set(FILES):
        fs.put(project + '/' + name, (ROOT / name).read_bytes())
    for name in TRAIN_SOURCES:
        fs.put(remote + '/' + name, (ROOT / name).read_bytes())
    if case == 'local_missing_evaluation_stale':
        fs.put(project + '/missing_evaluation.py', b'changed-evaluator')
    if case == 'remote_missing_evaluation_stale':
        fs.put(remote + '/missing_evaluation.py', b'changed-evaluator')
    if case == 'local_training_source_stale':
        fs.put(project + '/retrieval_utility_axis.py', b'changed')
    if case == 'remote_training_source_stale':
        fs.put(remote + '/retrieval_utility_axis.py', b'changed')
    if case == 'remote_new_source_exists':
        fs.put(remote + '/' + FILES[0], b'prior')
    if case == 'remote_output_exists':
        fs.dirs.add(output)
    if case == 'existing_remote_log':
        fs.put(output + '.log', b'prior log')
    if case != 'parent_exit_missing':
        fs.put(exit_file, {'exit_code': 7 if case == 'parent_exit_failed' else 0})
    fs.put(run + '/result.json', dict(status='RUNNING' if case == 'parent_incomplete' else 'COMPLETE', epochs=49 if case == 'parent_epochs49' else 50, arguments=dict(variant='wrong' if case == 'parent_wrong_variant' else 'axis_retrieval_utility_fullref')))
    if case != 'parent_checkpoint_missing':
        fs.put(run + '/best.pth', b'checkpoint-fixture')
    fs.put(run + '/best_dev_arrays.npz', b'arrays-fixture')

    def command(argv):
        assert argv[0] == 'scp'
        stages.append(argv)
        if case == 'scp_failure':
            raise RuntimeError('mock scp failed')
        src, dest = argv[-2:]
        assert dest.startswith('2026:')
        value = fs.Path(src).read_bytes()
        if case == 'staged_source_corrupt' and len(stages) == 1:
            value = b'corrupt staging'
        fs.put(dest.split(':', 1)[1], value)

    def remote_python(host, source):
        assert host == '2026'
        calls.append(source)
        boundary = 'initial' if len(calls) == 1 else 'second'
        if case.startswith('input_') and case.endswith('_between_guards') and (boundary == 'second'):
            target = {'input_best_between_guards': run + '/best.pth', 'input_arrays_between_guards': run + '/best_dev_arrays.npz', 'input_result_between_guards': run + '/result.json', 'input_exit_between_guards': exit_file}[case]
            fs.put(target, b'changed')

        def gpu_query(argv, **kwargs):
            assert argv == ['nvidia-smi', '-i', '1', '--query-gpu=memory.used', '--format=csv,noheader,nounits']
            gpu_calls.append(argv)
            return '500' if case == boundary + '_gpu_busy' else '499'

        def popen(argv, **kwargs):
            assert argv[2] == FILES[2] and kwargs['cwd'].path == remote and (kwargs['start_new_session'] is True)
            assert argv[argv.index('--training-root') + 1] == training and argv[argv.index('--output') + 1] == output
            spawns.append(argv)
            return SimpleNamespace(pid=9876)
        stdout = []
        namespace = dict(hashlib=hashlib, json=json, Path=fs.Path, subprocess=SimpleNamespace(check_output=gpu_query, Popen=popen, STDOUT=-2), shutil=SimpleNamespace(disk_usage=lambda path: SimpleNamespace(free=200000000 if case == boundary + '_disk_at_floor' else 200000001)), os=SimpleNamespace(environ={}), time=SimpleNamespace(time=lambda: 1), print=lambda value: stdout.append(value))
        exec(compile(no_imports(source), 'mock_remote_' + boundary, 'exec'), namespace)
        return '\n'.join(stdout)
    namespace = dict(datetime=datetime, hashlib=hashlib, json=json, Path=fs.Path, PROJECT=fs.Path(project), HOSTS={'2026': (remote, '/existing/python')}, OPTIONS=[], command=command, remote_python=remote_python, __file__=HELPER.as_posix(), print=lambda *args, **kwargs: None)
    error = None
    try:
        exec(compile(no_imports(helper_source), 'isolated_helper_AST_with_all_IO_replaced', 'exec'), namespace)
    except (AssertionError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    success = case == 'success'
    assert len(spawns) == int(success) and (error is None) == success, (case, spawns, error)
    assert (project + '/results/preflight/retrieval_utility_frozen_launch.json' in fs.files) == success
    if success:
        receipt = json.loads(fs.files[project + '/results/preflight/retrieval_utility_frozen_launch.json'])
        assert len(calls) == 2 and len(stages) == 3 and (receipt['source_sha256'] == {**TRAIN_SOURCES, **HASHES})
        assert receipt['gpu'] == 1 and receipt['planned_missing_conditions'] == 49 and (receipt['planned_four_state_conditions'] == 4)
        assert receipt['optimizer_updates'] == receipt['official_test_uses'] == 0 and len(receipt['availability']['first_inputs']) == 4
    if case == 'local_missing_evaluation_stale':
        assert not calls and not stages and not spawns
    if case == 'remote_missing_evaluation_stale':
        assert len(calls) == 1 and not stages and not spawns
    if case.startswith('parent_'):
        assert not stages and (not spawns)
    check('helper_' + case, True, dict(mock_remote_calls=len(calls), staged_files=len(stages), spawns=len(spawns), mock_gpu1_queries=len(gpu_calls), launch_receipt_written=success, exception=error))
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
assert HASHES == {name: sha(ROOT / name) for name in FILES} and HELPER_HASH == sha(HELPER)
result = {'status': 'PASS', 'checks': CHECKS, 'controller_mock_cases': len(launcher_cases), 'deployment_mock_cases': len(helper_cases), 'checked_source_sha256': HASHES, 'helper_sha256': HELPER_HASH, 'no_neural_imports': True, 'actual_SSH_calls': 0, 'actual_GPU_queries': 0, 'actual_launches': 0}
(TRACE / '005-helper-mock-checks.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: result[k] for k in ('status', 'controller_mock_cases', 'deployment_mock_cases', 'no_neural_imports', 'actual_SSH_calls', 'actual_GPU_queries', 'actual_launches')}))
