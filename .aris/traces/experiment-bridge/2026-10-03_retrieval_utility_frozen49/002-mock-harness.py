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


ROOT=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE=ROOT/'.aris/traces/experiment-bridge/2026-10-03_retrieval_utility_frozen49'
HELPER=Path('C:/Users/gb/.codex_tmp/demo_retrieval_utility_frozen26_deploy_20261003.py')
FILES=('diagnose_retrieval_utility_axis.py','missing_retrieval_utility_development.py','launch_retrieval_utility_frozen_evaluation.py')
CHECKS=[]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check(name, condition, evidence):
    assert condition, name
    CHECKS.append(dict(name=name, status='PASS', evidence=evidence))

def tree(name):
    return ast.parse((ROOT / name).read_text(encoding='utf-8'))

def function(module, name):
    return next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name)

def functions_only(module, names):
    return ast.fix_missing_locations(ast.Module(body=[function(module, n) for n in names], type_ignores=[]))

def no_imports(source):
    module = ast.parse(source)
    module.body = [n for n in module.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    module.body = [n for n in module.body if not (isinstance(n, ast.Expr) and ast.unparse(n).startswith('sys.path.insert('))]
    return ast.fix_missing_locations(module)

def dump(node):
    return ast.dump(node, include_attributes=False)


HASHES={name:sha(ROOT/name) for name in FILES}
HELPER_HASH=sha(HELPER)
TRAIN_RECEIPT=json.loads((ROOT/'results/preflight/retrieval_utility_launch.json').read_text(encoding='utf-8'))
TRAIN_SOURCES=TRAIN_RECEIPT['source_sha256']
class MemoryFS:
    def __init__(self):
        self.files = {}
        self.dirs = {'/'}
        fs = self
        class MemoryPath:
            def __init__(self, path): self.path = str(path).replace('\\', '/').rstrip('/') or '/'
            def __str__(self): return self.path
            def __truediv__(self, tail): return MemoryPath(self.path + '/' + str(tail))
            @property
            def name(self): return self.path.rsplit('/', 1)[-1]
            @property
            def parent(self): return MemoryPath(self.path.rsplit('/', 1)[0])
            def exists(self): return self.path in fs.files or self.path in fs.dirs
            def read_bytes(self):
                if self.path not in fs.files: raise FileNotFoundError(self.path)
                return fs.files[self.path]
            def read_text(self, encoding='utf-8'): return self.read_bytes().decode(encoding)
            def write_bytes(self, data): fs.files[self.path] = data; return len(data)
            def stat(self): return SimpleNamespace(st_size=len(self.read_bytes()))
            def mkdir(self, exist_ok=False, parents=False):
                if self.exists() and not exist_ok: raise FileExistsError(self.path)
                fs.dirs.add(self.path)
            def open(self, mode):
                assert mode == 'x'
                if self.exists(): raise FileExistsError(self.path)
                fs.files[self.path] = b''
                return io.StringIO()
        self.Path = MemoryPath
    def put(self, path, value):
        self.files[str(path)] = value if isinstance(value, bytes) else json.dumps(value).encode()

launcher = compile(functions_only(tree(FILES[2]), ('main',)), FILES[2], 'exec')
idle = compile(functions_only(tree('launch_axis_scaled.py'), ('idle',)), 'bound_real_idle', 'exec')
launcher_cases = (
    'success', 'wait_parent', 'busy_gpu', 'parent_exit', 'parent_incomplete', 'epochs49', 'wrong_variant', 'wrong_dataset',
    'child_exit', 'receipt_missing', 'smoke_bad', 'updates', 'versions', 'smoke_parity', 'full_parity',
    'three_states', 'forty_eight_missing', 'wrong_distance_source', 'missing_incomplete',
    'input_best.pth', 'input_best_dev_arrays.npz', 'input_result.json', 'input_exit.json', 'existing_output',
)
for case in launcher_cases:
    fs = MemoryFS(); calls = []; sleeps = []; queries = []
    training = '/training'; run = training + '/MSVR310_axis_retrieval_utility_fullref_s42'; output = '/frozen'
    exit_file = run + '_exit.json'
    terminal = dict(status='RUNNING' if case == 'parent_incomplete' else 'COMPLETE', epochs=49 if case == 'epochs49' else 50,
                    arguments=dict(variant='wrong' if case == 'wrong_variant' else 'axis_retrieval_utility_fullref', dataset='wrong' if case == 'wrong_dataset' else 'MSVR310'))
    fs.put(run + '/result.json', terminal)
    if case != 'wait_parent': fs.put(exit_file, {'exit_code': 1 if case == 'parent_exit' else 0})
    fs.put(run + '/best.pth', b'checkpoint-fixture')
    fs.put(run + '/best_dev_arrays.npz', b'arrays-fixture')
    if case == 'existing_output': fs.dirs.add(output)
    def sleep(seconds):
        assert seconds == 240 and not calls
        sleeps.append(seconds)
        if case == 'wait_parent': fs.put(exit_file, {'exit_code': 0})
    def gpu_query(argv, **kwargs):
        assert argv == ['nvidia-smi', '-i', '1', '--query-gpu=memory.used', '--format=csv,noheader,nounits']
        queries.append(argv)
        return '500' if case == 'busy_gpu' and len(queries) == 1 else '499'
    def popen(argv, **kwargs):
        smoke = '--smoke' in argv; four = argv[2] == FILES[0]
        stage = ('four_state' if four else 'missing') + ('_smoke' if smoke else '_full')
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '1'
        assert argv[argv.index('--run-dir') + 1] == run
        calls.append(stage)
        dest = argv[argv.index('--output') + 1]
        result = dict(status='PASS' if smoke else 'COMPLETE', optimizer_updates=int(case == 'updates'),
                      state_tensor_versions_unchanged=case != 'versions', normal_feature_max_error=int(case == 'smoke_parity'),
                      normal_inference_feature_max_error=int(case == 'full_parity'),
                      normal_distance_source='wrong' if case == 'wrong_distance_source' else 'saved best_dev_arrays.npz/distances',
                      metrics=dict.fromkeys(range(3 if case == 'three_states' else 4)),
                      measurements=dict.fromkeys(range(48 if case == 'forty_eight_missing' else 49)))
        if case == 'smoke_bad' and smoke: result['status'] = 'FAILED'
        if case == 'missing_incomplete' and not four and not smoke: result['status'] = 'RUNNING'
        if case != 'receipt_missing': fs.put(dest + ('/smoke.json' if smoke else '/diagnostic.json' if four else '/result.json'), result)
        if case.startswith('input_'):
            target = exit_file if case == 'input_exit.json' else run + '/' + case.removeprefix('input_')
            fs.put(target, b'changed-immutable-input')
        return SimpleNamespace(pid=123, wait=lambda: 7 if case == 'child_exit' else 0)
    namespace = dict(argparse=argparse, hashlib=hashlib, json=json, Path=fs.Path, os=SimpleNamespace(environ={}),
                     sys=SimpleNamespace(executable='/existing/python'), subprocess=SimpleNamespace(Popen=popen, check_output=gpu_query, STDOUT=-2),
                     time=SimpleNamespace(time=lambda: 1, sleep=sleep), write_json=fs.put, print=lambda *a, **kw: None)
    exec(idle, namespace); exec(launcher, namespace)
    previous = sys.argv; sys.argv = ['review', '--training-root', training, '--output', output, '--data-root', '/data', '--pretrained', '/clip']
    error = None
    try: namespace['main']()
    except (AssertionError, FileNotFoundError, FileExistsError) as exc: error = type(exc).__name__ + ': ' + str(exc)
    finally: sys.argv = previous
    success = case in ('success', 'wait_parent', 'busy_gpu')
    zero = ('parent_exit', 'parent_incomplete', 'epochs49', 'wrong_variant', 'wrong_dataset', 'existing_output')
    two = ('full_parity', 'three_states', 'wrong_distance_source')
    four = ('forty_eight_missing', 'missing_incomplete')
    expected = 4 if success or case in four else 0 if case in zero else 2 if case in two else 1
    assert len(calls) == expected, (case, calls, error)
    assert (output + '/controller_result.json' in fs.files) == success, (case, error)
    assert (error is None) == success, (case, error)
    assert sleeps == ([240] if case in ('wait_parent', 'busy_gpu') else [])
    if success:
        terminal = json.loads(fs.files[output + '/controller_result.json'])
        assert terminal['status'] == 'COMPLETE' and terminal['optimizer_updates'] == terminal['official_test_uses'] == 0
        assert calls == ['four_state_smoke', 'four_state_full', 'missing_smoke', 'missing_full']
    if case == 'child_exit':
        assert any(json.loads(v)['exit_code'] == 7 for k, v in fs.files.items() if k.endswith('/four_state_smoke_exit.json'))
    check('controller_' + case, True, dict(mock_stages=calls, wait_seconds=sleeps, terminal_written=success, exception=error))

helper_source = HELPER.read_text(encoding='utf-8')
helper_cases = (
    'success', 'review_blocked', 'review_has_blocker', 'review_source_stale', 'helper_stale', 'local_training_source_stale',
    'launch_missing', 'remote_training_source_stale', 'remote_new_source_exists', 'remote_output_exists',
    'parent_exit_missing', 'parent_exit_failed', 'parent_incomplete', 'parent_epochs49', 'parent_wrong_variant', 'parent_checkpoint_missing',
    'initial_gpu_busy', 'initial_disk_at_floor', 'second_gpu_busy', 'second_disk_at_floor',
    'staged_source_corrupt', 'scp_failure', 'input_best_between_guards', 'input_arrays_between_guards', 'input_result_between_guards', 'input_exit_between_guards', 'existing_remote_log',
)
for case in helper_cases:
    fs = MemoryFS(); calls = []; stages = []; spawns = []; gpu_calls = []
    project = ROOT.as_posix(); remote = '/data/gaob/Re-ID/DeMo-DualAxis'
    training = TRAIN_RECEIPT['output'] + '/development'; run = training + '/MSVR310_axis_retrieval_utility_fullref_s42'
    output = remote + '/runs/axis_collaboration_v10_retrieval_utility_frozen_trial'; exit_file = run + '_exit.json'
    review = dict(status='BLOCKED' if case == 'review_blocked' else 'PASS', blockers=['fixture'] if case == 'review_has_blocker' else [],
                  helper_sha256='stale' if case == 'helper_stale' else HELPER_HASH, checked_source_sha256=dict(HASHES))
    if case == 'review_source_stale': review['checked_source_sha256'][FILES[0]] = 'stale'
    fs.put(project + '/results/preflight/retrieval_utility_frozen_review.json', review)
    fs.put(HELPER.as_posix(), HELPER.read_bytes())
    if case != 'launch_missing': fs.put(project + '/results/preflight/retrieval_utility_launch.json', TRAIN_RECEIPT)
    for name in set(TRAIN_SOURCES) | set(FILES): fs.put(project + '/' + name, (ROOT / name).read_bytes())
    for name in TRAIN_SOURCES: fs.put(remote + '/' + name, (ROOT / name).read_bytes())
    if case == 'local_training_source_stale': fs.put(project + '/retrieval_utility_axis.py', b'changed')
    if case == 'remote_training_source_stale': fs.put(remote + '/retrieval_utility_axis.py', b'changed')
    if case == 'remote_new_source_exists': fs.put(remote + '/' + FILES[0], b'prior')
    if case == 'remote_output_exists': fs.dirs.add(output)
    if case == 'existing_remote_log': fs.put(output + '.log', b'prior log')
    if case != 'parent_exit_missing': fs.put(exit_file, {'exit_code': 7 if case == 'parent_exit_failed' else 0})
    fs.put(run + '/result.json', dict(status='RUNNING' if case == 'parent_incomplete' else 'COMPLETE',
           epochs=49 if case == 'parent_epochs49' else 50,
           arguments=dict(variant='wrong' if case == 'parent_wrong_variant' else 'axis_retrieval_utility_fullref')))
    if case != 'parent_checkpoint_missing': fs.put(run + '/best.pth', b'checkpoint-fixture')
    fs.put(run + '/best_dev_arrays.npz', b'arrays-fixture')
    def command(argv):
        assert argv[0] == 'scp'; stages.append(argv)
        if case == 'scp_failure': raise RuntimeError('mock scp failed')
        src, dest = argv[-2:]; assert dest.startswith('2026:')
        value = fs.Path(src).read_bytes()
        if case == 'staged_source_corrupt' and len(stages) == 1: value = b'corrupt staging'
        fs.put(dest.split(':', 1)[1], value)
    def remote_python(host, source):
        assert host == '2026'; calls.append(source)
        boundary = 'initial' if len(calls) == 1 else 'second'
        if case.startswith('input_') and case.endswith('_between_guards') and boundary == 'second':
            target = {'input_best_between_guards':run+'/best.pth', 'input_arrays_between_guards':run+'/best_dev_arrays.npz', 'input_result_between_guards':run+'/result.json', 'input_exit_between_guards':exit_file}[case]
            fs.put(target, b'changed')
        def gpu_query(argv, **kwargs):
            assert argv == ['nvidia-smi', '-i', '1', '--query-gpu=memory.used', '--format=csv,noheader,nounits']; gpu_calls.append(argv)
            return '500' if case == boundary + '_gpu_busy' else '499'
        def popen(argv, **kwargs):
            assert argv[2] == FILES[2] and kwargs['cwd'].path == remote and kwargs['start_new_session'] is True
            assert argv[argv.index('--training-root') + 1] == training and argv[argv.index('--output') + 1] == output
            spawns.append(argv); return SimpleNamespace(pid=9876)
        stdout = []
        namespace = dict(hashlib=hashlib, json=json, Path=fs.Path, subprocess=SimpleNamespace(check_output=gpu_query, Popen=popen, STDOUT=-2),
                         shutil=SimpleNamespace(disk_usage=lambda path: SimpleNamespace(free=200000000 if case == boundary + '_disk_at_floor' else 200000001)),
                         os=SimpleNamespace(environ={}), time=SimpleNamespace(time=lambda: 1), print=lambda value: stdout.append(value))
        exec(compile(no_imports(source), 'mock_remote_' + boundary, 'exec'), namespace)
        return '\n'.join(stdout)
    namespace = dict(datetime=datetime, hashlib=hashlib, json=json, Path=fs.Path, PROJECT=fs.Path(project),
                     HOSTS={'2026': (remote, '/existing/python')}, OPTIONS=[], command=command, remote_python=remote_python,
                     __file__=HELPER.as_posix(), print=lambda *args, **kwargs: None)
    error = None
    try: exec(compile(no_imports(helper_source), 'isolated_helper_AST_with_all_IO_replaced', 'exec'), namespace)
    except (AssertionError, FileNotFoundError, FileExistsError, RuntimeError) as exc: error = type(exc).__name__ + ': ' + str(exc)
    success = case == 'success'
    assert len(spawns) == int(success) and (error is None) == success, (case, spawns, error)
    assert (project + '/results/preflight/retrieval_utility_frozen_launch.json' in fs.files) == success
    if success:
        receipt = json.loads(fs.files[project + '/results/preflight/retrieval_utility_frozen_launch.json'])
        assert len(calls) == 2 and len(stages) == 3 and receipt['source_sha256'] == {**TRAIN_SOURCES, **HASHES}
        assert receipt['gpu'] == 1 and receipt['planned_missing_conditions'] == 49 and receipt['planned_four_state_conditions'] == 4
        assert receipt['optimizer_updates'] == receipt['official_test_uses'] == 0 and len(receipt['availability']['first_inputs']) == 4
    if case.startswith('parent_'): assert not stages and not spawns
    check('helper_' + case, True, dict(mock_remote_calls=len(calls), staged_files=len(stages), spawns=len(spawns),
                                      mock_gpu1_queries=len(gpu_calls), launch_receipt_written=success, exception=error))


assert 'torch' not in sys.modules and 'numpy' not in sys.modules
assert HASHES=={name:sha(ROOT/name) for name in FILES} and HELPER_HASH==sha(HELPER)
result={'status':'PASS','checks':CHECKS,'controller_mock_cases':len(launcher_cases),'deployment_mock_cases':len(helper_cases),'checked_source_sha256':HASHES,'helper_sha256':HELPER_HASH,'no_neural_imports':True,'actual_SSH_calls':0,'actual_GPU_queries':0,'actual_launches':0}
(TRACE/'002-mock-checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ('status','controller_mock_cases','deployment_mock_cases','no_neural_imports','actual_SSH_calls','actual_GPU_queries','actual_launches')}))
