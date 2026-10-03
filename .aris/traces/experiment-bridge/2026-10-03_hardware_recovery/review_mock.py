"""Local-only review harness. All generated remote code runs against memory fixtures."""
import ast
import builtins
import concurrent.futures
import copy
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import threading
import types

REPO = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_v4_hardware_recovery26_20261003.py')
TRACE = REPO / '.aris/traces/experiment-bridge/2026-10-03_hardware_recovery'
SOURCE = HELPER.read_text(encoding='utf-8')
HELPER_BYTES = HELPER.read_bytes()
SHA = hashlib.sha256(HELPER_BYTES).hexdigest()
tree = ast.parse(SOURCE)
evidence_names = ['driver_evidence', 'status_without_gpu', 'extra_cross26_launch', 'extra_cross26_terminal', 'deployment_launch']
evidence_bytes = {name: (REPO / ('results/preflight/axis_collaboration_v4_' + name + '.json')).read_bytes() for name in evidence_names}
evidence = {name: json.loads(data) for name, data in evidence_bytes.items()}
prior = evidence['extra_cross26_launch']
terminal = evidence['extra_cross26_terminal']['controller_result']
collect_tree = ast.parse((REPO / 'collect_results.py').read_text(encoding='utf-8'))
HOSTS = ast.literal_eval(next(n.value for n in collect_tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'HOSTS' for t in n.targets)))
source_bytes = {name: (REPO / name).read_bytes() for name in prior['source_sha256']}
root26, python26 = HOSTS['2026']
root25, _ = HOSTS['2025']
extract = {'root': root26, 'python': python26, 'HOSTS': HOSTS, 'prior': prior}
targets = {'output', 'jobs', 'guard26', 'pause25', 'controller_code', 'entry', 'launch_code'}
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in targets:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(HELPER), 'exec'), extract)
scripts = {name: extract[name] for name in ('guard26', 'pause25', 'controller_code', 'entry', 'launch_code')}
OUTPUT = extract['output']
JOBS = extract['jobs']
ROWS = {row['name']: row for row in evidence['status_without_gpu']['rows']}
FREQUENCY = 'RGBNT100_frequency_scaled_fullref_s42'
BASE = 'MSVR310_axis_scaled_base_s42'
RAW = 'MSVR310_axis_raw_fullref_s42'
old_campaign = root25 + '/runs/axis_collaboration_v4_development'
command25 = '/data2/gb/Re-ID/conda-envs/tri_reid/bin/python -u launch_axis_scaled.py --data-root /data2/gb/Re-ID/dataset --pretrained /data2/gb/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt'
RESULTS = []


def checked(name, fn):
    try:
        value = fn()
    except Exception as exc:
        RESULTS.append({'name': name, 'status': 'FAIL', 'error_type': type(exc).__name__, 'error': str(exc)})
    else:
        RESULTS.append({'name': name, 'status': 'PASS', 'details': value})


def require(value, message):
    assert value, message


class Sandbox:
    def __init__(self):
        self.files = {}
        self.dirs = set()
        self.writes = []
        self.spawns = []
        self.signals = []
        self.commands = []
        self.sleeps = []
        self.remote_calls = []
        self.lines = []
        self.prior_live = False
        self.memory = {0: [15], 1: [60], 2: [3000], 3: [3000]}
        self.kernel = '\n'.join(evidence['driver_evidence']['kernel']['selected_lines'])
        self.exit_codes = {}
        self.result_overrides = {}
        self.omit_results = set()
        self.pending = None
        self.waiting = threading.Event()
        self.release = threading.Event()
        self.entry_calls = []
        sandbox = self

        class MemPath:
            def __init__(self, value):
                self.value = str(PurePosixPath(str(value).replace('\\', '/')))

            def __str__(self):
                return self.value

            def __fspath__(self):
                return self.value

            def __truediv__(self, other):
                return MemPath(str(PurePosixPath(self.value) / str(other)))

            @property
            def parent(self):
                return MemPath(PurePosixPath(self.value).parent)

            def with_suffix(self, suffix):
                return MemPath(PurePosixPath(self.value).with_suffix(suffix))

            @property
            def suffix(self):
                return PurePosixPath(self.value).suffix

            def exists(self):
                return self.value in sandbox.files or self.value in sandbox.dirs

            def read_bytes(self):
                if self.value not in sandbox.files:
                    raise FileNotFoundError(self.value)
                return sandbox.files[self.value]

            def read_text(self, encoding='utf-8'):
                return self.read_bytes().decode(encoding)

            def write_bytes(self, data):
                assert str(self.parent) in sandbox.dirs, 'missing fixture parent: ' + self.value
                sandbox.files[self.value] = bytes(data)
                sandbox.writes.append(self.value)
                return len(data)

            def write_text(self, text, encoding='utf-8'):
                return self.write_bytes(text.encode(encoding))

            def mkdir(self, parents=False, exist_ok=False):
                if self.exists() and not exist_ok:
                    raise FileExistsError(self.value)
                if not parents:
                    assert str(self.parent) in sandbox.dirs, 'missing fixture parent: ' + self.value
                sandbox.add_dir(self.value)
                sandbox.writes.append(self.value + '/')

            def replace(self, target):
                sandbox.files[str(target)] = sandbox.files.pop(self.value)
                sandbox.writes.append(str(target))

            def open(self, mode='r', **kwargs):
                if mode in ('r', 'rb'):
                    return io.BytesIO(self.read_bytes()) if mode == 'rb' else io.StringIO(self.read_text())
                assert mode == 'x', mode
                if self.exists():
                    raise FileExistsError(self.value)
                self.write_bytes(b'')
                return io.StringIO()

        self.Path = MemPath
        self.sys = types.SimpleNamespace(path=[], executable=python26)
        self.os = types.SimpleNamespace(environ={'REVIEW_LOCAL_FIXTURE': '1'}, kill=lambda pid, sig: self.signals.append([pid, sig]))
        self.time = types.SimpleNamespace(time=lambda: 1790990000.0, sleep=lambda value: self.sleeps.append(value))
        self.subprocess = types.SimpleNamespace(run=self.run_command, check_output=self.check_output, Popen=self.popen, STDOUT=-2)
        self.collect = types.SimpleNamespace(PROJECT=MemPath(str(REPO)), HOSTS=HOSTS, remote_python=self.remote_python)
        self.modules = {'hashlib': hashlib, 'json': json, 'datetime': datetime, 'sys': self.sys, 'os': self.os,
                        'time': self.time, 'signal': types.SimpleNamespace(SIGSTOP=19), 'subprocess': self.subprocess,
                        'pathlib': types.SimpleNamespace(Path=MemPath), 'collect_results': self.collect,
                        'concurrent.futures': concurrent.futures,
                        'runpy': types.SimpleNamespace(run_path=self.run_path)}
        for filename, function in [('launch_axis_scaled.py', 'idle'), ('launch_runs.py', 'write_json')]:
            module_tree = ast.parse((REPO / filename).read_text(encoding='utf-8'))
            node = next(n for n in module_tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
            scope = {'subprocess': self.subprocess, 'time': self.time, 'json': json, 'print': self.print}
            exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), scope)
            self.modules[filename.removesuffix('.py')] = types.SimpleNamespace(**{function: scope[function]})
        for key, data in evidence_bytes.items():
            self.seed(str(REPO / ('results/preflight/axis_collaboration_v4_' + key + '.json')), data)
        self.seed(str(HELPER), HELPER_BYTES)
        self.seed_json(str(REPO / 'results/preflight/axis_collaboration_v4_hardware_recovery_review.json'), {'status': 'PASS', 'blockers': [], 'helper_sha256': SHA})
        for name, data in source_bytes.items():
            self.seed(root26 + '/' + name, data)
        self.seed_json(prior['output'] + '/controller_result.json', terminal)
        self.seed('/proc/1408150/cmdline', command25.replace(' ', '\x00').encode() + b'\x00')
        self.seed_json(old_campaign + '/' + FREQUENCY + '/status.json', ROWS[FREQUENCY]['status'])
        self.seed_json(old_campaign + '/' + BASE + '/result.json', ROWS[BASE]['result'])
        self.add_dir(root26 + '/runs')

    def add_dir(self, value):
        path = PurePosixPath(value)
        self.dirs.add(str(path))
        self.dirs.update(str(parent) for parent in path.parents)

    def seed(self, path, value):
        path = str(path).replace('\\', '/')
        self.add_dir(str(PurePosixPath(path).parent))
        self.files[str(PurePosixPath(path))] = value

    def seed_json(self, path, value):
        self.seed(path, json.dumps(value).encode('utf-8'))

    def read_json(self, path):
        return json.loads(self.files[str(path).replace('\\', '/')])

    def importer(self, name, globals=None, locals=None, fromlist=(), level=0):
        assert level == 0 and name in self.modules, 'unmocked import: ' + name
        return self.modules[name]

    def print(self, *args, **kwargs):
        self.lines.append(' '.join(str(arg) for arg in args))

    def execute(self, code, filename='<generated>'):
        scope = {'__name__': '__main__', '__file__': filename,
                 '__builtins__': {**vars(builtins), '__import__': self.importer,
                                  'open': lambda name, mode='r', **kw: self.Path(name).open(mode, **kw), 'print': self.print}}
        start = len(self.lines)
        exec(compile(code, filename, 'exec'), scope)
        return '\n'.join(self.lines[start:])

    def remote_python(self, host, code):
        self.remote_calls.append({'host': host, 'code_sha256': hashlib.sha256(code.encode()).hexdigest()})
        return self.execute(code)

    def run_command(self, command, **kwargs):
        assert command == ['ps', '-p', str(prior['pid']), '-o', 'pid=']
        self.commands.append(command)
        return types.SimpleNamespace(stdout=str(prior['pid']) if self.prior_live else '', returncode=0 if self.prior_live else 1)

    def check_output(self, command, **kwargs):
        self.commands.append(command)
        if command == ['dmesg', '--ctime']:
            return self.kernel
        if command == ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits']:
            return '\n'.join(str(gpu) + ', ' + str(values[0]) for gpu, values in self.memory.items()) + '\n'
        assert command[0:2] == ['nvidia-smi', '-i'] and command[3:] == ['--query-gpu=memory.used', '--format=csv,noheader,nounits'], command
        values = self.memory[int(command[2])]
        value = values.pop(0) if len(values) > 1 else values[0]
        return str(value)

    def popen(self, command, **kwargs):
        record = {'command': command, 'env': kwargs['env'], 'cwd': str(kwargs.get('cwd', '')), 'start_new_session': kwargs.get('start_new_session', False)}
        self.spawns.append(record)
        pid = 900000 + len(self.spawns)
        if 'run_experiment.py' not in command:
            assert command == [python26, '-u', '-c', scripts['entry']]
            return types.SimpleNamespace(pid=pid)
        flags = dict(zip(command[3::2], command[4::2]))
        key = (flags['--dataset'], flags['--mode'])

        def wait():
            output = flags['--output']
            mode = flags['--mode']
            self.add_dir(output)
            result = ({'status': 'SMOKE_PASS', 'steps': 3, 'strict_reload_equal': True} if mode == 'smoke' else {'status': 'COMPLETE', 'epochs': 50})
            result.update(self.result_overrides.get(key, {}))
            if key not in self.omit_results:
                self.seed_json(output + ('/smoke.json' if mode == 'smoke' else '/result.json'), result)
            if self.pending == key:
                self.waiting.set()
                assert self.release.wait(10), 'review harness pending process was not released'
            return self.exit_codes.get(key, 0)

        return types.SimpleNamespace(pid=pid, wait=wait)

    def run_path(self, path, run_name):
        self.entry_calls.append({'path': path, 'run_name': run_name, 'sys_path': list(self.sys.path)})
        self.execute(self.files[path].decode(), path)

    def stage_dirs(self):
        for stage in ('preflight', 'development'):
            self.add_dir(OUTPUT + '/' + stage)

    def task_spawns(self, dataset, mode):
        return [row for row in self.spawns if 'run_experiment.py' in row['command'] and row['command'][row['command'].index('--dataset') + 1] == dataset and row['command'][row['command'].index('--mode') + 1] == mode]


def should_block(code, mutate, no_signals=True, no_spawns=True):
    box = Sandbox()
    mutate(box)
    before = dict(box.files)
    try:
        box.execute(code, str(HELPER) if code == SOURCE else '<generated>')
    except (AssertionError, FileNotFoundError, FileExistsError, KeyError) as exc:
        error = type(exc).__name__
    else:
        raise AssertionError('invalid fixture was accepted')
    if no_signals:
        assert not box.signals, box.signals
    if no_spawns:
        assert not box.spawns, box.spawns
    assert box.files == before, 'blocked path changed fixture files'
    return {'blocked_with': error, 'mock_remote_calls': len(box.remote_calls), 'process_spawns': len(box.spawns), 'signals': box.signals}


for name, code in [('helper', SOURCE), *scripts.items()]:
    checked('ast_' + name, lambda code=code: {'nodes': len(list(ast.walk(ast.parse(code))))})
for name, digest in prior['source_sha256'].items():
    checked('local_source_sha256_' + name, lambda name=name, digest=digest: require(hashlib.sha256(source_bytes[name]).hexdigest() == digest, name))


def positive_guard():
    box = Sandbox()
    result = json.loads(box.execute(scripts['guard26']))
    assert result['source_sha256'] == prior['source_sha256']
    assert not box.spawns and not box.signals and not box.writes
    return {'selected_gpu_memory_mib': [15, 60], 'ignored_gpu_memory_mib': [3000, 3000], 'read_only': True}


checked('guard_valid_receipts_sources_selected_idle_cards', positive_guard)
def changed_terminal(field, value):
    def mutate(box):
        data = copy.deepcopy(terminal)
        data[field] = value
        box.seed_json(prior['output'] + '/controller_result.json', data)
    return mutate


checked('guard_wrong_status', lambda: should_block(scripts['guard26'], changed_terminal('status', 'FAIL')))
checked('guard_nonzero_optimizer_updates', lambda: should_block(scripts['guard26'], changed_terminal('optimizer_updates', 1)))
checked('guard_wrong_run_count', lambda: should_block(scripts['guard26'], changed_terminal('runs', terminal['runs'][:1])))
for r in range(2):
    data = copy.deepcopy(terminal['runs'])
    data[r]['checks'] = data[r]['checks'][:1]
    checked('guard_wrong_check_count_' + str(r), lambda data=data: should_block(scripts['guard26'], changed_terminal('runs', data)))
    for c in range(2):
        data = copy.deepcopy(terminal['runs'])
        data[r]['checks'][c]['exit_code'] = 1
        checked('guard_nonzero_exit_' + str(r) + '_' + str(c), lambda data=data: should_block(scripts['guard26'], changed_terminal('runs', data)))
checked('guard_prior_controller_live', lambda: should_block(scripts['guard26'], lambda b: setattr(b, 'prior_live', True)))
for name in source_bytes:
    checked('guard_changed_source_' + name, lambda name=name: should_block(scripts['guard26'], lambda b: b.seed(root26 + '/' + name, b'changed')))
checked('guard_existing_empty_output', lambda: should_block(scripts['guard26'], lambda b: b.add_dir(OUTPUT)))
checked('guard_existing_nonempty_output', lambda: should_block(scripts['guard26'], lambda b: b.seed(OUTPUT + '/original.json', b'original')))
for gpu in (0, 1):
    for used in (500, 12000):
        checked('guard_busy_gpu_' + str(gpu) + '_' + str(used), lambda gpu=gpu, used=used: should_block(scripts['guard26'], lambda b: b.memory.update({gpu: [used]})))


def positive_pause():
    box = Sandbox()
    before = dict(box.files)
    result = json.loads(box.execute(scripts['pause25']))
    assert box.signals == [[1408150, 19]], box.signals
    assert result['signal'] == 'SIGSTOP' and result['pid'] == evidence['deployment_launch']['pid']
    assert result['base_completed50_but_cleanup_exit_missing']
    assert box.files == before and not box.writes and not box.spawns
    assert box.commands == [['dmesg', '--ctime']]
    return {'signal_targets': box.signals, 'files_unchanged': True, 'gpu_commands': 0}


checked('pause_only_historical_project_controller_preserves_incident', positive_pause)
checked('pause_missing_controller', lambda: should_block(scripts['pause25'], lambda b: (b.dirs.remove('/proc/1408150'), b.files.pop('/proc/1408150/cmdline'))))
checked('pause_wrong_controller_command', lambda: should_block(scripts['pause25'], lambda b: b.seed('/proc/1408150/cmdline', b'python\x00other_project.py\x00')))
checked('pause_missing_xid79', lambda: should_block(scripts['pause25'], lambda b: setattr(b, 'kernel', 'healthy fixture')))
checked('pause_wrong_frequency_epoch', lambda: should_block(scripts['pause25'], lambda b: b.seed_json(old_campaign + '/' + FREQUENCY + '/status.json', {'status': 'RUNNING', 'epoch': 42})))
checked('pause_frequency_exit_already_exists', lambda: should_block(scripts['pause25'], lambda b: b.seed_json(old_campaign + '/' + FREQUENCY + '_exit.json', {'exit_code': 0})))
checked('pause_raw_already_started', lambda: should_block(scripts['pause25'], lambda b: b.add_dir(old_campaign + '/' + RAW)))
checked('pause_base_not_complete', lambda: should_block(scripts['pause25'], lambda b: b.seed_json(old_campaign + '/' + BASE + '/result.json', {'status': 'RUNNING', 'epochs': 50})))
checked('pause_base_not_50', lambda: should_block(scripts['pause25'], lambda b: b.seed_json(old_campaign + '/' + BASE + '/result.json', {'status': 'COMPLETE', 'epochs': 49})))
checked('pause_base_exit_already_exists', lambda: should_block(scripts['pause25'], lambda b: b.seed_json(old_campaign + '/' + BASE + '_exit.json', {'exit_code': 0})))


def controller_happy(entry=False):
    box = Sandbox()
    box.stage_dirs()
    if entry:
        box.seed(OUTPUT + '/controller.py', scripts['controller_code'].encode())
        box.execute(scripts['entry'])
        assert box.entry_calls == [{'path': OUTPUT + '/controller.py', 'run_name': '__main__', 'sys_path': [root26]}]
    else:
        box.execute(scripts['controller_code'])
    report = box.read_json(OUTPUT + '/controller_result.json')
    assert report['status'] == 'COMPLETE' and len(report['runs']) == 2
    assert report['original25_interrupted_attempt_separate']
    assert len(box.spawns) == 4 and not box.signals
    for dataset, variant, gpu in JOBS:
        for mode in ('smoke', 'train'):
            records = box.task_spawns(dataset, mode)
            assert len(records) == 1
            record = records[0]
            flags = dict(zip(record['command'][3::2], record['command'][4::2]))
            stage = 'preflight' if mode == 'smoke' else 'development'
            assert flags == {'--dataset': dataset, '--variant': variant, '--seed': '42', '--mode': mode, '--contribution-weight': '.05', '--data-root': '/data/gaob/Re-ID/dataset', '--pretrained': '/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt', '--output': OUTPUT + '/' + stage + '/' + dataset + '_' + variant + '_s42'}
            assert record['env']['CUDA_VISIBLE_DEVICES'] == str(gpu)
    return {'training_spawns': 2, 'smoke_spawns': 2, 'seed': 42, 'gpus': [0, 1], 'entry_executed': entry}


checked('controller_two_distinct_fresh_smoke_then_train_commands', controller_happy)
checked('entry_actual_run_path_and_import_root', lambda: controller_happy(entry=True))


def controller_reject(dataset, mode, defect, value):
    box = Sandbox()
    box.stage_dirs()
    key = (dataset, mode)
    if defect == 'exit_code':
        box.exit_codes[key] = value
    elif defect == 'missing_result':
        box.omit_results.add(key)
    else:
        box.result_overrides[key] = {defect: value}
    try:
        box.execute(scripts['controller_code'])
    except (AssertionError, FileNotFoundError):
        pass
    else:
        raise AssertionError('bad controller stage was accepted')
    assert OUTPUT + '/controller_result.json' not in box.files
    assert len(box.task_spawns(dataset, 'smoke')) == 1
    if mode == 'smoke':
        assert not box.task_spawns(dataset, 'train'), 'failed smoke launched same-task train'
    other = 'MSVR310' if dataset == 'RGBNT100' else 'RGBNT100'
    other_smoke = len(box.task_spawns(other, 'smoke'))
    other_train = len(box.task_spawns(other, 'train'))
    assert other_train <= other_smoke <= 1
    assert not box.signals
    return {'complete_written': False, 'failed_task_train_spawned': bool(box.task_spawns(dataset, 'train')), 'other_task_smoke_spawns': other_smoke, 'other_task_train_spawns': other_train}


for dataset, _, _ in JOBS:
    for defect, value in [('exit_code', 1), ('status', 'FAILED'), ('steps', 2), ('steps', 4), ('strict_reload_equal', False), ('missing_result', True)]:
        checked('controller_smoke_reject_' + dataset + '_' + defect + '_' + str(value), lambda dataset=dataset, defect=defect, value=value: controller_reject(dataset, 'smoke', defect, value))
    for defect, value in [('exit_code', 1), ('epochs', 49), ('status', 'RUNNING'), ('missing_result', True)]:
        checked('controller_train_reject_' + dataset + '_' + defect + '_' + str(value), lambda dataset=dataset, defect=defect, value=value: controller_reject(dataset, 'train', defect, value))


def pending_exit(dataset):
    box = Sandbox()
    box.stage_dirs()
    box.pending = (dataset, 'train')
    errors = []
    def target():
        try:
            box.execute(scripts['controller_code'])
        except Exception as exc:
            errors.append(str(exc))
    worker = threading.Thread(target=target)
    worker.start()
    try:
        assert box.waiting.wait(5), 'pending fake child did not reach wait'
        name = next(dataset + '_' + variant + '_s42' for d, variant, gpu in JOBS if d == dataset)
        report = box.read_json(OUTPUT + '/development/' + name + '/result.json')
        assert report == {'status': 'COMPLETE', 'epochs': 50}
        assert OUTPUT + '/development/' + name + '_exit.json' not in box.files
        assert OUTPUT + '/controller_result.json' not in box.files
        snapshot = {'result_status': report['status'], 'epochs': report['epochs'], 'exit_present': False, 'aggregate_complete_present': False}
    finally:
        box.release.set()
        worker.join(5)
    assert not worker.is_alive() and not errors, errors
    assert box.read_json(OUTPUT + '/controller_result.json')['status'] == 'COMPLETE'
    return {'while_child_wait_pending': snapshot, 'complete_only_after_child_exit0': True}


for dataset, _, _ in JOBS:
    checked('controller_complete50_without_exit_stays_pending_' + dataset, lambda dataset=dataset: pending_exit(dataset))


def idle_gate():
    box = Sandbox()
    box.stage_dirs()
    box.memory[0] = [12000, 500, 15]
    box.execute(scripts['controller_code'])
    assert box.sleeps == [240, 240]
    assert len(box.spawns) == 4 and not box.signals
    return {'mock_waits_seconds': box.sleeps, 'busy_values_mib': [12000, 500], 'release_value_mib': 15}


checked('controller_actual_idle_rechecks_busy_card_with_240s_waits', idle_gate)


def positive_launch():
    box = Sandbox()
    result = json.loads(box.execute(scripts['launch_code']))
    assert len(box.spawns) == 1 and not box.signals
    assert box.files[OUTPUT + '/controller.py'] == scripts['controller_code'].encode()
    record = box.spawns[0]
    assert record['cwd'] == root26 and record['start_new_session']
    assert result['status'] == 'HARDWARE_RECOVERY_STARTED_SMOKE_AND_50_EPOCH_RUNS_PENDING'
    assert result['jobs'] == [list(job) for job in JOBS]
    assert OUTPUT + '/controller_result.json' not in box.files
    return {'controller_spawns': 1, 'exclusive_output_directories': 3, 'status': result['status']}


checked('launch_new_output_exact_controller_source_detached_entry', positive_launch)
for name in source_bytes:
    checked('launch_rechecks_changed_source_' + name, lambda name=name: should_block(scripts['launch_code'], lambda b: b.seed(root26 + '/' + name, b'changed')))
checked('launch_refuses_existing_output', lambda: should_block(scripts['launch_code'], lambda b: b.add_dir(OUTPUT)))


def helper_happy():
    box = Sandbox()
    before = dict(box.files)
    box.execute(SOURCE, str(HELPER))
    assert [row['host'] for row in box.remote_calls] == ['2026', '2025', '2026']
    assert [row['code_sha256'] for row in box.remote_calls] == [hashlib.sha256(scripts[name].encode()).hexdigest() for name in ('guard26', 'pause25', 'launch_code')]
    assert len(box.spawns) == 1 and box.signals == [[1408150, 19]]
    receipt = box.read_json(str(REPO / 'results/preflight/axis_collaboration_v4_hardware_recovery26_launch.json'))
    assert receipt['original_interrupted_steps_not_counted_as_new_updates']
    assert receipt['incident_sha256'] == hashlib.sha256(evidence_bytes['driver_evidence']).hexdigest()
    assert receipt['status'] == 'HARDWARE_RECOVERY_STARTED_SMOKE_AND_50_EPOCH_RUNS_PENDING'
    assert all(box.files[path] == value for path, value in before.items())
    return {'remote_order_fully_mocked': ['guard26', 'pause25', 'launch26'], 'original_files_unchanged': True, 'new_attempt_status': receipt['status']}


checked('helper_full_mock_order_and_separate_cost_provenance', helper_happy)
def review_change(field, value):
    def mutate(box):
        path = str(REPO / 'results/preflight/axis_collaboration_v4_hardware_recovery_review.json')
        review = box.read_json(path)
        review[field] = value
        box.seed_json(path, review)
    return mutate


for field, value in [('status', 'BLOCKED'), ('blockers', ['actual blocker']), ('helper_sha256', 'wrong')]:
    checked('helper_review_gate_' + field, lambda field=field, value=value: should_block(SOURCE, review_change(field, value)))
checked('helper_missing_incident_xid79', lambda: should_block(SOURCE, lambda b: b.seed_json(str(REPO / 'results/preflight/axis_collaboration_v4_driver_evidence.json'), {'kernel': {'selected_lines': []}})))
checked('helper_busy26_prevents_pause25_and_launch26', lambda: should_block(SOURCE, lambda b: b.memory.update({0: [10000]})))
checked('helper_wrong25_command_prevents_launch26', lambda: should_block(SOURCE, lambda b: b.seed('/proc/1408150/cmdline', b'python\x00other_project.py\x00')))


def training_source_semantics():
    text = source_bytes['run_experiment.py'].decode('utf-8')
    training = ast.parse(text)
    main = next(n for n in training.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    smoke = next(n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == "args.mode == 'smoke'")
    smoke_text = ast.unparse(smoke)
    assert 'assert updated' in smoke_text and 'if steps == 3:' in smoke_text
    assert "model.load_state_dict(saved['model'], strict=True)" in smoke_text
    assert "optimizer.load_state_dict(saved['optimizer'])" in smoke_text
    assert 'assert torch.equal(before, after)' in smoke_text
    assert smoke_text.index('assert torch.equal(before, after)') < smoke_text.index("write_json(out / 'smoke.json', result)")
    loops = [n for n in ast.walk(main) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch']
    assert len(loops) == 1 and ast.unparse(loops[0].iter) == 'range(1, 51)'
    final_assert = next(n for n in main.body if isinstance(n, ast.Assert) and 'reloaded[k] - best[k]' in ast.unparse(n))
    final_write = next(n for n in main.body if isinstance(n, ast.Expr) and "write_json(out / 'result.json', terminal)" == ast.unparse(n))
    assert final_assert.lineno < final_write.lineno
    saves = [n for n in ast.walk(loops[0]) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.save']
    assert len(saves) == 1 and ast.unparse(saves[0].args[0]) == 'model.state_dict()'
    assert 'resume' not in {n.arg for n in ast.walk(main) if isinstance(n, ast.keyword)}
    return {'smoke_updates': 3, 'smoke_strict_model_optimizer_reload': True, 'train_epochs': 50, 'result_written_after_strict_reload_metric_equality': True, 'in_loop_checkpoint_model_only': True}


checked('run_experiment_actual_ast_smoke_fresh50_and_checkpoint_limits', training_source_semantics)


def observed_incident():
    rows = list(ROWS.values())
    completed = [row['name'] for row in rows if row.get('exit', {}).get('exit_code') == 0 and row.get('result', {}).get('status') == 'COMPLETE' and row['result']['epochs'] == 50]
    assert len(completed) == 8
    assert ROWS[FREQUENCY]['status']['epoch'] == 43 and 'exit' not in ROWS[FREQUENCY] and 'result' not in ROWS[FREQUENCY]
    assert ROWS[BASE]['result']['status'] == 'COMPLETE' and ROWS[BASE]['result']['epochs'] == 50 and 'exit' not in ROWS[BASE]
    assert RAW not in ROWS
    base = ROWS[BASE]['result']
    assert all(base['best'][k] == base['strict_reload'][k] for k in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10'))
    data = evidence['driver_evidence']['runs'][FREQUENCY]
    epochs = list(csv.DictReader(io.StringIO(data['epochs.csv']['tail'])))
    assert len(epochs) == 43 and int(epochs[-1]['optimizer_steps']) == 3471
    kernel_lines = [line for line in evidence['driver_evidence']['kernel']['selected_lines'] if 'Xid' in line and '79,' in line and 'GPU has fallen off the bus' in line]
    assert len(kernel_lines) == 1 and '07:17:14' in kernel_lines[0]
    assert 'State:\tD' in evidence['driver_evidence']['processes']['1574601']['status']
    return {'xid79_line': kernel_lines[0], 'normal_complete50_exit0_count': 8, 'interrupted_attempt': {'epochs_observed': 43, 'optimizer_steps_observed': 3471, 'sum_epoch_seconds': sum(float(row['seconds']) for row in epochs), 'no_exit_no_result': True}, 'base_complete50_strict_reload_without_exit': True, 'raw_not_started': True}


checked('actual_incident_state_and_original43_cost_separate', observed_incident)
checked('helper_unchanged_during_review', lambda: require(HELPER.read_bytes() == HELPER_BYTES, 'helper changed during review'))
for name, original in evidence_bytes.items():
    checked('evidence_unchanged_' + name, lambda name=name, original=original: require((REPO / ('results/preflight/axis_collaboration_v4_' + name + '.json')).read_bytes() == original, name))

result = {'helper_sha256': SHA, 'checks_total': len(RESULTS), 'checks_passed': sum(row['status'] == 'PASS' for row in RESULTS),
          'checks_failed': sum(row['status'] == 'FAIL' for row in RESULTS), 'checks': RESULTS,
          'generated_source_sha256': {name: hashlib.sha256(code.encode()).hexdigest() for name, code in scripts.items()},
          'evidence_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in evidence_bytes.items()},
          'actual_remote_calls': 0, 'actual_gpu_calls': 0, 'actual_signals': 0}
TRACE.mkdir(parents=True, exist_ok=True)
(TRACE / 'generated_scripts.json').write_text(json.dumps(scripts, indent=2), encoding='utf-8')
(TRACE / 'mock_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key not in ('checks', 'generated_source_sha256', 'evidence_sha256')}))
for row in RESULTS:
    if row['status'] == 'FAIL':
        print(json.dumps(row))
