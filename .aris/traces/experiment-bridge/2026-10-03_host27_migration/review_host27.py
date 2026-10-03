"""CPU/stdlib-only narrow host-27 migration review; all deployment I/O is virtual."""
from __future__ import annotations

import ast
import builtins
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import difflib
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import types

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TMP = Path('C:/Users/gb/.codex_tmp')
TRACE = PROJECT / '.aris/traces/experiment-bridge/2026-10-03_host27_migration'
PREFLIGHT = PROJECT / 'results/preflight'
SPEC_SHA = '4993690212b8f331ceac59c1cf8fe15751a69be89ffdcb2c046cdb7989cae5a0'
MODULE_SHA = '65f7eadb7fd0b05c939559626546dfe5f5686af6153958b98f10a5f552dbe91f'
TARGET_ROOT = '/data/gb/Re-ID/DeMo-DualAxis'
TARGET_PYTHON = '/data/gb/Re-ID/conda-envs/tri_reid/bin/python'
INPUT_ROOT = TARGET_ROOT + '/runs/axis_collaboration_v4_missing_development27_input'
TARGET_DATA = '/data/gb/Re-ID/dataset'
TARGET_CLIP = '/data/gb/Re-ID/pretrained/ViT-B-16.pt'
HOSTS = {
    '2025': ('/data2/gb/Re-ID/DeMo-DualAxis', '/data2/gb/Re-ID/conda-envs/tri_reid/bin/python'),
    '2026': ('/data/gaob/Re-ID/DeMo-DualAxis', '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'),
}
HELPERS = {
    'recovery': TMP / 'demo_axis_v4_hardware_recovery27_20261003.py',
    'missing': TMP / 'demo_axis_v4_missing_dev27_deploy_20261003.py',
}
OLD_HELPERS = {
    'recovery': TMP / 'demo_axis_v4_hardware_recovery26_20261003.py',
    'missing': TMP / 'demo_axis_v4_missing_dev26_deploy_20261003.py',
}
REPORT_NAMES = {
    'recovery': 'axis_collaboration_v4_hardware_recovery27_review.json',
    'missing': 'axis_collaboration_v4_missing_development27_review.json',
}
OLD_REPORT_NAMES = {
    'recovery': 'axis_collaboration_v4_hardware_recovery_review.json',
    'missing': 'axis_collaboration_v4_missing_development_review.json',
}
SOURCE = {kind: path.read_bytes() for kind, path in HELPERS.items()}
OLD_SOURCE = {kind: path.read_bytes() for kind, path in OLD_HELPERS.items()}
OLD_REPORTS = {kind: json.loads((PREFLIGHT/name).read_bytes()) for kind, name in OLD_REPORT_NAMES.items()}
PRIOR_BYTES = (PREFLIGHT/'axis_collaboration_v4_extra_cross26_launch.json').read_bytes()
PRIOR = json.loads(PRIOR_BYTES)
INCIDENT_BYTES = (PREFLIGHT/'axis_collaboration_v4_driver_evidence.json').read_bytes()
MODULE_BYTES = (PROJECT/'missing_development.py').read_bytes()
CORE_BYTES = {name: (PROJECT/name).read_bytes() for name in PRIOR['source_sha256']}
STAGE = json.loads((PREFLIGHT/'axis_collaboration_v4_code27_stage.json').read_bytes())
CLIP_STAGE = json.loads((PREFLIGHT/'axis_collaboration_v4_clip27_stage.json').read_bytes())
SPEC_BYTES = (PREFLIGHT/'axis_collaboration_v4_env27_spec.json').read_bytes()
SPEC = json.loads(SPEC_BYTES)
SOURCE_SHA = {kind: hashlib.sha256(value).hexdigest() for kind, value in SOURCE.items()}
CHECKS = {'recovery': [], 'missing': []}
RESULTS = {}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def verify(kind, name, condition, **evidence):
    CHECKS[kind].append(dict(name=name, status='PASS' if condition else 'FAIL', **evidence))


def encode(value):
    return json.dumps(value, sort_keys=True).encode()


class MemoryFS:
    def __init__(self, host, events):
        self.host, self.events = host, events
        self.files, self.dirs = {}, {'/'}

    def seed(self, path, content):
        path = str(PurePosixPath(path))
        self.files[path] = content
        self.dirs.update(str(parent) for parent in PurePosixPath(path).parents)

    def path(self, path):
        return MemoryPath(self, str(path))

    def event(self, operation, path, **kwargs):
        self.events.append(dict(host=self.host, operation=operation, path=str(path), **kwargs))


class MemoryPath:
    def __init__(self, fs, path):
        self.fs, self.value = fs, str(PurePosixPath(path))

    def __str__(self):
        return self.value

    def __repr__(self):
        return f'MemoryPath({self.fs.host!r}, {self.value!r})'

    def __truediv__(self, value):
        return MemoryPath(self.fs, str(PurePosixPath(self.value)/value))

    @property
    def parent(self):
        return MemoryPath(self.fs, str(PurePosixPath(self.value).parent))

    @property
    def name(self):
        return PurePosixPath(self.value).name

    def exists(self):
        self.fs.event('exists', self)
        return self.value in self.fs.files or self.value in self.fs.dirs

    def read_bytes(self):
        self.fs.event('read', self)
        if self.value not in self.fs.files:
            raise FileNotFoundError(self.value)
        return self.fs.files[self.value]

    def read_text(self, encoding='utf-8'):
        return self.read_bytes().decode(encoding)

    def stat(self):
        self.fs.event('stat', self)
        if self.value not in self.fs.files:
            raise FileNotFoundError(self.value)
        return types.SimpleNamespace(st_size=len(self.fs.files[self.value]))

    def write_bytes(self, content):
        assert self.parent.value in self.fs.dirs, self.parent
        self.fs.files[self.value] = content
        self.fs.event('write', self, bytes=len(content))
        return len(content)

    def write_text(self, content, encoding='utf-8'):
        return self.write_bytes(content.encode(encoding))

    def mkdir(self, exist_ok=False, parents=False):
        if self.value in self.fs.files or (self.value in self.fs.dirs and not exist_ok):
            raise FileExistsError(self.value)
        if not parents and self.parent.value not in self.fs.dirs:
            raise FileNotFoundError(self.parent.value)
        self.fs.dirs.add(self.value)
        if parents:
            self.fs.dirs.update(str(p) for p in PurePosixPath(self.value).parents)
        self.fs.event('mkdir', self, exist_ok=exist_ok)

    def open(self, mode='r', encoding='utf-8'):
        if mode == 'x':
            if self.value in self.fs.files:
                raise FileExistsError(self.value)
            self.write_bytes(b'')
            return io.StringIO()
        if mode == 'rb':
            return io.BytesIO(self.read_bytes())
        if mode == 'r':
            return io.StringIO(self.read_text(encoding))
        raise AssertionError(('unmocked file mode', mode))


class World:
    def __init__(self, kind, mutation=None, run_controller=False):
        self.kind, self.mutation = kind, mutation
        self.events, self.remote_calls, self.transfers, self.spawns = [], [], [], []
        self.generated, self.controllers, self.entry_calls = [], [], []
        self.fs = {h: MemoryFS(h, self.events) for h in ('local', '2025', '2026', '2027')}
        self.hosts = deepcopy(HOSTS)
        self.output = TARGET_ROOT + '/runs/axis_collaboration_v4_' + ('hardware_recovery27' if kind == 'recovery' else 'missing_development27')
        self.run_controller_after_helper = run_controller
        self.module = types.SimpleNamespace(PROJECT=self.fs['local'].path(str(PROJECT).replace('\\', '/')),
            HOSTS=self.hosts, OPTIONS=['-o', 'BatchMode=yes'], command=self.command, remote_python=self.remote_python)
        self._seed()

    def _seed(self):
        local = self.fs['local']
        base = str(PROJECT).replace('\\', '/') + '/results/preflight/'
        local.seed(base+'axis_collaboration_v4_extra_cross26_launch.json', PRIOR_BYTES)
        local.seed(base+'axis_collaboration_v4_driver_evidence.json', INCIDENT_BYTES)
        local.seed(base+REPORT_NAMES[self.kind], encode(dict(status='PASS', blockers=[], helper_sha256=SOURCE_SHA[self.kind], module_sha256=MODULE_SHA)))
        local.seed(str(HELPERS[self.kind]).replace('\\', '/'), SOURCE[self.kind])
        local.seed(str(PROJECT/'missing_development.py').replace('\\', '/'), MODULE_BYTES)
        target = self.fs['2027']
        target.dirs.update((TARGET_ROOT, TARGET_ROOT+'/runs', TARGET_ROOT+'/setup'))
        target.seed(TARGET_ROOT+'/setup/env27_doc_validation.json', encode(dict(status='PASS', spec_sha256=SPEC_SHA)))
        for name, content in CORE_BYTES.items():
            target.seed(TARGET_ROOT+'/'+name, content)
        self.expected_proofs = {}
        for dataset, variant, host, campaign in [
            ('MSVR310', 'demo', '2025', 'dynamic_amp_comparison'),
            ('RGBNT201', 'demo', '2025', 'dynamic_amp_comparison'),
            ('RGBNT100', 'demo', '2026', 'dynamic_amp_comparison'),
            *[(d, 'axis_scaled_fullref', '2026', 'axis_collaboration_v4_cross26_input') for d in ('MSVR310', 'RGBNT201', 'RGBNT100')],
        ]:
            name = dataset+'_'+variant+'_s42'
            original = HOSTS[host][0]+'/runs/'+campaign+'/'+name
            files = {
                'best.pth': ('checkpoint fixture '+host+' '+name).encode(),
                'best_dev_arrays.npz': ('arrays fixture '+host+' '+name).encode(),
                'result.json': encode(dict(status='COMPLETE', epochs=50, arguments=dict(dataset=dataset, variant=variant, seed=42))),
                'exit.json' if variant == 'demo' else name+'_exit.json': encode(dict(exit_code=0, source=name)),
            }
            for filename, value in files.items():
                path = original+'/'+filename if not filename.endswith('_exit.json') else original.rsplit('/', 1)[0]+'/'+filename
                self.fs[host].seed(path, value)
            self.expected_proofs[name] = dict(source_host=host, source_path=original,
                files={filename: dict(sha256=digest(value), bytes=len(value)) for filename, value in files.items()})
        old25 = self.fs['2025']
        old25.dirs.add('/proc/1408150')
        old25.seed('/proc/1408150/cmdline', b'python\x00launch_axis_scaled.py\x00--data-root\x00/data2/gb/Re-ID/dataset\x00')
        campaign = HOSTS['2025'][0]+'/runs/axis_collaboration_v4_development'
        old25.seed(campaign+'/RGBNT100_frequency_scaled_fullref_s42/status.json', encode(dict(epoch=43)))
        old25.seed(campaign+'/MSVR310_axis_scaled_base_s42/result.json', encode(dict(status='COMPLETE', epochs=50)))
        self.original_sources = {host: deepcopy(self.fs[host].files) for host in ('2025', '2026')}
        self.memories = {gpu: 0 for gpu in range(4)}
        doc_path = TARGET_ROOT+'/setup/env27_doc_validation.json'
        if self.mutation == 'missing_doc':
            del target.files[doc_path]
        elif self.mutation == 'not_pass_doc':
            target.files[doc_path] = encode(dict(status='TIER1_IMPORTS_TIER2_SEEDED_KERNEL_PASS_DOC_AGENT_PENDING', spec_sha256=SPEC_SHA))
        elif self.mutation == 'wrong_spec_doc':
            target.files[doc_path] = encode(dict(status='PASS', spec_sha256='0'*64))
        elif self.mutation == 'source_drift':
            target.files[TARGET_ROOT+'/'+next(iter(CORE_BYTES))] += b'changed target source'
        elif self.mutation and self.mutation.startswith('gpu_busy_'):
            self.memories[int(self.mutation.rsplit('_', 1)[1])] = 500
        elif self.mutation == 'other_gpus_busy':
            for gpu in ((2, 3) if self.kind == 'recovery' else (0, 1)):
                self.memories[gpu] = 24000

    def _subprocess(self, host):
        def check_output(argv, **kwargs):
            self.events.append(dict(host=host, operation='check_output', argv=argv))
            if argv[0] == 'nvidia-smi':
                assert host == '2027'
                return '\n'.join(f'{gpu}, {used}' for gpu, used in self.memories.items())+'\n'
            if argv == ['dmesg', '--ctime']:
                assert host == '2025'
                return 'NVRM: Xid (PCI:0000:3b:00): 79, GPU has fallen off the bus.\n'
            raise AssertionError(('unmocked command', argv))

        def popen(argv, **kwargs):
            assert host == '2027'
            call = dict(host=host, argv=list(argv), env=deepcopy(kwargs.get('env', {})),
                cwd=str(kwargs.get('cwd', '')), start_new_session=kwargs.get('start_new_session', False))
            self.spawns.append(call)
            self.events.append(dict(host=host, operation='spawn', argv=list(argv)))
            if '-c' in argv:
                assert argv[0] == TARGET_PYTHON
            elif argv[2] == 'run_experiment.py':
                out = argv[argv.index('--output')+1]
                mode = argv[argv.index('--mode')+1]
                if mode == 'smoke':
                    data = dict(status='SMOKE_PASS', steps=3, strict_reload_equal=True)
                    filename = 'smoke.json'
                else:
                    data = dict(status='COMPLETE', epochs=50)
                    filename = 'result.json'
                self.fs[host].seed(out+'/'+filename, encode(data))
            elif argv[2] == 'missing_development.py':
                out = argv[argv.index('--output')+1]
                smoke = '--smoke' in argv
                data = dict(status='PASS' if smoke else 'COMPLETE', optimizer_updates=0, normal_feature_max_error=0, measurements=list(range(13)))
                self.fs[host].seed(out+('/smoke.json' if smoke else '/result.json'), encode(data))
            else:
                raise AssertionError(('unexpected subprocess', argv))
            return types.SimpleNamespace(pid=900000+len(self.spawns), wait=lambda: 0)
        return types.SimpleNamespace(check_output=check_output, Popen=popen, STDOUT=-2)

    def _imports(self, host):
        mock_sys = types.SimpleNamespace(executable=TARGET_PYTHON, path=[])
        def kill(pid, sig):
            self.events.append(dict(host=host, operation='mock_signal', pid=pid, signal=sig))
        def idle(gpu):
            self.events.append(dict(host=host, operation='mock_idle', gpu=gpu))
        def write_json(path, value):
            path.write_bytes(encode(value))
        def run_path(path, run_name):
            self.entry_calls.append(dict(host=host, path=path, run_name=run_name, sys_path=list(mock_sys.path)))
        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            mapped = {
                'collect_results': self.module,
                'pathlib': types.SimpleNamespace(Path=self.fs[host].path),
                'subprocess': self._subprocess(host),
                'os': types.SimpleNamespace(environ={}, kill=kill),
                'signal': types.SimpleNamespace(SIGSTOP='SIGSTOP'),
                'sys': mock_sys,
                'launch_axis_scaled': types.SimpleNamespace(idle=idle),
                'launch_runs': types.SimpleNamespace(write_json=write_json),
                'runpy': types.SimpleNamespace(run_path=run_path),
            }
            if name in mapped:
                return mapped[name]
            assert name in {'datetime', 'hashlib', 'json', 'time', 'concurrent.futures'}, name
            return builtins.__import__(name, globals, locals, fromlist, level)
        return mock_import

    def execute(self, code, host, filename):
        ast.parse(code, filename=filename)
        bytecode = compile(code, filename, 'exec')
        sandbox_builtins = dict(vars(builtins), __import__=self._imports(host),
            open=lambda p, mode='r': self.fs[host].path(str(p)).open(mode))
        scope = dict(__builtins__=sandbox_builtins, __file__=filename, __name__='__main__')
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exec(bytecode, scope)
        return scope, stdout.getvalue()

    def command(self, argv):
        assert argv[0] == 'scp'
        self.transfers.append(list(argv))
        self.events.append(dict(host='local', operation='mock_scp', argv=list(argv)))
        source, dest = argv[-2:]
        dest_host, dest_path = dest.split(':', 1)
        assert dest_host == '2027'
        if '-3' in argv:
            source_host, source_path = source.split(':', 1)
            assert source_host in ('2025', '2026')
            value = self.fs[source_host].path(source_path).read_bytes()
        else:
            value = self.fs['local'].path(source.replace('\\', '/')).read_bytes()
        self.fs[dest_host].path(dest_path).write_bytes(value)
        return ''

    def remote_python(self, host, code):
        index = len(self.remote_calls)
        self.remote_calls.append(dict(host=host, code=code))
        self.generated.append(dict(host=host, label=f'remote_{index:02}', code=code))
        is_launch = 'child=subprocess.Popen' in code and 'output.mkdir' in code
        if is_launch and self.mutation and self.mutation.startswith(('corrupt_', 'move_exit_')):
            target = self.fs['2027']
            variant = 'demo' if self.mutation.endswith('_baseline') else 'axis_scaled_fullref'
            name = 'MSVR310_'+variant+'_s42'
            if self.mutation.startswith('corrupt_'):
                p = INPUT_ROOT+'/'+name+'/best.pth'
                value = target.files[p]
                target.files[p] = bytes([value[0] ^ 1])+value[1:]
            elif variant == 'demo':
                p = INPUT_ROOT+'/'+name+'/exit.json'
                target.files[INPUT_ROOT+'/'+name+'_exit.json'] = target.files.pop(p)
            else:
                p = INPUT_ROOT+'/'+name+'_exit.json'
                target.files[INPUT_ROOT+'/'+name+'/exit.json'] = target.files.pop(p)
        scope, stdout = self.execute(code, host, f'{self.kind}_remote_{index:02}.py')
        return stdout

    def run(self):
        self.error = None
        try:
            self.scope, self.stdout = self.execute(SOURCE[self.kind].decode('utf-8'), 'local', str(HELPERS[self.kind]).replace('\\', '/'))
            for key in ('controller_code', 'entry'):
                value = self.scope[key]
                ast.parse(value)
                compile(value, key+'.py', 'exec')
                self.generated.append(dict(host='2027', label=key, code=value))
            self.execute(self.scope['entry'], '2027', self.kind+'_entry.py')
            if self.run_controller_after_helper:
                self.execute(self.scope['controller_code'], '2027', self.kind+'_controller.py')
        except Exception as exc:
            self.error = dict(type=type(exc).__name__, message=str(exc))
        return self

    def mutations(self):
        return [event for event in self.events if event['operation'] in ('mkdir', 'write', 'spawn', 'mock_signal', 'mock_scp')]


def option(argv, name):
    return argv[argv.index(name)+1]


def guard_checks(kind):
    rows = []
    for case in ('missing_doc', 'not_pass_doc', 'wrong_spec_doc', 'source_drift'):
        w = World(kind, mutation=case).run()
        expected_error = 'FileNotFoundError' if case == 'missing_doc' else 'AssertionError'
        verify(kind, case+'_blocks_before_mutation', w.error and w.error['type'] == expected_error and not w.mutations(), error=w.error, remote_calls=len(w.remote_calls), mutations=len(w.mutations()))
        verify(kind, case+'_only_initial27_guard_runs', len(w.remote_calls) == 1 and w.remote_calls[0]['host'] == '2027' and not any(e['operation'] == 'check_output' for e in w.events))
        rows.append(dict(case=case, error=w.error, remote_hosts=[x['host'] for x in w.remote_calls], mutations=w.mutations()))
    for gpu in ((0, 1) if kind == 'recovery' else (2, 3)):
        w = World(kind, mutation='gpu_busy_'+str(gpu)).run()
        verify(kind, f'target_gpu{gpu}_500_blocks_before_mutation', w.error and w.error['type'] == 'AssertionError' and not w.mutations(), error=w.error)
        rows.append(dict(case='gpu_busy_'+str(gpu), error=w.error, mutations=w.mutations()))
    w = World(kind, mutation='other_gpus_busy').run()
    verify(kind, 'other_controller_gpus_do_not_block_this_pair', w.error is None and len(w.spawns) == 1, error=w.error)
    rows.append(dict(case='other_gpus_busy', error=w.error, controller_spawns=len(w.spawns)))
    return rows


def static_checks(kind):
    ast.parse(SOURCE[kind])
    compile(SOURCE[kind], str(HELPERS[kind]), 'exec')
    verify(kind, 'helper_ast_parse_and_compile', True)
    old = OLD_REPORTS[kind]
    verify(kind, 'prior_pass_is_for_exact_unchanged26_helper', old['status'] == 'PASS' and not old['blockers'] and old['helper_sha256'] == digest(OLD_SOURCE[kind]), prior_checks=117 if kind == 'recovery' else 80, previous_checks_reexecuted=False)
    verify(kind, 'canonical_spec_matches_new_guard', digest(json.dumps(SPEC, sort_keys=True, separators=(',', ':')).encode()) == SPEC_SHA and SPEC_SHA in SOURCE[kind].decode(), canonical_sha256=SPEC_SHA, pretty_file_sha256=digest(SPEC_BYTES))
    verify(kind, 'local_source14_and_stage_metadata_match_prior_manifest', len(CORE_BYTES) == 14 and all(digest(data) == PRIOR['source_sha256'][name] == STAGE['source_sha256'][name] for name, data in CORE_BYTES.items()), entries=len(CORE_BYTES), remote_reread=False)
    verify(kind, 'public_clip_stage_metadata_has_expected27_target', CLIP_STAGE['target'] == TARGET_CLIP and CLIP_STAGE['sha256'] == '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f', remote_rehash=False)
    if kind == 'missing':
        verify(kind, 'module_is_exact_prior80_pass_hash_no_model_review_repeated', digest(MODULE_BYTES) == MODULE_SHA == OLD_REPORTS['missing']['module_sha256'], module_sha256=MODULE_SHA)
        verify(kind, 'obsolete_same_host_copy_branch_absent', 'shutil' not in SOURCE[kind].decode() and "if host=='2026'" not in SOURCE[kind].decode())
    else:
        new_ast, old_ast = ast.parse(SOURCE[kind]), ast.parse(OLD_SOURCE[kind])
        def assignment(tree, name):
            return next(node.value for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))
        verify(kind, 'pause25_template_unchanged_from_prior117_pass', ast.dump(assignment(new_ast, 'pause25')) == ast.dump(assignment(old_ast, 'pause25')))


def positive_checks(kind):
    w = World(kind, run_controller=True).run()
    verify(kind, 'actual_helper_generated_scripts_and_controller_mock_execute', w.error is None, error=w.error)
    if w.error:
        return w
    verify(kind, 'new27_mapping_preserves25_26_source_mappings', w.hosts['2027'] == (TARGET_ROOT, TARGET_PYTHON) and all(w.hosts[h] == HOSTS[h] for h in ('2025', '2026')))
    first_guard = w.remote_calls[0]
    remote_events = [e for e in w.events if e['host'] != 'local']
    doc_read = next(i for i, e in enumerate(remote_events) if e['operation'] == 'read' and e['path'].endswith('/setup/env27_doc_validation.json'))
    gpu_query = next(i for i, e in enumerate(remote_events) if e['operation'] == 'check_output' and e['argv'][0] == 'nvidia-smi')
    core_reads = [i for i, e in enumerate(remote_events) if e['operation'] == 'read' and e.get('path') in {TARGET_ROOT+'/'+name for name in CORE_BYTES} and i < gpu_query]
    verify(kind, 'first27_guard_reads_doc_then_source14_then_gpu', first_guard['host'] == '2027' and doc_read == 0 and len(core_reads) == 14 and doc_read < min(core_reads) <= max(core_reads) < gpu_query, guard_source14_reads=len(core_reads))
    verify(kind, 'source25_26_bytes_not_changed', all(w.fs[h].files == w.original_sources[h] for h in ('2025', '2026')))
    launch_spawns = [s for s in w.spawns if '-c' in s['argv']]
    verify(kind, 'exactly_one_detached_controller_uses_target27_python_cwd', len(launch_spawns) == 1 and launch_spawns[0]['argv'][0] == TARGET_PYTHON and launch_spawns[0]['cwd'] == TARGET_ROOT and launch_spawns[0]['start_new_session'])
    verify(kind, 'entry_binds27_project_and_controller', w.entry_calls == [dict(host='2027', path=w.output+'/controller.py', run_name='__main__', sys_path=[TARGET_ROOT])])
    receipt_path = str(PROJECT).replace('\\', '/')+'/results/preflight/'+REPORT_NAMES[kind].replace('_review.json', '_launch.json')
    receipt = json.loads(w.fs['local'].files[receipt_path])
    verify(kind, 'receipt_host_output_and_availability_are27', receipt['host'] == '2027' and receipt['output'] == w.output and 'availability27' in receipt and 'availability26' not in receipt and 'PENDING' in receipt['status'])
    commands = [s for s in w.spawns if '-c' not in s['argv']]
    verify(kind, 'all_worker_commands_use27_env_data_clip', all(s['argv'][0] == TARGET_PYTHON and option(s['argv'], '--data-root') == TARGET_DATA and option(s['argv'], '--pretrained') == TARGET_CLIP for s in commands), worker_commands=len(commands))
    verify(kind, 'all_output_directories_remain_exclusive', all(e['exist_ok'] is False for e in w.events if e['operation'] == 'mkdir'))
    if kind == 'recovery':
        verify(kind, 'recovery_remote_hosts_are27_guard25_pause27_launch', [x['host'] for x in w.remote_calls] == ['2027', '2025', '2027'])
        signals = [e for e in w.events if e['operation'] == 'mock_signal']
        verify(kind, 'unchanged_pause_targets_only_mock25_pid1408150_sigstop', signals == [dict(host='2025', operation='mock_signal', pid=1408150, signal='SIGSTOP')])
        verify(kind, 'removed26_predecessor_paths_not_queried_on27', not any(e['operation'] in ('read', 'exists', 'stat') and '/data/gaob/' in e.get('path', '') for e in w.events if e['host'] == '2027'))
        verify(kind, 'recovery_gpu0_1_job_bindings_and_smoke_train_commands', len(commands) == 4 and sorted((option(s['argv'], '--dataset'), option(s['argv'], '--variant'), int(s['env']['CUDA_VISIBLE_DEVICES']), option(s['argv'], '--mode')) for s in commands) == sorted((dataset, variant, gpu, mode) for dataset, variant, gpu in [('RGBNT100','frequency_scaled_fullref',0),('MSVR310','axis_raw_fullref',1)] for mode in ('smoke','train')))
        verify(kind, 'recovery_seed42_contribution_and_fresh_stage_paths_unchanged', all(option(s['argv'],'--seed') == '42' and option(s['argv'],'--contribution-weight') == '.05' and option(s['argv'],'--output') == w.output+('/preflight/' if option(s['argv'],'--mode') == 'smoke' else '/development/')+option(s['argv'],'--dataset')+'_'+option(s['argv'],'--variant')+'_s42' for s in commands))
        verify(kind, 'recovery_receipt_retains_incident_and_new_attempt_metadata', receipt['original25_controller']['pid'] == 1408150 and receipt['original_interrupted_steps_not_counted_as_new_updates'] is True and receipt['incident_sha256'] == digest(INCIDENT_BYTES))
        verify(kind, 'recovery_does_not_upload_or_copy_source', not w.transfers)
    else:
        proof_calls = [call for call in w.remote_calls if "terminal=json.loads" in call['code']]
        bindings = []
        for call in proof_calls:
            tree = ast.parse(call['code'])
            assign = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'root' for t in n.targets))
            bindings.append((call['host'], ast.literal_eval(assign.value.args[0])))
        verify(kind, 'six_generated_proofs_bind_original25_25_26_and26_26_26', bindings == [(p['source_host'], p['source_path']) for p in w.expected_proofs.values()], bindings=bindings)
        verify(kind, 'six_original_proof_metadata_records_preserved', w.scope['proofs'] == w.expected_proofs and receipt['input_files'] == w.expected_proofs)
        scp3 = [a for a in w.transfers if '-3' in a]
        uploads = [a for a in w.transfers if '-3' not in a]
        expected_transfers = []
        for name, record in w.expected_proofs.items():
            for filename in record['files']:
                source = record['source_path']+'/'+filename
                dest = INPUT_ROOT+'/'+name+'/'+filename
                if filename.endswith('_exit.json'):
                    source = record['source_path'].rsplit('/',1)[0]+'/'+filename
                    dest = INPUT_ROOT+'/'+filename
                expected_transfers.append((record['source_host']+':'+source, '2027:'+dest))
        verify(kind, 'exactly24_scp3_transfers_from25_26_to_unique27_paths', len(scp3) == 24 and [tuple(a[-2:]) for a in scp3] == expected_transfers and len({a[-1] for a in scp3}) == 24, source_host_counts=dict(Counter(a[-2].split(':')[0] for a in scp3)), transfer_count=len(scp3))
        verify(kind, 'only_new_module_is_uploaded', len(uploads) == 1 and uploads[0][-2].replace('\\','/') == str(PROJECT/'missing_development.py').replace('\\','/') and uploads[0][-1] == '2027:'+TARGET_ROOT+'/missing_development.py', source_upload_count=len(uploads))
        for name, record in w.expected_proofs.items():
            expected_paths = [(INPUT_ROOT+'/'+filename if filename.endswith('_exit.json') else INPUT_ROOT+'/'+name+'/'+filename, proof) for filename, proof in record['files'].items()]
            verify(kind, 'copied_input_and_exit_layout_'+name, all(p in w.fs['2027'].files and digest(w.fs['2027'].files[p]) == proof['sha256'] and len(w.fs['2027'].files[p]) == proof['bytes'] for p, proof in expected_paths), paths=[p for p,_ in expected_paths])
        launch_start = next(i for i,e in enumerate(w.events) if e['operation'] == 'mkdir' and e['path'] == w.output)
        last_scp = max(i for i,e in enumerate(w.events) if e['operation'] == 'mock_scp')
        launch_reads = [e['path'] for e in w.events[last_scp+1:launch_start] if e['host'] == '2027' and e['operation'] == 'read']
        verify(kind, 'launch_rechecks15_target_sources_and_all24_target_inputs', len(launch_reads) == 39 and set(launch_reads) == {TARGET_ROOT+'/'+name for name in (*CORE_BYTES, 'missing_development.py')} | {dest.split(':',1)[1] for _,dest in expected_transfers}, reads=len(launch_reads))
        verify(kind, 'no_original_source_path_is_used_for_target27_intake', not any(e['operation'] in ('read','stat','exists') and '/data/gaob/' in e.get('path','') for e in w.events if e['host'] == '2027'))
        verify(kind, 'six_smokes_and_six_full_commands_all_read_new27_inputs', len(commands) == 12 and Counter((option(s['argv'],'--run-dir'), '--smoke' in s['argv']) for s in commands) == Counter((INPUT_ROOT+'/'+name, smoke) for name in w.expected_proofs for smoke in (True,False)))
        verify(kind, 'gpu2_axis_gpu3_demo_three_dataset_worker_bindings', all(int(s['env']['CUDA_VISIBLE_DEVICES']) == (3 if option(s['argv'],'--run-dir').endswith('_demo_s42') else 2) for s in commands) and {int(s['env']['CUDA_VISIBLE_DEVICES']) for s in commands} == {2,3})
        for gpu, variant in ((2,'axis_scaled_fullref'), (3,'demo')):
            seq = [(option(s['argv'],'--run-dir').rsplit('/',1)[1], '--smoke' in s['argv']) for s in commands if int(s['env']['CUDA_VISIBLE_DEVICES']) == gpu]
            verify(kind, f'gpu{gpu}_per_dataset_smoke_full_order', seq == [(dataset+'_'+variant+'_s42', smoke) for dataset in ('MSVR310','RGBNT201','RGBNT100') for smoke in (True, False)])
        verify(kind, 'receipt_source_manifest_contains_only_prior14_plus_new_module', receipt['source_sha256'] == {**PRIOR['source_sha256'], 'missing_development.py':MODULE_SHA})
    RESULTS[kind] = dict(
        helper_stdout=w.stdout,
        remote_hosts=[x['host'] for x in w.remote_calls],
        generated_script_count=len(w.generated),
        generated_script_sha256={r['label']:digest(r['code'].encode()) for r in w.generated},
        transfers=w.transfers,
        spawns=w.spawns,
        entry_calls=w.entry_calls,
        receipt=receipt,
        events=w.events,
    )
    return w


def migration_input_failure_checks():
    cases = []
    for mutation in ('corrupt_baseline', 'corrupt_axis', 'move_exit_baseline', 'move_exit_axis'):
        w = World('missing', mutation=mutation).run()
        expected = 'AssertionError' if mutation.startswith('corrupt') else 'FileNotFoundError'
        verify('missing', mutation+'_blocks_before_controller_output_or_spawn', w.error and w.error['type'] == expected and not w.spawns and w.output not in w.fs['2027'].dirs, error=w.error, mock_transfers=len(w.transfers))
        cases.append(dict(case=mutation, error=w.error, spawns=len(w.spawns), output_exists=w.output in w.fs['2027'].dirs))
    return cases


def main():
    positives = {}
    for kind in ('recovery', 'missing'):
        static_checks(kind)
        positives[kind] = positive_checks(kind)
        RESULTS.setdefault(kind, {})['negative_guard_cases'] = guard_checks(kind)
    RESULTS['missing']['input_failure_cases'] = migration_input_failure_checks()
    for kind, w in positives.items():
        if w.error is None:
            other = positives['missing' if kind == 'recovery' else 'recovery']
            left = {int(s['env']['CUDA_VISIBLE_DEVICES']) for s in w.spawns if '-c' not in s['argv']}
            right = {int(s['env']['CUDA_VISIBLE_DEVICES']) for s in other.spawns if '-c' not in s['argv']}
            verify(kind, 'controllers_use_disjoint_gpu_pairs_union_four', left.isdisjoint(right) and left | right == {0,1,2,3}, this_gpu_set=sorted(left), other_gpu_set=sorted(right))
            verify(kind, 'all_generated_instances_compiled_including_each_loop_binding', all(compile(row['code'], row['label']+'.py','exec') for row in w.generated), generated_instances=len(w.generated))
        verify(kind, 'helper_and_model_module_bytes_unchanged_after_review', HELPERS[kind].read_bytes() == SOURCE[kind] and (PROJECT/'missing_development.py').read_bytes() == MODULE_BYTES)
        diff = ''.join(difflib.unified_diff(OLD_SOURCE[kind].decode().splitlines(True), SOURCE[kind].decode().splitlines(True), fromfile=str(OLD_HELPERS[kind]), tofile=str(HELPERS[kind])))
        (TRACE/(kind+'_migration.diff')).write_text(diff, encoding='utf-8')
        (TRACE/(kind+'_generated_scripts.json')).write_text(json.dumps(w.generated, indent=2), encoding='utf-8')
    result = dict(status='PASS' if all(c['status']=='PASS' for rows in CHECKS.values() for c in rows) else 'BLOCKED',
        actual_python='stdlib CPython 3.13', checks=CHECKS, results=RESULTS,
        boundary=dict(actual_ssh=0, actual_scp=0, actual_gpu=0, actual_signals=0, actual_remote_spawns=0, actual_training=0,
            production_source_edits=0, human_document_edits=0, third_party_imports=0,
            all_remote_and_helper_local_writes='memory-only fixtures'))
    (TRACE/'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], checks={k:dict(total=len(v),passed=sum(c['status']=='PASS' for c in v),failed=sum(c['status']=='FAIL' for c in v)) for k,v in CHECKS.items()}, failures={k:[c for c in v if c['status']=='FAIL'] for k,v in CHECKS.items()}),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
