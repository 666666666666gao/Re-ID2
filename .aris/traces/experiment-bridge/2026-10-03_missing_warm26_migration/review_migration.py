"""Narrow warm-host migration review. Only stdlib AST and isolated in-memory mocks."""
import ast
import copy
from datetime import datetime, timezone
import difflib
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import traceback
from types import SimpleNamespace

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = PROJECT / '.aris/traces/experiment-bridge/2026-10-03_missing_warm26_migration'
PREFIX = PROJECT / 'results/preflight'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_v4_missing_dev26_repaired_deploy_20261003.py')
OLD_HELPER = HELPER.with_name('demo_axis_v4_missing_dev27_repaired_deploy_20261003.py')
MODULE = PROJECT / 'missing_development.py'
REPORT = PREFIX / 'axis_collaboration_v4_missing_development26_repaired_review.json'
prior = json.loads((PREFIX / 'axis_collaboration_v4_missing_development27_launch.json').read_text(encoding='utf-8'))
failure = json.loads((PREFIX / 'axis_collaboration_v4_missing_development27_failure_snapshot.json').read_text(encoding='utf-8'))
previous_review = json.loads((PREFIX / 'axis_collaboration_v4_missing_development27_repaired_review.json').read_text(encoding='utf-8'))
helper_source, old_source = HELPER.read_text(encoding='utf-8'), OLD_HELPER.read_text(encoding='utf-8')
helper_tree = ast.parse(helper_source)
helper_hash = hashlib.sha256(HELPER.read_bytes()).hexdigest()
module_hash = hashlib.sha256(MODULE.read_bytes()).hexdigest()
root26 = '/data/gaob/Re-ID/DeMo-DualAxis'
root27 = '/data/gb/Re-ID/DeMo-DualAxis'
output26 = root26 + '/runs/axis_collaboration_v4_missing_development26_repaired'
inputs = {n: r for n, r in prior['input_files'].items() if r['source_host'] == '2026'}
run_paths = {n: r['source_path'] for n, r in inputs.items()}
collect_tree = ast.parse((PROJECT / 'collect_results.py').read_text(encoding='utf-8'))
hosts = ast.literal_eval(next(n.value for n in collect_tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'HOSTS' for t in n.targets)))
checks, errors = [], []

# Reuse only reviewed, side-effect-free mock infrastructure; none of the old 25 checks execute.
support_path = PROJECT / '.aris/traces/experiment-bridge/2026-10-03_missing_distance_repair/review_delta_mock.py'
support_tree = ast.parse(support_path.read_text(encoding='utf-8'))
support_names = {'require', 'rejects', 'execute', 'strip_imports', 'FS', 'VPath', 'Pool'}
support = [n for n in support_tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in support_names]
exec(compile(ast.Module(body=support, type_ignores=[]), str(support_path), 'exec'), globals())

def proof_path(name, filename):
    base = PurePosixPath(run_paths[name])
    return str(base.parent / filename if filename.endswith('_exit.json') else base / filename)

class Harness:
    def __init__(self):
        self.local, self.remotes = FS(), {'2026': FS(), '2027': FS()}
        self.events, self.commands, self.spawns, self.codes, self.read_stages = [], [], [], [], []
        self.live, self.used = False, {0: 15, 1: 60, 2: 12591, 3: 153}
        self.after_upload = lambda: None
        self.review = dict(status='PASS', blockers=[], helper_sha256=helper_hash, module_sha256=module_hash)
        for name, value in [('axis_collaboration_v4_missing_development27_launch.json', prior), ('axis_collaboration_v4_missing_development27_failure_snapshot.json', failure)]:
            self.local.put_json((PREFIX / name).as_posix(), value)
        self.local.put(MODULE.as_posix(), MODULE.read_bytes())
        for name in prior['source_sha256']:
            old = (PREFIX / 'axis_collaboration_v4_missing_development_original.py').read_bytes() if name == 'missing_development.py' else (PROJECT / name).read_bytes()
            self.remotes['2027'].put(root27 + '/' + name, old)
            if name != 'missing_development.py':
                self.remotes['2026'].put(root26 + '/' + name, old)
        for name, record in inputs.items():
            for filename, proof in record['files'].items():
                value = json.dumps({'status': 'COMPLETE', 'epochs': 50}).encode() if filename == 'result.json' else json.dumps({'exit_code': 0}).encode() if filename.endswith('exit.json') else ('input:' + name + ':' + filename).encode()
                self.put_proof(name, filename, value)
        for row in failure['rows']:
            base = prior['output'] + '/' + row['name']
            if row.get('exit'):
                self.remotes['2027'].put_json(base + '/' + row['stage'] + '_exit.json', row['exit'])
            if row.get('result'):
                self.remotes['2027'].put_json(base + '/' + row['stage'] + ('/result.json' if row['stage'] == 'full' else '/smoke.json'), row['result'])
    def put_proof(self, name, filename, value):
        fs = self.remotes['2026']
        path, proof = proof_path(name, filename), inputs[name]['files'][filename]
        # Include identifying whitespace so mocked result/exit contents cannot share a digest key.
        if filename.endswith('.json'):
            value += b' ' * (list(inputs).index(name) + 1)
        fs.put(path, value)
        fs.sizes[path] = proof['bytes']
        fs.digest_map[value] = proof['sha256']
    def remote_python(self, host, code):
        phase = ['reuse_guard', 'guard26', 'launch26'][len(self.codes)]
        self.events.append((phase, host))
        self.codes.append((host, code))
        fs, output = self.remotes[host], []
        start = len(fs.reads)
        def ps(argv, **kwargs):
            require(host == '2027' and argv == ['ps', '-p', str(prior['pid']), '-o', 'pid='])
            return SimpleNamespace(stdout='alive' if self.live else '')
        def gpu(argv, **kwargs):
            require(host == '2026' and argv[0] == 'nvidia-smi')
            return ''.join(f'{i}, {v}\n' for i, v in self.used.items())
        ns = dict(hashlib=SimpleNamespace(sha256=fs.hash), json=json, Path=fs.path,
                  subprocess=SimpleNamespace(run=ps, check_output=gpu, Popen=self.spawn, STDOUT=-2),
                  os=SimpleNamespace(environ={}), time=SimpleNamespace(time=lambda: 1.), print=output.append)
        execute(strip_imports(ast.parse(code)), ns)
        self.read_stages.append((phase, host, fs.reads[start:]))
        return output[-1]
    def spawn(self, argv, **kwargs):
        self.spawns.append((argv, kwargs))
        return SimpleNamespace(pid=1234)
    def command(self, argv):
        self.commands.append(argv)
        self.events.append(('upload', '2026'))
        require(argv == ['scp', 'MOCK_OPTIONS', MODULE.as_posix(), '2026:' + root26 + '/missing_development.py'])
        fs = self.remotes['2026']
        fs.put(root26 + '/missing_development.py', MODULE.read_bytes())
        fs.writes.append(root26 + '/missing_development.py')
        self.after_upload()
        return ''
    def run(self):
        self.local.put_json(REPORT.as_posix(), self.review)
        ns = dict(datetime=datetime, hashlib=hashlib, json=json, sys=SimpleNamespace(path=[]), PROJECT=self.local.path(PROJECT.as_posix()),
                  HOSTS=copy.deepcopy(hosts), OPTIONS=['MOCK_OPTIONS'], command=self.command, remote_python=self.remote_python,
                  __file__=str(HELPER), open=lambda *a: io.BytesIO(HELPER.read_bytes()), print=lambda *a, **k: None)
        execute(strip_imports(helper_tree), ns)
        self.ns = ns
        return self

def check(name, fn):
    try:
        detail = fn()
        row = dict(name=name, status='PASS', detail=detail)
    except Exception as exc:
        row = dict(name=name, status='FAIL', detail=f'{type(exc).__name__}: {exc}')
        errors.append(traceback.format_exc())
    checks.append(row)
    print(json.dumps(row), flush=True)

def no_mutation(h):
    require(not h.commands and not h.spawns and all(not fs.writes for fs in h.remotes.values()) and not h.local.writes)

def prerequisite():
    require(previous_review['status'] == 'PASS' and previous_review['checks_count'] == 25)
    require(previous_review['helper_sha256'] == hashlib.sha256(OLD_HELPER.read_bytes()).hexdigest())
    require(module_hash == previous_review['module_sha256'] == '9cbb3440b61ecf3bddfe88bf071328632c9ad3903b2d02aa0bf5a36d57cf8973')
    require(len(prior['source_sha256']) == 15)
    require(all(hashlib.sha256((PROJECT / n).read_bytes()).hexdigest() == s for n, s in prior['source_sha256'].items() if n != 'missing_development.py'))
    return 'Prior 25-check PASS binds unchanged helper/module; the other 14 source hashes remain exact. No previous suite or module review rerun.'

def resource_evidence():
    inv = json.loads((PREFIX / 'axis_collaboration_v4_next_resource_inventory.json').read_text(encoding='utf-8'))
    gpu = {int(r.split(',')[0]): int(r.split(',')[-1]) for r in inv['2026']['gpu'].strip().splitlines()}
    rp = json.loads((PREFIX / 'axis_collaboration_v4_missing_development27_repaired_resource_probe.json').read_text(encoding='utf-8'))
    gp = json.loads((PREFIX / 'axis_collaboration_v4_missing_development27_repaired_guard_probe.json').read_text(encoding='utf-8'))
    require(gpu[0] < 500 and gpu[1] < 500 and gp['returncode'] == 1 and not rp['new_output_exists'])
    require(rp['old_module_sha256'] == prior['source_sha256']['missing_development.py'])
    require(sum('/envs/gola/bin/python' in r['executable'] for r in rp['process_entries']) == 3)
    require(hosts['2026'] == (root26, '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'))
    return {'inventory_at': inv['observed_at'], 'selected_gpu_memory_MiB': [gpu[0], gpu[1]], 'old27_resource_guard_failed_without_upload': True}

happy = None
def happy_sequence():
    global happy
    happy = Harness().run()
    require(happy.events == [('reuse_guard', '2027'), ('guard26', '2026'), ('upload', '2026'), ('launch26', '2026')])
    require(happy.ns['run_paths'] == run_paths and len(run_paths) == 4)
    require(len(happy.spawns) == 1)
    argv, kw = happy.spawns[0]
    require(argv[:3] == [hosts['2026'][1], '-u', '-c'] and str(kw['cwd']) == root26 and kw['start_new_session'])
    receipt = happy.ns['launch']
    require(receipt['host'] == '2026' and receipt['input_files'] == inputs and receipt['reused_host'] == '2027')
    return 'Actual helper AST: read-only 2027 reuse check -> 2026 guard -> one module upload -> one 2026 controller spawn with original warm interpreter/cwd.'

def host_paths():
    expected = {proof_path(n, f) for n, r in inputs.items() for f in r['files']}
    require(len(expected) == 16)
    for stage, host, reads in happy.read_stages:
        if host == '2026':
            require(expected <= set(reads))
            require(all(p.startswith(root26 + '/') for p in reads))
        else:
            require(all(p.startswith(root27 + '/') for p in reads))
    require(not happy.remotes['2027'].writes)
    require(happy.ns['source_sha'] == {**prior['source_sha256'], 'missing_development.py': module_hash})
    require(happy.ns['availability']['existing_input_proof_count'] == 16)
    require(all(r['source_host'] == '2026' and PurePosixPath(r['source_path']).is_absolute() for r in inputs.values()))
    return 'All 16 proofs read on physical host 2026 twice; baseline exit inside run; three axis exits outside. Reused 2027 results never enter 2026 input reads.'

def only_module_copy():
    require(len(happy.commands) == 1 and happy.commands[0][-2].endswith('/missing_development.py'))
    require(all(p in (root26 + '/missing_development.py', output26) or p.startswith(output26 + '/') for p in happy.remotes['2026'].writes))
    require(not happy.remotes['2027'].writes)
    return 'Exactly one SCP of patched missing_development.py; zero checkpoint/feature/result copies; writes confined to that module and new 2026 output.'

def resource_reject(gpu):
    h = Harness(); h.used[gpu] = 500
    rejects(h.run); no_mutation(h)
    require(h.events == [('reuse_guard', '2027'), ('guard26', '2026')])
    return f'2026 GPU{gpu}=500 MiB rejects before any upload, directory, receipt or Popen.'

def new_path_rejects():
    for path in (root26 + '/missing_development.py', output26):
        h = Harness(); h.remotes['2026'].dirs.add(path)
        rejects(h.run); no_mutation(h)
    return 'Existing module or existing output on 2026 blocks all mutations (2 cases).'

def terminal_exit_rejects():
    trials = 0
    for name, record in inputs.items():
        for field, bad in [('status', 'RUNNING'), ('epochs', 49)]:
            h = Harness(); doc = dict(status='COMPLETE', epochs=50); doc[field] = bad
            h.put_proof(name, 'result.json', json.dumps(doc).encode())
            rejects(h.run); no_mutation(h); trials += 1
        filename = next(f for f in record['files'] if f.endswith('exit.json'))
        h = Harness(); h.put_proof(name, filename, b'{"exit_code":1}')
        rejects(h.run); no_mutation(h); trials += 1
    return f'{trials} migration-path terminal checks: each original run must be COMPLETE/50 epochs/exit0 at its native exit location.'

def new_path_proofs():
    # These paths changed in this delta; each proof is tested once, not the prior 24-file suite.
    for name, record in inputs.items():
        for filename in record['files']:
            h = Harness(); h.remotes['2026'].sizes[proof_path(name, filename)] += 1
            rejects(h.run); no_mutation(h)
    return 'All 16 newly bound source-path size proofs individually reject before mutation.'

def reuse_host_rejects():
    h = Harness(); h.live = True; rejects(h.run); no_mutation(h)
    h = Harness(); p = prior['output'] + '/MSVR310_demo_s42/full_exit.json'
    h.remotes['2027'].put_json(p, {'exit_code': 1}); rejects(h.run); no_mutation(h)
    require(h.events == [('reuse_guard', '2027')])
    return 'Live old controller or invalid reused baseline on 2027 prevents reaching 2026; 2027 remains read-only.'

def source26_reject():
    h = Harness(); h.remotes['2026'].files[root26 + '/run_experiment.py'] += b'x'
    rejects(h.run); no_mutation(h)
    return 'Changed original source on target 2026 rejects before upload.'

def launch_recheck():
    for path in (root26 + '/missing_development.py', proof_path('RGBNT100_demo_s42', 'best.pth')):
        h = Harness()
        def change(): h.remotes['2026'].files[path] += b'x'
        h.after_upload = change
        rejects(h.run)
        require(not h.spawns and h.remotes['2026'].writes == [root26 + '/missing_development.py'])
        require(not h.remotes['2027'].writes)
    return 'Post-upload module or native input drift blocks output/Popen at repeated 2026 proof gate (2 cases).'

def controller(fault=None, slot=None):
    fs, events = FS(), []
    fs.dirs.add(output26)
    def popen(argv, **kw):
        stage = 'smoke' if '--smoke' in argv else 'full'
        run = argv[argv.index('--run-dir') + 1]
        name = PurePosixPath(run).name
        gpu = int(kw['env']['CUDA_VISIBLE_DEVICES'])
        require(run == run_paths[name])
        require(argv[argv.index('--data-root') + 1] == '/data/gaob/Re-ID/dataset')
        require(argv[argv.index('--pretrained') + 1] == '/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt')
        require(argv[:3] == [hosts['2026'][1], '-u', 'missing_development.py'])
        events.append(('spawn', gpu, name, stage))
        def wait():
            events.append(('wait', gpu, name, stage))
            out = argv[argv.index('--output') + 1]
            fs.put_json(out + ('/smoke.json' if stage == 'smoke' else '/result.json'), dict(status='PASS' if stage == 'smoke' else 'COMPLETE', optimizer_updates=0, normal_feature_max_error=0, measurements=dict.fromkeys(range(13)), normal_distance_source='saved best_dev_arrays.npz/distances'))
            return 1 if fault == (gpu, stage) else 0
        return SimpleNamespace(pid=99, wait=wait)
    ns = dict(ThreadPoolExecutor=Pool, json=json, os=SimpleNamespace(environ={}), subprocess=SimpleNamespace(Popen=popen, STDOUT=-2),
              sys=SimpleNamespace(executable=hosts['2026'][1]), time=SimpleNamespace(time=lambda: 1.), Path=fs.path,
              idle=lambda gpu: events.append(('idle', gpu)), write_json=lambda path, value: fs.put_json(str(path), value))
    tree = ast.parse(happy.ns['controller'])
    failure = None
    try:
        if slot is None: execute(strip_imports(tree), ns)
        else:
            execute([n for n in strip_imports(tree) if isinstance(n, (ast.Assign, ast.FunctionDef))], ns)
            ns['slot'](slot)
    except AssertionError as exc:
        failure = str(exc)
    return fs, events, failure

def controller_schedule():
    fs, events, error = controller()
    require(error is None)
    expected = []
    expected_jobs = {0: [('MSVR310', 'axis_scaled_fullref'), ('RGBNT201', 'axis_scaled_fullref')], 1: [('RGBNT100', 'demo'), ('RGBNT100', 'axis_scaled_fullref')]}
    require(happy.ns['jobs'] == expected_jobs)
    for gpu, jobs in expected_jobs.items():
        for dataset, variant in jobs:
            for stage in ('smoke', 'full'):
                name = dataset + '_' + variant + '_s42'
                expected.extend([('idle', gpu), ('spawn', gpu, name, stage), ('wait', gpu, name, stage)])
    require(events == expected)
    result = json.loads(fs.files[output26 + '/controller_result.json'])
    require(result['status'] == 'COMPLETE' and len(result['runs']) == 4 and result['conditions'] == 78 and result['new_conditions'] == 52 and result['reused_conditions'] == 26)
    require(result['reused_host'] == '2027' and result['reused_output'] == prior['output'] and result['reused_complete'] == ['MSVR310_demo_s42', 'RGBNT201_demo_s42'])
    require(all(p.startswith(output26 + '/') for p in fs.files))
    return 'GPU0 MSVR->RGB201 axis; GPU1 RGB100 demo->axis. All 8 stages idle->spawn->wait, smoke before full; COMPLETE binds 52 new + 26 reused on host2027.'

def controller_failures():
    for gpu in (0, 1):
        for stage in ('smoke', 'full'):
            fs, events, error = controller((gpu, stage), gpu)
            require(error is not None)
            require(sum(e[0] == 'spawn' for e in events) == (1 if stage == 'smoke' else 2))
            require(output26 + '/controller_result.json' not in fs.files)
    return '4 failures on migrated slots stop before next stage/job and never write aggregate COMPLETE; other submitted slot retains previous ThreadPoolExecutor semantics.'

def unchanged_controller_flow():
    # Evaluate only assignments needed to render old controller, with no helper entrypoint or old tests.
    keep = []
    wanted = {'root', 'python', 'inputs', 'output', 'reused', 'jobs', 'controller'}
    for n in ast.parse(old_source).body:
        if isinstance(n, ast.Assign):
            names = {t.id for target in n.targets for t in ast.walk(target) if isinstance(t, ast.Name)}
            if names & wanted: keep.append(n)
    old_ns = {'HOSTS': {'2027': (root27, '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')}, 'prior': prior}
    execute(keep, old_ns)
    class Normalize(ast.NodeTransformer):
        def visit_Assign(self, node):
            names = {t.id for target in node.targets for t in ast.walk(target) if isinstance(t, ast.Name)}
            if names == {'root'}: node.value = ast.Constant('ROOT')
            if names == {'argv'}:
                # Only these three CLI values changed: run path, data provider, pretrained provider.
                for i in (4, 8, 10): node.value.elts[i] = ast.Constant('PROVIDER')
            return self.generic_visit(node)
        def visit_Dict(self, node):
            if node.keys and all(isinstance(k, ast.Constant) and isinstance(k.value, int) for k in node.keys):
                node.keys = [ast.Constant(i) for i in range(len(node.keys))]
            return self.generic_visit(node)
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'map': node.args[1] = ast.List(elts=[ast.Constant(0), ast.Constant(1)], ctx=ast.Load())
            node.keywords = [k for k in node.keywords if k.arg != 'reused_host']
            return self.generic_visit(node)
    a = Normalize().visit(ast.parse(old_ns['controller']))
    b = Normalize().visit(ast.parse(happy.ns['controller']))
    require(ast.dump(a) == ast.dump(b), 'unexpected controller-flow change outside provider, GPU numbers or reused_host')
    return 'Generated old/new controller ASTs identical after normalizing only root, 3 provider CLI values, GPU indices and new reused_host field. All waits, result gates, and 2-worker execution inherited unchanged.'

def syntax_scope():
    codes = [helper_source] + [happy.ns[k] for k in ('reuse_guard', 'guard', 'controller', 'entry', 'launch_code')]
    for code in codes:
        tree = ast.parse(code, feature_version=(3, 10)); compile(tree, '<migration review>', 'exec')
        require(not any(isinstance(n, ast.Try) for n in ast.walk(tree)))
    require(all(s not in helper_source for s in ('pip install', 'conda install', 'os.kill', 'SIGTERM', 'SIGKILL')))
    return 'Helper plus all 5 generated scripts compile for Python3.10; no environment setup/install, signal, additional exception machinery, or unrelated source write.'

check('prior_review_and_pinned_unchanged_module', prerequisite)
check('actual_capacity_and_warm_provider_evidence', resource_evidence)
check('physical_hosts_guard_upload_launch_order', happy_sequence)
check('native_absolute_paths_16_proofs_and_reused_host_isolation', host_paths)
check('zero_weight_copy_and_narrow_write_scope', only_module_copy)
check('gpu0_resource_reject_zero_mutation', lambda: resource_reject(0))
check('gpu1_resource_reject_zero_mutation', lambda: resource_reject(1))
check('target_module_and_output_absence_guards', new_path_rejects)
check('four_native_COMPLETE50_exit0_guards', terminal_exit_rejects)
check('all_16_migrated_input_proof_paths_reject_changes', new_path_proofs)
check('old2027_reuse_guards_block_target_mutation', reuse_host_rejects)
check('target2026_original_source_guard', source26_reject)
check('postupload_target_source_and_input_rechecks', launch_recheck)
check('dual_gpu_order_wait_and_cross_host_COMPLETE_schema', controller_schedule)
check('dual_gpu_failure_stops_own_slot', controller_failures)
check('controller_only_provider_migration_AST_delta', unchanged_controller_flow)
check('all_generated_python310_AST_compile_and_scope', syntax_scope)

blockers = [r for r in checks if r['status'] != 'PASS']
report = dict(status='BLOCKED' if blockers else 'PASS', blockers=blockers, helper_sha256=helper_hash, module_sha256=module_hash,
              review_independence='same-family', acceptance_status='provisional', reviewer_model='gpt-6-astra', reasoning_effort='max',
              scope='Only host/provider migration from repaired2027 helper to warm2026 helper. Prior80/51/25 checks and unchanged patched module not re-reviewed.',
              checks_count=len(checks), passed_checks=sum(r['status'] == 'PASS' for r in checks), checks=checks,
              trace_path=str(TRACE), harness=str(TRACE / 'review_migration.py'),
              execution=dict(stdlib_only=True, python=sys.version.split()[0], real_ssh_calls=0, real_gpu_calls=0, real_signals=0, package_installs=0, project_source_edits=0, human_doc_edits=0),
              limitations=['AST and in-memory controlled mocks only; actual deployment must pass live guards and all four smoke/full stages.', 'Old25 PASS remains same-family provisional; no new module metric or numerical parity acceptance inferred.', 'Failure stops its own GPU slot and blocks aggregate COMPLETE; the other submitted slot is not killed.'],
              observed_at=datetime.now(timezone.utc).isoformat())
TRACE.mkdir(parents=True, exist_ok=True)
(TRACE / 'checks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in checks), encoding='utf-8')
(TRACE / 'harness_errors.txt').write_text('\n'.join(errors), encoding='utf-8')
(TRACE / 'migration.diff').write_text(''.join(difflib.unified_diff(old_source.splitlines(True), helper_source.splitlines(True), fromfile=str(OLD_HELPER), tofile=str(HELPER))), encoding='utf-8')
(TRACE / 'generated_scripts.json').write_text(json.dumps({k: happy.ns[k] for k in ('reuse_guard', 'guard', 'controller', 'entry', 'launch_code')}, indent=2), encoding='utf-8')
(TRACE / 'review.result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
REPORT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: report[k] for k in ('status', 'checks_count', 'passed_checks', 'helper_sha256', 'module_sha256')}))
