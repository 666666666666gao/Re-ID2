"""Fresh narrow review: stdlib/AST only; never imports project runtime or opens SSH."""
import ast
import builtins
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import traceback
from types import SimpleNamespace

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = PROJECT / '.aris/traces/experiment-bridge/2026-10-03_missing_distance_repair'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_v4_missing_dev27_repaired_deploy_20261003.py')
MODULE = PROJECT / 'missing_development.py'
ORIGINAL = PROJECT / 'results/preflight/axis_collaboration_v4_missing_development_original.py'
REPORT = PROJECT / 'results/preflight/axis_collaboration_v4_missing_development27_repaired_review.json'
PREFIX = PROJECT / 'results/preflight'
prior = json.loads((PREFIX / 'axis_collaboration_v4_missing_development27_launch.json').read_text(encoding='utf-8'))
failure = json.loads((PREFIX / 'axis_collaboration_v4_missing_development27_failure_snapshot.json').read_text(encoding='utf-8'))
probe = json.loads((PREFIX / 'axis_collaboration_v4_distance27_cpu_probe.json').read_text(encoding='utf-8'))
source = MODULE.read_text(encoding='utf-8')
helper_source = HELPER.read_text(encoding='utf-8')
original = ORIGINAL.read_text(encoding='utf-8')
module_hash = hashlib.sha256(MODULE.read_bytes()).hexdigest()
helper_hash = hashlib.sha256(HELPER.read_bytes()).hexdigest()
module_tree = ast.parse(source)
helper_tree = ast.parse(helper_source)
original_tree = ast.parse(original)
checks = []
LOG = []


def check(name, fn):
    try:
        detail = fn()
    except Exception as exc:
        checks.append(dict(name=name, status='FAIL', detail=f'{type(exc).__name__}: {exc}'))
        LOG.append(traceback.format_exc())
    else:
        checks.append(dict(name=name, status='PASS', detail=detail))
    print(json.dumps(checks[-1], ensure_ascii=False), flush=True)


def require(value, message='assertion failed'):
    assert value, message


def rejects(fn, errors=(AssertionError,)):
    try:
        fn()
    except errors:
        return
    raise AssertionError('expected rejection was not raised')


def body(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def execute(nodes, namespace):
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(tree, '<reviewed AST>', 'exec'), namespace)


def strip_imports(tree):
    return [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]


def exact_delta():
    expected = original.replace(
        'def measure(name, q, g, missing_query, missing_gallery):',
        'def measure(name, distances, missing_query, missing_gallery):')
    expected = expected.replace('full_metrics(distance(q, g), ids[query]', 'full_metrics(distances, ids[query]')
    expected = expected.replace("measure('clean', clean[query], clean, [], [])", "measure('clean', saved['distances'], [], [])")
    expected = expected.replace("measure('both_missing_' + code, feature[query], feature,", "measure('both_missing_' + code, distance(feature[query], feature),")
    expected = expected.replace("measure('query_missing_' + code, feature[query], clean,", "measure('query_missing_' + code, distance(feature[query], clean),")
    expected = expected.replace("               'state_tensor_versions_unchanged': True, 'original_mask_check': original_masks,\n               'original_input_files': original_inputs,", "               'normal_distance_source': 'saved best_dev_arrays.npz/distances after exact clean feature parity; missing conditions use full_evaluation.distance',\n               'state_tensor_versions_unchanged': True, 'original_mask_check': original_masks,\n               'original_input_files': original_inputs,")
    require(ast.dump(ast.parse(expected)) == ast.dump(module_tree), 'unexpected module AST change')
    return 'Only measure distance argument, clean saved distance, two missing call sites and provenance field changed; every other AST node unchanged.'


def evidence():
    require(not failure['controller_live'])
    require(failure['failures'] == ['MSVR310_axis_scaled_fullref_s42/full', 'RGBNT100_demo_s42/full'])
    complete = [r for r in failure['rows'] if r['stage'] == 'full' and r.get('result', {}).get('status') == 'COMPLETE']
    require([r['name'] for r in complete] == ['MSVR310_demo_s42', 'RGBNT201_demo_s42'])
    require(all(r['exit']['exit_code'] == 0 and len(r['result']['measurements']) == 13 and r['result']['normal_feature_max_error'] == 0 for r in complete))
    require(probe['gpu_forwards'] == probe['optimizer_updates'] == 0)
    require(probe['different_rank_rows'] == 7 and len(probe['AP_changed_queries']) == 1)
    require(abs(probe['mAP']['saved'] - probe['mAP']['recomputed']) > 1e-8)
    failed = [r for r in failure['rows'] if r.get('exit', {}).get('exit_code') == 1]
    require(all("terminal['strict_reload'][key]) < 1e-8" in r['log_tail'] for r in failed))
    return 'Archive: two full failures at clean strict metric assertion; two completed baselines/26 conditions; one AP query changed from same saved float32 features.'


def source_preservation():
    old = prior['source_sha256']
    require(len(old) == 15)
    require(hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == old['missing_development.py'])
    require(all(hashlib.sha256((PROJECT / n).read_bytes()).hexdigest() == h for n, h in old.items() if n != 'missing_development.py'))
    require(sum(len(v['files']) for v in prior['input_files'].values()) == 24)
    return 'Original module equals launch hash; all other 14 source entries match; original manifest has 24 input proofs.'


main = body(module_tree, 'main')
parity_guard = next(n for n in main.body if isinstance(n, ast.Assert) and ast.unparse(n.test) == 'parity == 0')
metric_guard = next(n for n in main.body if isinstance(n, ast.Assert) and 'strict_reload' in ast.unparse(n))
measure_def = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == 'measure')
clean_call = next(n for n in main.body if isinstance(n, ast.Expr) and ast.unparse(n).startswith("measure('clean'"))
missing_loop = next(n for n in main.body if isinstance(n, ast.For) and ast.unparse(n.iter) == 'MISSING.items()')
missing_tree = ast.parse((PROJECT / 'missing_evaluation.py').read_text(encoding='utf-8'))
literals = {n.targets[0].id: ast.literal_eval(n.value) for n in missing_tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('MISSING', 'METRICS')}


def feature_guard():
    execute([parity_guard], {'parity': 0})
    rejects(lambda: execute([parity_guard], {'parity': 1e-15}))
    return 'Actual unchanged parity assertion accepts zero and rejects 1e-15.'


def strict_metric_guard():
    keys = ('mAP', 'Rank-1', 'Rank-5', 'Rank-10')
    expected = dict.fromkeys(keys, 0.)
    def run(delta, key):
        got = dict(expected)
        got[key] = delta
        execute([metric_guard], dict(measurements={'clean': {'metrics': got}}, terminal={'strict_reload': expected}))
    for key in keys:
        run(0, key)
        rejects(lambda: run(1e-8, key))
    rejects(lambda: run(abs(probe['mAP']['saved'] - probe['mAP']['recomputed']), 'mAP'))
    return 'All four actual metric guards reject equality to 1e-8; evidenced AP drift also rejects. Tolerance was not relaxed.'


class Array:
    def __init__(self, name):
        self.name = name
        self.shape = (360, 5632)
    def __getitem__(self, key):
        return Array(self.name + '[query]')


def measurement_wiring():
    calls, distances = [], []
    clean = Array('clean')
    saved_distance = object()
    def metric(distances_, qids, gids, qexclude, gexclude, names, cameras, scenes, output):
        calls.append((distances_, str(output)))
        return dict.fromkeys(literals['METRICS'], 0.)
    def distance(q, g):
        result = (q.name, g.name)
        distances.append(result)
        return result
    def extract(model, dev, cfg, seed, missing):
        return Array('+'.join(missing)), {}
    ns = dict(json=json, print=lambda *a, **k: None, measurements={}, full_metrics=metric, distance=distance,
              ids=Array('ids'), exclusion=Array('exclude'), names=Array('names'), cams=Array('cams'), scenes=Array('scenes'),
              query=object(), output=PurePosixPath('/output'), saved={'distances': saved_distance}, clean=clean,
              torch=SimpleNamespace(isfinite=lambda f: SimpleNamespace(all=lambda: True)),
              MISSING=literals['MISSING'], METRICS=literals['METRICS'], extract_missing=extract, model=object(), dev=object(), cfg=object(),
              arguments=SimpleNamespace(seed=42), diagnostics={})
    execute([measure_def, clean_call, missing_loop], ns)
    require(calls[0][0] is saved_distance and len(calls) == 13 and len(distances) == 12)
    expected = []
    for missing in literals['MISSING'].values():
        name = '+'.join(missing)
        expected.extend([(name + '[query]', name), (name + '[query]', 'clean')])
    require(distances == expected)
    require(len(ns['measurements']) == 13)
    return 'Actual AST: clean forwards exact saved object; all 12 missing conditions use original distance(q,g), including clean gallery for query-only.'


def signature_check():
    signatures = {}
    for filename, funcs in {'collect_results.py': ('command', 'remote_python'), 'launch_axis_scaled.py': ('idle',), 'launch_runs.py': ('write_json',), 'full_evaluation.py': ('distance', 'full_metrics')}.items():
        tree = ast.parse((PROJECT / filename).read_text(encoding='utf-8'))
        for name in funcs:
            fn = body(tree, name)
            signatures[name] = [arg.arg for arg in fn.args.args]
    require(signatures == {'command': ['argv'], 'remote_python': ['host', 'code'], 'idle': ['gpu'], 'write_json': ['path', 'value'], 'distance': ['q', 'g'], 'full_metrics': ['distances', 'qids', 'gids', 'qexclude', 'gexclude', 'names', 'cameras', 'scenes', 'output']})
    return signatures


class FS:
    def __init__(self):
        self.files, self.dirs, self.sizes, self.digest_map = {}, set(), {}, {}
        self.reads, self.writes = [], []
    def path(self, value):
        return VPath(self, value)
    def put(self, path, value):
        name = str(path)
        self.files[name] = value.encode() if isinstance(value, str) else value
        parent = PurePosixPath(name).parent
        self.dirs.add(str(parent))
    def put_json(self, path, value):
        self.put(path, json.dumps(value))
    def hash(self, value):
        return SimpleNamespace(hexdigest=lambda: self.digest_map[value] if value in self.digest_map else hashlib.sha256(value).hexdigest())


class VPath:
    def __init__(self, fs, value):
        self.fs, self.value = fs, str(PurePosixPath(str(value)))
    def __str__(self):
        return self.value
    def __truediv__(self, value):
        return VPath(self.fs, PurePosixPath(self.value) / str(value))
    @property
    def parent(self):
        return VPath(self.fs, PurePosixPath(self.value).parent)
    @property
    def name(self):
        return PurePosixPath(self.value).name
    @property
    def suffix(self):
        return PurePosixPath(self.value).suffix
    def with_suffix(self, suffix):
        return VPath(self.fs, PurePosixPath(self.value).with_suffix(suffix))
    def read_bytes(self):
        self.fs.reads.append(self.value)
        return self.fs.files[self.value]
    def read_text(self, **kwargs):
        return self.read_bytes().decode('utf-8')
    def stat(self):
        return SimpleNamespace(st_size=self.fs.sizes.get(self.value, len(self.fs.files[self.value])))
    def exists(self):
        return self.value in self.fs.files or self.value in self.fs.dirs
    def mkdir(self, exist_ok=False):
        if self.exists() and not exist_ok:
            raise FileExistsError(self.value)
        self.fs.dirs.add(self.value)
        self.fs.writes.append(self.value)
    def write_bytes(self, value):
        self.fs.put(self.value, value)
        self.fs.writes.append(self.value)
    def write_text(self, value):
        self.write_bytes(value.encode())
    def open(self, mode):
        require(mode == 'x')
        if self.exists():
            raise FileExistsError(self.value)
        self.write_bytes(b'')
        return io.StringIO()
    def replace(self, target):
        self.fs.files[str(target)] = self.fs.files.pop(self.value)
        self.fs.writes.append(str(target))


class Harness:
    def __init__(self):
        self.remote = FS()
        self.local = FS()
        self.root = '/data/gb/Re-ID/DeMo-DualAxis'
        self.output = self.root + '/runs/axis_collaboration_v4_missing_development27_repaired'
        self.inputs = self.root + '/runs/axis_collaboration_v4_missing_development27_input'
        self.events, self.remote_codes, self.popen = [], [], []
        self.live = False
        self.used = {2: 20, 3: 20}
        self.after_upload = lambda: None
        self.review = dict(status='PASS', blockers=[], module_sha256=module_hash, helper_sha256=helper_hash)
        for name, value in [('axis_collaboration_v4_missing_development27_launch.json', prior), ('axis_collaboration_v4_missing_development27_failure_snapshot.json', failure)]:
            self.local.put_json(str(PREFIX / name).replace('\\', '/'), value)
        self.local.put(str(MODULE).replace('\\', '/'), MODULE.read_bytes())
        self.remote.put_json(self.root + '/setup/env27_doc_validation.json', dict(status='PASS', spec_sha256='4993690212b8f331ceac59c1cf8fe15751a69be89ffdcb2c046cdb7989cae5a0'))
        for name in prior['source_sha256']:
            self.remote.put(self.root + '/' + name, ORIGINAL.read_bytes() if name == 'missing_development.py' else (PROJECT / name).read_bytes())
        for name, row in prior['input_files'].items():
            for filename, proof in row['files'].items():
                path = self.inputs + '/' + (filename if filename.endswith('_exit.json') else name + '/' + filename)
                token = ('input:' + path).encode()
                self.remote.put(path, token)
                self.remote.sizes[path] = proof['bytes']
                self.remote.digest_map[token] = proof['sha256']
        for row in failure['rows']:
            if row.get('exit'):
                self.remote.put_json(prior['output'] + '/' + row['name'] + '/' + row['stage'] + '_exit.json', row['exit'])
            if row.get('result'):
                self.remote.put_json(prior['output'] + '/' + row['name'] + '/' + row['stage'] + ('/result.json' if row['stage'] == 'full' else '/smoke.json'), row['result'])
    def remote_python(self, host, code):
        require(host == '2027')
        self.events.append('guard' if not self.remote_codes else 'launch')
        self.remote_codes.append(code)
        output = []
        subprocess = SimpleNamespace(run=lambda argv, **kw: SimpleNamespace(stdout='1014886\n' if self.live else ''),
                                     check_output=lambda argv, **kw: '0, 349\n1, 20\n2, %s\n3, %s\n' % (self.used[2], self.used[3]),
                                     Popen=self.spawn, STDOUT=-2)
        ns = dict(hashlib=SimpleNamespace(sha256=self.remote.hash), json=json, subprocess=subprocess, Path=self.remote.path,
                  os=SimpleNamespace(environ={}), time=SimpleNamespace(time=lambda: 1.), print=lambda value: output.append(value))
        execute(strip_imports(ast.parse(code)), ns)
        return output[-1]
    def spawn(self, argv, **kwargs):
        self.popen.append((argv, kwargs))
        return SimpleNamespace(pid=2345)
    def command(self, argv):
        self.events.append('upload')
        require(argv[0] == 'scp' and argv[-2] == str(self.local.path(str(MODULE).replace('\\', '/'))))
        require(argv[-1] == '2027:' + self.root + '/missing_development.py')
        self.remote.put(self.root + '/missing_development.py', MODULE.read_bytes())
        self.after_upload()
        return ''
    def run(self):
        self.local.put_json(str(REPORT).replace('\\', '/'), self.review)
        ns = dict(datetime=datetime, hashlib=hashlib, json=json, sys=SimpleNamespace(path=[]),
                  PROJECT=self.local.path(str(PROJECT).replace('\\', '/')), HOSTS={}, OPTIONS=[], command=self.command,
                  remote_python=self.remote_python, __file__=str(HELPER), open=lambda p, mode: io.BytesIO(HELPER.read_bytes()),
                  print=lambda *a, **k: None)
        execute(strip_imports(helper_tree), ns)
        self.ns = ns
        return self


happy = None


def helper_happy():
    global happy
    happy = Harness().run()
    require(happy.events == ['guard', 'upload', 'launch'])
    require(len(happy.popen) == 1)
    require(happy.ns['jobs'] == {2: [('MSVR310', 'axis_scaled_fullref'), ('RGBNT201', 'axis_scaled_fullref')], 3: [('RGBNT100', 'demo'), ('RGBNT100', 'axis_scaled_fullref')]})
    require(happy.ns['reused'] == ['MSVR310_demo_s42', 'RGBNT201_demo_s42'])
    require(happy.ns['source_sha'] == {**prior['source_sha256'], 'missing_development.py': module_hash})
    require(happy.ns['availability']['input_proof_count'] == 24)
    argv, kwargs = happy.popen[0]
    require(argv[:3] == ['/data/gb/Re-ID/conda-envs/tri_reid/bin/python', '-u', '-c'])
    require(str(kwargs['cwd']) == happy.root and kwargs['start_new_session'] is True)
    require(str(REPORT).replace('\\', '/') not in happy.local.writes)
    return 'Full helper AST mock: one preflight, one module upload, one controller spawn/receipt; expected GPU2/3 schedule and new source manifest.'


def guard_rejection(kind):
    h = Harness()
    if kind == 'live': h.live = True
    elif kind == 'environment': h.remote.put_json(h.root + '/setup/env27_doc_validation.json', dict(status='FAIL', spec_sha256='invalid'))
    elif kind == 'output_exists': h.remote.dirs.add(h.output)
    elif kind == 'failed_exit': h.remote.put_json(prior['output'] + '/MSVR310_axis_scaled_fullref_s42/full_exit.json', dict(exit_code=0))
    elif kind == 'unstarted_exists': h.remote.dirs.add(prior['output'] + '/RGBNT100_axis_scaled_fullref_s42')
    elif kind in ('gpu2', 'gpu3'): h.used[int(kind[-1])] = 500
    rejects(h.run)
    require(not h.popen and 'upload' not in h.events and not h.remote.writes)
    return f'{kind} rejected before source upload or launch.'


def proof_rejections():
    count = 0
    baseline = Harness()
    for path in baseline.remote.sizes:
        h = Harness()
        h.remote.sizes[path] += 1
        rejects(h.run)
        require('upload' not in h.events)
        h = Harness()
        h.remote.files[path] += b'x'
        rejects(h.run)
        require('upload' not in h.events)
        count += 2
    return f'{count} negative probes: every one of 24 input size/hash proofs rejects before upload.'


def source_rejections():
    for name in prior['source_sha256']:
        h = Harness()
        h.remote.files[h.root + '/' + name] += b'x'
        rejects(h.run)
        require('upload' not in h.events)
    return 'All 15 old source entries independently corrupted: each rejected before upload.'


def reused_rejections():
    count = 0
    for name in ['MSVR310_demo_s42', 'RGBNT201_demo_s42']:
        for field, value in [('status', 'INCOMPLETE'), ('measurements', {}), ('normal_feature_max_error', 1e-15), ('optimizer_updates', 1)]:
            h = Harness()
            path = prior['output'] + '/' + name + '/full/result.json'
            doc = json.loads(h.remote.files[path])
            doc[field] = value
            h.remote.put_json(path, doc)
            rejects(h.run)
            require('upload' not in h.events)
            count += 1
        h = Harness()
        h.remote.put_json(prior['output'] + '/' + name + '/full_exit.json', dict(exit_code=1))
        rejects(h.run)
        require('upload' not in h.events)
        count += 1
    return f'{count} reused-result negative probes blocked before upload; old schema accepted without new distance provenance field.'


def review_rejections():
    for field, value in [('status', 'BLOCKED'), ('blockers', ['blocker']), ('helper_sha256', 'wrong'), ('module_sha256', 'wrong')]:
        h = Harness()
        h.review[field] = value
        rejects(h.run)
        require(not h.events)
    return 'Status, blockers, helper digest and module digest gates independently reject before any remote call.'


def second_guard_rejections():
    for kind in ('source', 'input'):
        h = Harness()
        path = h.root + '/missing_development.py' if kind == 'source' else next(iter(h.remote.sizes))
        def change(): h.remote.files[path] += b'x'
        h.after_upload = change
        rejects(h.run)
        require(h.events == ['guard', 'upload', 'launch'] and not h.popen and not h.remote.writes)
    return 'New source and repeated input proofs both reject after upload, before output creation/Popen.'


class Pool:
    def __init__(self, max_workers):
        require(max_workers == 2)
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def map(self, fn, values): return map(fn, values)


def run_controller(fault=None, only_slot=None):
    fs = FS()
    fs.dirs.add(happy.output)
    events, rows = [], []
    def idle(gpu): events.append(('idle', gpu))
    def write_json(path, value):
        fs.put_json(str(path), value)
        rows.append((str(path), value))
    def popen(argv, **kwargs):
        stage = 'smoke' if '--smoke' in argv else 'full'
        name = PurePosixPath(argv[argv.index('--run-dir') + 1]).name
        gpu = kwargs['env']['CUDA_VISIBLE_DEVICES']
        events.append(('spawn', name, stage, gpu))
        result = dict(status='PASS' if stage == 'smoke' else 'COMPLETE', optimizer_updates=0, normal_feature_max_error=0,
                      measurements=dict.fromkeys(range(13)), normal_distance_source='saved best_dev_arrays.npz/distances after exact clean feature parity')
        code = 0
        if fault and name == fault[0] and stage == fault[1]:
            if fault[2] == 'exit': code = 1
            else: result[fault[2]] = fault[3]
        output = argv[argv.index('--output') + 1]
        fs.put_json(output + ('/smoke.json' if stage == 'smoke' else '/result.json'), result)
        require(argv[:3] == ['python-mock', '-u', 'missing_development.py'])
        require(argv[argv.index('--data-root') + 1] == '/data/gb/Re-ID/dataset')
        require(argv[argv.index('--pretrained') + 1] == '/data/gb/Re-ID/pretrained/ViT-B-16.pt')
        return SimpleNamespace(pid=3456, wait=lambda: code)
    tree = ast.parse(happy.ns['controller'])
    ns = dict(ThreadPoolExecutor=Pool, json=json, os=SimpleNamespace(environ={}), subprocess=SimpleNamespace(Popen=popen, STDOUT=-2),
              sys=SimpleNamespace(executable='python-mock'), time=SimpleNamespace(time=lambda: 1.), Path=fs.path, idle=idle, write_json=write_json)
    error = None
    try:
        if only_slot is None:
            execute(strip_imports(tree), ns)
        else:
            prefix = [n for n in strip_imports(tree) if isinstance(n, (ast.Assign, ast.FunctionDef))]
            execute(prefix, ns)
            ns['slot'](only_slot)
    except AssertionError as exc:
        error = str(exc)
    return fs, events, rows, error


def controller_happy():
    fs, events, rows, error = run_controller()
    require(error is None)
    expected = []
    for gpu, jobs in happy.ns['jobs'].items():
        for dataset, variant in jobs:
            for stage in ('smoke', 'full'):
                expected += [('idle', gpu), ('spawn', f'{dataset}_{variant}_s42', stage, str(gpu))]
    require(events == expected)
    result = json.loads(fs.files[happy.output + '/controller_result.json'])
    require(result['status'] == 'COMPLETE' and len(result['runs']) == 4 and result['conditions'] == 78 and result['new_conditions'] == 52 and result['reused_conditions'] == 26)
    require(result['reused_output'] == prior['output'] and result['reused_complete'] == happy.ns['reused'])
    require(all(path.startswith(happy.output + '/') for path in fs.files))
    return 'Four jobs/eight child stages: per-card smoke then full and exact next-job order; idle before every spawn; 52+26=78; output exclusively new root.'


def controller_failures():
    trials = []
    for gpu, name in [(2, 'MSVR310_axis_scaled_fullref_s42'), (3, 'RGBNT100_demo_s42')]:
        for stage, expected in [('smoke', 1), ('full', 2)]:
            fs, events, rows, error = run_controller((name, stage, 'exit', 1), gpu)
            require(error is not None and len([e for e in events if e[0] == 'spawn']) == expected)
            require(happy.output + '/controller_result.json' not in fs.files)
            require(json.loads(fs.files[happy.output + '/' + name + '/' + stage + '_exit.json'])['exit_code'] == 1)
            trials.append(f'gpu{gpu}/{stage}')
    return {'per_slot_fail_stop_cases': trials, 'semantics': 'No next stage/job in failed slot, failed exit retained, no COMPLETE aggregate. Existing ThreadPoolExecutor does not kill the other slot.'}


def controller_schema_rejections():
    name = 'MSVR310_axis_scaled_fullref_s42'
    trials = [('smoke', 'status', 'COMPLETE'), ('smoke', 'optimizer_updates', 1), ('smoke', 'normal_feature_max_error', 1e-15),
              ('full', 'status', 'PASS'), ('full', 'optimizer_updates', 1), ('full', 'normal_feature_max_error', 1e-15),
              ('full', 'measurements', {}), ('full', 'normal_distance_source', 'recomputed')]
    for stage, key, value in trials:
        fs, events, rows, error = run_controller((name, stage, key, value), 2)
        require(error is not None)
        require(len([e for e in events if e[0] == 'spawn']) == (1 if stage == 'smoke' else 2))
        require(happy.output + '/controller_result.json' not in fs.files)
    return 'Eight invalid smoke/full schema probes stop before the next stage/job and never report overall COMPLETE.'


def idle_wait():
    fn = body(ast.parse((PROJECT / 'launch_axis_scaled.py').read_text(encoding='utf-8')), 'idle')
    used = iter(['500', '700', '499'])
    sleeps, calls = [], []
    def output(argv, **kwargs):
        calls.append(argv)
        return next(used)
    ns = dict(subprocess=SimpleNamespace(check_output=output), time=SimpleNamespace(sleep=sleeps.append), print=lambda *a, **k: None)
    execute([fn], ns)
    ns['idle'](2)
    require(sleeps == [240, 240] and len(calls) == 3)
    require(all(c[0:3] == ['nvidia-smi', '-i', '2'] for c in calls))
    return 'Actual idle() AST with 500,700,499 MiB sleeps exactly 240 seconds twice then returns; sleeps are mocked.'


def generated_and_scope():
    trees = [module_tree, helper_tree] + [ast.parse(happy.ns[name]) for name in ['guard', 'controller', 'entry', 'launch_code']]
    for tree in trees:
        compile(tree, '<review target>', 'exec')
    require(not any(isinstance(node, ast.Try) for tree in trees for node in ast.walk(tree)))
    require(not any(isinstance(node, ast.While) for tree in trees for node in ast.walk(tree)))
    require('kill' not in helper_source and 'SIG' not in helper_source)
    final_write = next(n for n in reversed(main.body) if isinstance(n, ast.Expr) and ast.unparse(n).startswith('write_json('))
    require('normal_distance_source' in ast.unparse(final_write))
    return 'Module, helper and four generated code strings compile with Python 3.10; no new try/except, retry loop or signal; new distance provenance schema present.'


check('actual_failure_evidence', evidence)
check('original_15_source_and_24_input_manifest', source_preservation)
check('module_exact_narrow_AST_delta', exact_delta)
check('unchanged_feature_parity_zero_guard', feature_guard)
check('unchanged_strict_metric_1e_8_guard', strict_metric_guard)
check('clean_saved_and_all_12_missing_distance_wiring', measurement_wiring)
check('actual_API_signatures', signature_check)
check('helper_happy_guard_upload_launch_sequence', helper_happy)
for kind in ['live', 'environment', 'output_exists', 'failed_exit', 'unstarted_exists', 'gpu2', 'gpu3']:
    check('preflight_reject_' + kind, lambda kind=kind: guard_rejection(kind))
check('all_original_source_mismatches_rejected', source_rejections)
check('all_24_input_size_and_hash_mismatches_rejected', proof_rejections)
check('two_reused_results_completion_guards', reused_rejections)
check('local_review_status_and_exact_hash_gates', review_rejections)
check('post_upload_new_source_and_input_guards', second_guard_rejections)
check('controller_four_jobs_smoke_full_wait_and_78_schema', controller_happy)
check('controller_failure_stops_own_slot_no_false_complete', controller_failures)
check('controller_result_schema_guards', controller_schema_rejections)
check('actual_idle_240_second_wait_boundary', idle_wait)
check('generated_AST_compile_and_no_retry_or_extra_try', generated_and_scope)

blockers = [row for row in checks if row['status'] != 'PASS']
report = dict(status='BLOCKED' if blockers else 'PASS', blockers=blockers, helper_sha256=helper_hash, module_sha256=module_hash,
              review_independence='same-family', acceptance_status='provisional', reviewer_route='fresh Codex review',
              scope='Narrow post-failure module delta and new repaired continuation helper only; original 80/51 suites not rerun.',
              checks_count=len(checks), passed_checks=sum(row['status'] == 'PASS' for row in checks), checks=checks,
              check_log=str(TRACE / 'checks.jsonl'), harness=str(TRACE / 'review_delta_mock.py'),
              evidence_files=[str(PREFIX / name) for name in ['axis_collaboration_v4_distance27_cpu_probe.json','axis_collaboration_v4_missing_development27_failure_snapshot.json','axis_collaboration_v4_missing_development27_launch.json']],
              execution=dict(stdlib_only=True, python='3.10.19', real_ssh_calls=0, gpu_forwards=0, optimizer_updates=0, real_signals=0, package_installs=0),
              limitations=['Static and controlled mock review only; deployment must run actual recorded guards and smoke stages.', 'Same saved-feature CPU probe establishes distance/rank/AP differences, not the dispatch cause.', 'Failure stops its own GPU slot and blocks COMPLETE; ThreadPoolExecutor does not kill the other submitted slot.'],
              observed_at=datetime.now(timezone.utc).isoformat())
(TRACE / 'checks.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in checks) + '\n', encoding='utf-8')
(TRACE / 'harness_errors.txt').write_text('\n'.join(LOG), encoding='utf-8')
(TRACE / 'review.result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=report['status'], checks_count=report['checks_count'], passed_checks=report['passed_checks'], helper_sha256=helper_hash, module_sha256=module_hash)))
