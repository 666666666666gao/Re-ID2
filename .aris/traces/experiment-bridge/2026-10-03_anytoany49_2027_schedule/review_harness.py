"""Source-only secondary review: stdlib AST and in-memory mocked side effects."""
import argparse
import ast
import contextlib
import difflib
import hashlib
import io
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path as RealPath, PurePosixPath
from types import SimpleNamespace as NS

project = RealPath('C:/Users/gb/projects/demo_dual_axis_20261002')
trace = project / '.aris/traces/experiment-bridge/2026-10-03_anytoany49_2027_schedule'
prior_trace = project / '.aris/traces/experiment-bridge/2026-10-03_anytoany49'
helper = RealPath('C:/Users/gb/.codex_tmp/demo_anytoany27_deploy_20261003.py')
old_helper = RealPath('C:/Users/gb/.codex_tmp/demo_anytoany_deploy_20261003.py')
stamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
checks = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_result(name, value):
    (trace / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def check(name, detail):
    checks.append(dict(name=name, status='PASS', detail=detail))


def parse(name):
    return ast.parse((project / name).read_text(encoding='utf-8'))


def main_node(name):
    return next(n for n in parse(name).body if isinstance(n, ast.FunctionDef) and n.name == 'main')


def run_nodes(nodes, env):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<review-AST>', 'exec'), env)


prior_review = json.loads((project / 'results/preflight/anytoany49_review.json').read_text(encoding='utf-8'))
assert prior_review['status'] == 'PASS' and not prior_review['blockers']
assert sha(project / 'missing_anytoany_development.py') == prior_review['checked_source_sha256']['missing_anytoany_development.py']
assert sha(project / 'launch_anytoany_frozen.py') == prior_review['checked_source_sha256']['launch_anytoany_frozen.py']
assert sha(old_helper) == prior_review['helper_sha256']
preserved_paths = [project / name for name in (
    'missing_anytoany_development.py', 'launch_anytoany_frozen.py',
    'results/preflight/anytoany49_review.json', 'results/preflight/anytoany49_2026_launch.json',
    'results/preflight/anytoany49_2027_failed_guard_diagnostic.json',
    'results/preflight/axis_collaboration_v4_missing_development27_launch.json')]
preserved_hashes = {str(p): sha(p) for p in preserved_paths}
check('original_reviewed_files_match', {'leaf': sha(project / 'missing_anytoany_development.py'), 'original_controller': sha(project / 'launch_anytoany_frozen.py'), 'original_helper': sha(old_helper)})

old_launch_source = (project / 'launch_anytoany_frozen.py').read_text(encoding='utf-8')
launch_source = (project / 'launch_anytoany_frozen27.py').read_text(encoding='utf-8')
expected_launch_source = old_launch_source.replace(
    '"""Three independent frozen dataset jobs on idle cards; smoke before full49."""',
    '"""Frozen jobs on two currently idle 2027 cards; serialize MSVR and201 onGPU2."""').replace(
    "jobs=[('MSVR310',0),('RGBNT201',2),('RGBNT100',3)]",
    "groups={2:[('MSVR310',2),('RGBNT201',2)],1:[('RGBNT100',1)]}").replace(
    '    with ThreadPoolExecutor(max_workers=3) as pool:runs=list(pool.map(slot,jobs))',
    '    def worker(jobs):return [slot(job) for job in jobs]\n\n'
    '    with ThreadPoolExecutor(max_workers=2) as pool:groups_done=list(pool.map(worker,groups.values()))\n'
    '    runs=[run for group in groups_done for run in group]')
assert launch_source == expected_launch_source
launch_main = main_node('launch_anytoany_frozen27.py')
old_launch_main = main_node('launch_anytoany_frozen.py')
slot = next(n for n in launch_main.body if isinstance(n, ast.FunctionDef) and n.name == 'slot')
old_slot = next(n for n in old_launch_main.body if isinstance(n, ast.FunctionDef) and n.name == 'slot')
assert ast.dump(slot) == ast.dump(old_slot)
groups = ast.literal_eval(next(n.value for n in launch_main.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'groups' for t in n.targets)))
assert groups == {2: [('MSVR310', 2), ('RGBNT201', 2)], 1: [('RGBNT100', 1)]}
check('controller_changes_exactly_requested_schedule', {'groups': groups, 'workers': 2, 'slot_AST_identical': True, 'all_non_schedule_text_identical': True})

old_helper_source = old_helper.read_text(encoding='utf-8')
helper_source = helper.read_text(encoding='utf-8')
expected_helper_source = old_helper_source.replace('anytoany49_review.json', 'anytoany49_2027_schedule_review.json').replace('launch_anytoany_frozen.py', 'launch_anytoany_frozen27.py').replace(
    "[('2026','axis_mass_fullref','runs/axis_collaboration_v5_mass/development'),\n                               ('2027','demo','runs/axis_collaboration_v4_missing_development27_input')]",
    "[('2027','demo','runs/axis_collaboration_v4_missing_development27_input')]").replace('(0,2,3)', '(1,2)').replace('gpu=[0,2,3]', 'gpu=[1,2]')
assert helper_source == expected_helper_source
helper_tree = ast.parse(helper_source)


def assigned(name):
    return ast.literal_eval(next(n.value for n in helper_tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)))


files = assigned('files')
dependencies = assigned('dependencies')
assert files == ('missing_anytoany_development.py', 'launch_anytoany_frozen27.py')
outer_loop = next(n for n in helper_tree.body if isinstance(n, ast.For))
assert ast.literal_eval(outer_loop.iter) == [('2027', 'demo', 'runs/axis_collaboration_v4_missing_development27_input')]
source_hashes = {name: sha(project / name) for name in files}
helper_hash = sha(helper)
actual26 = json.loads((project / 'results/preflight/anytoany49_2026_launch.json').read_text(encoding='utf-8'))
assert actual26['host'] == '2026' and actual26['pid'] == 3301608
assert all(sha(project / name) == actual26['source_sha256'][name] for name in dependencies)
check('helper_changes_exactly_requested_27_only_schedule', {'hosts': ['2027'], 'variant': 'demo', 'source_bindings': source_hashes, 'all_non_schedule_text_identical': True, 'dependency_hashes_equal_existing26_receipt': len(dependencies)})

diagnostic = json.loads((project / 'results/preflight/anytoany49_2027_failed_guard_diagnostic.json').read_text(encoding='utf-8'))
assert diagnostic['input_differences'] == {name + '_demo_s42': [] for name in ('MSVR310', 'RGBNT201', 'RGBNT100')}
assert diagnostic['output_exists'] is False and diagnostic['new_sources_exist'] == []
assert not (project / 'results/preflight/anytoany49_2027_launch.json').exists()
check('known_failed_guard_state', {'diagnostic': diagnostic, 'existing26_pid': actual26['pid'], 'receipt27_absent_at_review': True, 'no_new_runtime_observation': True})

leaf = main_node('missing_anytoany_development.py')
assertions = {ast.unparse(n.test): n.lineno for n in ast.walk(leaf) if isinstance(n, ast.Assert)}
for expression in (
    "np.array_equal(saved['query_indices'], query)",
    "np.array_equal(saved['ids'], [row[1] for row in dev])",
    "np.array_equal(saved['cameras'], [row[2] for row in dev])",
    "np.array_equal(saved['scenes'], [row[3] for row in dev])",
    'parity == 0', 'len(measurements) == 49'):
    assert expression in assertions, expression
assert any(test.startswith("np.array_equal(saved['names']") for test in assertions)
assert assertions['parity == 0'] < assertions['len(measurements) == 49']
check('unchanged_clean_GT_and49_prerequisites', {'leaf_exact_previous_review_hash': True, 'saved_query_ids_cameras_scenes_names_checked': True, 'exact_clean_feature_parity_before_grid': True, 'all49_required': True, 'previous_review_evidence_reused': str(prior_trace / '004-semantic-checks.json')})

write_result('001-static-checks.json', {'status': 'PASS', 'checks': checks, 'launcher_diff': list(difflib.unified_diff(old_launch_source.splitlines(), launch_source.splitlines(), fromfile='launch_anytoany_frozen.py', tofile='launch_anytoany_frozen27.py', lineterm='')), 'helper_diff': list(difflib.unified_diff(old_helper_source.splitlines(), helper_source.splitlines(), fromfile=old_helper.name, tofile=helper.name, lineterm=''))})

# Reuse only previously recorded stdlib fake path classes and actual-slot test function.
prior_harness = ast.parse(json.loads((prior_trace / '002-mock-harness.json').read_text(encoding='utf-8'))['source'])
mock_nodes = [n for n in prior_harness.body if isinstance(n, ast.ClassDef) and n.name in ('MemoryFS', 'MemoryPath', 'NSContext') or isinstance(n, ast.FunctionDef) and n.name in ('dump', 'controller_case')]
run_nodes(mock_nodes, globals())
slot_cases = [controller_case('demo', dataset, gpu) for jobs in groups.values() for dataset, gpu in jobs]
slot_cases += [controller_case('demo', 'MSVR310', 2, fault) for fault in (
    'training_exit', 'terminal_status', 'terminal_epochs', 'wrong_variant', 'wrong_dataset',
    'smoke_exit', 'smoke_status', 'updates', 'versions', 'parity',
    'mutate_0', 'mutate_1', 'mutate_2', 'mutate_3', 'full48', 'full_exit')]


def controller_main_case(fail_smoke=False):
    fs = MemoryFS()
    for dataset in ('MSVR310', 'RGBNT201', 'RGBNT100'):
        run = fs.path('/train') / (dataset + '_demo_s42')
        dump(run / 'result.json', {'status': 'COMPLETE', 'epochs': 50, 'arguments': {'dataset': dataset, 'variant': 'demo', 'seed': 42}})
        dump(run / 'exit.json', {'exit_code': 0})
        (run / 'best.pth').write_bytes(b'checkpoint ' + dataset.encode())
        (run / 'best_dev_arrays.npz').write_bytes(b'arrays ' + dataset.encode())
    options = NS(training_root='/train', variant='demo', data_root='/data', pretrained='/clip', output='/suite')
    parser = NS(add_argument=lambda *a, **k: None, parse_args=lambda: options)
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = {1: 0, 2: 0}
    maximum = {1: 0, 2: 0}
    events = []
    idle_calls = []
    pools = []

    def event(*row):
        with lock:
            events.append(row)

    def pool(max_workers):
        assert max_workers == 2
        pools.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)

    def spawn(argv, stdout, stderr, env):
        gpu = int(env['CUDA_VISIBLE_DEVICES'])
        assert gpu in (1, 2)
        folder = fs.path(argv[argv.index('--output') + 1])
        dataset = folder.parent.name.split('_')[0]
        smoke = '--smoke' in argv
        stage = 'smoke' if smoke else 'full'
        assert argv[:3] == ['/python', '-u', 'missing_anytoany_development.py']
        assert (dataset == 'RGBNT100') == (gpu == 1)
        with lock:
            active[gpu] += 1
            maximum[gpu] = max(maximum[gpu], active[gpu])
            assert active[gpu] == 1
            events.append(('start', dataset, stage, gpu))
        dump(folder / ('smoke.json' if smoke else 'result.json'), {'status': 'PASS' if smoke else 'COMPLETE', 'optimizer_updates': 0, 'state_tensor_versions_unchanged': True, 'normal_feature_max_error': 0, 'measurements': {str(i): {} for i in range(49)}})

        def wait():
            if smoke and dataset in ('MSVR310', 'RGBNT100'):
                barrier.wait(timeout=5)
            with lock:
                active[gpu] -= 1
                events.append(('end', dataset, stage, gpu))
            return 1 if fail_smoke and dataset == 'MSVR310' and smoke else 0

        return NS(pid=123, wait=wait)

    def write_json(path, value):
        dump(path, value)
        if path.name == 'controller_result.json':
            event('record', path.parent.name, value['status'])

    env = dict(argparse=NS(ArgumentParser=lambda: parser), Path=fs.path, json=json, hashlib=hashlib,
               write_json=write_json, idle=idle_calls.append, sys=NS(executable='/python'),
               time=NS(time=lambda: 1), os=NS(environ={}), subprocess=NS(Popen=spawn, STDOUT=-2),
               ThreadPoolExecutor=pool)
    run_nodes([launch_main], env)
    rejected = False
    try:
        env['main']()
    except AssertionError:
        rejected = True
    assert rejected == fail_smoke
    assert pools == [2] and active == {1: 0, 2: 0} and maximum == {1: 1, 2: 1}
    assert {row[3] for row in events if row[0] == 'start'} == {1, 2}
    root_result = fs.path('/suite/controller_result.json')
    if fail_smoke:
        assert not root_result.exists()
        assert not any(row[0] == 'start' and row[1] == 'RGBNT201' for row in events)
        assert not (fs.path('/suite/MSVR310_demo_s42') / 'controller_result.json').exists()
    else:
        result = json.loads(root_result.read_text())
        assert result['status'] == 'COMPLETE' and result['availability_pairs'] == 147
        assert [(r['name'], r['gpu']) for r in result['runs']] == [('MSVR310_demo_s42', 2), ('RGBNT201_demo_s42', 2), ('RGBNT100_demo_s42', 1)]
        assert all(r['status'] == 'PASS' and r['availability_pairs'] == 49 and [s['stage'] for s in r['stages']] == ['smoke', 'full'] and all(s['exit_code'] == 0 for s in r['stages']) for r in result['runs'])
        assert events.index(('record', 'MSVR310_demo_s42', 'PASS')) < events.index(('start', 'RGBNT201', 'smoke', 2))
        assert events[-1] == ('record', 'suite', 'COMPLETE')
    return {'fault': 'MSVR310_smoke_exit1' if fail_smoke else None, 'rejected': rejected, 'max_active_per_GPU': maximum, 'worker_count': 2, 'GPU2_second_dataset_requires_first_PASS_record': True, 'root_COMPLETE_written': root_result.exists(), 'events': events, 'real_child_processes': 0}


main_cases = [controller_main_case(False), controller_main_case(True)]
write_result('002-controller-mock-checks.json', {'status': 'PASS', 'slot_cases': slot_cases, 'main_cases': main_cases, 'scope': 'Actual controller AST with real stdlib two-thread pool and in-memory files; all subprocess/GPU/model work mocked'})

helper_nodes = [n for n in helper_tree.body if not isinstance(n, (ast.Import, ast.ImportFrom)) and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'sys.path.insert')]
helper_ast = compile(ast.Module(body=helper_nodes, type_ignores=[]), '<deployment-helper-AST>', 'exec')
hosts = {'2026': ('/data/gaob/Re-ID/DeMo-DualAxis', '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'),
         '2027': ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')}
training = 'runs/axis_collaboration_v4_missing_development27_input'


def deploy_case(fault=None):
    local = MemoryFS()
    remote = {host: MemoryFS() for host in hosts}
    root, python = hosts['2027']
    gpu = {0: 4176, 1: 20, 2: 20, 3: 853}
    disk_free = 2_000_000_000
    expected_demo = {}
    copies, launches, remote_calls, queries, scripts = [], [], [], [], []
    for name in set(dependencies) | set(files):
        local.path(str(project / name)).write_bytes((project / name).read_bytes())
    local.path(str(helper)).write_bytes(helper.read_bytes())
    receipt26 = local.path(str(project / 'results/preflight/anytoany49_2026_launch.json'))
    receipt26_bytes = (project / 'results/preflight/anytoany49_2026_launch.json').read_bytes()
    receipt26.write_bytes(receipt26_bytes)
    remote['2026'].path('/preserve-existing-controller-3301608').write_bytes(b'untouched')
    before26 = dict(remote['2026'].files)
    for name in dependencies:
        remote['2027'].path(root + '/' + name).write_bytes((project / name).read_bytes())
    for dataset in ('MSVR310', 'RGBNT201', 'RGBNT100'):
        name = dataset + '_demo_s42'
        run = remote['2027'].path(root + '/' + training + '/' + name)
        dump(run / 'result.json', {'status': 'COMPLETE', 'epochs': 50, 'arguments': {'dataset': dataset, 'variant': 'demo', 'seed': 42}})
        dump(run / 'exit.json', {'exit_code': 0})
        (run / 'best.pth').write_bytes(b'synthetic checkpoint ' + name.encode())
        (run / 'best_dev_arrays.npz').write_bytes(b'synthetic arrays ' + name.encode())
        inputs = [run / key for key in ('best.pth', 'best_dev_arrays.npz', 'result.json', 'exit.json')]
        expected_demo[name] = {'files': {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs}}
    review = {'status': 'PASS', 'blockers': [], 'helper_sha256': helper_hash, 'checked_source_sha256': dict(source_hashes)}
    if fault == 'review_blocked':
        review.update(status='BLOCKED', blockers=['synthetic'])
    if fault == 'review_wrong_helper':
        review['helper_sha256'] = 'wrong'
    if fault == 'review_wrong_source':
        review['checked_source_sha256'][files[0]] = 'wrong'
    if fault == 'review_extra_source':
        review['checked_source_sha256']['launch_anytoany_frozen.py'] = 'extra'
    if fault == 'review_missing_source':
        del review['checked_source_sha256'][files[1]]
    dump(local.path(str(project / 'results/preflight/anytoany49_2027_schedule_review.json')), review)
    dump(local.path(str(project / 'results/preflight/axis_collaboration_v4_missing_development27_launch.json')), {'input_files': expected_demo})
    if fault in ('gpu1_500_before_copy', 'gpu2_500_before_copy'):
        gpu[int(fault[3])] = 500
    if fault == 'disk_exactly_1GB':
        disk_free = 1_000_000_000
    if fault == 'dependency_changed':
        remote['2027'].path(root + '/run_experiment.py').write_bytes(b'changed')
    if fault in ('leaf_already_staged', 'launcher_already_staged'):
        name = files[0 if fault == 'leaf_already_staged' else 1]
        remote['2027'].path(root + '/' + name).write_bytes((project / name).read_bytes())
    output = root + '/runs/anytoany49_demo_20261003'
    if fault == 'output_exists':
        remote['2027'].path(output).mkdir()
    if fault == 'exclusive_log_exists':
        remote['2027'].path(output + '.log').write_bytes(b'original log')
    run = remote['2027'].path(root + '/' + training + '/MSVR310_demo_s42')
    if fault == 'mirror_checkpoint_changed':
        (run / 'best.pth').write_bytes(b'changed')
    if fault == 'training_exit1':
        dump(run / 'exit.json', {'exit_code': 1})
    if fault in ('terminal_not_complete', 'terminal49epochs', 'wrong_dataset', 'wrong_variant'):
        terminal = json.loads((run / 'result.json').read_text())
        if fault == 'terminal_not_complete':
            terminal['status'] = 'RUNNING'
        if fault == 'terminal49epochs':
            terminal['epochs'] = 49
        if fault == 'wrong_dataset':
            terminal['arguments']['dataset'] = 'RGBNT201'
        if fault == 'wrong_variant':
            terminal['arguments']['variant'] = 'axis_mass_fullref'
        dump(run / 'result.json', terminal)
    frozen_inputs_before = {key: value for key, value in remote['2027'].files.items() if '/' + training + '/' in key}

    def command(argv):
        assert argv[0] == 'scp'
        name = PurePosixPath(argv[-2]).name
        host, destination = argv[-1].split(':', 1)
        assert host == '2027' and name in files and destination == root + '/' + name
        remote[host].path(destination).write_bytes(local.path(argv[-2]).read_bytes())
        copies.append((host, name))
        if len(copies) == 2:
            if fault == 'copied_source_changed':
                remote[host].path(root + '/' + files[0]).write_bytes(b'changed')
            if fault in ('gpu1_500_before_start', 'gpu2_500_before_start'):
                gpu[int(fault[3])] = 500
        return ''

    def remote_python(host, source):
        assert host == '2027'
        remote_calls.append(host)
        is_start = 'start_new_session' in source
        scripts.append({'stage': 'start' if is_start else 'guard', 'source': source})
        parsed = ast.parse(source)
        stripped = [n for n in parsed.body if not isinstance(n, (ast.Import, ast.ImportFrom))]

        def check_output(argv, text):
            assert argv == ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits']
            queries.append('start' if is_start else 'guard')
            return ''.join(str(i) + ', ' + str(m) + '\n' for i, m in gpu.items())

        def spawn(argv, cwd, start_new_session, stdout, stderr, env):
            assert str(cwd) == root and start_new_session is True
            assert argv[:3] == [python, '-u', 'launch_anytoany_frozen27.py']
            assert argv[argv.index('--variant') + 1] == 'demo'
            assert argv[argv.index('--training-root') + 1] == root + '/' + training
            assert argv[argv.index('--data-root') + 1] == '/data/gb/Re-ID/dataset'
            assert argv[argv.index('--pretrained') + 1] == '/data/gb/Re-ID/pretrained/ViT-B-16.pt'
            assert env['OMP_NUM_THREADS'] == env['MKL_NUM_THREADS'] == env['OPENBLAS_NUM_THREADS'] == '4'
            remote[host].path(argv[argv.index('--output') + 1]).mkdir()
            launches.append({'host': host, 'argv': argv})
            return NS(pid=2001)

        env = dict(hashlib=hashlib, json=json, Path=remote[host].path, shutil=NS(disk_usage=lambda path: NS(free=disk_free)),
                   subprocess=NS(check_output=check_output, Popen=spawn, STDOUT=-2), os=NS(environ={}), time=NS(time=lambda: 1234))
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            run_nodes(stripped, env)
        return stream.getvalue()

    env = dict(__file__=str(helper), Path=local.path, PROJECT=local.path(str(project)), HOSTS=dict(hosts), OPTIONS=[],
               command=command, remote_python=remote_python, json=json, hashlib=hashlib, datetime=datetime)
    error = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(helper_ast, env)
    except (AssertionError, FileExistsError) as exc:
        error = type(exc).__name__
    receipt27 = local.path(str(project / 'results/preflight/anytoany49_2027_launch.json'))
    rerun_rejected = None
    if fault is None:
        assert error is None and len(launches) == 1 and launches[0]['host'] == '2027'
        assert copies == [('2027', name) for name in files]
        receipt = json.loads(receipt27.read_text())
        assert receipt['host'] == '2027' and receipt['availability']['gpus'] == {'1': 20, '2': 20}
        assert set(receipt['source_sha256']) == set(dependencies) | set(files)
        receipt_bytes = receipt27.read_bytes()
        prior_copies, prior_launches = len(copies), len(launches)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(helper_ast, env)
        except AssertionError:
            rerun_rejected = True
        assert rerun_rejected and len(copies) == prior_copies and len(launches) == prior_launches and receipt27.read_bytes() == receipt_bytes
    else:
        assert error is not None, fault
        assert not launches and not receipt27.exists(), fault
        after_copy = fault in ('copied_source_changed', 'gpu1_500_before_start', 'gpu2_500_before_start', 'exclusive_log_exists')
        assert len(copies) == (2 if after_copy else 0), (fault, copies)
    assert receipt26.read_bytes() == receipt26_bytes and remote['2026'].files == before26
    assert set(remote_calls) <= {'2027'} and all(h == '2027' and n in files and n.endswith('.py') for h, n in copies)
    assert {key: value for key, value in remote['2027'].files.items() if '/' + training + '/' in key} == frozen_inputs_before
    if fault is None:
        write_result('004-generated-remote-AST.json', {'scope': 'Mock-generated source only; never sent remotely', 'scripts': scripts[:2]})
    return {'fault': fault, 'exception': error, 'mock_controller_launch_hosts': [x['host'] for x in launches], 'source_copies': copies,
            'mock_remote_call_hosts': remote_calls, 'receipt26_preserved': True, 'remote26_untouched': True,
            'frozen_inputs_unchanged': True, 'receipt27_written': receipt27.exists(), 'successful_rerun_rejected_without_copy_or_launch': rerun_rejected,
            'actual_SSH_calls': 0, 'actual_GPU_calls': 0}


deploy_cases = [deploy_case()] + [deploy_case(fault) for fault in (
    'review_blocked', 'review_wrong_helper', 'review_wrong_source', 'review_extra_source', 'review_missing_source',
    'gpu1_500_before_copy', 'gpu2_500_before_copy', 'disk_exactly_1GB', 'dependency_changed',
    'leaf_already_staged', 'launcher_already_staged', 'output_exists', 'exclusive_log_exists',
    'mirror_checkpoint_changed', 'training_exit1', 'terminal_not_complete', 'terminal49epochs',
    'wrong_dataset', 'wrong_variant', 'copied_source_changed', 'gpu1_500_before_start', 'gpu2_500_before_start')]
write_result('003-deployment-mock-checks.json', {'status': 'PASS', 'cases': deploy_cases,
    'scope': 'Actual helper and generated remote AST with in-memory files and fake subprocesses; no network or neural imports'})

assert not any(n == 'torch' or n.startswith('torch.') or n == 'numpy' or n.startswith('numpy.') for n in sys.modules)
assert {str(p): sha(p) for p in preserved_paths} == preserved_hashes
assert source_hashes == {name: sha(project / name) for name in files} and helper_hash == sha(helper)
write_result('005-verification-summary.json', {'status': 'PASS', 'static_check_groups': len(checks), 'slot_mock_cases': len(slot_cases),
    'real_threadpool_mock_cases': len(main_cases), 'deployment_mock_cases': len(deploy_cases), 'checked_source_sha256': source_hashes,
    'helper_sha256': helper_hash, 'preserved_local_artifact_sha256': preserved_hashes, 'torch_imported': False, 'numpy_imported': False,
    'NN_forwards': 0, 'actual_SSH_calls': 0, 'actual_GPU_calls': 0, 'implementation_edits': False, 'reviewed_at': stamp})
print(json.dumps({'status': 'PASS', 'static_checks': len(checks), 'slot_cases': len(slot_cases), 'threadpool_cases': len(main_cases), 'deployment_cases': len(deploy_cases), 'source_sha256': source_hashes, 'helper_sha256': helper_hash}, indent=2))
