"""Independent stdlib-only source and orchestration mocks; never SSH/Torch/CUDA."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_axis_v5_frozen_eval'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_v5_frozen26_deploy_20261003.py')
NEW = ('missing_mass_development.py', 'diagnose_mass_axis.py', 'launch_mass_frozen_evaluation.py')
CHECKS = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(name, detail, cases=None):
    value = dict(name=name, status='PASS', detail=detail)
    if cases is not None:
        value['cases'] = cases
    CHECKS.append(value)
    print(json.dumps(value), flush=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def function(path, name, namespace):
    source = ast.parse(path.read_text(encoding='utf-8'))
    node = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


initial = {name: sha(ROOT / name) for name in NEW}
helper_sha = sha(HELPER)
launch = json.loads((ROOT / 'results/preflight/axis_collaboration_v5_mass_launch.json').read_text(encoding='utf-8'))
assert len(launch['source_sha256']) == 18
assert all(sha(ROOT / name) == value for name, value in launch['source_sha256'].items())
for path in (*(ROOT / name for name in NEW), HELPER):
    compile(path.read_text(encoding='utf-8'), str(path), 'exec')
record('syntax_and_existing18_source_preservation', 'Three new leaves and private helper compile; all18 launch-bound existing sources remain byte-exact.')

original = (ROOT / 'missing_development.py').read_text(encoding='utf-8')
expected = original.replace('MISSING, METRICS, extract_missing, verify_original_mask', 'MISSING, METRICS, extract_missing')
expected = expected.replace('from run_experiment import build, configuration, write_json', 'from run_mass_experiment import build, configuration, write_json')
expected = expected.replace('from scaled_axis_collaboration import VARIANTS\n', '')
expected = expected.replace("arguments.variant == 'demo' or arguments.variant in VARIANTS", "arguments.variant == 'axis_mass_fullref'")
expected = expected.replace("run / 'exit.json' if arguments.variant == 'demo' else run.parent / (run.name + '_exit.json')", "run.parent / (run.name + '_exit.json')")
expected = expected.replace("verify_original_mask(model, records, cfg, arguments.seed) if arguments.variant == 'demo' else None", 'None')
assert expected == (ROOT / NEW[0]).read_text(encoding='utf-8')
record('missing_leaf_exact_scoped_delta', 'Exact text comparison confirms only V5 build/variant/exit and inapplicable DeMo-mask-check removal; repaired clean saved-distance logic retained.')

original = (ROOT / 'diagnose_axis_collaboration.py').read_text(encoding='utf-8')
expected = original.replace('from run_experiment import build, configuration, write_json', 'from run_mass_experiment import build, configuration, write_json')
expected = expected.replace('from scaled_axis_collaboration import VARIANTS\n', '')
expected = expected.replace("    if arguments.variant in VARIANTS:\n        exit_file = run.parent / (run.name + '_exit.json')\n    else:\n        assert arguments.variant == 'axis_collaboration'\n        exit_file = run / 'exit.json'", "    assert arguments.variant == 'axis_mass_fullref'\n    exit_file = run.parent / (run.name + '_exit.json')")
expected = expected.replace("        metrics[key] = full_metrics(distance(value[query], value), ids[query], ids, exclusion[query], exclusion,", "        distances = saved['distances'] if key == '11' else distance(value[query], value)\n        metrics[key] = full_metrics(distances, ids[query], ids, exclusion[query], exclusion,")
expected = expected.replace("              'metrics': metrics,", "              'normal_distance_source': 'saved best_dev_arrays.npz/distances after exact full11 feature parity; 00/10/01 use full_evaluation.distance',\n              'metrics': metrics,")
assert expected == (ROOT / NEW[1]).read_text(encoding='utf-8')
record('four_state_leaf_exact_scoped_delta', 'Exact comparison confirms V5 build/variant/exit plus state11 saved distances and explicit provenance; strict parity and1e-8 metric gates unchanged.')

actual_terminal = json.loads((ROOT / 'results/axis_collaboration_v5_mass/development/MSVR310_axis_mass_fullref_s42/result.json').read_text(encoding='utf-8'))
actual_exit = json.loads((ROOT / 'results/axis_collaboration_v5_mass/development/MSVR310_axis_mass_fullref_s42_exit.json').read_text(encoding='utf-8'))
assert actual_terminal['status'] == 'COMPLETE' and actual_terminal['epochs'] == 50 and actual_exit['exit_code'] == 0
assert actual_terminal['arguments']['variant'] == 'axis_mass_fullref'
assert actual_terminal['arguments']['output'] == launch['output'] + '/development/MSVR310_axis_mass_fullref_s42'
assert actual_terminal['descriptor_dim'] == 5632 and actual_terminal['dev_records'] >= 64
for key in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10'):
    assert key in actual_terminal['best'] and key in actual_terminal['strict_reload']
record('actual_V5_terminal_schema_and_paths', 'Existing MSVR terminal contains required argument/build/epoch/best/strict_reload fields; adjacent exit and launch-derived run paths agree. No historical training replay/audit.')

all_sources = dict(launch['source_sha256'])
all_sources['missing_evaluation.py'] = sha(ROOT / 'missing_evaluation.py')
launcher_source = ROOT / NEW[2]


def launcher_case(sandbox, label, fault=None, fault_stage=0, waiting=False):
    area = sandbox / label
    training = area / 'training'
    training.mkdir(parents=True)
    output = area / 'output'
    launches, idle_calls, sleeps = [], [], []
    for dataset in ('MSVR310', 'RGBNT201', 'RGBNT100'):
        name = dataset + '_axis_mass_fullref_s42'
        run = training / name
        run.mkdir()
        terminal = dict(status='COMPLETE', epochs=50, arguments=dict(variant='axis_mass_fullref', dataset=dataset))
        exit_value = dict(exit_code=0)
        if dataset == 'MSVR310':
            if fault == 'training_exit':
                exit_value['exit_code'] = 9
            if fault == 'terminal_status':
                terminal['status'] = 'RUNNING'
            if fault == 'epochs':
                terminal['epochs'] = 49
            if fault == 'variant':
                terminal['arguments']['variant'] = 'axis_scaled_fullref'
            if fault == 'dataset':
                terminal['arguments']['dataset'] = 'RGBNT201'
        write_json(run / 'result.json', terminal)
        (run / 'best.pth').write_bytes(b'checkpoint-fixture')
        (run / 'best_dev_arrays.npz').write_bytes(b'frozen-arrays-fixture')
        if not (waiting and dataset == 'RGBNT201'):
            write_json(training / (name + '_exit.json'), exit_value)

    def sleep(delay):
        assert delay == 240 and len(launches) == 4
        sleeps.append(delay)
        write_json(training / 'RGBNT201_axis_mass_fullref_s42_exit.json', dict(exit_code=0))

    def idle(gpu):
        assert gpu == 3
        idle_calls.append(gpu)

    class Child:
        pid = 12345

        def __init__(self, argv, env, stdout, stderr):
            assert env['CUDA_VISIBLE_DEVICES'] == '3'
            assert all(env[key] == '4' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
            assert argv[:2] == [sys.executable, '-u']
            self.argv = argv
            self.index = len(launches)
            launches.append(dict(argv=argv, env=env))
            stdout.write('mock-child-only\n')

        def wait(self):
            if fault == 'child_failure' and self.index == fault_stage:
                return 17
            smoke = '--smoke' in self.argv
            four = self.argv[2] == 'diagnose_mass_axis.py'
            stage_output = Path(self.argv[self.argv.index('--output') + 1])
            stage_output.mkdir()
            result = dict(optimizer_updates=0, state_tensor_versions_unchanged=True)
            parity_key = 'normal_inference_feature_max_error' if four and not smoke else 'normal_feature_max_error'
            result[parity_key] = 0
            if smoke:
                filename = 'smoke.json'
                result['status'] = 'PASS'
            elif four:
                filename = 'diagnostic.json'
                result.update(metrics={key: {} for key in ('00', '10', '01', '11')}, normal_distance_source='saved best_dev_arrays.npz/distances after exact feature parity')
            else:
                filename = 'result.json'
                result.update(status='COMPLETE', measurements={str(index): {} for index in range(13)})
            if self.index == fault_stage:
                if fault == 'updates':
                    result['optimizer_updates'] = 1
                elif fault == 'versions':
                    result['state_tensor_versions_unchanged'] = False
                elif fault == 'parity':
                    result[parity_key] = 1e-12
                elif fault == 'smoke_status':
                    result['status'] = 'FAIL'
                elif fault == 'four_count':
                    result['metrics'].pop('00')
                elif fault == 'missing_count':
                    result['measurements'].pop('0')
                elif fault == 'distance_source':
                    result['normal_distance_source'] = 'recomputed'
                elif fault and fault.startswith('input:'):
                    run = Path(self.argv[self.argv.index('--run-dir') + 1])
                    target = run.parent / (run.name + '_exit.json') if fault == 'input:exit' else run / fault.split(':', 1)[1]
                    target.write_bytes(target.read_bytes() + b'changed')
            write_json(stage_output / filename, result)
            return 0

    namespace = dict(argparse=argparse, hashlib=hashlib, json=json, os=os, Path=Path,
                     subprocess=types.SimpleNamespace(Popen=Child, STDOUT='STDOUT'),
                     sys=types.SimpleNamespace(executable=sys.executable),
                     time=types.SimpleNamespace(time=lambda: 123.0, sleep=sleep), idle=idle, write_json=write_json)
    main = function(launcher_source, 'main', namespace)
    previous = sys.argv
    sys.argv = ['launcher', '--training-root', str(training), '--output', str(output), '--data-root', 'fixture-data', '--pretrained', 'fixture-clip']
    raised = None
    try:
        main()
    except AssertionError as error:
        raised = repr(error)
    finally:
        sys.argv = previous
    if fault is None:
        assert raised is None and len(launches) == 12 and len(idle_calls) == 12
        summary = json.loads((output / 'controller_result.json').read_text())
        assert summary['status'] == 'COMPLETE' and summary['missing_conditions'] == 39 and summary['four_state_conditions'] == 12
        assert summary['optimizer_updates'] == summary['official_test_uses'] == 0
        assert len(summary['runs']) == 3
        for row in summary['runs']:
            assert [stage['stage'] for stage in row['stages']] == ['four_state_smoke', 'four_state_full', 'missing_smoke', 'missing_full']
            proofs = json.loads((output / row['name'] / 'frozen_inputs.json').read_text())
            assert len(proofs) == 4
        assert sleeps == ([240] if waiting else [])
    else:
        assert raised is not None and not (output / 'controller_result.json').exists()
        expected_count = 0 if fault in ('training_exit', 'terminal_status', 'epochs', 'variant', 'dataset') else fault_stage + 1
        assert len(launches) == expected_count
        if expected_count:
            stage = ('four_state_smoke', 'four_state_full', 'missing_smoke', 'missing_full')[fault_stage]
            folder = output / 'MSVR310_axis_mass_fullref_s42'
            assert (folder / (stage + '.log')).exists() and (folder / (stage + '_launch.json')).exists()
            receipt = json.loads((folder / (stage + '_exit.json')).read_text())
            assert receipt['exit_code'] == (17 if fault == 'child_failure' else 0)
            assert not (folder / 'controller_result.json').exists()
    return dict(case=label, child_count=len(launches), waits=sleeps, rejected=raised is not None)


helper_tree = ast.parse(HELPER.read_text(encoding='utf-8'))
helper_body = [node for node in helper_tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
               and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'sys.path.insert')]
helper_code = compile(ast.Module(body=helper_body, type_ignores=[]), str(HELPER), 'exec')


def helper_case(sandbox, label, fault=None):
    area = sandbox / label
    project = area / 'project'
    remote = area / 'remote'
    project.mkdir(parents=True)
    remote.mkdir()
    for name in (*all_sources, *NEW):
        for destination in (project,) if name in NEW else (project, remote):
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
    preflight = project / 'results/preflight'
    preflight.mkdir(parents=True)
    review = dict(status='PASS', blockers=[], helper_sha256=helper_sha, checked_source_sha256=initial)
    if fault == 'review_status':
        review['status'] = 'BLOCKED'
    if fault == 'review_blocker':
        review['blockers'] = ['fixture']
    if fault == 'review_helper_sha':
        review['helper_sha256'] = 'wrong'
    if fault == 'review_source_sha':
        review['checked_source_sha256'] = {}
    write_json(preflight / 'axis_collaboration_v5_frozen_review.json', review)
    fixture_launch = dict(launch, output=str(remote / 'runs/axis_collaboration_v5_mass'))
    write_json(preflight / 'axis_collaboration_v5_mass_launch.json', fixture_launch)
    run = Path(fixture_launch['output']) / 'development/MSVR310_axis_mass_fullref_s42'
    run.mkdir(parents=True)
    terminal = dict(status='COMPLETE', epochs=50, arguments=dict(variant='axis_mass_fullref'))
    exit_value = dict(exit_code=0)
    if fault == 'terminal':
        terminal['epochs'] = 49
    if fault == 'training_exit':
        exit_value['exit_code'] = 9
    write_json(run / 'result.json', terminal)
    write_json(run.parent / (run.name + '_exit.json'), exit_value)
    (run / 'best.pth').write_bytes(b'checkpoint-fixture')
    (run / 'best_dev_arrays.npz').write_bytes(b'frozen-arrays-fixture')
    frozen_output = remote / 'runs/axis_collaboration_v5_mass_frozen_development'
    if fault == 'local_old_source':
        (project / 'run_experiment.py').write_bytes(b'changed')
    if fault == 'remote_old_source':
        (remote / 'run_experiment.py').write_bytes(b'changed')
    if fault == 'new_exists':
        (remote / NEW[0]).write_bytes(b'existing')
    if fault == 'output_exists':
        frozen_output.mkdir()
    if fault == 'log_exists':
        Path(str(frozen_output) + '.log').write_text('existing', encoding='utf-8')
    calls, uploads, children = [], [], []

    def command(argv):
        assert argv[:2] == ['scp', 'fixture-option'] and argv[3].startswith('2026:')
        source, destination = Path(argv[2]), Path(argv[3].split(':', 1)[1])
        assert source.name in NEW and source.name == destination.name
        destination.write_bytes(source.read_bytes())
        uploads.append(source.name)
        return ''

    def check_output(argv, text):
        assert argv == ['nvidia-smi', '-i', '3', '--query-gpu=memory.used', '--format=csv,noheader,nounits'] and text
        return '500\n' if (fault == 'gpu_guard' or fault == 'gpu_start' and len(calls) == 2) else '153\n'

    def popen(argv, **kwargs):
        assert kwargs['cwd'] == remote and kwargs['start_new_session']
        assert argv[:3] == ['fixture-python', '-u', 'launch_mass_frozen_evaluation.py']
        assert argv[argv.index('--training-root') + 1] == fixture_launch['output'] + '/development'
        assert argv[argv.index('--data-root') + 1] == '/data/gaob/Re-ID/dataset'
        assert argv[argv.index('--pretrained') + 1] == '/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt'
        children.append(argv)
        return types.SimpleNamespace(pid=23456)

    def remote_python(host, code):
        assert host == '2026'
        calls.append(code)
        if len(calls) == 2:
            if fault == 'upload_changed':
                (remote / NEW[0]).write_bytes(b'changed')
            if fault == 'input_changed':
                (run / 'best.pth').write_bytes(b'changed')
        printed = []
        tree = ast.parse(code)
        body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        namespace = dict(hashlib=hashlib, json=json, os=os, Path=Path,
                         subprocess=types.SimpleNamespace(check_output=check_output, Popen=popen, STDOUT='STDOUT'),
                         time=types.SimpleNamespace(time=lambda: 321.0), print=lambda value: printed.append(value))
        exec(compile(ast.Module(body=body, type_ignores=[]), label + '-embedded', 'exec'), namespace)
        assert len(printed) == 1
        return printed[0]

    namespace = dict(__file__=str(HELPER), hashlib=hashlib, json=json, Path=Path,
                     PROJECT=project, HOSTS={'2026': (str(remote), 'fixture-python')}, OPTIONS=['fixture-option'],
                     command=command, remote_python=remote_python,
                     datetime=__import__('datetime').datetime, print=lambda *args, **kwargs: None)
    raised = None
    try:
        exec(helper_code, namespace)
    except AssertionError as error:
        raised = repr(error)
    if fault is None:
        assert raised is None and len(calls) == 2 and uploads == list(NEW) and len(children) == 1
        assert all(sha(remote / name) == digest for name, digest in all_sources.items())
        receipt = json.loads((preflight / 'axis_collaboration_v5_frozen_launch.json').read_text())
        assert receipt['host'] == '2026' and receipt['gpu'] == 3 and receipt['optimizer_updates'] == 0
        assert receipt['planned_missing_conditions'] == 39 and receipt['planned_four_state_conditions'] == 12
        assert len(receipt['source_sha256']) == len(all_sources) + 3
    else:
        assert raised is not None and not children
        assert not (preflight / 'axis_collaboration_v5_frozen_launch.json').exists()
        local_faults = ('review_status', 'review_blocker', 'review_helper_sha', 'review_source_sha', 'local_old_source')
        start_faults = ('upload_changed', 'input_changed', 'gpu_start', 'log_exists')
        assert len(calls) == (0 if fault in local_faults else 2 if fault in start_faults else 1)
        assert len(uploads) == (3 if fault in start_faults else 0)
    return dict(case=label, remote_mocks=len(calls), upload_mocks=len(uploads), process_mocks=len(children), rejected=raised is not None)


TRACE.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='fixtures_', dir=TRACE) as temporary:
    sandbox = Path(temporary).resolve()
    # Verify the sole generated directory before TemporaryDirectory's recursive cleanup.
    assert sandbox.parent == TRACE.resolve() and sandbox.name.startswith('fixtures_')
    record('launcher_full_happy_path', 'Actual launcher function: 3 datasets,12 child stages, each smoke before its full counterpart, GPU3 only, four-input proofs, totals39missing/12states.', [launcher_case(sandbox, 'launcher_happy')])
    record('launcher_waits_for_existing_training', 'Missing RGBNT201 exit produces one240second wait only after MSVR four stages; no training restart.', [launcher_case(sandbox, 'launcher_wait', waiting=True)])
    cases = [launcher_case(sandbox, 'launcher_' + fault, fault) for fault in ('training_exit', 'terminal_status', 'epochs', 'variant', 'dataset')]
    record('launcher_terminal_dependency_rejections', 'Nonzero exit/incomplete/49epochs/wrong variant/wrong dataset reject before any child.', cases)
    cases = [launcher_case(sandbox, 'launcher_child_' + str(index), 'child_failure', index) for index in (0, 1, 2, 3)]
    record('launcher_child_failure_trace_and_no_retry', 'Each stage nonzero exit preserves exclusive stdout/stderr log plus launch/exit JSON, then stops with no subsequent child.', cases)
    cases = [launcher_case(sandbox, 'launcher_gate_' + fault, fault, index) for fault, index in (('updates', 0), ('versions', 0), ('parity', 0), ('smoke_status', 0), ('four_count', 1), ('distance_source', 1), ('missing_count', 3))]
    record('launcher_result_semantic_gates', 'Rejects reported update/version change/nonzero feature error/failed smoke/incomplete counts/recomputed normal distance before next stage.', cases)
    cases = [launcher_case(sandbox, 'launcher_input_' + str(index), 'input:' + key) for index, key in enumerate(('best.pth', 'best_dev_arrays.npz', 'result.json', 'exit'))]
    record('launcher_four_frozen_input_proofs', 'Mutation of each of four frozen files after a stage is detected; no next stage or completed controller claim.', cases)
    record('helper_actual_embedded_guard_and_start', 'Actual helper and both embedded Python scripts run only against fixture paths/mock subprocess: guard,3new uploads,source/input/resource recheck,exclusive log,one isolated-session controller.', [helper_case(sandbox, 'helper_happy')])
    cases = [helper_case(sandbox, 'helper_' + fault, fault) for fault in ('review_status', 'review_blocker', 'review_helper_sha', 'review_source_sha', 'local_old_source')]
    record('helper_local_review_and_source_gates', 'Review status/blockers/helper SHA/exact3source map and old-source mismatch each prevent all remote calls.', cases)
    cases = [helper_case(sandbox, 'helper_' + fault, fault) for fault in ('remote_old_source', 'new_exists', 'output_exists', 'terminal', 'training_exit', 'gpu_guard')]
    record('helper_readonly_guard_rejections', 'Remote old-source/new-leaf/output/dependency/GPU guard rejects before uploads or process creation.', cases)
    cases = [helper_case(sandbox, 'helper_' + fault, fault) for fault in ('upload_changed', 'input_changed', 'gpu_start', 'log_exists')]
    record('helper_postupload_recheck_rejections', 'Changed uploaded source/frozen input/new GPU occupancy/existing log prevents process creation after3uploads; no retries or overwrite.', cases)

memory_reads = iter(('500', '1300', '153'))
idle_waits, idle_queries = [], []


def idle_check_output(argv, text):
    idle_queries.append(argv)
    assert argv == ['nvidia-smi', '-i', '3', '--query-gpu=memory.used', '--format=csv,noheader,nounits'] and text
    return next(memory_reads)


idle_fn = function(ROOT / 'launch_axis_scaled.py', 'idle', dict(subprocess=types.SimpleNamespace(check_output=idle_check_output), time=types.SimpleNamespace(sleep=idle_waits.append), print=lambda *args, **kwargs: None))
idle_fn(3)
assert idle_waits == [240, 240] and len(idle_queries) == 3
record('existing_idle_GPU3_nonpreemption', 'Actual reused idle function polls240seconds while memory>=500MiB, returns at153MiB; no signal, kill or other-GPU command.')

assert initial == {name: sha(ROOT / name) for name in NEW}
assert helper_sha == sha(HELPER)
assert all(sha(ROOT / name) == value for name, value in launch['source_sha256'].items())
record('final_source_bindings_unchanged', 'All3 reviewed leaf hashes, private helper and preserved18 source hashes still match their starting bytes.')
result = dict(status='PASS', checks_count=len(CHECKS), checks=CHECKS, helper_sha256=helper_sha,
              checked_source_sha256=initial, existing_source_count=18,
              real_remote_calls=0, real_process_spawns=0, torch_imports=0, gpu_forwards=0, optimizer_updates=0)
write_json(TRACE / '001-harness-results.json', result)
print('STDLIB_MOCK_PASS', len(CHECKS), flush=True)
