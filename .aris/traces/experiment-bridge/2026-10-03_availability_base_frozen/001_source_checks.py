"""Reviewer-owned source/stdlib-only checks. Never import the experiment modules."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = Path(__file__).resolve().parent
HELPER = Path('C:/Users/gb/.codex_tmp/demo_available_base_deploy_20261003.py')
FILES = ('availability_base_intervention.py', 'verify_availability_base_intervention.py',
         'missing_available_base_development.py', 'launch_available_base_frozen.py')
PLAN = ROOT / 'results/preflight/availability_base_plan.json'
CHECKS = []


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add(name, evidence, kind='AST_OR_SOURCE_CHECK'):
    CHECKS.append(dict(name=name, status='PASS', kind=kind, evidence=evidence))


def extracted(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def capture_sources():
    snapshot = TRACE / 'sources_001'
    snapshot.mkdir(exist_ok=False)
    names = list(FILES) + ['modeling/moe/AttnMOE.py', 'modeling/make_model.py', 'modeling/meta_arch.py',
        'mass_axis_collaboration.py', 'scaled_axis_collaboration.py', 'axis_collaboration.py',
        'dual_axis.py', 'missing_anytoany_development.py', 'full_evaluation.py', 'missing_evaluation.py',
        'run_experiment.py', 'run_mass_experiment.py', 'experiment_data.py', 'utils/reid_evaluation.py',
        'launch_axis_scaled.py', 'launch_runs.py', 'collect_results.py',
        'configs/RGBNT100/DeMo.yml', 'configs/RGBNT201/DeMo.yml', 'configs/MSVR310/DeMo.yml',
        'results/preflight/availability_base_plan.json', 'results/preflight/anytoany49_2026_launch.json',
        'results/preflight/anytoany49_2027_launch.json']
    manifest = {}
    for name in names:
        path = ROOT / name
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        manifest[name] = dict(sha256=sha(path), bytes=path.stat().st_size)
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    (snapshot / HELPER.name).write_bytes(HELPER.read_bytes())
    ast.parse(HELPER.read_text(encoding='utf-8'), filename=str(HELPER))
    manifest[str(HELPER)] = dict(sha256=sha(HELPER), bytes=HELPER.stat().st_size)
    write_json(TRACE / '001_source_manifest.json', manifest)
    for name in FILES:
        result = subprocess.run(['git', 'diff', '--no-index', '--', '/dev/null', name],
                                cwd=ROOT, capture_output=True)
        assert result.returncode == 1
        (TRACE / ('001_' + name + '.diff')).write_bytes(result.stdout)
        (TRACE / ('001_' + name + '.diff.stderr.txt')).write_bytes(result.stderr)
    result = subprocess.run(['git', 'diff', '--no-index', '--', 'missing_anytoany_development.py',
                             'missing_available_base_development.py'], cwd=ROOT, capture_output=True)
    assert result.returncode == 1
    (TRACE / '001_evaluator_against_parent.diff').write_bytes(result.stdout)
    (TRACE / '001_evaluator_against_parent.diff.stderr.txt').write_bytes(result.stderr)
    for label, argv in [('head', ['git', 'rev-parse', 'HEAD']), ('status', ['git', 'status', '--short'])]:
        result = subprocess.run(argv, cwd=ROOT, capture_output=True)
        assert result.returncode == 0
        (TRACE / ('001_git_' + label + '.txt')).write_bytes(result.stdout)
    add('syntax_and_raw_source_capture', dict(source_count=len(manifest), sources='sources_001',
        source_diff='001_*.diff', no_source_module_import=True))
    return manifest


def check_source_binding():
    tree = ast.parse(HELPER.read_text(encoding='utf-8'))
    outer = next(node for node in tree.body if isinstance(node, ast.For))
    extension = next(node for node in outer.body if isinstance(node, ast.For))
    extras = ast.literal_eval(extension.iter)
    assert extras == ('modeling/moe/AttnMOE.py', 'modeling/meta_arch.py', 'launch_axis_scaled.py', 'launch_runs.py')
    new_sources = {name: sha(ROOT / name) for name in FILES}
    details = []
    for host in ('2026', '2027'):
        parent = json.loads((ROOT / f'results/preflight/anytoany49_{host}_launch.json').read_text())
        original = parent['source_sha256']
        assert not set(extras).intersection(original), 'Fresh overrides must not replace parent-bound digests'
        assert not set(new_sources).intersection(original)
        mismatches = [name for name, expected in original.items() if sha(ROOT / name) != expected]
        assert not mismatches, mismatches
        scope = {**original, **{name: sha(ROOT / name) for name in extras}, **new_sources}
        direct = set()
        for name in FILES:
            for node in ast.walk(ast.parse((ROOT / name).read_text(encoding='utf-8'))):
                if isinstance(node, ast.ImportFrom) and node.module:
                    target = node.module.replace('.', '/') + '.py'
                    if (ROOT / target).exists():
                        direct.add(target)
        if host == '2027':
            direct.remove('run_mass_experiment.py')  # demo-only jobs never enter this import branch
        assert direct <= set(scope), direct - set(scope)
        details.append(dict(host=host, parent_source_count=len(original), checked_total=len(scope),
                            direct_local_imports=sorted(direct), parent_mismatches=mismatches,
                            parent_digest_overwrites=[]))
    add('parent_bound_sources_and_actual_direct_imports', details, 'STDLIB_HASH_AND_AST_CHECK')
    write_json(TRACE / '001_initial_concern_and_resolution.json', dict(
        initial_finding=dict(file=str(HELPER), line=24,
            original_line="for name in ('modeling/moe/AttnMOE.py','modeling/meta_arch.py'):",
            issue='Actual scheduler import launch_axis_scaled.idle -> launch_runs.write_json was absent from either parent source map and fresh extension.',
            source='Original source read returned by review tool before implementation-owner correction; no original byte snapshot was taken before correction.'),
        resolution=dict(file=str(HELPER), line=24, current_extension=list(extras),
            scope='The implementation owner added only the two evidenced scheduler sources.',
            status='RESOLVED_SOURCE_ONLY', helper_sha256=sha(HELPER)),
        original_tool_trace=dict(command='rg --files .aris/traces/experiment-bridge/2026-10-03_availability_base_frozen',
            exit_code=1, output='Directory did not exist before this reviewer created it.',
            classification='Read-only inventory result, not a production failure')))


def controller_case(label, variant='demo', failure=None):
    folder = TRACE / 'controller_mocks_001' / label
    folder.mkdir(parents=True, exist_ok=False)
    dataset = 'MSVR310'
    run = folder / (dataset + '_' + variant + '_s42')
    run.mkdir()
    terminal = dict(status='COMPLETE', epochs=50, arguments=dict(dataset=dataset, variant=variant, seed=42))
    write_json(run / 'result.json', terminal)
    (run / 'best.pth').write_bytes(b'REVIEW_MOCK_BYTES_NOT_A_CHECKPOINT')
    (run / 'best_dev_arrays.npz').write_bytes(b'REVIEW_MOCK_BYTES_NOT_AN_ARRAY')
    exit_file = run / 'exit.json' if variant == 'demo' else run.parent / (run.name + '_exit.json')
    write_json(exit_file, dict(exit_code=0))
    inputs = [run / 'best.pth', run / 'best_dev_arrays.npz', run / 'result.json', exit_file]
    proofs = {path.name: dict(bytes=path.stat().st_size, sha256=sha(path)) for path in inputs}
    jobs = [dict(run=str(run), gpu=2, dataset=dataset, variant=variant, frozen_inputs=proofs)]
    jobs_path = folder / 'jobs.json'
    write_json(jobs_path, jobs)
    output = folder / 'output'
    launches, waits = [], []
    if failure == 'input_mismatch':
        (run / 'best.pth').write_bytes(b'PRELAUNCH_CHANGED_MOCK')

    class Child:
        def __init__(self, argv, stdout, stderr, env):
            smoke = '--smoke' in argv
            stage = 'smoke' if smoke else 'full'
            assert argv[2] == 'missing_available_base_development.py'
            assert env['CUDA_VISIBLE_DEVICES'] == '2'
            assert all(env[key] == '4' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
            launches.append(dict(stage=stage, argv=argv, gpu=env['CUDA_VISIBLE_DEVICES']))
            self.pid = 10000 + len(launches)
            self.code = 7 if failure == stage + '_exit' else 0
            stdout.write('REVIEW STDLIB MOCK; NO CHILD PROCESS OR NEURAL EXECUTION\n')
            result_path = Path(argv[argv.index('--output') + 1])
            result_path.mkdir()
            result = dict(status='PASS' if smoke else 'COMPLETE', normal_feature_max_error=0,
                          optimizer_updates=0, state_tensor_versions_unchanged=True,
                          parameter_objects_and_state_keys_unchanged=True,
                          base_mask_tensor_contract=dict(status='PASS'),
                          measurements={str(i): {} for i in range(49)})
            if smoke and failure == 'smoke_parity':
                result['normal_feature_max_error'] = 0.01
            if smoke and failure == 'smoke_tensor_contract':
                result['base_mask_tensor_contract']['status'] = 'FAIL'
            if smoke and failure == 'smoke_input_mutation':
                (run / 'best.pth').write_bytes(b'POSTSMOKE_CHANGED_MOCK')
            if not smoke and failure == 'full_pair_count':
                result['measurements'].pop('48')
            if failure != stage + '_missing_result':
                write_json(result_path / ('smoke.json' if smoke else 'result.json'), result)

        def wait(self):
            return self.code

    argv = ['review-mock', '--jobs-json', str(jobs_path), '--output', str(output),
            '--data-root', '/REVIEW_MOCK_DATA', '--pretrained', '/REVIEW_MOCK_PRETRAINED']
    ns = dict(argparse=argparse, ThreadPoolExecutor=ThreadPoolExecutor, hashlib=hashlib, json=json,
              os=os, Path=Path, subprocess=SimpleNamespace(Popen=Child, STDOUT=-2),
              sys=SimpleNamespace(executable=sys.executable), time=time, write_json=write_json,
              idle=lambda gpu: waits.append(gpu))
    extracted(ROOT / 'launch_available_base_frozen.py', ['main'], ns)
    old_argv = sys.argv
    sys.argv = argv
    error = None
    try:
        ns['main']()
    except (AssertionError, FileNotFoundError) as exc:
        error = type(exc).__name__
        (folder / 'expected_failure_trace.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        sys.argv = old_argv
    if failure:
        assert error is not None, (label, 'expected guard did not stop')
        assert not (output / 'controller_result.json').exists()
        expected = 0 if failure == 'input_mismatch' else 2 if failure.startswith('full_') else 1
        assert len(launches) == expected, (label, launches)
    else:
        assert error is None
        assert [row['stage'] for row in launches] == ['smoke', 'full']
        result = json.loads((output / 'controller_result.json').read_text())
        assert result['status'] == 'COMPLETE' and result['availability_pairs'] == 49
        assert {path.name: dict(bytes=path.stat().st_size, sha256=sha(path)) for path in inputs} == proofs
    if failure in ('smoke_exit', 'full_exit'):
        stage = failure.removesuffix('_exit')
        result = json.loads((output / run.name / (stage + '_exit.json')).read_text())
        assert result['exit_code'] == 7
    write_json(folder / 'mock_check.json', dict(label=label, expected_failure=failure,
        actual_error=error, launched=launches, idle_calls=waits, passed=True,
        execution='Actual AST-extracted controller main with stdlib child/result mocks; no process launch'))
    add('controller_' + label, dict(actual_error=error, launches=[item['stage'] for item in launches],
        trace=str(folder.relative_to(TRACE))), 'EXECUTED_STDLIB_MOCK')


def idle_check():
    used = iter(('500\n', '900\n', '499\n'))
    waits = []
    ns = dict(subprocess=SimpleNamespace(check_output=lambda *a, **kw: next(used)),
              time=SimpleNamespace(sleep=lambda seconds: waits.append(seconds)))
    extracted(ROOT / 'launch_axis_scaled.py', ['idle'], ns)
    ns['idle'](3)
    assert waits == [240, 240]
    add('idle_waits_240_and_returns_only_below500', dict(memory_sequence=[500, 900, 499],
        observed_sleep_requests=waits, actual_sleep_seconds=0), 'EXECUTED_STDLIB_MOCK')


def static_invariants():
    plan = json.loads(PLAN.read_text())
    jobs = plan['jobs']
    assert len(jobs) == 6
    assert {(job['host'], job['gpu']) for job in jobs} == {
        ('2026', 0), ('2026', 1), ('2026', 2), ('2026', 3), ('2027', 1), ('2027', 2)}
    demo100 = next(job for job in jobs if job['dataset'] == 'RGBNT100' and job['variant'] == 'demo')
    assert demo100['host'] == '2026' and demo100['run'] == 'runs/dynamic_amp_comparison/RGBNT100_demo_s42'
    assert {(job['dataset'], job['variant']) for job in jobs} == {
        (dataset, variant) for dataset in ('MSVR310', 'RGBNT201', 'RGBNT100')
        for variant in ('demo', 'axis_mass_fullref')}
    add('six_jobs_and_samehost_RGBNT100_canonical_path', jobs)
    relations = next(node for node in ast.parse((ROOT / 'dual_axis.py').read_text()).body
                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id == 'RELATIONS')
    assert ast.literal_eval(relations.value) == ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
    fusion = (ROOT / 'availability_base_intervention.py').read_text()
    assert 'return original_fusion(*inputs)' in fusion
    assert 'logits.masked_fill(~eligible[:, None, None, :], -torch.inf).softmax(-1)' in fusion
    assert 'torch.cat(heads, -1) * eligible[..., None]' in fusion
    assert '(model.rgb_reduce, model.nir_reduce, model.tir_reduce)' in fusion
    add('relation_order_and_guard_operators_present', dict(relations=ast.literal_eval(relations.value),
        scope='Static source operators only; no numerical tensor execution'))
    forbidden = []
    for name in FILES:
        tree = ast.parse((ROOT / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = ast.unparse(node.func)
                if target.endswith(('.backward', '.step', '.train', '.save')) or target == 'official_records':
                    forbidden.append(dict(file=name, line=node.lineno, target=target))
    assert not forbidden
    add('new_files_no_training_checkpoint_write_or_test_calls', dict(forbidden_calls=forbidden,
        qualification='Existing imported run modules contain training definitions; only build/configuration/write_json are called.'))


def main():
    assert sys.version_info[:2] == (3, 13)
    before_modules = set(sys.modules)
    manifest = capture_sources()
    check_source_binding()
    static_invariants()
    controller_case('demo_success')
    controller_case('v5_success', variant='axis_mass_fullref')
    for failure in ('input_mismatch', 'smoke_exit', 'smoke_missing_result', 'smoke_parity',
                    'smoke_tensor_contract', 'smoke_input_mutation', 'full_exit', 'full_pair_count'):
        controller_case(failure, failure=failure)
    idle_check()
    imported = sorted(set(sys.modules) - before_modules)
    assert not any(name == 'torch' or name.startswith('torch.') or name == 'numpy' or name.startswith('numpy.')
                   or name.startswith('modeling') for name in sys.modules)
    for name in FILES:
        assert sha(ROOT / name) == manifest[name]['sha256']
    assert sha(HELPER) == manifest[str(HELPER)]['sha256']
    result = dict(status='PASS', observed_at=datetime.now(timezone.utc).isoformat(),
        python=sys.executable, python_version=sys.version, checks=CHECKS,
        check_count=len(CHECKS), neural_imports=[], runtime_tensor_execution=False,
        real_gpu_execution=False, ssh_or_network_calls=0, real_experiment_child_processes=0,
        real_read_only_git_processes=7,
        imported_during_checks=imported,
        checked_source_sha256={name: sha(ROOT / name) for name in FILES},
        helper_sha256=sha(HELPER), plan_sha256=sha(PLAN))
    write_json(TRACE / '001_checks_result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
