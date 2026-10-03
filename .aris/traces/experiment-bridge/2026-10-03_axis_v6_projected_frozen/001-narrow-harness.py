"""V6 frozen MSVR review: exact V5 delta plus stdlib orchestration mocks."""
import argparse
import ast
import base64
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_axis_v6_projected_frozen'
V5TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_axis_v5_frozen_eval'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_v6_projected_frozen26_deploy_20261003.py')
OBSERVER = Path('C:/Users/gb/.codex_tmp/demo_axis_v6_projected_frozen26_observe_20261003.py')
NEW = ('missing_projected_mass_development.py', 'diagnose_projected_mass_axis.py', 'launch_projected_mass_frozen_evaluation.py')
OLD = ('missing_mass_development.py', 'diagnose_mass_axis.py', 'launch_mass_frozen_evaluation.py')
CHECKS = []
old_harness_text = (V5TRACE / '001-narrow-harness.py').read_text(encoding='utf-8')
old_harness_ast = ast.parse(old_harness_text)


def previous_function_source(name):
    node = next(node for node in old_harness_ast.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.get_source_segment(old_harness_text, node)


for utility in ('sha', 'record', 'write_json', 'function'):
    exec(compile(previous_function_source(utility), 'V5-reviewed-harness-utility', 'exec'), globals())


v5_report_path = ROOT / 'results/preflight/axis_collaboration_v5_frozen_review.json'
v5_report_hash = sha(v5_report_path)
v5_report = json.loads(v5_report_path.read_text(encoding='utf-8'))
assert v5_report['status'] == 'PASS' and v5_report['checks_count'] == 22
assert all(sha(ROOT / name) == v5_report['checked_source_sha256'][name] for name in OLD)
initial = {name: sha(ROOT / name) for name in NEW}
helper_sha, observer_sha = sha(HELPER), sha(OBSERVER)
for path in (*(ROOT / name for name in NEW), HELPER, OBSERVER):
    compile(path.read_text(encoding='utf-8'), str(path), 'exec')
record('syntax_and_prior_review_preservation', 'Three new leaves and both private helpers compile. Prior PASS22 V5 report and all three V5 reviewed leaves remain unchanged.')

for old, new in zip(OLD[:2], NEW[:2]):
    expected = (ROOT / old).read_text(encoding='utf-8').replace('from run_mass_experiment import build, configuration, write_json', 'from run_projected_mass_experiment import build, configuration, write_json').replace("assert arguments.variant == 'axis_mass_fullref'", "assert arguments.variant == 'axis_mass_projected_fullref'")
    assert expected == (ROOT / new).read_text(encoding='utf-8')
    old_main = next(node for node in ast.parse((ROOT / old).read_text(encoding='utf-8')).body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    new_main = next(node for node in ast.parse((ROOT / new).read_text(encoding='utf-8').replace("'axis_mass_projected_fullref'", "'axis_mass_fullref'")).body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    assert ast.dump(old_main) == ast.dump(new_main)
record('two_evaluation_leaves_exact_delta_and_main_AST', 'Both files differ only in builder import and exact variant assertion. Their normalized main ASTs are identical, preserving all masks, GT, gallery, metric, frozen-input and parity logic.')

expected = (ROOT / OLD[2]).read_text(encoding='utf-8')
for before, after in (
    ('One frozen V5 checkpoint per dataset, four states and thirteen missing conditions.', 'Frozen V6 MSVR checkpoint, four states and thirteen missing conditions.'),
    ("('MSVR310','RGBNT201','RGBNT100')", "('MSVR310',)"),
    ('axis_mass_fullref', 'axis_mass_projected_fullref'),
    ('WAIT_V5_CLEAN50', 'WAIT_V6_CLEAN50'),
    ('diagnose_mass_axis.py', 'diagnose_projected_mass_axis.py'),
    ('missing_mass_development.py', 'missing_projected_mass_development.py'),
    ('missing_conditions=39,four_state_conditions=12', 'missing_conditions=13,four_state_conditions=4')):
    expected = expected.replace(before, after)
assert expected == (ROOT / NEW[2]).read_text(encoding='utf-8')
record('single_MSVR_launcher_exact_delta', 'Only dataset tuple, V6 variant/names, console label and final13/4 totals change. Original four stage gates, GPU3 idle and four-input before/after proofs are text-identical.')

failed_launch = json.loads((ROOT / 'results/preflight/axis_collaboration_v6_projected_launch.json').read_text(encoding='utf-8'))
expected22 = dict(failed_launch['source_sha256'])
assert len(expected22) == 22
assert sha(ROOT / 'projected_mass_axis_collaboration.py') == '559c9512b47c48b5047743177b1d1454595876852fa644aa8440f2720f17ae0a'
expected22['projected_mass_axis_collaboration.py'] = sha(ROOT / 'projected_mass_axis_collaboration.py')
assert all(sha(ROOT / name) == digest for name, digest in expected22.items())
launch = dict(failed_launch, source_sha256=expected22,
              output='/data/gaob/Re-ID/DeMo-DualAxis/runs/axis_collaboration_v6_projected_mass_trial_ampfix')
all_sources = dict(expected22, **{'missing_evaluation.py': sha(ROOT / 'missing_evaluation.py')})
receipt_present_at_review_start = (ROOT / 'results/preflight/axis_collaboration_v6_projected_ampfix_launch.json').exists()
record('prepared22_sources_without_fabricating_launch', 'Current prepared22 mapping equals failed receipt names with only the authorized projected-model two-cast hash replacement; other21 sources match. This mapping is a mock fixture, not a real AMPfix launch or NN success.')


def top_function(path, name):
    return next(node for node in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(node, ast.FunctionDef) and node.name == name)


for name in ('configuration', 'evaluate'):
    assert ast.dump(top_function(ROOT / 'run_mass_experiment.py', name)) == ast.dump(top_function(ROOT / 'run_projected_mass_experiment.py', name))
v5_class = next(node for node in ast.parse((ROOT / 'mass_axis_collaboration.py').read_text(encoding='utf-8')).body if isinstance(node, ast.ClassDef))
v6_class = next(node for node in ast.parse((ROOT / 'projected_mass_axis_collaboration.py').read_text(encoding='utf-8')).body if isinstance(node, ast.ClassDef))
assert ast.unparse(v6_class.bases[0]) == 'MassAxisCollaborationDeMo'
assert [node.name for node in v6_class.body if isinstance(node, ast.FunctionDef)] == ['forward']


def eval_prefix(cls):
    forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    index = next(index for index, node in enumerate(forward.body) if isinstance(node, ast.If) and ast.unparse(node.test) == 'not self.training')
    return ast.dump(ast.Module(body=forward.body[:index + 1], type_ignores=[]))


assert eval_prefix(v5_class) == eval_prefix(v6_class)
record('V6_eval_prefix_configuration_and_inherited_states', 'V6 evaluation forward through return is AST-identical to V5; class overrides only forward and inherits fuse/controlled_states. Configuration/evaluate ASTs are identical. Training auxiliary casts are outside the evaluation prefix and not accepted as NN evidence here.')

# Reuse only existing reviewed mock function definitions, with explicit V6 fixture deltas.
launcher_source = ROOT / NEW[2]
launcher_fixture = previous_function_source('launcher_case')
for before, after in (
    ("('MSVR310', 'RGBNT201', 'RGBNT100')", "('MSVR310',)"),
    ('axis_mass_fullref', 'axis_mass_projected_fullref'),
    ("waiting and dataset == 'RGBNT201'", "waiting and dataset == 'MSVR310'"),
    ("training / 'RGBNT201_axis_mass_projected_fullref_s42_exit.json'", "training / 'MSVR310_axis_mass_projected_fullref_s42_exit.json'"),
    ('delay == 240 and len(launches) == 4', 'delay == 240 and len(launches) == 0'),
    ('len(launches) == 12 and len(idle_calls) == 12', 'len(launches) == 4 and len(idle_calls) == 4'),
    ("summary['missing_conditions'] == 39 and summary['four_state_conditions'] == 12", "summary['missing_conditions'] == 13 and summary['four_state_conditions'] == 4"),
    ("len(summary['runs']) == 3", "len(summary['runs']) == 1"),
    ('diagnose_mass_axis.py', 'diagnose_projected_mass_axis.py')):
    launcher_fixture = launcher_fixture.replace(before, after)
exec(compile(launcher_fixture, 'V6-adapted-launcher-fixture', 'exec'), globals())

helper_tree = ast.parse(HELPER.read_text(encoding='utf-8'))
helper_body = [node for node in helper_tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
               and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'sys.path.insert')]
helper_code = compile(ast.Module(body=helper_body, type_ignores=[]), str(HELPER), 'exec')
helper_fixture = previous_function_source('helper_case')
for before, after in (
    ('axis_collaboration_v5_frozen_review.json', 'axis_collaboration_v6_projected_frozen_review.json'),
    ('axis_collaboration_v5_mass_launch.json', 'axis_collaboration_v6_projected_ampfix_launch.json'),
    ('axis_collaboration_v5_mass_frozen_development', 'axis_collaboration_v6_projected_mass_frozen_trial'),
    ('axis_collaboration_v5_frozen_launch.json', 'axis_collaboration_v6_projected_frozen_launch.json'),
    ('axis_collaboration_v5_mass', 'axis_collaboration_v6_projected_mass_trial_ampfix'),
    ('axis_mass_fullref', 'axis_mass_projected_fullref'),
    ('launch_mass_frozen_evaluation.py', 'launch_projected_mass_frozen_evaluation.py'),
    ("receipt['planned_missing_conditions'] == 39 and receipt['planned_four_state_conditions'] == 12", "receipt['planned_missing_conditions'] == 13 and receipt['planned_four_state_conditions'] == 4"),
    ('except AssertionError as error:', 'except (AssertionError, FileNotFoundError) as error:'),
    ("'review_source_sha', 'local_old_source')", "'review_source_sha', 'local_old_source', 'launch_missing')"),
    ("'gpu_start', 'log_exists')", "'gpu_start', 'log_exists', 'disk_start')"),
    ("write_json(preflight / 'axis_collaboration_v6_projected_ampfix_launch.json', fixture_launch)", "if fault != 'launch_missing':\n        write_json(preflight / 'axis_collaboration_v6_projected_ampfix_launch.json', fixture_launch)"),
    ("namespace = dict(hashlib=hashlib, json=json, os=os, Path=Path,", "namespace = dict(hashlib=hashlib, json=json, os=os, Path=Path,\n                         shutil=types.SimpleNamespace(disk_usage=lambda root: types.SimpleNamespace(free=200000000 if fault == 'disk_guard' or fault == 'disk_start' and len(calls) == 2 else 500000000)),")):
    helper_fixture = helper_fixture.replace(before, after)
exec(compile(helper_fixture, 'V6-adapted-helper-fixture', 'exec'), globals())


def observer_case(sandbox, label, fault=None, advance=False):
    area = sandbox / label
    project, remote = area / 'project', area / 'remote'
    project.mkdir(parents=True)
    remote.mkdir()
    preflight = project / 'results/preflight'
    preflight.mkdir(parents=True)
    sources = dict(all_sources, **initial)
    for name in sources:
        path = remote / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    output = remote / 'runs/axis_collaboration_v6_projected_mass_frozen_trial'
    output.mkdir(parents=True)
    name = 'MSVR310_axis_mass_projected_fullref_s42'
    folder = output / name
    folder.mkdir()
    logfile = Path(str(output) + '.log')
    logfile.write_text('fixture-controller-log\n', encoding='utf-8')
    observer_launch = dict(output=str(output), log=str(logfile), pid=12345, source_sha256=sources)
    write_json(preflight / 'axis_collaboration_v6_projected_frozen_launch.json', observer_launch)
    target = project / 'results/axis_collaboration_v6_projected_mass_frozen_trial'
    stages = ('four_state_smoke', 'four_state_full', 'missing_smoke', 'missing_full')

    def create_stage(stage, code=0):
        stage_dir = folder / stage
        stage_dir.mkdir(exist_ok=True)
        write_json(folder / (stage + '_launch.json'), dict(pid=3333, gpu=3))
        write_json(folder / (stage + '_exit.json'), dict(exit_code=code))
        (folder / (stage + '.log')).write_text('fixture-stage-output\n', encoding='utf-8')
        result_name = 'smoke.json' if stage.endswith('smoke') else 'diagnostic.json' if stage == 'four_state_full' else 'result.json'
        write_json(stage_dir / result_name, dict(status='PASS' if stage.endswith('smoke') else 'COMPLETE', optimizer_updates=0))
        (stage_dir / 'per_query.csv').write_text('query_index,AP\n0,1\n', encoding='utf-8')
        (stage_dir / 'states.npz').write_bytes(b'fixture-binary-never-transferred')

    def complete():
        for stage in stages:
            create_stage(stage)
        write_json(folder / 'controller_result.json', dict(status='PASS', optimizer_updates=0))
        write_json(folder / 'frozen_inputs.json', dict(fixture='four-input-proofs'))
        write_json(output / 'controller_result.json', dict(status='COMPLETE', missing_conditions=13, four_state_conditions=4, optimizer_updates=0))

    if advance or fault in ('stage_failure', 'controller_dead'):
        create_stage(stages[0])
        if fault == 'stage_failure':
            create_stage(stages[1], 17)
    else:
        complete()
    if fault == 'source_mismatch':
        (remote / 'run_experiment.py').write_bytes(b'changed')
    protected = target / name / stages[0] / 'smoke.json'
    if fault == 'raw_exists':
        protected.parent.mkdir(parents=True)
        protected.write_bytes(b'preserve-existing')
    calls, ps_calls, sleeps, printed, file_counts = [], [], [], [], []

    def ps(argv, **kwargs):
        assert argv == ['ps', '-p', '12345', '-o', 'pid=']
        assert kwargs == dict(capture_output=True, text=True)
        ps_calls.append(argv)
        return types.SimpleNamespace(stdout='' if fault in ('stage_failure', 'controller_dead') else '12345\n')

    def remote_python(host, code):
        assert host == '2026'
        calls.append(code)
        tree = ast.parse(code)
        body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        captured = []
        namespace = dict(base64=base64, hashlib=hashlib, json=json, Path=Path,
                         subprocess=types.SimpleNamespace(run=ps), print=captured.append)
        exec(compile(ast.Module(body=body, type_ignores=[]), label + '-remote-readonly-mock', 'exec'), namespace)
        assert len(captured) == 1
        value = json.loads(captured[0])
        file_counts.append(len(value['files']))
        assert all(not path.endswith(('.npz', '.pth')) for path in value['files'])
        if fault == 'transfer_sha':
            proof = next(iter(value['files'].values()))
            proof['data'] = base64.b64encode(b'X' * proof['bytes']).decode()
        return json.dumps(value)

    def sleep(seconds):
        assert seconds == 240 and advance and len(sleeps) == 0
        sleeps.append(seconds)
        complete()

    source = ast.parse(OBSERVER.read_text(encoding='utf-8'))
    body = [node for node in source.body if not isinstance(node, (ast.Import, ast.ImportFrom))
            and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'sys.path.insert')]
    namespace = dict(base64=base64, datetime=datetime, hashlib=hashlib, json=json, PROJECT=project,
                     remote_python=remote_python, time=types.SimpleNamespace(sleep=sleep),
                     print=lambda *args, **kwargs: printed.append(args))
    raised = None
    try:
        exec(compile(ast.Module(body=body, type_ignores=[]), str(OBSERVER), 'exec'), namespace)
    except AssertionError as error:
        raised = repr(error)
    if fault is None:
        assert raised is None and sleeps == ([240] if advance else [])
        assert len(list(target.glob('*/*/intake.json'))) == 4
        raw_paths = [path for path in target.rglob('*') if path.is_file() and (path.parent.name in stages or path.suffix == '.log' or path.name.endswith(('_launch.json', '_exit.json')))]
        snapshots = {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for path in raw_paths}
        assert namespace['collect']() is True
        assert file_counts[-1] == 0
        assert snapshots == {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for path in raw_paths}
        assert all(not path.name.endswith(('.npz', '.pth')) for path in target.rglob('*'))
        assert any(args[0] == 'V6_PROJECTED_FROZEN_ALL13_MISSING_AND4_FOUR_STATES_COLLECTED' for args in printed)
        assert len(json.loads((preflight / 'axis_collaboration_v6_projected_frozen_latest_snapshot.json').read_text())['rows']) == 4
    else:
        assert raised is not None and not sleeps
        if fault in ('stage_failure', 'controller_dead'):
            snapshot = json.loads((preflight / 'axis_collaboration_v6_projected_frozen_latest_snapshot.json').read_text())
            assert not snapshot['controller_live']
            assert (target / name / stages[0] / 'intake.json').exists()
            if fault == 'stage_failure':
                intake = json.loads((target / name / stages[1] / 'intake.json').read_text())
                assert intake['exit_code'] == 17
                assert any(args[0] == 'V6_PROJECTED_FROZEN_PRIMARY_FAILURE' for args in printed)
        if fault == 'source_mismatch':
            assert not ps_calls and not target.exists()
        if fault == 'transfer_sha':
            assert not target.exists()
        if fault == 'raw_exists':
            assert protected.read_bytes() == b'preserve-existing'
            assert not (protected.parent / 'intake.json').exists()
    return dict(case=label, remote_mocks=len(calls), ps_mocks=len(ps_calls), transfer_file_counts=file_counts, sleeps=sleeps, rejected=raised is not None)


TRACE.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix='fixtures_', dir=TRACE) as temporary:
    sandbox = Path(temporary).resolve()
    assert sandbox.parent == TRACE.resolve() and sandbox.name.startswith('fixtures_')
    record('launcher_single_MSVR_happy_and_wait', 'Actual V6 launcher produces four GPU3 stages and totals13/4; missing fresh50 exit causes240second wait before any child.', [launcher_case(sandbox, 'launch_happy'), launcher_case(sandbox, 'launch_wait', waiting=True)])
    record('launcher_training_dependency_rejections', 'Nonzero exit, incomplete status,49epochs, wrong variant or dataset reject before child creation.', [launcher_case(sandbox, 'launch_' + fault, fault) for fault in ('training_exit', 'terminal_status', 'epochs', 'variant', 'dataset')])
    record('launcher_four_stage_failure_no_retry', 'Each failed stage keeps its log, launch and exit JSON then stops without later stages.', [launcher_case(sandbox, 'launch_fail_' + str(index), 'child_failure', index) for index in range(4)])
    record('launcher_semantic_and_frozen_file_gates', 'All original update/version/parity/status/count/distance provenance and four frozen input guards remain effective.', [launcher_case(sandbox, 'launch_gate_' + fault, fault, index) for fault, index in (('updates', 0), ('versions', 0), ('parity', 0), ('smoke_status', 0), ('four_count', 1), ('distance_source', 1), ('missing_count', 3))] + [launcher_case(sandbox, 'launch_input_' + str(index), 'input:' + name) for index, name in enumerate(('best.pth', 'best_dev_arrays.npz', 'result.json', 'exit'))])
    record('helper_happy_single_MSVR_source_scope', 'Actual helper guards, uploads exactly three new leaves, rechecks26 total sources/four inputs/GPU3/disk and starts one isolated controller.', [helper_case(sandbox, 'helper_happy')])
    record('helper_local_and_future_receipt_gates', 'Missing real prerequisite receipt, review failures or local source mismatch stop before all remote calls.', [helper_case(sandbox, 'helper_' + fault, fault) for fault in ('launch_missing', 'review_status', 'review_blocker', 'review_helper_sha', 'review_source_sha', 'local_old_source')])
    record('helper_guard_and_postupload_failures', 'Old source, existing outputs/new file, incomplete training/nonzero exit, occupied GPU3 and disk200MB boundary reject; changed uploads/frozen inputs and postupload GPU/disk/log guards reject without spawn or retry.', [helper_case(sandbox, 'helper_' + fault, fault) for fault in ('remote_old_source', 'new_exists', 'output_exists', 'terminal', 'training_exit', 'gpu_guard', 'disk_guard', 'upload_changed', 'input_changed', 'gpu_start', 'disk_start', 'log_exists')])
    record('observer_complete_and_incremental_one_time_intake', 'Actual observer receives only four MSVR stages, retains raw text byte proofs once, skips all closed intakes on reread, excludes NPZ/PTH and polls240seconds only while incomplete.', [observer_case(sandbox, 'observer_complete'), observer_case(sandbox, 'observer_incremental', advance=True)])
    record('observer_failure_trace_and_fail_closed', 'Failed stage and dead controller retain snapshot/raw primary failure before stopping. Source mismatch, transit SHA mismatch and existing raw files reject without overwrite/retry.', [observer_case(sandbox, 'observer_' + fault, fault) for fault in ('stage_failure', 'controller_dead', 'source_mismatch', 'transfer_sha', 'raw_exists')])

assert v5_report_hash == sha(v5_report_path)
assert all(sha(ROOT / name) == v5_report['checked_source_sha256'][name] for name in OLD)
assert initial == {name: sha(ROOT / name) for name in NEW}
assert helper_sha == sha(HELPER) and observer_sha == sha(OBSERVER)
assert all(sha(ROOT / name) == digest for name, digest in expected22.items())
record('final_source_and_prior_artifact_preservation', 'Reviewed V6 leaves/private helpers, current prepared22 dependencies and all original V5 leaves/report remain byte-exact through review.')
result = dict(status='PASS', checks_count=len(CHECKS), checks=CHECKS, helper_sha256=helper_sha,
              observer_sha256=observer_sha, checked_source_sha256=initial,
              prepared_dependency_sha256=expected22, ampfix_launch_present_at_review_start=receipt_present_at_review_start,
              prior_review_sha256=v5_report_hash, real_remote_calls=0, real_process_spawns=0, torch_imports=0, gpu_forwards=0, optimizer_updates=0)
write_json(TRACE / '001-harness-results.json', result)
print('V6_STDLIB_MOCK_PASS', len(CHECKS), flush=True)
