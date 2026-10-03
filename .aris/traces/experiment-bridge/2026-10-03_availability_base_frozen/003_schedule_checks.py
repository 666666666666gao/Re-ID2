"""Supplementary stdlib-only review of the later global-four scheduling change."""
import ast
import contextlib
from datetime import datetime, timezone
import difflib
import hashlib
import io
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = Path(__file__).resolve().parent
HELPER = Path('C:/Users/gb/.codex_tmp/demo_available_base_deploy_20261003.py')
PLAN = ROOT / 'results/preflight/availability_base_plan.json'
CASES = ('RGBNT201_axis_mass_fullref_s42', 'MSVR310_axis_mass_fullref_s42')
CHECKS = []


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loop_and_gate(helper):
    outer = next(node for node in ast.parse(helper.read_text(encoding='utf-8')).body if isinstance(node, ast.For))
    gate = outer.body[0]
    assert isinstance(gate, ast.If)
    return outer, gate


def release_code(helper, folder):
    _, gate = loop_and_gate(helper)
    ns = dict(receipts=[dict(output=str(folder))])
    exec(compile(ast.Module(body=gate.body[:2], type_ignores=[]), str(helper), 'exec'), ns)
    ast.parse(ns['code'])
    return ns['code']


def predicate_case(label, mode, helper=HELPER, expected_release=None, expected_error=None):
    folder = TRACE / 'schedule_mocks_003' / label
    folder.mkdir(parents=True, exist_ok=False)
    for name in CASES:
        (folder / name).mkdir()
    if mode in ('one_complete', 'both_complete', 'controller_fail', 'controller_missing_full'):
        write_json(folder / CASES[0] / 'controller_result.json', dict(status='FAIL' if mode == 'controller_fail' else 'PASS'))
        if mode != 'controller_missing_full':
            write_json(folder / CASES[0] / 'full_exit.json', dict(exit_code=0))
    if mode == 'both_complete':
        write_json(folder / CASES[1] / 'controller_result.json', dict(status='PASS'))
        write_json(folder / CASES[1] / 'full_exit.json', dict(exit_code=0))
    if mode in ('full_exit_pending', 'full_failed'):
        write_json(folder / CASES[0] / 'full_exit.json', dict(exit_code=1 if mode == 'full_failed' else 0))
    if mode == 'smoke_failed':
        write_json(folder / CASES[0] / 'smoke_exit.json', dict(exit_code=1))
    stdout = io.StringIO()
    error = None
    code = release_code(helper, folder)
    (folder / 'extracted_predicate.py').write_text(code, encoding='utf-8')
    try:
        with contextlib.redirect_stdout(stdout):
            exec(compile(code, str(folder / 'extracted_predicate.py'), 'exec'), {})
    except (AssertionError, FileNotFoundError) as exc:
        error = type(exc).__name__
        (folder / 'original_failure_trace.txt').write_text(traceback.format_exc(), encoding='utf-8')
    text = stdout.getvalue()
    (folder / 'stdout.txt').write_text(text, encoding='utf-8')
    released = json.loads(text)['released'] if text else None
    assert error == expected_error, (label, error, expected_error)
    assert released == expected_release, (label, released, expected_release)
    record = dict(name=label, status='PASS', execution='AST_EXTRACTED_STDLIB_PREDICATE',
                  actual_released=released, actual_error=error, trace=str(folder.relative_to(TRACE)),
                  no_ssh=True, no_neural_execution=True)
    write_json(folder / 'mock_check.json', record)
    CHECKS.append(record)


def wait_case():
    _, gate = loop_and_gate(HELPER)
    events = []
    states = iter((False, False, True))

    def remote_python(host, code):
        assert host == '2026'
        ast.parse(code)
        value = next(states)
        events.append(['POLL', value])
        return json.dumps(dict(released=value))

    namespace = dict(host='2027', receipts=[dict(output='/MOCK_PREVIOUS_OUTPUT')],
                     remote_python=remote_python, json=json,
                     time=SimpleNamespace(sleep=lambda seconds: events.append(['SLEEP_REQUEST', seconds])))
    exec(compile(ast.Module(body=[gate], type_ignores=[]), str(HELPER), 'exec'), namespace)
    events.append(['READY_FOR_2027', True])
    assert events == [['POLL', False], ['SLEEP_REQUEST', 240], ['POLL', False],
                      ['SLEEP_REQUEST', 240], ['POLL', True], ['READY_FOR_2027', True]]
    CHECKS.append(dict(name='actual_wait_loop_defers2027_and_requests240', status='PASS',
        execution='AST_EXTRACTED_STDLIB_MOCK', events=events, actual_sleep_seconds=0, no_ssh=True))
    namespace['host'] = '2026'
    prior = list(events)
    exec(compile(ast.Module(body=[gate], type_ignores=[]), str(HELPER), 'exec'), namespace)
    assert events == prior
    CHECKS.append(dict(name='first_host_not_gated_on_itself', status='PASS',
                      execution='AST_EXTRACTED_STDLIB_MOCK', extra_polls=0))


def compile_remote_programs(plan, four):
    outer, _ = loop_and_gate(HELPER)
    hosts = {'2026': ('/data/gaob/Re-ID/DeMo-DualAxis', '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'),
             '2027': ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')}
    entries = []
    for host in hosts:
        namespace = dict(host=host, PROJECT=ROOT, HOSTS=hosts, Path=Path, hashlib=hashlib, json=json,
                         plan=plan, files=tuple(four), new_sources=four)
        prefix = []
        for node in outer.body[1:]:
            prefix.append(node)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'guard' for t in node.targets):
                break
        exec(compile(ast.Module(body=prefix, type_ignores=[]), str(HELPER), 'exec'), namespace)
        namespace['availability'] = dict(jobs=namespace['jobs'])
        suffix = []
        recording = False
        for node in outer.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'all_sources' for t in node.targets):
                recording = True
            if recording:
                suffix.append(node)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'start' for t in node.targets):
                break
        exec(compile(ast.Module(body=suffix, type_ignores=[]), str(HELPER), 'exec'), namespace)
        for kind in ('guard', 'start'):
            code = namespace[kind]
            ast.parse(code)
            (TRACE / ('003_remote_' + host + '_' + kind + '_NOT_EXECUTED.py')).write_text(code, encoding='utf-8')
        parent_keys = set(namespace['parent']['source_sha256'])
        extensions = {'modeling/moe/AttnMOE.py', 'modeling/meta_arch.py', 'launch_axis_scaled.py', 'launch_runs.py'}
        assert not parent_keys.intersection(extensions)
        assert not parent_keys.intersection(four)
        assert set(namespace['all_sources']) == parent_keys | extensions | set(four)
        entries.append(dict(host=host, sources=len(namespace['all_sources']),
                            parent_keys_overwritten=[], jobs=len(namespace['jobs']),
                            remote_programs_parsed_not_executed=True))
    CHECKS.append(dict(name='current_remote_source_checks_and_templates', status='PASS',
                      execution='AST_AND_STDLIB_ONLY', hosts=entries))


def main():
    assert sys.version_info[:2] == (3, 13)
    snapshot = TRACE / 'sources_003_final'
    snapshot.mkdir(exist_ok=False)
    (snapshot / HELPER.name).write_bytes(HELPER.read_bytes())
    (snapshot / PLAN.name).write_bytes(PLAN.read_bytes())
    for old in ('sources_001', 'sources_002_pending_gate'):
        before = (TRACE / old / HELPER.name).read_text(encoding='utf-8').splitlines(keepends=True)
        after = HELPER.read_text(encoding='utf-8').splitlines(keepends=True)
        diff = difflib.unified_diff(before, after, fromfile=old + '/' + HELPER.name, tofile='sources_003_final/' + HELPER.name)
        (TRACE / ('003_' + old + '_helper.diff')).write_text(''.join(diff), encoding='utf-8')
    checks1 = json.loads((TRACE / '001_checks_result.json').read_text())
    four = checks1['checked_source_sha256']
    assert len(four) == 4 and all(sha(ROOT / name) == expected for name, expected in four.items())
    plan = json.loads(PLAN.read_text())
    plan1 = json.loads((TRACE / 'sources_001/results/preflight/availability_base_plan.json').read_text())
    assert {k: v for k, v in plan.items() if k != 'schedule'} == plan1
    old = TRACE / 'sources_002_pending_gate' / HELPER.name
    predicate_case('original_smoke_failure_left_pending', 'smoke_failed', helper=old, expected_release=False)
    predicate_case('original_full_failure_left_pending', 'full_failed', helper=old, expected_release=False)
    for label, mode, released, error in [
        ('no_terminals_waits', 'none', False, None),
        ('full_exit_before_controller_waits', 'full_exit_pending', False, None),
        ('one_completed_waits', 'one_complete', False, None),
        ('two_completed_releases', 'both_complete', True, None),
        ('smoke_failure_stops', 'smoke_failed', None, 'AssertionError'),
        ('full_failure_stops', 'full_failed', None, 'AssertionError'),
        ('invalid_controller_stops', 'controller_fail', None, 'AssertionError'),
        ('missing_full_exit_stops', 'controller_missing_full', None, 'FileNotFoundError')]:
        predicate_case(label, mode, expected_release=released, expected_error=error)
    wait_case()
    compile_remote_programs(plan, four)
    assert sha(HELPER) == sha(snapshot / HELPER.name)
    assert sha(PLAN) == sha(snapshot / PLAN.name)
    assert not any(name == 'torch' or name.startswith('torch.') or name == 'numpy' or name.startswith('numpy.')
                   or name.startswith('modeling') for name in sys.modules)
    result = dict(status='PASS', observed_at=datetime.now(timezone.utc).isoformat(),
        check_count=len(CHECKS), checks=CHECKS, checked_source_sha256=four,
        helper_sha256=sha(HELPER), plan_sha256=sha(PLAN), runtime_tensor_execution=False,
        real_gpu_execution=False, ssh_or_network_calls=0, neural_imports=[],
        unchanged_scientific_plan=True, schedule_only_plan_addition=True,
        original_failure_trace_preserved='schedule_mocks_003/original_*',
        qualifier='Two original-version cases reproduce the pending-on-failure defect. Remaining cases verify the corrected current source with stdlib mocks, not neural runtime.')
    write_json(TRACE / '003_schedule_checks_result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
