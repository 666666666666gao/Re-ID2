"""Stdlib-only source delta and orchestration gates; no production import."""
import argparse
import ast
import base64
import contextlib
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_axis_metric_frozen'
TMP = Path('C:/Users/gb/.codex_tmp')
LEAVES = ('diagnose_metric_mass_axis.py', 'missing_metric_mass_development.py', 'launch_metric_mass_frozen_evaluation.py')
OLD = ('diagnose_projected_mass_axis.py', 'missing_projected_mass_development.py', 'launch_projected_mass_frozen_evaluation.py')
HELPER = TMP / 'demo_axis_metric_frozen26_deploy_20261003.py'
OBSERVER = TMP / 'demo_axis_metric_frozen26_observe_20261003.py'
CPU = TMP / 'demo_axis_metric_complete_metrics_20261003.py'
checks = []

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')

def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)

initial = {name: sha(ROOT / name) for name in LEAVES}
helpers = {str(p): sha(p) for p in (HELPER, OBSERVER, CPU)}
prior_path = ROOT / 'results/preflight/axis_collaboration_v6_projected_frozen_review.json'
prior = json.loads(prior_path.read_text())
assert prior['status'] == 'PASS' and not prior['blockers']
replacement = (
    ('run_projected_mass_experiment', 'run_metric_mass_experiment'),
    ('axis_mass_projected_fullref', 'axis_metric_fullref'),
    ('diagnose_projected_mass_axis', 'diagnose_metric_mass_axis'),
    ('missing_projected_mass_development', 'missing_metric_mass_development'),
    ('launch_projected_mass_frozen_evaluation', 'launch_metric_mass_frozen_evaluation'),
    ('WAIT_V6_CLEAN50', 'WAIT_METRIC_CLEAN50'),
    ('Frozen V6 MSVR checkpoint', 'Frozen P1 metric-interface MSVR checkpoint'),
    ('axis_collaboration_v6_projected_frozen_review', 'axis_metric_frozen_review'),
    ('axis_collaboration_v6_projected_ampfix_launch', 'axis_metric_interface_launch'),
    ('axis_collaboration_v6_projected_mass_frozen_trial', 'axis_collaboration_v7_metric_frozen_trial'),
    ('axis_collaboration_v6_projected_frozen_launch', 'axis_metric_frozen_launch'),
    ('axis_collaboration_v6_projected_frozen_latest_snapshot', 'axis_metric_frozen_latest_snapshot'),
    ('new V6 model sources', 'new metric-interface model sources'),
    ('V6_PROJECTED_FROZEN', 'METRIC_INTERFACE_FROZEN'),
)
for old, new in zip(OLD, LEAVES):
    assert sha(ROOT / old) == prior['checked_source_sha256'][old]
    expected = (ROOT / old).read_text()
    for a, b in replacement:
        expected = expected.replace(a, b)
    assert expected == (ROOT / new).read_text(), new
    checks.append({'case': 'exact_V6_delta_' + new, 'status': 'PASS'})
for path, suffix, key in ((HELPER, 'deploy', 'helper_sha256'), (OBSERVER, 'observe', 'observer_sha256')):
    old = TMP / ('demo_axis_v6_projected_frozen26_' + suffix + '_20261003.py')
    assert sha(old) == prior[key]
    expected = old.read_text()
    for a, b in replacement:
        expected = expected.replace(a, b)
    assert expected == path.read_text(), str(path)
    checks.append({'case': 'exact_V6_delta_' + suffix, 'status': 'PASS'})
for path in (*(ROOT / n for n in LEAVES), HELPER, OBSERVER, CPU):
    compile(path.read_text(), str(path), 'exec')
for name in ('configuration', 'evaluate'):
    assert ast.dump(function(ROOT / 'run_mass_experiment.py', name)) == ast.dump(function(ROOT / 'run_metric_mass_experiment.py', name))
assert 'MetricMassAxisCollaborationDeMo(classes, cfg, cameras).float().cuda()' in ast.unparse(function(ROOT / 'run_metric_mass_experiment.py', 'build'))
checks.append({'case': 'syntax_builder_and_configuration', 'status': 'PASS'})

launcher = compile(ast.Module(body=[function(ROOT / LEAVES[2], 'main')], type_ignores=[]), LEAVES[2], 'exec')
def launcher_case(area, fault):
    training, output = area / 'training', area / 'output'
    run = training / 'MSVR310_axis_metric_fullref_s42'
    run.mkdir(parents=True)
    exit_file = training / (run.name + '_exit.json')
    terminal = {'status': 'RUNNING' if fault == 'incomplete' else 'COMPLETE', 'epochs': 49 if fault == 'epochs49' else 50,
                'arguments': {'variant': 'bad' if fault == 'variant' else 'axis_metric_fullref', 'dataset': 'MSVR310'}}
    write_json(run / 'result.json', terminal)
    if fault != 'wait':
        write_json(exit_file, {'exit_code': 1 if fault == 'parent_exit' else 0})
    for name in ('best.pth', 'best_dev_arrays.npz'):
        (run / name).write_bytes(b'reviewer-fixture-only')
    launches, idle_calls, sleeps = [], [], []
    def sleep(seconds):
        assert seconds == 240 and not launches
        sleeps.append(seconds)
        write_json(exit_file, {'exit_code': 0})
    def popen(argv, env, stdout, stderr):
        assert env['CUDA_VISIBLE_DEVICES'] == '3'
        launches.append(argv)
        dest = Path(argv[argv.index('--output') + 1])
        is_smoke = '--smoke' in argv
        four = argv[2] == LEAVES[0]
        result = {'optimizer_updates': int(fault == 'updates'), 'state_tensor_versions_unchanged': fault != 'versions',
                  'normal_feature_max_error': int(fault == 'parity'), 'normal_inference_feature_max_error': 0,
                  'normal_distance_source': 'saved best_dev_arrays.npz/distances',
                  'status': 'PASS' if is_smoke else 'COMPLETE',
                  'metrics': dict.fromkeys(range(3 if fault == 'states' else 4)),
                  'measurements': dict.fromkeys(range(12 if fault == 'missing' else 13))}
        filename = 'smoke.json' if is_smoke else ('diagnostic.json' if four else 'result.json')
        write_json(dest / filename, result)
        if fault == 'input_change':
            (run / 'best.pth').write_bytes(b'changed-review-fixture-only')
        return types.SimpleNamespace(pid=123, wait=lambda: int(fault == 'child_exit'))
    ns = dict(argparse=argparse, hashlib=hashlib, json=json, os=os, Path=Path,
              subprocess=types.SimpleNamespace(Popen=popen, STDOUT=-2), sys=sys,
              time=types.SimpleNamespace(time=lambda: 1, sleep=sleep), idle=idle_calls.append, write_json=write_json)
    argv = sys.argv
    sys.argv = ['review', '--training-root', str(training), '--output', str(output), '--data-root', 'fixture', '--pretrained', 'fixture']
    rejected = False
    try:
        exec(launcher, ns)
        ns['main']()
    except AssertionError:
        rejected = True
    finally:
        sys.argv = argv
    expected_count = 0 if fault in ('incomplete', 'epochs49', 'variant', 'parent_exit') else 2 if fault == 'states' else 4 if fault in ('happy', 'wait', 'missing') else 1
    assert len(launches) == expected_count, (fault, len(launches))
    assert rejected == (fault not in ('happy', 'wait')), fault
    assert idle_calls == [3] * len(launches)
    if not rejected:
        result = json.loads((output / 'controller_result.json').read_text())
        assert result['status'] == 'COMPLETE' and result['four_state_conditions'] == 4 and result['missing_conditions'] == 13
    checks.append({'case': 'launcher_' + fault, 'status': 'PASS', 'mock_children': len(launches), 'rejected': rejected, 'waits': sleeps})

# Execute only the helper prefix before any remote call, against isolated files.
helper_ast = ast.parse(HELPER.read_text())
prefix = []
for node in helper_ast.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'root' for t in ast.walk(node.targets[0])):
        break
    if isinstance(node, (ast.Import, ast.ImportFrom)) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'sys.path.insert'):
        continue
    prefix.append(node)
helper_prefix = compile(ast.Module(body=prefix, type_ignores=[]), str(HELPER), 'exec')
def helper_case(area, fault):
    project = area / 'project'
    for name in (*LEAVES, 'missing_evaluation.py', 'metric_mass_axis_collaboration.py'):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    review = {'status': 'FAIL' if fault == 'review' else 'PASS', 'blockers': ['fixture'] if fault == 'blocker' else [],
              'helper_sha256': 'bad' if fault == 'helper_sha' else sha(HELPER),
              'checked_source_sha256': initial.copy()}
    if fault == 'leaf_sha':
        review['checked_source_sha256'][LEAVES[0]] = 'bad'
    write_json(project / 'results/preflight/axis_metric_frozen_review.json', review)
    if fault != 'launch_missing':
        write_json(project / 'results/preflight/axis_metric_interface_launch.json', {'source_sha256': {'metric_mass_axis_collaboration.py': 'bad' if fault == 'training_sha' else sha(ROOT / 'metric_mass_axis_collaboration.py')}})
    rejected = False
    try:
        exec(helper_prefix, dict(PROJECT=project, Path=Path, hashlib=hashlib, json=json, __file__=str(HELPER)))
    except (AssertionError, FileNotFoundError):
        rejected = True
    assert rejected == (fault != 'happy'), fault
    checks.append({'case': 'helper_prefix_' + fault, 'status': 'PASS', 'rejected': rejected, 'remote_calls': 0})

collect_code = compile(ast.Module(body=[function(OBSERVER, 'collect')], type_ignores=[]), str(OBSERVER), 'exec')
def observer_case(area, fault):
    project = area / 'project'
    target = project / 'results/frozen'
    (project / 'results/preflight').mkdir(parents=True)
    target.mkdir(parents=True)
    row = {'name': 'MSVR310_axis_metric_fullref_s42', 'stage': 'four_state_smoke', 'log_tail': 'primary fixture failure'}
    record = {'controller_live': fault != 'dead', 'rows': [row], 'files': {}, 'controller_log_tail': 'fixture controller'}
    if fault == 'stage_failure':
        row['exit'] = {'exit_code': 1}
    elif fault == 'complete':
        record['controller_result'] = {'status': 'COMPLETE'}
    ns = dict(PROJECT=project, target=target, launch={'output': '/fixture/runs/frozen', 'source_sha256': {}, 'pid': 1, 'log': '/fixture/controller.log'},
              base64=base64, datetime=datetime, hashlib=hashlib, json=json, remote_python=lambda host, code: json.dumps(record))
    rejected = False
    try:
        exec(collect_code, ns)
        complete = ns['collect']()
        assert complete == (fault == 'complete')
    except AssertionError:
        rejected = True
    assert rejected == (fault in ('stage_failure', 'dead'))
    snapshot = json.loads((project / 'results/preflight/axis_metric_frozen_latest_snapshot.json').read_text())
    assert snapshot['rows'][0]['log_tail'] == 'primary fixture failure'
    checks.append({'case': 'observer_' + fault, 'status': 'PASS', 'rejected': rejected, 'snapshot_preserved': True})

with tempfile.TemporaryDirectory(prefix='mf_', dir=TMP) as tmp, contextlib.redirect_stdout(io.StringIO()):
    area = Path(tmp)
    for fault in ('happy', 'wait', 'incomplete', 'epochs49', 'variant', 'parent_exit', 'updates', 'versions', 'parity', 'states', 'missing', 'input_change', 'child_exit'):
        launcher_case(area / ('launch_' + fault), fault)
    for fault in ('happy', 'review', 'blocker', 'helper_sha', 'leaf_sha', 'launch_missing', 'training_sha'):
        helper_case(area / ('helper_' + fault), fault)
    for fault in ('complete', 'running', 'stage_failure', 'dead'):
        observer_case(area / ('observer_' + fault), fault)
assert all(sha(ROOT / name) == digest for name, digest in initial.items())
assert all(sha(Path(name)) == digest for name, digest in helpers.items())
result = {'status': 'PASS', 'checks': checks, 'checked_source_sha256': initial,
          'helper_sha256': sha(HELPER), 'observer_sha256': sha(OBSERVER), 'cpu_helper_sha256': sha(CPU),
          'prior_review_sha256': sha(prior_path), 'torch_imported': 'torch' in sys.modules,
          'real_remote_calls': 0, 'real_child_processes': 0, 'gpu_forwards': 0, 'optimizer_updates': 0}
assert not result['torch_imported']
write_json(TRACE / '001-check-results.json', result)
print(json.dumps({'status': result['status'], 'checks': len(checks), 'torch_imported': result['torch_imported']}))
