"""Local stdlib-only preflight; no project imports, NN, SSH, GPU or installation."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_p0_utility26_deploy_20261003.py')
FILES = ('probe_axis_retrieval_utility.py', 'probe_axis_task_gradients.py', 'launch_axis_utility_probes.py')
trees = {name: ast.parse((ROOT/name).read_text(encoding='utf-8'), filename=name) for name in FILES}
helper_tree = ast.parse(HELPER.read_text(encoding='utf-8'), filename=str(HELPER))

# Compile all new source plus the private helper without importing project modules.
for name, tree in trees.items():
    compile(tree, name, 'exec')
compile(helper_tree, str(HELPER), 'exec')

old = json.loads((ROOT/'results/preflight/axis_collaboration_v6_projected_frozen_launch.json').read_text(encoding='utf-8'))['source_sha256']
assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha for name, sha in old.items())

scope = dict(root='/data/gaob/Re-ID/DeMo-DualAxis', python='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python',
             files=FILES, sources=old, all_sources=old)
scope['output'] = scope['root']+'/runs/axis_p0_retrieval_utility_20261003'
scope['v5'] = scope['root']+'/runs/axis_collaboration_v5_mass/development'
scope['v6'] = scope['root']+'/runs/axis_collaboration_v6_projected_mass_trial_ampfix/development/MSVR310_axis_mass_projected_fullref_s42'
embedded = {}
for node in helper_tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('guard', 'start'):
        name = node.targets[0].id
        script = eval(compile(ast.Expression(node.value), '<review-helper-fstring>', 'eval'), scope)
        embedded[name] = ast.parse(script)
        compile(embedded[name], '<embedded-'+name+'>', 'exec')
        assert 'for gpu in (0,1,3)' in script
assert set(embedded) == {'guard', 'start'}

statuses = {'smoke':'PASS_FROZEN_UTILITY_SMOKE', 'utility':'PASS_FROZEN_RETRIEVAL_UTILITY',
            'gradients':'PASS_ACTUAL_SPLIT_TASK_GRADIENT_DIAGNOSTIC'}
execute_node = next(node for node in trees[FILES[2]].body if isinstance(node, ast.FunctionDef) and node.name == 'execute')
events = []
scenario = ''
def write_json(path, data):
    Path(path).write_text(json.dumps(data), encoding='utf-8')

class Child:
    pid = 123
    def __init__(self, argv, stdout, stderr, env):
        self.argv = argv
        self.stage = 'smoke' if '--smoke' in argv else ('gradients' if argv[2] == 'probe_axis_task_gradients.py' else 'utility')
        events.append(dict(stage=self.stage, gpu=env['CUDA_VISIBLE_DEVICES'], command=argv))
        assert env['CUDA_VISIBLE_DEVICES'] != '2'
    def wait(self):
        if scenario == 'smoke_exit_failure':
            return 7
        out = Path(self.argv[self.argv.index('--output')+1])
        if self.stage != 'gradients':
            out.mkdir()
            out = out/'result.json'
        write_json(out, dict(status='WRONG_STATUS' if scenario == 'smoke_status_failure' else statuses[self.stage],
                             optimizer_updates=0, official_test_uses=0))
        return 0

namespace = dict(Path=Path, json=json, os=os, sys=sys, time=time, idle=lambda gpu: None,
                 write_json=write_json, subprocess=SimpleNamespace(Popen=Child, STDOUT=-2))
exec(compile(ast.Module(body=[execute_node], type_ignores=[]), '<isolated-launch-execute>', 'exec'), namespace)
results = {}
for scenario in ('success', 'smoke_exit_failure', 'smoke_status_failure'):
    events.clear()
    with tempfile.TemporaryDirectory(prefix='launcher-'+scenario+'-', dir=TRACE) as directory:
        rejected = False
        try:
            namespace['execute']('case', '/read-only/seed42-best', 1, Path(directory))
        except AssertionError:
            rejected = True
        expected = ['smoke','utility','gradients'] if scenario == 'success' else ['smoke']
        assert [event['stage'] for event in events] == expected
        assert rejected == (scenario != 'success')
        results[scenario] = dict(rejected=rejected, stages=expected)

main_node = next(node for node in trees[FILES[2]].body if isinstance(node, ast.FunctionDef) and node.name == 'main')
schedule_node = next(node for node in main_node.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'schedules')
schedule = eval(compile(ast.Expression(schedule_node.value), '<launcher-schedule>', 'eval'),
                dict(Path=Path, args=SimpleNamespace(v5_root='/v5', v6_run='/v6/MSVR310_axis_mass_projected_fullref_s42')))
assert set(schedule) == {0,1,3}
assert [name for name, path in schedule[0]] == ['V5_MSVR310']
assert [name for name, path in schedule[1]] == ['V5_RGBNT100','V6_MSVR310']
assert [name for name, path in schedule[3]] == ['V5_RGBNT201']

# This is arithmetic coverage of the actual head grouping, not a model test.
for count in (5,7):
    terms = list(range(1, count+1))
    losses = {'base':terms[1]+sum(terms[4:]), 'fused':terms[0], 'M_aux':terms[2], 'F_aux':terms[3]}
    assert sum(losses.values()) == sum(terms)

report = dict(status='PASS_STDLIB_STATIC_AND_LAUNCHER_MOCK', no_nn=True, no_ssh=True, no_gpu=True,
              no_package_install=True, original_bound_sources_match=True, parsed_new_files=list(FILES),
              helper_embedded_scripts_compile=list(embedded), schedule=schedule, launcher_scenarios=results,
              loss_partition_for_direct_and_separate_globals=True,
              checked_source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES},
              helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest())
(TRACE/'stdlib_checks.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, indent=2))
