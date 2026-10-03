"""Focused source and stdlib mocks; never import torch or run SSH/GPU."""
import argparse
import ast
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest import mock

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_axis_contribution_gradient_probe'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_contribution_probe26_deploy_20261003.py')
COLLECTOR = Path('C:/Users/gb/.codex_tmp/demo_axis_contribution_probe26_observe_20261003.py')
FILES = ('probe_contribution_gradients.py', 'launch_contribution_gradient_probe.py')
CHECKS = []


def check(name, passed, evidence):
    assert passed, (name, evidence)
    CHECKS.append({'check': name, 'status': 'PASS', 'evidence': evidence})


def read(name):
    return (ROOT / name).read_text(encoding='utf-8')


def function(node, name):
    return next(x for x in node.body if isinstance(x, ast.FunctionDef) and x.name == name)


def assigned(node, name):
    return next(x for x in ast.walk(node) if isinstance(x, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in x.targets))


def dump(node):
    return ast.dump(node, include_attributes=False)


source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
helper_sha = hashlib.sha256(HELPER.read_bytes()).hexdigest()
collector_sha = hashlib.sha256(COLLECTOR.read_bytes()).hexdigest()
source = read(FILES[0])
controller_source = read(FILES[1])
collector_source = COLLECTOR.read_text(encoding='utf-8')
for path in [*(ROOT / name for name in FILES), HELPER, COLLECTOR]:
    compile(path.read_text(encoding='utf-8'), str(path), 'exec')
check('new_two_leaves_helper_collector_compile', True, 'Compilation only; no neural imports.')
frozen = json.loads(read('results/preflight/axis_collaboration_v6_projected_frozen_launch.json'))
protected = frozen['source_sha256']
check('existing_frozen_launch26_sources_unchanged', len(protected) == 26 and all(
    hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in protected.items()),
    'Reuses the existing receipt; no source framework added and no old experiment rerun.')
run_paths = {'V5': 'results/axis_collaboration_v5_mass/development/MSVR310_axis_mass_fullref_s42',
             'V6': 'results/axis_collaboration_v6_projected_mass_trial_ampfix/development/MSVR310_axis_mass_projected_fullref_s42'}
terminals = {key: json.loads(read(value + '/result.json')) for key, value in run_paths.items()}
check('actual_completed_V5_V6_same_MSVR_config',
      all(row['status'] == 'COMPLETE' and row['epochs'] == 50 and row['arguments']['seed'] == 42
          and row['arguments']['dataset'] == 'MSVR310' for row in terminals.values())
      and terminals['V5']['config'] == terminals['V6']['config'],
      'Existing completed run metadata only; same configuration, seed42 and MSVR data, no new checkpoint selection.')
leaf = function(ast.parse(source), 'main')
step = function(ast.parse(read('run_mass_experiment.py')), 'step')


class LossNames(ast.NodeTransformer):
    def visit_Name(self, node):
        names = {'values': 'output', 'labels': 'target'}
        if node.id in names:
            node.id = names[node.id]
        return node

    def visit_Attribute(self, node):
        if ast.unparse(node) == 'model.loss_weights':
            return ast.Name(id='weights', ctx=ast.Load())
        return self.generic_visit(node)


task_ast = LossNames().visit(ast.parse(ast.unparse(assigned(leaf, 'task').value), mode='eval').body)
check('task_loss_sum_AST_exact_original_step_without_C',
      dump(task_ast) == dump(assigned(step, 'loss').value)
      and 'assert len(values)%2==1' in source and 'end=len(values)-1' in source,
      'Original full weighted head-pair sum matches exactly after variable-name normalization. Odd output assertion makes end=len-1 equal native len-len%2; finalC excluded.')
check('weighted_C_is_exact_returned_last_scalar_once',
      ast.unparse(assigned(leaf, 'contribution').value) == 'values[-1]'
      and 'output.append(self.contribution_loss_weight * contribution)' in read('mass_axis_collaboration.py')
      and 'output.append(self.contribution_loss_weight * contribution)' in read('projected_mass_axis_collaboration.py'),
      'Both models already weightC in output[-1]; the probe neither drops that coefficient nor multiplies it twice.')
check('strict_saved_best_and_one_seeded_training_B64',
      "model.load_state_dict(torch.load(run/'best.pth',map_location='cuda',weights_only=True),strict=True)" in source
      and 'fit,_,_,classes,cameras=split_records(arguments.data_root,arguments.dataset)' in source
      and 'next(iter(make_loader(fit,cfg,True,arguments.seed)))' in source
      and 'assert len(labels)==64' in source and 'model.train()' in source
      and sum(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == 'model' for x in ast.walk(leaf)) == 1,
      'Strict existing dev-best state load; actual fit split and training augmentations; one seed42 B64 forward per checkpoint, no dev/test evaluation.')
check('four_inputs_hashed_before_and_after_diagnostic',
      "inputs=(run/'best.pth',run/'best_dev_arrays.npz',run/'result.json',exit_file)" in source
      and source.index('proofs=') < source.index('values=model(')
      and source.index('for path in inputs:') > source.index('contribution_grad=torch.autograd.grad')
      and "path.stat().st_size==proof['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==proof['sha256']" in source,
      'Original model, dev arrays, result and exit receipt are byte/size checked before and after; only a new output JSON is written.')
grad_calls = [x for x in ast.walk(leaf) if isinstance(x, ast.Call) and ast.unparse(x.func) == 'torch.autograd.grad']
check('both_predictor_gradients_same_forward_scale512',
      len(grad_calls) == 2 and ast.unparse(grad_calls[0].args[0]) == 'task * scale'
      and ast.unparse(grad_calls[1].args[0]) == 'contribution * scale'
      and ast.unparse(grad_calls[0].args[1]) == ast.unparse(grad_calls[1].args[1]) == 'parameters'
      and ast.unparse(grad_calls[0].keywords[0].value) == 'True'
      and ast.unparse(assigned(leaf, 'scale').value) == '512.0'
      and 'model.calibrator.predictor.named_parameters()' in source,
      'Two autograd.grad calls select only predictor parameters; first retains the shared graph, both losses scaled by512, no extra model forward.')
call_names = {ast.unparse(x.func) for x in ast.walk(leaf) if isinstance(x, ast.Call)}
check('no_optimizer_or_checkpoint_writer_training_BN_scope_explicit',
      not any(name == 'torch.save' or name == 'make_optimizer' or name.endswith('.step')
              or name.endswith('.backward') or name.startswith('torch.optim') for name in call_names)
      and 'optimizer_constructed=False' in source and 'temporary in-memory BatchNorm buffers' in source
      and 'those buffers are discarded' in source and 'target_requires_grad' in source,
      'No optimizer construction/step, backward accumulation or checkpoint save. TrainingBN buffer changes are explicitly temporary; four checkpoint/input files must remain identical.')


class Vector:
    def __init__(self, values):
        self.values = [float(x) for x in values]

    def detach(self):
        return self

    def float(self):
        return self

    def flatten(self):
        return self

    def __truediv__(self, value):
        return Vector([x / value for x in self.values])

    def norm(self):
        return math.sqrt(math.fsum(x*x for x in self.values))


torch_mock = types.SimpleNamespace(cat=lambda seq: Vector([x for item in seq for x in item.values]),
                                  dot=lambda a, b: math.fsum(x*y for x, y in zip(a.values, b.values)))
report_call = next(x for x in ast.walk(leaf) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                   and x.func.id == 'dict' and any(k.arg == 'gradient_cosine' for k in x.keywords))
cosine_ast = next(k.value for k in report_call.keywords if k.arg == 'gradient_cosine')
ratio_ast = next(k.value for k in report_call.keywords if k.arg == 'task_to_weighted_contribution_gradient_L2_ratio')
env = {'torch': torch_mock, 'scale': 512.0, 'task_grad': [Vector([1536.,2048.]), Vector([0.])],
       'contribution_grad': [Vector([2048.,-1536.]), Vector([0.])]}
for name in ('task_vector', 'contribution_vector'):
    node = assigned(leaf, name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), 'actual_vector_expression.py', 'exec'), env)
env['task_norm'], env['contribution_norm'] = env['task_vector'].norm(), env['contribution_vector'].norm()
ratio = eval(compile(ast.Expression(ratio_ast), 'actual_ratio.py', 'eval'), env)
cosine = eval(compile(ast.Expression(cosine_ast), 'actual_cosine.py', 'eval'), env)
check('actual_unscale_norm_ratio_cosine_formulas_stdlib_numeric_witness',
      env['task_vector'].values == [3.,4.,0.] and env['contribution_vector'].values == [4.,-3.,0.]
      and env['task_norm'] == env['contribution_norm'] == 5. and ratio == 1. and cosine == 0.,
      'Actual expressions evaluated on synthetic scaled vectors: division by512 occurs before global norm/dot; unit ratio and orthogonal cosine recovered. This is not an autograd test.')
check('per_parameter_gradnorms_and_statistics_are_correct_scope',
      'float(left.float().norm()/scale)' in source and 'float(right.float().norm()/scale)' in source
      and 'target.std(0,unbiased=False)' in source and 'prediction.std(0,unbiased=False)' in source
      and '(prediction-target).abs().mean(0)' in source and 'target.shape==prediction.shape==(64,3)' in source
      and 'local diagnostic, not a convergence, generalization or causal conclusion' in source,
      'Per-parameter L2 is unscaled once; mean/biased-population std/MAE uses64x3 targets/predictions; no convergence or causal claim.')

controller_tree = ast.parse(controller_source)
controller_fn = function(controller_tree, 'main')
controller_code = compile(ast.fix_missing_locations(ast.Module(body=[controller_fn], type_ignores=[])), FILES[1], 'exec')
check('controller_compares_all_five_input_proofs',
      "('names','labels','cameras','scenes','image_tensor_sha256')" in controller_source,
      'Identity order, labels, camera, scene and exact augmented CPU image tensor bytes are compared across cases before COMPLETE.')


def controller_case(label, exits=(0,0), changed=None):
    with tempfile.TemporaryDirectory(prefix='cgctrl_', dir='C:/Users/gb/.codex_tmp') as folder:
        root = Path(folder).resolve()
        assert root.is_relative_to(Path('C:/Users/gb/.codex_tmp').resolve())
        output = root / 'output'
        events, idle_gpus = [], []

        def write_json(path, value):
            Path(path).write_text(json.dumps(value), encoding='utf-8')

        class Child:
            def __init__(self, argv, stdout, stderr, env):
                self.index = len([x for x in events if x.startswith('spawn')])
                self.name = ('V5','V6')[self.index]
                self.pid = 100 + self.index
                assert env['CUDA_VISIBLE_DEVICES'] == str(self.index)
                assert argv[2] == 'probe_contribution_gradients.py'
                assert stdout.mode == 'x'
                events.append('spawn_' + self.name)
                stdout.write('MOCK ONLY ' + self.name)
                result = {'status': 'PASS_ACTUAL_GRADIENT_DIAGNOSTIC', 'optimizer_updates': 0,
                          'optimizer_constructed': False, 'target_requires_grad': False,
                          'names': ['same'], 'labels': [0], 'cameras': [1], 'scenes': [2],
                          'image_tensor_sha256': {'RGB': 'same', 'NI': 'same', 'TI': 'same'}}
                if changed and self.name == 'V6':
                    result.update(changed)
                write_json(argv[-1], result)

            def wait(self):
                events.append('wait_' + self.name)
                return exits[self.index]

        ns = {'argparse': argparse, 'json': json, 'os': os, 'Path': Path,
              'subprocess': types.SimpleNamespace(Popen=Child, STDOUT=-2), 'sys': types.SimpleNamespace(executable='MOCK_PYTHON'),
              'time': types.SimpleNamespace(time=lambda: 1), 'idle': idle_gpus.append, 'write_json': write_json}
        exec(controller_code, ns)
        failed = False
        with mock.patch.object(sys, 'argv', ['controller', '--v5-run', 'V5_RUN', '--v6-run', 'V6_RUN', '--output', str(output)]):
            try:
                ns['main']()
            except AssertionError:
                failed = True
        expected_fail = any(exits) or changed is not None
        check('controller_mock_' + label,
              events == ['spawn_V5','spawn_V6','wait_V5','wait_V6'] and idle_gpus == [0,1]
              and failed == expected_fail and (output / 'controller_result.json').exists() == (not expected_fail)
              and all((output / (name + '_exit.json')).exists() and (output / (name + '.log')).exists() for name in ('V5','V6')),
              {'events': events, 'exit_receipts': exits, 'both_children_reaped_before_failure': True,
               'complete': not expected_fail, 'mock_only': True})


controller_case('both_pass')
controller_case('first_child_fails_still_wait_second', exits=(1,0))
controller_case('second_child_fails_both_exits_saved', exits=(0,1))
controller_case('different_image_batch_rejected', changed={'image_tensor_sha256': {'RGB': 'different'}})
controller_case('optimizer_update_misreport_rejected', changed={'optimizer_updates': 1})


def helper_case(label, busy=None, bad_variant=False, existing_leaf=False):
    with tempfile.TemporaryDirectory(prefix='cgdply_', dir='C:/Users/gb/.codex_tmp') as folder:
        root = Path(folder).resolve()
        assert root.is_relative_to(Path('C:/Users/gb/.codex_tmp').resolve())
        local, remote = root/'local', root/'remote'
        (local/'results/preflight').mkdir(parents=True)
        remote.mkdir()
        for name in protected:
            for base in (local, remote):
                path = base/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes((ROOT/name).read_bytes())
        for name in FILES:
            (local/name).write_bytes((ROOT/name).read_bytes())
        (local/'results/preflight/axis_collaboration_v6_projected_frozen_launch.json').write_text(json.dumps(frozen),encoding='utf-8')
        (local/'results/preflight/axis_contribution_gradient_probe_review.json').write_text(json.dumps(
            {'status':'PASS','blockers':[],'helper_sha256':helper_sha,'checked_source_sha256':source_hashes}),encoding='utf-8')
        variants = [('axis_collaboration_v5_mass','axis_mass_fullref'),('axis_collaboration_v6_projected_mass_trial_ampfix','axis_mass_projected_fullref')]
        inputs = []
        for campaign, variant in variants:
            run=remote/'runs'/campaign/'development'/('MSVR310_'+variant+'_s42')
            run.mkdir(parents=True)
            values={'status':'COMPLETE','epochs':50,'arguments':{'variant':'wrong' if bad_variant else variant}}
            (run/'result.json').write_text(json.dumps(values),encoding='utf-8')
            (run/'best.pth').write_bytes(b'NO_NEURAL_WEIGHTS_MOCK_ONLY')
            (run/'best_dev_arrays.npz').write_bytes(b'NO_NUMERICAL_ARRAYS_MOCK_ONLY')
            exit_path=run.parent/(run.name+'_exit.json')
            exit_path.write_text(json.dumps({'exit_code':0}),encoding='utf-8')
            inputs.extend([run/'result.json',run/'best.pth',run/'best_dev_arrays.npz',exit_path])
        input_bytes={str(path):path.read_bytes() for path in inputs}
        if existing_leaf:
            (remote/FILES[0]).write_bytes(b'EXISTING_LEAF_DO_NOT_OVERWRITE')
        uploads,starts,scripts=[],[],[]
        collector=types.ModuleType('collect_results')
        collector.PROJECT,collector.HOSTS,collector.OPTIONS=local,{'2026':(str(remote),'MOCK_REMOTE_PYTHON')},[]

        def command(argv):
            assert argv[0]=='scp' and len(argv)==3
            path=Path(argv[1]);assert path.parent==local and path.name in FILES
            assert argv[2]=='2026:'+str(remote)+'/'+path.name
            uploads.append(path.name);(remote/path.name).write_bytes(path.read_bytes())
            return ''

        class Child:
            pid=987654
            def __init__(self,argv,**kwargs):
                assert argv[:3]==['MOCK_REMOTE_PYTHON','-u','launch_contribution_gradient_probe.py']
                assert kwargs['cwd']==remote and kwargs['start_new_session'] is True and kwargs['stdout'].mode=='x'
                assert 'axis_collaboration_v5_mass' in argv[argv.index('--v5-run')+1]
                assert 'axis_collaboration_v6_projected_mass_trial_ampfix' in argv[argv.index('--v6-run')+1]
                starts.append(argv)

        def remote_python(host,code):
            assert host=='2026'
            scripts.append(code);stage=len(scripts)
            def gpu(argv,**kwargs):
                assert argv[0]=='nvidia-smi' and argv[1]=='-i' and int(argv[2]) in (0,1)
                return '900' if busy==(stage,int(argv[2])) else '60'
            buffer=io.StringIO()
            if label=='success':
                (TRACE/('generated_remote_'+('guard' if stage==1 else 'start')+'.py')).write_text(code,encoding='utf-8')
            with mock.patch('subprocess.check_output',side_effect=gpu),mock.patch('subprocess.Popen',Child),contextlib.redirect_stdout(buffer):
                exec(compile(code,'mock_remote.py','exec'),{'__name__':'mock_remote'})
            return buffer.getvalue()

        collector.command,collector.remote_python=command,remote_python
        failed=False
        with mock.patch.dict(sys.modules,{'collect_results':collector}),contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(compile(HELPER.read_text(encoding='utf-8'),str(HELPER),'exec'),{'__name__':'mock_helper','__file__':str(HELPER)})
            except AssertionError:
                failed=True
        first_gate_fails=bad_variant or existing_leaf or (busy is not None and busy[0]==1)
        expected_uploads=0 if first_gate_fails else 2
        expected_starts=0 if busy or bad_variant or existing_leaf else 1
        check('helper_mock_'+label,len(uploads)==expected_uploads and len(starts)==expected_starts
              and failed==(expected_starts==0)
              and all(path.read_bytes()==input_bytes[str(path)] for path in inputs)
              and all(hashlib.sha256((remote/name).read_bytes()).hexdigest()==sha for name,sha in protected.items()),
              {'uploads':uploads,'controller_starts':len(starts),'all_original26_sources_and8_checkpoint_inputs_unchanged':True,'real_remote_actions':0})


helper_case('success')
helper_case('GPU0_busy_before_upload',busy=(1,0))
helper_case('GPU1_busy_before_upload',busy=(1,1))
helper_case('GPU0_busy_after_upload',busy=(2,0))
helper_case('GPU1_busy_after_upload',busy=(2,1))
helper_case('wrong_checkpoint_variant',bad_variant=True)
helper_case('new_leaf_already_exists',existing_leaf=True)
check('collector_archives_raw_and_intake_before_failure_assert',
      collector_source.index('path.write_bytes(data)') < collector_source.index('assert not failures')
      and collector_source.index("path=target/(name+'_intake.json')") < collector_source.index('assert not failures')
      and "('.json','.log','_launch.json','_exit.json')" in collector_source
      and "if 'exit' in row and name not in {collected!r}" in collector_source,
      'Static scan: both case raw siblings and byte proofs are written before failed-exit assertion; existing intake markers prevent payload overwrite.')
check('collector_240_seconds_only_read_remote_no_retry',
      'time.sleep(240)' in collector_source and 'assert record[\'controller_live\'] or complete' in collector_source
      and "remote_python('2026',code)" in collector_source and 'subprocess.Popen' not in collector_source
      and "complete=record.get('controller_result',{}).get('status')=='COMPLETE'" in collector_source,
      '240-second polling; COMPLETE or live-controller requirement; logs retained on failure; no NN, retry or upload operation in collector.')

assert 'torch' not in sys.modules
result={'status':'PASS_SOURCE_AND_STDLIB_MOCKS','observed_at':datetime.now(timezone.utc).isoformat(),
        'checks_count':len(CHECKS),'checks':CHECKS,'checked_source_sha256':source_hashes,
        'helper_path':str(HELPER),'helper_sha256':helper_sha,'collector_path':str(COLLECTOR),'collector_sha256':collector_sha,
        'no_NN_model_import':True,'real_SSH_calls':0,'real_GPU_calls':0,'real_optimizer_updates':0,
        'scope':'Two one-forward diagnostic leaves, loss/gradient formulas, child exit semantics, resource/source guards and collector static scan. No old61/24 reruns.'}
(TRACE/'checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({key:result[key] for key in ('status','checks_count','helper_sha256','collector_sha256','no_NN_model_import')}))
