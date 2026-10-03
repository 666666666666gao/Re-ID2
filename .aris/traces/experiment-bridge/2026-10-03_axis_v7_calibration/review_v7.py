"""Narrow scalar-trial review. Stdlib mocks only; no SSH, GPU, or NN imports."""
import argparse
import ast
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest import mock

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT/'.aris/traces/experiment-bridge/2026-10-03_axis_v7_calibration'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_v7_calibration26_deploy_20261003.py')
OBSERVER = Path('C:/Users/gb/.codex_tmp/demo_axis_v7_calibration26_observe_20261003.py')
NAME = 'launch_mass_calibration_trial.py'
checks = []

def check(name, good, evidence):
    assert good, (name, evidence)
    checks.append(dict(check=name, status='PASS', evidence=evidence))

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

source = (ROOT/NAME).read_text(encoding='utf-8')
observer_source = OBSERVER.read_text(encoding='utf-8')
source_hashes = {NAME: sha(ROOT/NAME)}
helper_sha, observer_sha = sha(HELPER), sha(OBSERVER)
for path in (ROOT/NAME, HELPER, OBSERVER):
    compile(path.read_text(encoding='utf-8'), str(path), 'exec')
check('launcher_helper_observer_compile', True, 'Compilation only; no production module imported.')
frozen = load(ROOT/'results/preflight/axis_collaboration_v5_mass_launch.json')
protected = frozen['source_sha256']
check('original_V5_18_sources_unchanged', len(protected)==18 and all(sha(ROOT/name)==value for name,value in protected.items()),
      'Existing V5 launch receipt reused, including original model, runner, sampler, metrics and configuration.')
plan = load(ROOT/'results/preflight/axis_collaboration_v7_calibration_plan.json')
check('plan_binds_only_new_launcher_and_single_scalar', plan['source_sha256']==source_hashes and plan['budget_epochs']==50
      and plan['engineering_updates']==3 and plan['official_test_uses']==0,
      'One MSVR seed42 development trial; no new model, runner, target units or projected auxiliary taps.')
runner = (ROOT/'run_mass_experiment.py').read_text(encoding='utf-8')
check('original_scalar_assignment_and_fresh50_paths',
      "parser.add_argument('--contribution-weight', type=float, default=.05)" in runner
      and 'model.contribution_loss_weight = args.contribution_weight' in runner
      and 'cfg.SOLVER.IMS_PER_BATCH = 64' in runner and 'cfg.SOLVER.MAX_EPOCHS = 50' in runner
      and 'cfg.MODEL.PRETRAIN_PATH_T = args.pretrained' in runner
      and "if metrics['mAP'] > best['mAP']:" in runner,
      'Original CLI sets the existing scalar. Original builder, B64, 50 epochs and earliest-max dev-mAP selection remain bound by18-source equality.')
check('native_smoke_full_optimizer_and_AMP_contract_preserved',
      "scaler = torch.amp.GradScaler('cuda', init_scale=512)" in runner
      and "assert updated, 'smoke optimizer step skipped'" in runner
      and 'if steps == 3:' in runner and 'assert all(gradients.values())' in runner
      and "torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, out / 'smoke.pth')" in runner
      and "model.load_state_dict(saved['model'], strict=True)" in runner
      and "optimizer.load_state_dict(saved['optimizer'])" in runner and 'assert torch.equal(before, after)' in runner,
      'Existing native smoke saves/reloads full model and optimizer and compares model outputs. Actual changed-scalar AMP remains pending.')
main = next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='main')
main_code = compile(ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[])),NAME,'exec')
baseline = load(ROOT/'results/axis_collaboration_v5_mass/development/MSVR310_axis_mass_fullref_s42_launch.json')['command']

def options(argv):
    return dict(zip(argv[3::2],argv[4::2]))

baseline_options=options(baseline)

def controller_case(label, fail_stage=None, patch=None):
    with tempfile.TemporaryDirectory(prefix='v7ctrl_',dir='C:/Users/gb/.codex_tmp') as folder:
        base=Path(folder).resolve()
        assert base.is_relative_to(Path('C:/Users/gb/.codex_tmp').resolve())
        output=base/'output';events=[];commands=[]
        class Child:
            pid=12345
            def __init__(self,argv,stdout,stderr,env):
                opt=options(argv);self.mode=opt['--mode']
                assert argv[:3]==['MOCK_PYTHON','-u','run_mass_experiment.py']
                assert env['CUDA_VISIBLE_DEVICES']=='1' and stdout.mode=='x'
                target=Path(opt['--output']);target.mkdir()
                wanted=dict(baseline_options,**{'--contribution-weight':'1.0','--mode':self.mode,'--output':str(target)})
                assert opt==wanted,(opt,wanted)
                if self.mode=='train':
                    assert load(output/'preflight/controller_result.json')['status']=='PASS'
                    assert not (target/'smoke.pth').exists()
                events.append('spawn_'+self.mode);commands.append(argv)
                result=dict(arguments={'contribution_weight':1.0})
                result.update(dict(status='SMOKE_PASS',steps=3,gradients={'parameter':True},strict_reload_equal=True)
                              if self.mode=='smoke' else dict(status='COMPLETE',epochs=50))
                if patch and patch[0]==self.mode:result.update(patch[1])
                write(target/('smoke.json' if self.mode=='smoke' else 'result.json'),result)
                stdout.write('MOCK ONLY')
            def wait(self):
                events.append('wait_'+self.mode)
                return int(self.mode==fail_stage)
        def idle(gpu):
            assert gpu==1;events.append('idle1')
        ns=dict(argparse=argparse,json=json,os=os,Path=Path,subprocess=types.SimpleNamespace(Popen=Child,STDOUT=-2),
                sys=types.SimpleNamespace(executable='MOCK_PYTHON'),time=types.SimpleNamespace(time=lambda:1),idle=idle,write_json=write)
        exec(main_code,ns)
        failed=False
        argv=['controller','--data-root',baseline_options['--data-root'],'--pretrained',baseline_options['--pretrained'],'--output',str(output)]
        with mock.patch.object(sys,'argv',argv):
            try:ns['main']()
            except AssertionError:failed=True
        expected_fail=fail_stage is not None or patch is not None
        expected_modes=['smoke'] if fail_stage=='smoke' or (patch and patch[0]=='smoke') else ['smoke','train']
        assert [options(a)['--mode'] for a in commands]==expected_modes
        for mode in expected_modes:
            stage='preflight' if mode=='smoke' else 'development'
            name='MSVR310_axis_mass_fullref_s42'+('_smoke' if mode=='smoke' else '')
            assert (output/stage/(name+'_exit.json')).exists()
            assert (output/stage/(name+'.log')).exists()
        check('controller_mock_'+label, failed==expected_fail and (output/'controller_result.json').exists()==(not expected_fail),
              dict(events=events,commands_match_V5_except_scalar_mode_output=True,all_started_child_exits_saved=True,real_processes=0))

controller_case('smoke_then_fresh50_complete')
controller_case('smoke_exit_failure_blocks50',fail_stage='smoke')
controller_case('smoke_wrong_scalar_blocks50',patch=('smoke',dict(arguments={'contribution_weight':.05})))
controller_case('smoke_wrong_step_count_blocks50',patch=('smoke',dict(steps=2)))
controller_case('smoke_bad_gradient_blocks50',patch=('smoke',dict(gradients={'parameter':False})))
controller_case('smoke_reload_failure_blocks50',patch=('smoke',dict(strict_reload_equal=False)))
controller_case('train_exit_failure_no_COMPLETE',fail_stage='train')
controller_case('train49_no_COMPLETE',patch=('train',dict(epochs=49)))

def helper_case(label,busy=None,low_disk=None):
    with tempfile.TemporaryDirectory(prefix='v7dep_',dir='C:/Users/gb/.codex_tmp') as folder:
        base=Path(folder).resolve();assert base.is_relative_to(Path('C:/Users/gb/.codex_tmp').resolve())
        local,remote=base/'local',base/'remote'
        (local/'results/preflight').mkdir(parents=True);(remote/'runs').mkdir(parents=True)
        for name in protected:
            for target in (local,remote):
                path=target/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
        (local/NAME).write_bytes((ROOT/NAME).read_bytes())
        write(local/'results/preflight/axis_collaboration_v5_mass_launch.json',frozen)
        write(local/'results/preflight/axis_collaboration_v7_calibration_review.json',
              dict(status='PASS',blockers=[],helper_sha256=helper_sha,checked_source_sha256=source_hashes))
        collector=types.ModuleType('collect_results')
        collector.PROJECT=local;collector.HOSTS={'2026':(str(remote),'MOCK_PYTHON')};collector.OPTIONS=[]
        uploads=[];starts=[];scripts=[]
        def command(argv):
            assert argv==['scp',str(local/NAME),'2026:'+str(remote)+'/'+NAME]
            assert not (remote/NAME).exists()
            (remote/NAME).write_bytes((local/NAME).read_bytes());uploads.append(NAME)
        class Child:
            pid=23456
            def __init__(self,argv,**kwargs):
                assert argv[:3]==['MOCK_PYTHON','-u',NAME]
                assert argv[3:7]==['--data-root',baseline_options['--data-root'],'--pretrained',baseline_options['--pretrained']]
                assert kwargs['cwd']==remote and kwargs['start_new_session'] and kwargs['stdout'].mode=='x'
                starts.append(argv)
        def remote_python(host,code):
            assert host=='2026';scripts.append(code);stage=len(scripts)
            def gpu(argv,**kwargs):
                assert argv[:3]==['nvidia-smi','-i','1']
                return '500' if busy==stage else '60'
            disk=types.SimpleNamespace(free=3200000000 if low_disk==stage else 9142000000)
            if label=='success':(TRACE/('generated_guard.py' if stage==1 else 'generated_start.py')).write_text(code,encoding='utf-8')
            capture=io.StringIO()
            with mock.patch('subprocess.check_output',side_effect=gpu),mock.patch('subprocess.Popen',Child),mock.patch('shutil.disk_usage',return_value=disk),contextlib.redirect_stdout(capture):
                exec(compile(code,'mock_remote.py','exec'),{'__name__':'mock_remote'})
            return capture.getvalue()
        collector.command=command;collector.remote_python=remote_python
        failed=False
        with mock.patch.dict(sys.modules,{'collect_results':collector}),contextlib.redirect_stdout(io.StringIO()):
            try:exec(compile(HELPER.read_text(encoding='utf-8'),str(HELPER),'exec'),{'__name__':'mock_helper','__file__':str(HELPER)})
            except AssertionError:failed=True
        expected_fail=busy is not None or low_disk is not None
        expected_uploads=0 if busy==1 or low_disk==1 else 1
        check('helper_mock_'+label,failed==expected_fail and len(uploads)==expected_uploads and len(starts)==(0 if expected_fail else 1)
              and all(sha(remote/name)==value for name,value in protected.items()),
              dict(uploads=uploads,controller_starts=len(starts),original18_unchanged=True,real_remote_actions=0))

helper_case('success')
helper_case('GPU1_busy_before_upload',busy=1)
helper_case('GPU1_busy_before_start',busy=2)
helper_case('disk_at32e8_before_upload',low_disk=1)
helper_case('disk_at32e8_before_start',low_disk=2)
disk_proof=load(ROOT/'results/preflight/axis_collaboration_v6_actual_disk_probe.json')
sizes={Path(row['path']).name:row['bytes'] for row in disk_proof['checkpoints'] if 'MSVR310_' in row['path']}
check('full_optimizer_disk_budget_grounded_in_measured_sizes',sum(sizes.values())==2792932497 and 3200000000-sum(sizes.values())==407067503,
      dict(checkpoint_bytes=sizes,minimum_headroom_after_three_checkpoints=407067503,
           note='Live free space is rechecked by helper. No deletion or serialization change is introduced.'))
check('observer_two_rows_incremental_raw_before_failure',
      "rows=[('preflight','MSVR310_axis_mass_fullref_s42_smoke',1),('development','MSVR310_axis_mass_fullref_s42',1)]" in observer_source
      and "if 'exit' in row and stage+'/'+name not in {collected!r}:" in observer_source
      and "assert not path.exists(),relative" in observer_source
      and observer_source.index('path.write_bytes(data)')<observer_source.index('assert not failures')
      and observer_source.index("path=output/relative/'intake.json'")<observer_source.index('assert not failures'),
      'Exactly smoke and development rows; exited raw text and intake are stored once before failures are asserted. Checkpoint binaries are not collected.')
check('observer_near_end_or240_poll_read_only',
      "delay=max(240,min(remaining)-180) if remaining else 240" in observer_source
      and "remaining.append((50-row['status']['epoch'])*row['status']['latest']['seconds'])" in observer_source
      and "assert record['controller_live'] or terminal" in observer_source
      and "remote_python('2026',code)" in observer_source and 'subprocess.Popen' not in observer_source,
      'Polls at least240 seconds or approximately180 seconds before the current epoch-time estimate ends; reads existing controller with failure logs and no retry.')
assert 'torch' not in sys.modules
assert sha(HELPER)==helper_sha and sha(OBSERVER)==observer_sha and source_hashes=={NAME:sha(ROOT/NAME)}
assert all(sha(ROOT/name)==value for name,value in protected.items())
report=dict(status='PASS',blockers=[],reviewed_at=datetime.now(timezone.utc).isoformat(),checks_count=len(checks),checks=checks,
            experiment_status='SHELVED_USER_STEERING',deployment_status='NOT_DEPLOYED',deployment_authorized_by_this_review=False,
            steering_note='Parent reported user prioritizes P0 expert retrieval and shared-CLIP/projection gradients; scalar trial is archived and not being launched.',
            helper_path=str(HELPER),helper_sha256=helper_sha,checked_source_sha256=source_hashes,
            observer_path=str(OBSERVER),observer_sha256=observer_sha,review_independence='same-family',acceptance_status='provisional',
            requested_model='gpt-6-astra',requested_reasoning_effort='max',requested_fork_turns='none',
            serving_identity_verified=False,actual_serving_identity=None,
            route_note='Parent-requested route only; continued same reviewer agent. Supplier execution identity is not inferred.',
            scope='Single new launcher, original V5 CLI scalar .05 to1.0, native smoke-to-fresh50 sequencing, deploy/source/resource guards and observer. No old NN/core reruns.',
            runtime_validation=dict(status='NOT_RUN_TRIAL_SHELVED',changed_scalar_AMP_run=False,fresh50_run=False,real_SSH_calls=0,real_GPU_calls=0,NN_model_imports=0,real_optimizer_updates=0),
            limitations=['Static/stdlib checks do not establish changed-scalar AMP success or retrieval improvement.',
                         'The task/C ratio is one-batch diagnostic evidence;20x scalar change does not establish global gradient dominance.',
                         'Predictor-only gradient diagnostics do not establish shared-backbone gradient conflict.',
                         'Existing ordinary controls do not match the cw1 optimization protocol.'],
            trace_directory=str(TRACE))
write(TRACE/'checks.json',report)
write(TRACE/'review.result.json',report)
write(TRACE/'001-review.response.json',report)
write(ROOT/'results/preflight/axis_collaboration_v7_calibration_review.json',report)
meta=dict(skill='experiment-bridge',purpose='V7_scalar_loss_trial_narrow_code_review',agent_id='/root/review_v6_projected_aux',
          requested_model='gpt-6-astra',requested_reasoning_effort='max',requested_fork_turns='none',
          review_independence='same-family',acceptance_status='provisional',serving_identity_verified=False,
          status='PASS',experiment_status='SHELVED_USER_STEERING',checks_count=len(checks),timestamp=report['reviewed_at'])
write(TRACE/'001-review.meta.json',meta);write(TRACE/'run.meta.json',meta)
events=ROOT/'.aris/meta/events.jsonl'
with events.open('a',encoding='utf-8') as handle:
    handle.write(json.dumps(dict(event='review_trace',**meta,trace_path=str(TRACE)))+'\n')
print(json.dumps({key:report[key] for key in ('status','checks_count','helper_sha256','checked_source_sha256','observer_sha256','runtime_validation')},indent=2))
