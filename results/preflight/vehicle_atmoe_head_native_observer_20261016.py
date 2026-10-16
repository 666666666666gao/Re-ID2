"""One estimated native-admission node; never launches or retries a neural run."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import importlib.util
import json
import shlex
import subprocess
import time

T = Path('C:/Users/gb/.codex_tmp')
P = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
PF = P/'results/preflight'
spec = importlib.util.spec_from_file_location('transport', T/'publish_RGBNT100_N5_s44_closed_s45_started_20261015.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def main():
    session = json.loads((PF/'vehicle_atmoe_head_actual_session_20261016.json').read_text('utf-8'))
    output = PF/'vehicle_atmoe_head_native_gate_actual_20261016.json'
    assert not output.exists()
    due = session['remote_owner']['started']+session['first_native_node_seconds']
    time.sleep(max(0., due-time.time()))
    root = session['remote_root']+'/experiments'
    owner = session['remote_owner']['pid']
    code = f'''import json,time
from pathlib import Path
r=Path({root!r});native=r/'native_acceptance.json';terminal=r/'controller_result.json';children=[];diagnostics=[]
for phase in ('parity','native','training','audit'):
 for f in sorted((r/phase).glob('*_launch.json')):
  d=json.loads(f.read_text());e=f.with_name(f.name.replace('_launch.json','_exit.json'))
  children.append(dict(phase=phase,name=f.name,pid=d['pid'],gpu=d['gpu'],started=d['started'],live=Path('/proc/'+str(d['pid'])).exists(),exit=json.loads(e.read_text()) if e.exists() else None))
for f in sorted((r/'native').glob('*/smoke.json')):
 d=json.loads(f.read_text());diagnostics.append(dict(dataset=d['arguments']['dataset'],heads=d['arguments']['atmoe_heads'],steps=d['steps'],attempts=d['attempts'],amp_skipped_steps=d['amp_skipped_steps'],strict_reload_equal=d['strict_reload_equal'],all_active_gradients=all(d['gradients'].values()),parameters=d['parameters'],descriptor_shape=d['descriptor_shape']))
print(json.dumps(dict(observed_epoch=time.time(),controller_pid={owner},controller_live=Path('/proc/{owner}').exists(),native=json.loads(native.read_text()) if native.exists() else None,controller_result=json.loads(terminal.read_text()) if terminal.exists() else None,children=children,native_diagnostics=diagnostics)))'''
    stdout = T/'vehicle_atmoe_head_native_observation_private_stdout_20261016.log'
    stderr = T/'vehicle_atmoe_head_native_observation_private_stderr_20261016.log'
    capture = T/'vehicle_atmoe_head_native_observation_transport_actual_20261016.json'
    assert not stdout.exists() and not stderr.exists() and not capture.exists()
    started = time.time()
    capture.write_text(json.dumps(dict(status='ATTEMPT_STARTED', started=started,
        stdout=str(stdout), stderr=str(stderr)), indent=2)+'\n', 'utf-8')
    with stdout.open('x', encoding='utf-8') as out, stderr.open('x', encoding='utf-8') as err:
        child = subprocess.run(['ssh', *prior.OPTIONS, '2026', shlex.quote('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python')+' -'],
            input=code, text=True, encoding='utf-8', stdout=out, stderr=err, timeout=180)
    transport = dict(status='RETURNED', started=started, finished=time.time(), exit_code=child.returncode, stdout=str(stdout), stderr=str(stderr))
    capture.write_text(json.dumps(transport, indent=2)+'\n', 'utf-8')
    child.check_returncode()
    actual = json.loads(stdout.read_text('utf-8'))
    accepted = actual['native'] is not None and actual['native']['status'] == 'PASS'
    expected = dict(RGBNT100=(32, 97780224, 5120), MSVR310=(16, 99867648, 5632))
    if accepted:
        native = actual['native']
        assert native['actual_updates'] == 6 and native['parity_forward_record_executions'] == 64
        assert native['strict_reload'] and native['all_active_gradients']
        earlier = [row for row in actual['children'] if row['phase'] in ('parity', 'native')]
        assert len(earlier) == 4 and all(not row['live'] and row['exit']['exit_code'] == 0 for row in earlier)
        assert {row['dataset'] for row in actual['native_diagnostics']} == set(expected)
        for row in actual['native_diagnostics']:
            heads, parameters, dimension = expected[row['dataset']]
            assert row['heads'] == heads and row['parameters'] == parameters and row['descriptor_shape'] == [8, dimension]
            assert row['steps'] == row['attempts'] == 3 and row['amp_skipped_steps'] == 0
            assert row['strict_reload_equal'] and row['all_active_gradients']
    training = [row for row in actual['children'] if row['phase'] == 'training']
    both_live = accepted and len(training) == 2 and all(row['live'] for row in training)
    proof = dict(status='ACTUAL_VEHICLE_HEAD_SINGLE_NATIVE_NODE_OBSERVATION',
        at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'), scheduled_epoch=due,
        actual=actual, transport=transport, native_accepted=accepted, both_formal_models_live_at_observation=both_live,
        new_neural_calls=0, training_epoch_queries=0, temperature_power_reads=0, NN_restarts=0,
        limits='One native/child-admission observation. Estimates from historical full50 wall times are not current speed measurements. No training or observer restart.')
    if both_live:
        starts = {next(dataset for dataset in expected if row['name'].startswith(dataset+'_')): row['started'] for row in training}
        proof['actual_formal_starts'] = starts
        proof['estimated_completion_nodes'] = {dataset: starts[dataset]+seconds*session['completion_estimate_multiplier']
            for dataset, seconds in session['historical_full_training_seconds'].items()}
    output.write_text(json.dumps(proof, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    print(json.dumps(dict(status=proof['status'], native_accepted=accepted, both_formal_models_live=both_live)), flush=True)


if __name__ == '__main__':
    main()
