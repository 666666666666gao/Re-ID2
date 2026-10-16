"""One estimated-node observation of the new RGBNT100 B128 controller only."""
import argparse
from datetime import datetime, timezone, timedelta
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

P = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
PF = P/'results/preflight'
T = Path('C:/Users/gb/.codex_tmp')
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('startup', 'completion'), required=True)
    parser.add_argument('--due-epoch', type=float, required=True)
    args = parser.parse_args()
    receipt = PF/('rgbnt100_paper_batch_'+args.phase+'_actual_20261016.json')
    assert not receipt.exists()
    remaining = args.due_epoch-time.time()
    while remaining > 0:
        time.sleep(min(300, remaining))
        remaining = args.due_epoch-time.time()
    session = json.loads((PF/'rgbnt100_paper_batch_actual_session_20261016.json').read_text())
    spec = importlib.util.spec_from_file_location('transport', T/'publish_RGBNT100_N5_s44_closed_s45_started_20261015.py')
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    script = f'''import json,time
from pathlib import Path
r=Path({session['remote_root']!r});p=r/'experiments';owner=json.loads((r/'remote_owner.json').read_text())
result=dict(observed=time.time(),controller_pid=owner['pid'],controller_live=Path('/proc/'+str(owner['pid'])).exists(),children=[],native_acceptance=None,controller_result=None,unrelated_jobs_queried=False)
for phase in ('native','training','audit'):
 for file in sorted((p/phase).glob('*_launch.json')):
  launch=json.loads(file.read_text());end=file.with_name(file.name.replace('_launch.json','_exit.json'))
  result['children'].append(dict(phase=phase,name=file.stem,pid=launch['pid'],gpu=launch['gpu'],live=Path('/proc/'+str(launch['pid'])).exists(),exit=json.loads(end.read_text()) if end.exists() else None))
for key in ('native_acceptance','controller_result'):
 file=p/(key+'.json')
 if file.exists():result[key]=json.loads(file.read_text())
if not result['controller_live'] and result['controller_result'] is None:
 result['closed_controller_log_tail']=(r/'controller.log').read_text()[-12000:]
print(json.dumps(result))'''
    prefix = T/('rgbnt100_paper_batch_'+args.phase+'_observation_transport_20261016')
    out, err, transport = [prefix.with_suffix(suffix) for suffix in ('.stdout.log', '.stderr.log', '.json')]
    assert not any(file.exists() for file in (out, err, transport))
    started = time.time()
    transport.write_text(json.dumps(dict(status='ATTEMPT_STARTED', started=started, stdout=str(out), stderr=str(err)), indent=2)+'\n', 'utf-8')
    with out.open('x', encoding='utf-8') as stdout, err.open('x', encoding='utf-8') as stderr:
        child = subprocess.run(['ssh', *prior.OPTIONS, '2026', shlex.quote(PYTHON)+' -'], input=script,
            text=True, encoding='utf-8', stdout=stdout, stderr=stderr, timeout=180)
    captured = dict(status='RETURNED', started=started, finished=time.time(), exit_code=child.returncode, stdout=str(out), stderr=str(err))
    transport.write_text(json.dumps(captured, indent=2)+'\n', 'utf-8')
    child.check_returncode()
    actual = json.loads(out.read_text('utf-8'))
    value = dict(status='ACTUAL_SCHEDULED_RGBNT100_PAPER_BATCH_'+args.phase.upper()+'_OBSERVATION',
        at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'), due_epoch=args.due_epoch,
        actual=actual, transport=captured, unrelated_jobs_queried=False)
    receipt.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    print(json.dumps(dict(status=value['status'], controller_live=actual['controller_live'],
        native_accepted=actual['native_acceptance'] is not None, complete=actual['controller_result'] is not None)), flush=True)


if __name__ == '__main__':
    main()
