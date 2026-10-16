"""Observe one of the two estimated component-completion nodes, with no new NN."""
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('RGBNT100','MSVR310'), required=True)
    args = parser.parse_args()
    native = json.loads((PF/'vehicle_atmoe_head_native_gate_actual_20261016.json').read_text('utf-8'))
    session = json.loads((PF/'vehicle_atmoe_head_actual_session_20261016.json').read_text('utf-8'))
    assert native['native_accepted'] and native['both_formal_models_live_at_observation']
    output = PF/('vehicle_atmoe_head_'+args.dataset+'_completion_node_actual_20261016.json')
    assert not output.exists()
    due = native['estimated_completion_nodes'][args.dataset]
    time.sleep(max(0., due-time.time()))
    root = session['remote_root']+'/experiments'
    owner = session['remote_owner']['pid']
    code = f'''import csv,json,time
from pathlib import Path
r=Path({root!r});terminal=r/'controller_result.json';component=r/{(args.dataset+'_component_complete.json')!r};children=[];progress=[]
for phase in ('parity','native','training','audit'):
 for f in sorted((r/phase).glob({(args.dataset+'*_launch.json')!r})):
  d=json.loads(f.read_text());e=f.with_name(f.name.replace('_launch.json','_exit.json'))
  children.append(dict(phase=phase,name=f.name,pid=d['pid'],gpu=d['gpu'],started=d['started'],live=Path('/proc/'+str(d['pid'])).exists(),exit=json.loads(e.read_text()) if e.exists() else None))
for folder in sorted((r/'training').glob({(args.dataset+'_*')!r})):
 if folder.is_dir():
  f=folder/'epochs.csv';rows=list(csv.DictReader(f.open())) if f.exists() else []
  progress.append(dict(name=folder.name,completed_epochs=len(rows),last_complete_epoch=rows[-1] if rows else None))
print(json.dumps(dict(observed_epoch=time.time(),controller_pid={owner},controller_live=Path('/proc/{owner}').exists(),component=json.loads(component.read_text()) if component.exists() else None,controller_result=json.loads(terminal.read_text()) if terminal.exists() else None,children=children,progress=progress)))'''
    prefix='vehicle_atmoe_head_'+args.dataset+'_completion_observation_'
    stdout=T/(prefix+'private_stdout_20261016.log');stderr=T/(prefix+'private_stderr_20261016.log')
    capture=T/(prefix+'transport_actual_20261016.json')
    assert not stdout.exists() and not stderr.exists() and not capture.exists()
    started=time.time()
    capture.write_text(json.dumps(dict(status='ATTEMPT_STARTED', started=started,
        stdout=str(stdout), stderr=str(stderr)), indent=2)+'\n','utf-8')
    with stdout.open('x',encoding='utf-8') as out, stderr.open('x',encoding='utf-8') as err:
        child=subprocess.run(['ssh',*prior.OPTIONS,'2026',shlex.quote('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python')+' -'],input=code,text=True,encoding='utf-8',stdout=out,stderr=err,timeout=180)
    transport=dict(status='RETURNED',started=started,finished=time.time(),exit_code=child.returncode,stdout=str(stdout),stderr=str(stderr))
    capture.write_text(json.dumps(transport,indent=2)+'\n','utf-8')
    child.check_returncode()
    actual=json.loads(stdout.read_text('utf-8'))
    proof=dict(status='ACTUAL_VEHICLE_HEAD_ONE_ESTIMATED_COMPONENT_NODE',dataset=args.dataset,
        at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),scheduled_epoch=due,
        actual=actual,transport=transport,new_neural_calls=0,NN_restarts=0,temperature_power_reads=0,
        limits='One estimated completion-node read of the original controller. No component completion inferred from time, no NN restart. Necessary followup uses the same job and180-300s or a revised evidence-based completion estimate.')
    output.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(dict(status=proof['status'],dataset=args.dataset,component_present=actual['component'] is not None,controller_complete=actual['controller_result'] is not None)),flush=True)


if __name__ == '__main__':
    main()
