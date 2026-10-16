"""One scheduled formal-start witness; no training restart or intermediate epoch read."""
from datetime import datetime,timezone,timedelta
from pathlib import Path
import importlib.util
import json
import shlex
import subprocess
import time

P = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
PF = P/'results/preflight'
T = Path('C:/Users/gb/.codex_tmp')
spec = importlib.util.spec_from_file_location('transport',T/'publish_RGBNT100_N5_s44_closed_s45_started_20261015.py')
prior = importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)


def main():
    output = PF/'vehicle_atmoe_head_dynamic_amp_formal_start_actual_20261016.json'
    assert not output.exists()
    session = prior.load(PF/'vehicle_atmoe_head_dynamic_amp_actual_session_20261016.json')
    due = session['remote_owner']['started']+session['first_formal_witness_seconds']
    time.sleep(max(0.,due-time.time()))
    root = session['remote_root'];pid = session['remote_owner']['pid']
    code = f'''import json,time
from pathlib import Path
r=Path({root!r});e=r/'experiments';a=e/'native_acceptance.json';children=[]
for f in sorted((e/'training').glob('*_launch.json')):
 d=json.loads(f.read_text());x=f.with_name(f.name.replace('_launch.json','_exit.json'))
 children.append(dict(name=f.name,pid=d['pid'],gpu=d['gpu'],started=d['started'],live=Path('/proc/'+str(d['pid'])).exists(),exit=json.loads(x.read_text()) if x.exists() else None))
print(json.dumps(dict(observed_epoch=time.time(),controller_live=Path('/proc/{pid}').exists(),native=json.loads(a.read_text()) if a.exists() else None,children=children,controller_log=(r/'controller.log').read_text() if not Path('/proc/{pid}').exists() else None)))'''
    out = T/'vehicle_atmoe_dynamic_amp_start_20261016.stdout.log'
    err = T/'vehicle_atmoe_dynamic_amp_start_20261016.stderr.log'
    receipt = T/'vehicle_atmoe_dynamic_amp_start_transport_20261016.json'
    assert not any(file.exists() for file in (out,err,receipt))
    started = time.time()
    receipt.write_text(json.dumps(dict(status='ATTEMPT_STARTED',started=started,stdout=str(out),stderr=str(err)),indent=2)+'\n','utf-8')
    with out.open('x',encoding='utf-8') as stdout,err.open('x',encoding='utf-8') as stderr:
        child = subprocess.run(['ssh',*prior.OPTIONS,'2026',shlex.quote('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python')+' -'],input=code,text=True,encoding='utf-8',stdout=stdout,stderr=stderr,timeout=180)
    transport = dict(status='RETURNED',started=started,finished=time.time(),exit_code=child.returncode,stdout=str(out),stderr=str(err))
    receipt.write_text(json.dumps(transport,indent=2)+'\n','utf-8');child.check_returncode()
    actual = json.loads(out.read_text('utf-8'));native=actual['native']
    accepted = native is not None and native['status']=='PASS_ORIGINAL_NATIVE_DYNAMIC_AMP_ACCOUNTING'
    if accepted:
        assert native['evidence_reused'] and native['new_native_updates']==0
        assert native['successful_optimizer_updates']==6 and native['attempts']==7 and native['amp_skips']==1
    children=actual['children']
    live = accepted and len(children)==2 and all(row['live'] for row in children)
    proof = dict(status='ACTUAL_DYNAMIC_AMP_SINGLE_FORMAL_START_OBSERVATION',
        at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),scheduled_epoch=due,
        actual=actual,transport=transport,original_native_reaccepted=accepted,both_formal_models_live=live,
        new_neural_calls=0,training_epoch_queries=0,NN_restarts=0,temperature_power_reads=0)
    if live:
        assert {row['gpu'] for row in children}=={2,3}
        starts={next(dataset for dataset in ('RGBNT100','MSVR310') if row['name'].startswith(dataset+'_')):row['started'] for row in children}
        proof['actual_formal_starts']=starts
        proof['estimated_completion_nodes']={dataset:starts[dataset]+seconds*session['completion_estimate_multiplier'] for dataset,seconds in session['historical_full_training_seconds'].items()}
    output.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(dict(status=proof['status'],native_reaccepted=accepted,both_formal_models_live=live)),flush=True)


if __name__=='__main__':
    main()
