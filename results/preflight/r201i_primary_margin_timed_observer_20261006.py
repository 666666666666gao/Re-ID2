"""Observe the original sequential three-dataset I receiver at estimated milestones."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import time

project = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0,str(project))
from collect_results import OPTIONS,command
parser = argparse.ArgumentParser()
parser.add_argument('--primary-session',type=int,required=True)
args = parser.parse_args()
pf = project/'results/preflight'
started = pf/'r201i_primary_margin_timed_observer_started_20261006.json'
assert not started.exists()
assert json.loads((pf/'r201g_other_two_missing49_closeout_queue_actual_session_20261006.json').read_text())['exit_code']==0
initial = json.loads((pf/'r201i_primary_margin_current_receiver_20261006.json').read_text())
assert initial['dataset']=='RGBNT201'
first_due = datetime.fromisoformat(initial['started_at']).timestamp()+180
started.write_text(json.dumps(dict(status='ACTUAL_SINGLE_I_THREE_NORMAL_PASSIVE_TIMER_STARTED',pid=os.getpid(),
    started_at=datetime.now().isoformat(timespec='seconds'),first_due=datetime.fromtimestamp(first_due).isoformat(timespec='seconds'),
    primary_session=args.primary_session,original_local_ssh_pid=initial['local_ssh_pid'],remote_queries_so_far=0,
    policy='Original receiver only; first180s, estimatedE20 thennearfullend, 240s only whenneeded. No NN/GPU/power/temperature/source changes.'),indent=2)+'\n',encoding='utf-8')
print('I_NORMAL_PASSIVE_FIRST_DUE',datetime.fromtimestamp(first_due).isoformat(timespec='seconds'),flush=True)
code = '''import json,shutil,subprocess,time
from pathlib import Path
root=Path('/data/gaob/Re-ID/DeMo-DualAxis/runs/r201i_primary_margin_20261006')
def load(path):return json.loads(path.read_text())
def process(pid):
 result=subprocess.run(['ps','-p',str(pid),'-o','pid=,stat=,args='],text=True,capture_output=True)
 assert result.returncode in (0,1)
 return result.stdout.strip()
datasets=[]
for dataset,steps in [('RGBNT201',2647),('MSVR310',705),('RGBNT100',6357)]:
 folder=root/dataset
 if not folder.exists():
  datasets.append(dict(dataset=dataset,stage='NOT_STARTED'))
  continue
 launch=load(folder/'controller_launch.json');native=folder/'native_acceptance.json';result=folder/'controller_result.json';runs=[]
 for variant,gpu in [('frequency_shared',2),('axis_shared',3)]:
  name=dataset+'_r201i_'+variant+'_s42';job=folder/'training'/name;start=folder/'training'/(name+'_launch.json')
  if not start.exists():
   runs.append(dict(name=name,gpu=gpu,phase='NATIVE_OR_WAITING_FORMAL'))
   continue
  child=load(start);status=job/'status.json';exit=folder/'training'/(name+'_exit.json')
  row=dict(name=name,gpu=gpu,pid=child['pid'],started=child['started'],process=process(child['pid']),exit=load(exit) if exit.exists() else None)
  if status.exists():
   data=load(status);row.update(phase=data['status'],epoch=data['epoch'] if 'epoch' in data else data['epochs'],
    latest=data['latest'] if 'latest' in data else dict(optimizer_steps=data['optimizer_steps'],amp_skipped_steps=data['amp_skipped_steps']),
    best={key:data['best'][key] for key in ('epoch','mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')})
  else:row['phase']='FORMAL_STARTING'
  runs.append(row)
 datasets.append(dict(dataset=dataset,stage='COMPLETE' if result.exists() else 'LIVE',controller_pid=launch['pid'],started=launch['started'],
  parent_process=process(launch['pid']),native=load(native) if native.exists() else None,result=load(result) if result.exists() else None,runs=runs,expected_updates_each=steps))
print(json.dumps(dict(observed_epoch_time=time.time(),datasets=datasets,remote_free=shutil.disk_usage(root).free)))'''
milestone_seen = set()
due = first_due
queries = 0
historical_max = {'RGBNT201':60.0,'MSVR310':26.225719690322876,'RGBNT100':182.15861129760742}
while True:
    while time.time()<due:
        time.sleep(min(60,due-time.time()))
    data = json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python')+' -'],input=code))
    queries += 1
    data.update(observed_at=datetime.now().isoformat(timespec='seconds'),observer_pid=os.getpid(),primary_session=args.primary_session,
        local_D_free=shutil.disk_usage('D:/').free,remote_queries=queries)
    with (pf/'r201i_primary_margin_timed_observations_20261006.jsonl').open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(data)+'\n')
    (pf/'r201i_primary_margin_timed_latest_20261006.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print('I_NORMAL_ACTUAL_OBSERVATION',data['observed_at'],[(d['dataset'],d['stage'],[(r['name'],r.get('epoch'),r['phase']) for r in d.get('runs',[])]) for d in data['datasets']],flush=True)
    if all(d['stage']=='COMPLETE' for d in data['datasets']):
        assert all(d['result']['status']=='COMPLETE' and d['result']['successful_updates']==2*d['expected_updates_each'] and d['result']['normal_archives_local_verified']==2 for d in data['datasets'])
        break
    active = next(d for d in data['datasets'] if d['stage']!='COMPLETE')
    if active['stage']=='NOT_STARTED':
        due = time.time()+240
        print('I_NORMAL_NEXT_ESTIMATED_OBSERVATION',datetime.fromtimestamp(due).isoformat(timespec='seconds'),flush=True)
        continue
    assert active['stage']=='LIVE' and active['parent_process']
    assert all(r['exit'] is None or r['exit']['exit_code']==0 for r in active['runs'] if 'exit' in r)
    running = [r for r in active['runs'] if r['phase']=='RUNNING']
    now = time.time()
    if len(running)==2:
        if active['dataset'] not in milestone_seen and min(r['epoch'] for r in running)<20:
            due = max(now+240,max(r['started']+20*r['latest']['seconds']+60 for r in running))
        else:
            milestone_seen.add(active['dataset'])
            due = now+max(240,max((50-r['epoch'])*r['latest']['seconds'] for r in running)-180)
    elif running:
        due = now+max(240,max((50-r['epoch'])*r['latest']['seconds'] for r in running)-180)
    else:
        due = max(now+240,active['started']+120+3*historical_max[active['dataset']])
    print('I_NORMAL_NEXT_ESTIMATED_OBSERVATION',datetime.fromtimestamp(due).isoformat(timespec='seconds'),flush=True)
result = dict(status='ACTUAL_I_THREE_NORMAL_PASSIVE_OBSERVER_COMPLETE',finished_at=datetime.now().isoformat(timespec='seconds'),observations=queries,
    primary_session=args.primary_session,new_neural_jobs=0,successful_updates_observed=19418)
(pf/'r201i_primary_margin_timed_observer_terminal_20261006.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result),flush=True)
