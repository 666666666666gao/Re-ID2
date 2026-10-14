"""One estimated native gate observation; no interim metrics or NN launch."""
import json,shlex,sys,time
from datetime import datetime
from pathlib import Path
P=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(P))
from collect_results import OPTIONS
import subprocess
def command(argv,**kwargs):
 r=subprocess.run(argv,check=False,capture_output=True,text=True,encoding='utf-8',timeout=300,**kwargs)
 if r.returncode:
  print(r.stdout,flush=True);print(r.stderr,file=sys.stderr,flush=True);r.check_returncode()
 return r.stdout
PF=P/'results/preflight';TAG='r201az_RGB_corrected_state_metric_20261014'
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
f=PF/(TAG+'_actual_session.json');started=json.loads(f.read_text('utf-8-sig'))
assert started['status']=='ACTUAL_AZ_DETACHED_CONTROLLER_STARTED_NATIVE_NOT_YET_CONFIRMED'
owner=started['remote_owner']['pid'];deadline=started['remote_owner']['started']+180
assert not (PF/'r201az_native_gate_actual_20261014.json').exists()
time.sleep(max(0,deadline-time.time()))
code=f'''import json,time
from pathlib import Path
r=Path({REMOTE!r})/'runs'/{TAG!r}/'RGBNT100'
assert Path('/proc/{owner}').exists()
native=json.loads((r/'native_acceptance.json').read_text());assert native['status']=='PASS' and native['actual_updates']==6
smokes={{}};children=[]
for variant,gpu in (('frequency_shared',2),('axis_shared',3)):
 name='RGBNT100_r201az_'+variant+'_s42';s=json.loads((r/'native'/(name+'_smoke')/'smoke.json').read_text())
 assert s['status']=='SMOKE_PASS' and s['steps']==s['attempts']==3 and s['strict_reload_equal'] and s['amp_skipped_steps']==0
 assert s['parameters']==99585158 and s['trainable_parameters']==1887366 and s['descriptor_dim']==5120
 for v in s['details']:
  assert v['M_F_auxiliary_Triplet_outlet']=='existing_corrected10_01_relation_coordinates_unit3584'
  metric=v['corrected_state_metric_audit'];grad=v['corrected_state_metric_native_gradients']
  assert metric['states']==['10','01'] and metric['relationship_slice']==[1536,5120] and metric['metric_unit_norm_error']<1e-6
  assert metric['feature_requires_grad'] and metric['extra_module_forwards']==0 and metric['existing_CE512_unchanged']
  assert set(grad)=={'M','F'} and all(g['finite'] and g['scaled_backward_l1']>0 and g['native_hook_visits']==3 and g['actual_backward_visit']=='scaled_full_loss_backward' for g in grad.values())
  assert v['modality_projected_query_normalization']['attention_calls']==84 and 'frequency_projected_query_normalization' not in v
 smokes[variant]=s
 launch=json.loads((r/'training'/(name+'_launch.json')).read_text());pid=launch['pid'];p=Path('/proc')/str(pid)
 assert launch['gpu']==gpu and p.exists() and not (r/'training'/(name+'_exit.json')).exists()
 argv=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
 env=(p/'environ').read_bytes().split(b'\\0');visible=next(x.decode().split('=',1)[1] for x in env if x.startswith(b'CUDA_VISIBLE_DEVICES='))
 assert visible==str(gpu) and 'run_r201az_RGB_corrected_state_metric.py' in argv and '--mode train' in argv
 children.append(dict(launch=launch,live=True,physical_gpu=gpu,CUDA_VISIBLE_DEVICES=visible))
assert [(d['names'],d['partial_set']) for d in smokes['frequency_shared']['details']]==[(d['names'],d['partial_set']) for d in smokes['axis_shared']['details']]
print(json.dumps(dict(observed_epoch=time.time(),controller_pid={owner},controller_live=True,native=native,smokes=smokes,training_children=children)))'''
actual=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
proof=dict(status='ACTUAL_AZ_NATIVE6_PASS_TWO_FORMAL50_CHILDREN_STARTED',
    at=datetime.now().astimezone().isoformat(timespec='seconds'),scheduled_epoch=deadline,actual=actual,
    early_progress_queries=0,NN_relaunches=0,physical_gpus=[2,3],goal='ACTIVE_UNMET')
(PF/'r201az_native_gate_actual_20261014.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n','utf-8')
started.update(status=proof['status'],native_gate=str(PF/'r201az_native_gate_actual_20261014.json'),
    training_children=actual['training_children'],
    next_near_end_epoch=max(v['launch']['started'] for v in actual['training_children'])+9140)
f.write_text(json.dumps(started,ensure_ascii=False,indent=2)+'\n','utf-8')
print(json.dumps(dict(status=proof['status'],native_updates=6,formal_completed=False,
    next_near_end_epoch=started['next_near_end_epoch'])),flush=True)
