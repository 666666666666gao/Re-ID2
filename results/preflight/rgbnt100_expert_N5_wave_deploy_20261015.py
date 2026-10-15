"""Admit one bounded seed wave only after BC closure, selection lock and source review."""
import argparse
from datetime import datetime
import hashlib,json,os,shlex,shutil,subprocess,sys
from pathlib import Path

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
PF=PROJECT/'results/preflight'
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def command(argv,**kwargs):
    r=subprocess.run(argv,check=False,capture_output=True,text=True,encoding='utf-8',timeout=300,**kwargs)
    if r.returncode:
        print(r.stdout,flush=True);print(r.stderr,file=sys.stderr,flush=True);r.check_returncode()
    return r.stdout


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=(43,44,45,46),required=True)
    args=parser.parse_args()
    selection_path=PF/'rgbnt100_expert_N5_selected_recipe_lock_20261015.json';selection=load(selection_path)
    assert selection['status']=='ACTUAL_CLOSED_RECIPE_LOCKED_FOR_BOUNDED_EXPERT_N5'
    assert selection['expert_seeds']==[42,43,44,45,46] and selection['teacher_seed']==42
    key=selection['selected_recipe'];assert key in ('AV','BC')
    closed=load(PROJECT/selection['BC_intake_path'])
    assert sha(PROJECT/selection['BC_intake_path'])==selection['BC_intake_sha256']
    assert closed['status']=='ACTUAL_BC_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE' and closed['terminal']['all_children_and_controller_absent']
    BC_owner=load(PF/'r201bc_RGB_reference_interaction_20261015_actual_session.json')['remote_owner']['pid']
    assert sha(PROJECT/selection['BC_audit_path'])==selection['BC_audit_sha256']
    if args.seed>43:
        prior=load(PF/('rgbnt100_N5_'+key+'_s'+str(args.seed-1)+'_complete_intake_20261015.json'))
        assert prior['status']=='ACTUAL_N5_WAVE_D_ARCHIVED_COMPLETE' and prior['seed']==args.seed-1
    review=load(PF/'rgbnt100_expert_N5_source_review_20261015.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources,reuse=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert all(sha(PROJECT/name)==value for name,value in (sources|reuse).items())
    assert shutil.disk_usage('D:/').free>2_000_000_000
    tag='rgbnt100_N5_'+key+'_s'+str(args.seed)
    proof_path=PF/(tag+'_actual_session_20261015.json');assert not proof_path.exists()
    root=selection['remote_family_root']+'/s'+str(args.seed)
    protected=load(PF/'r201ad_capacity_anchors_actual_20261012.json')['protected_weights']
    code=f'''import hashlib,json,shutil,subprocess
from pathlib import Path
p=Path({REMOTE!r});assert not Path({root!r}).exists()
assert not Path('/proc/{BC_owner}').exists()
selected={{g:subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip() for g in (2,3)}}
assert not any(selected.values()) and shutil.disk_usage(p).free>2_000_000_000
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {reuse!r}.items() if not n.startswith('results/'))
print(json.dumps(dict(project_free=shutil.disk_usage(p).free,selected_GPU_processes=selected,protected_four_exact=True,temperature_power_read=False)))'''
    capacity=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    for name in sources:
        if not name.startswith('results/'):
            command(['scp',*OPTIONS,str(PROJECT/name),'2026:'+REMOTE+'/'+name])
    remote_selection=REMOTE+'/results/preflight/'+selection_path.name
    command(['scp',*OPTIONS,str(selection_path),'2026:'+remote_selection])
    code=f'''import hashlib,json
from pathlib import Path
p=Path({REMOTE!r})
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {(sources|reuse)!r}.items() if not n.startswith('results/'))
assert hashlib.sha256(Path({remote_selection!r}).read_bytes()).hexdigest()=={sha(selection_path)!r}
print(json.dumps(dict(reviewed_sources_exact=True,selection_lock_exact=True)))'''
    exact=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    warm_code=f'''import importlib,json,sys
sys.path.insert(0,{REMOTE!r})
import torch
import rgbnt100_expert_seed_recipe as selected
runner,parent=selected.configure_runner({key!r})
import identity_coordinate_rgbnt100_expert_seeds as factory
import run_r201g_normal_priority as g
import run_r201ac_RGB_hard_identity as sampler
assert g.coordinate_build is factory.build and runner.make_loader is sampler.make_loader
assert runner.make_optimizer is parent.make_optimizer and runner.configuration is parent.configuration
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='ACTUAL_SELECTED_SEED_MAIN_ALIASES_CPU_PASS',recipe={key!r},CUDA_initialized=False,models=0,NN_calls=0,updates=0)))'''
    warm=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=warm_code))
    argv=[PYTHON,'-u','launch_rgbnt100_expert_N5_wave.py','--selection-file',remote_selection,'--seed',str(args.seed),
        '--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
        '--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',root+'/RGBNT100']
    code=f'''import json,subprocess,time
from pathlib import Path
r=Path({root!r});r.mkdir(parents=True,exist_ok=False)
with (r/'controller.log').open('w') as log:
 p=subprocess.Popen({argv!r},cwd={REMOTE!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
owner=dict(pid=p.pid,started=time.time(),argv={argv!r})
(r/'remote_owner.json').write_text(json.dumps(owner,indent=2)+'\\n')
print(json.dumps(owner))'''
    owner=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    row=dict(status='ACTUAL_N5_WAVE_CONTROLLER_STARTED_NATIVE_NOT_YET_CONFIRMED',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        recipe=key,seed=args.seed,remote_root=root,remote_owner=owner,local_deploy_pid=os.getpid(),capacity=capacity,
        exact_sources=exact,CPU_main_aliases=warm,selection_sha256=sha(selection_path),sources_sha256=sources,reused_sources_sha256=reuse,
        protected_weights=protected,physical_gpus=[2,3],maximum_NN=2,native_updates_planned=6,formal_epochs_planned=100,
        first_native_node_seconds=180,near_end_after_actual_formal_start_seconds=10440,necessary_followup_seconds=300,
        formal_complete=False,new_metrics_claimed=False,goal='ACTIVE_UNMET')
    proof_path.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({name:row[name] for name in ('status','recipe','seed','remote_owner','capacity')}),flush=True)


if __name__=='__main__':main()
