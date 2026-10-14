"""Admit one matched AZ controller after reviewed sources and closed AZ negative result."""
from datetime import datetime
from pathlib import Path
import hashlib,json,os,shlex,shutil,sys
PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS
import subprocess

def command(argv, **kwargs):
    result=subprocess.run(argv,check=False,capture_output=True,text=True,encoding='utf-8',timeout=300,**kwargs)
    if result.returncode:
        print(result.stdout,flush=True);print(result.stderr,file=sys.stderr,flush=True)
        result.check_returncode()
    return result.stdout
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
TAG='r201ba_RGB_teacher_head_20261015';ROOT=REMOTE+'/runs/'+TAG


def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text('utf-8-sig'))
    closed=load(pf/'r201az_RGB_corrected_state_metric_20261014_complete_intake.json')
    assert closed['status']=='ACTUAL_AZ_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE'
    audit=load(pf/'r201az_closed_normal_integrity_audit_20261014.json')
    assert audit['status'] in ('PASS','WARN') and not audit['blocking_findings']
    bank=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201ab_RGB_training_neighbors_20261012/training_features.npz')
    assert bank.stat().st_size==167098445 and hashlib.sha256(bank.read_bytes()).hexdigest()=='9d1f92ec7b3755debf2cecee4980a574d70f3b6c1ce37ffc760d0a7197e2cba9'
    review=load(pf/'r201ba_RGB_teacher_head_source_review_20261015.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources,reuse=review['sources_sha256'],review['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==h for n,h in (sources|reuse).items())
    proof=pf/(TAG+'_actual_session.json');assert not proof.exists()
    assert not load(Path('C:/Users/gb/.codex_tmp/r201l_uniform_k8_continuation_20261007.json'))['live_native_sessions']
    assert shutil.disk_usage('D:/').free>3_300_000_000
    protected=load(pf/'r201ad_capacity_anchors_actual_20261012.json')['protected_weights']
    code=f'''import hashlib,json,shutil,subprocess
from pathlib import Path
r=Path({REMOTE!r});assert not Path({ROOT!r}).exists()
selected={{g:subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip() for g in (2,3)}}
assert not any(selected.values()) and shutil.disk_usage(r).free>3_300_000_000
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {reuse!r}.items() if not n.startswith('results/'))
print(json.dumps(dict(project_free=shutil.disk_usage(r).free,selected_GPU_processes=selected,protected_four_exact=True,temperature_power_read=False)))'''
    ready=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    for n in sources:
        if not n.startswith('results/'):command(['scp',*OPTIONS,str(PROJECT/n),'2026:'+REMOTE+'/'+n])
    code=f'''import hashlib,json
from pathlib import Path
r=Path({REMOTE!r});assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {(sources|reuse)!r}.items() if not n.startswith('results/'))
print(json.dumps(dict(reviewed_sources_exact=True)))'''
    admitted=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    bank_remote=REMOTE+'/inputs/r201ba_RGB_training_bank_20261015/training_features.npz'
    code=f'''import json
from pathlib import Path
p=Path({bank_remote!r});assert not p.exists();p.parent.mkdir(parents=True,exist_ok=False)
print(json.dumps(dict(required_bank_directory_created=True)))'''
    command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code)
    command(['scp',*OPTIONS,str(bank),'2026:'+bank_remote])
    code=f'''import hashlib,json
from pathlib import Path
p=Path({bank_remote!r});assert p.stat().st_size==167098445 and hashlib.sha256(p.read_bytes()).hexdigest()=='9d1f92ec7b3755debf2cecee4980a574d70f3b6c1ce37ffc760d0a7197e2cba9'
print(json.dumps(dict(TRAIN_only_bank_exact=True,bytes=p.stat().st_size,sha256='9d1f92ec7b3755debf2cecee4980a574d70f3b6c1ce37ffc760d0a7197e2cba9')))'''
    restored=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    warm_code=f'''import importlib,json,sys
sys.path.insert(0,{REMOTE!r})
import torch
recipe=importlib.import_module('run_r201ba_RGB_teacher_head')
importlib.import_module('launch_r201ba_RGB_teacher_head')
importlib.import_module('evaluate_r201ba_RGB_normal_states')
importlib.import_module('analyze_r201ba_RGB_teacher_head_normal')
assert recipe.configuration is recipe.parent.configuration
assert recipe.make_optimizer is recipe.parent.make_optimizer
assert recipe.parent.parent.modality_query_unit_attention is importlib.import_module('run_r201as_RGB_query_normalization').modality_query_unit_attention
assert recipe.parent.parent.parent.original_make_optimizer is importlib.import_module('run_full_official_frozen_anchor_experiment').make_optimizer
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='ACTUAL_BA_CPU_IMPORT_ALIASES_PASS',cuda_initialized=False,models_built=0,NN_calls=0,updates=0)))'''
    warm=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=warm_code))
    (pf/'r201ba_actual_CPU_import_aliases_20261015.json').write_text(json.dumps(warm,indent=2)+'\n','utf-8')
    argv=[PYTHON,'-u','launch_r201ba_RGB_teacher_head.py','--data-root','/data/gaob/Re-ID/dataset',
        '--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
        '--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',ROOT+'/RGBNT100']
    code=f'''import json,subprocess,time
from pathlib import Path
r=Path({ROOT!r});r.mkdir(exist_ok=False)
with (r/'controller.log').open('w') as log:
 p=subprocess.Popen({argv!r},cwd={REMOTE!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
owner=dict(pid=p.pid,started=time.time(),argv={argv!r},status='REMOTE_CONTROLLER_STARTED_NATIVE_AND_FORMAL_NOT_YET_CONFIRMED')
(r/'remote_owner.json').write_text(json.dumps(owner,indent=2)+'\\n')
print(json.dumps(owner))'''
    owner=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    row=dict(status='ACTUAL_BA_DETACHED_CONTROLLER_STARTED_NATIVE_NOT_YET_CONFIRMED',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        local_deploy_pid=os.getpid(),remote_owner=owner,capacity=ready,admission=admitted,required_bank_restore=restored,sources_sha256=sources,
        reused_sources_sha256=reuse,protected_weights=protected,physical_gpus=[2,3],maximum_NN=2,
        native_updates_planned=6,formal_updates_planned=13126,epochs_planned=100,goal='ACTIVE_UNMET',
        estimated_training_hours=10620/3600,first_near_end_observation_after_formal_seconds=10440,necessary_followup_seconds=300,
        limits='BA only changes initial directions of the existing final 50x5120 classifier versus AV, using cached originalteacher42 full-official TRAIN class means centered/whitened by TRAIN population variance and existing BN epsilon/gamma, retaining original random row norms. The classifier remains trainable. Other heads/BN/experts/route/losses/optimizer/sampling/params/RNG and final5120 inference graph remain AV. Known prototype initialization; not evidence that random heads caused failure, no guaranteed loss/retrieval improvement. Required immutable AB8675 TRAIN bank; no QG features or labels, no new initialization encoder passes. Extra CPU preprocessing/bank I/O and existing cached-bank creation cost disclosed; no equal-startup-cost claim. Failed AZ metric change is not retained. Only26physical23/max2/noT-P/no201NN/missing49/seedhunt. Popen is not native or formal success. PreserveK201/AV/MSstrong/Dunique.')
    proof.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:row[k] for k in ('status','at','local_deploy_pid','remote_owner','capacity')}),flush=True)


if __name__=='__main__':main()
