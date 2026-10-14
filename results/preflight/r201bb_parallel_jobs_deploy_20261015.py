"""Explicit deferred BB admission after all BA consumers and its result audit close."""
from datetime import datetime
from pathlib import Path
import hashlib,json,os,shlex,subprocess,sys
P=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(P))
from collect_results import OPTIONS
PF=P/'results/preflight';REMOTE='/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python';TAG='r201bb_parallel_jobs_20261015';ROOT=REMOTE+'/runs/'+TAG
BANK=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201ab_RGB_training_neighbors_20261012/training_features.npz')
BANK_SHA='9d1f92ec7b3755debf2cecee4980a574d70f3b6c1ce37ffc760d0a7197e2cba9'
BANK_REMOTE=REMOTE+'/inputs/r201ba_RGB_training_bank_20261015/training_features.npz'
load=lambda p:json.loads(p.read_text('utf-8-sig'))


def command(argv,**kwargs):
    d=subprocess.run(argv,check=False,capture_output=True,text=True,encoding='utf-8',timeout=300,**kwargs)
    if d.returncode:
        print(d.stdout,flush=True);print(d.stderr,file=sys.stderr,flush=True);d.check_returncode()
    return d.stdout


def main():
    closed=load(PF/'r201ba_RGB_teacher_head_20261015_complete_intake.json')
    assert closed['status']=='ACTUAL_BA_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE'
    assert closed['bank_cleanup']['TRAIN_bank_duplicate_retired']==167098445
    audit=load(PF/'r201ba_closed_normal_integrity_audit_20261015.json')
    assert audit['status'] in ('PASS','WARN') and not audit['blocking_findings']
    review=load(PF/'r201bb_parallel_jobs_source_review_20261015.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources=review['sources_sha256'];reuse=review['directly_reused_sources_sha256']
    assert all(hashlib.sha256((P/n).read_bytes()).hexdigest()==h for n,h in (sources|reuse).items())
    assert BANK.stat().st_size==167098445 and hashlib.sha256(BANK.read_bytes()).hexdigest()==BANK_SHA
    proof=PF/(TAG+'_actual_session.json');assert not proof.exists()
    protected=load(PF/'r201ad_capacity_anchors_actual_20261012.json')['protected_weights']
    code=f'''import hashlib,json,subprocess
from pathlib import Path
r=Path({REMOTE!r});bank=Path({BANK_REMOTE!r});assert not bank.exists() and not Path({ROOT!r}).exists()
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
apps=subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not apps
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {reuse!r}.items() if not n.startswith('results/'))
bank.parent.mkdir(parents=True,exist_ok=False)
print(json.dumps(dict(selected_GPU2_idle=True,protected_four_exact=True,temperature_power_read=False)))'''
    ready=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    for n in sources:
        if not n.startswith('results/'):command(['scp',*OPTIONS,str(P/n),'2026:'+REMOTE+'/'+n])
    command(['scp',*OPTIONS,str(BANK),'2026:'+BANK_REMOTE])
    code=f'''import hashlib,importlib,json,sys
from pathlib import Path
sys.path.insert(0,{REMOTE!r});r=Path({REMOTE!r});bank=Path({BANK_REMOTE!r})
assert bank.stat().st_size==167098445 and hashlib.sha256(bank.read_bytes()).hexdigest()=={BANK_SHA!r}
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {(sources|reuse)!r}.items() if not n.startswith('results/'))
import torch
w=importlib.import_module('profile_r201bb_BA_job');importlib.import_module('profile_r201bb_parallel_jobs');importlib.import_module('analyze_r201bb_parallel_jobs')
assert w.recipe is importlib.import_module('run_r201ba_RGB_teacher_head')
assert w.make_loader is importlib.import_module('run_r201ac_RGB_hard_identity').make_loader
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='ACTUAL_BB_CPU_IMPORT_AND_DEPENDENCY_PASS',CUDA_initialized=False,models_built=0,NN_calls=0,updates=0)))'''
    warm=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    argv=[PYTHON,'-u','profile_r201bb_parallel_jobs.py','--data-root','/data/gaob/Re-ID/dataset',
        '--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
        '--anchor-run-dir',REMOTE+'/runs/full_official_baselines_20261004/training/RGBNT100_demo_s42',
        '--closed-BA-root',REMOTE+'/runs/r201ba_RGB_teacher_head_20261015','--gpu','2','--output',ROOT+'/profile_output']
    code=f'''import json,subprocess,time
from pathlib import Path
r=Path({ROOT!r});r.mkdir(exist_ok=False)
with (r/'controller.log').open('w') as log:
 p=subprocess.Popen({argv!r},cwd={REMOTE!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=p.pid,started=time.time(),argv={argv!r},status='CONTROLLER_STARTED_NOT_BENCHMARK_SUCCESS')
(r/'remote_owner.json').write_text(json.dumps(d,indent=2)+'\\n');print(json.dumps(d))'''
    owner=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    data=dict(status='ACTUAL_BB_CONTROLLER_STARTED_NOT_COMPLETE',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        owner=owner,local_deployer_pid=os.getpid(),capacity=ready,CPU_warm=warm,sources_sha256=sources,reused_sources_sha256=reuse,
        protected_weights=protected,required_bank=dict(remote=BANK_REMOTE,local=str(BANK),bytes=167098445,sha256=BANK_SHA),
        selected_physical_gpu=2,max_NN=2,no_GPU3_NN=True,temperature_power_read=False,
        actual_BA_result_audit=audit['status'],estimated_first_terminal_observation_seconds=1800,automatic_retry=False,
        limits='Detached controller start is not benchmark completion. Current BA NN/consumers already closed. Only finite speed workload/no method result or weight writes.')
    proof.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=data['status'],owner=owner,CPU_warm=warm)),flush=True)


if __name__=='__main__':main()
