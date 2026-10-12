"""Admit AH only after actual AG terminal/archive and selected GPU23 are free."""
from datetime import datetime
from pathlib import Path
import hashlib,json,os,shlex,shutil,sys
PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
TAG='r201ah_MS_full_identity_20261012';ROOT=REMOTE+'/runs/'+TAG


def main():
    pf=PROJECT/'results/preflight';load=lambda f:json.loads(f.read_text(encoding='utf-8-sig'))
    ag=load(pf/'r201ag_RGB_coordinate_metric_20261012_complete_intake.json')
    assert ag['status']=='ACTUAL_AG_TWO_FULL50_NORMAL_GT_D_ARCHIVED_COMPLETE'
    assert ag['terminal']['all_children_and_controller_absent'] and ag['unique_best_deleted']==0
    assert all(Path(r['local']).stat().st_size==r['bytes'] and hashlib.sha256(Path(r['local']).read_bytes()).hexdigest()==r['sha256'] for r in ag['archives'].values())
    review=load(pf/'r201ah_MS_full_identity_source_review_20261012.json');assert review['status']=='PASS' and not review['blocking_findings']
    sources=review['sources_sha256'];reuse=review['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==h for n,h in (sources|reuse).items())
    proof=pf/(TAG+'_actual_session.json');assert not proof.exists()
    assert not load(Path('C:/Users/gb/.codex_tmp/r201l_uniform_k8_continuation_20261007.json'))['live_native_sessions']
    assert shutil.disk_usage('D:/').free>1_000_000_000
    protected=load(pf/'r201ad_capacity_anchors_actual_20261012.json')['protected_weights']
    cache=load(pf/'msvr_strong_shared_arrays_present_CPU_actual_20261012.json')['record']
    old_owner=load(pf/'r201ag_RGB_coordinate_metric_20261012_actual_session.json')['remote_owner']['pid']
    code=f'''import hashlib,json,shutil,subprocess
from pathlib import Path
r=Path({REMOTE!r});assert not Path({ROOT!r}).exists() and not Path('/proc/{old_owner}').exists()
selected={{g:subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip() for g in (2,3)}}
assert not any(selected.values()) and shutil.disk_usage(r).free>1_400_000_000
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
c={cache!r};f=Path(c['path']);assert f.stat().st_size==c['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==c['sha256']
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {reuse!r}.items() if not n.startswith('results/'))
print(json.dumps(dict(project_free=shutil.disk_usage(r).free,selected_GPU_processes=selected,protected_bests_exact=True,strong_cache_exact=True,temperature_power_read=False)))'''
    ready=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    for n in sources:
        if not n.startswith('results/'):command(['scp',*OPTIONS,str(PROJECT/n),'2026:'+REMOTE+'/'+n])
    code=f'''import hashlib,json
from pathlib import Path
r=Path({REMOTE!r});assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {(sources|reuse)!r}.items() if not n.startswith('results/'))
print(json.dumps(dict(reviewed_sources_exact=True)))'''
    admitted=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    argv=[PYTHON,'-u','launch_r201ah_MS_full_identity.py','--data-root','/data/gaob/Re-ID/dataset',
        '--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
        '--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',ROOT+'/MSVR310']
    code=f'''import json,subprocess,time
from pathlib import Path
r=Path({ROOT!r});r.mkdir(exist_ok=False)
with (r/'controller.log').open('w') as log:
 p=subprocess.Popen({argv!r},cwd={REMOTE!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
owner=dict(pid=p.pid,started=time.time(),argv={argv!r},status='REMOTE_CONTROLLER_STARTED_REFERENCE_NATIVE_FORMAL_NOT_CONFIRMED')
(r/'remote_owner.json').write_text(json.dumps(owner,indent=2)+'\\n');print(json.dumps(owner))'''
    owner=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    row=dict(status='ACTUAL_AH_DETACHED_CONTROLLER_STARTED_REFERENCE_NATIVE_NOT_CONFIRMED',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        local_deploy_pid=os.getpid(),remote_owner=owner,capacity=ready,admission=admitted,sources_sha256=sources,reused_sources_sha256=reuse,
        protected_weights=protected,physical_gpus=[2,3],maximum_NN=2,native_updates_planned=6,formal_updates_planned=1410,epochs_planned=100,
        trained_normal_state_cases_planned=8,goal='ACTIVE_UNMET',estimated_training_seconds=1500,first_metric_observation_after_formal_seconds=1260,
        limits='Popen is not reference/native/formal success. AG already truly terminal/Dverified; max2 physicalGPU23 NN. '
            'Strongteacher retained/inherited gains disclosed; actual full00GT parity before native3/formal50. '
            'Detached controller returncode unobserved; child exit records and controller COMPLETE/absence separate. '
            'No ACK/autoretry/201NN/missing专项/new49/seedsearch/T-P/GPU01.')
    proof.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:row[k] for k in ('status','at','local_deploy_pid','remote_owner','capacity')}),flush=True)


if __name__=='__main__':main()
