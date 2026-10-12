"""Start two-card AE waves after AD completed hard pair and user-stopped random pair are recovered."""
from datetime import datetime
from pathlib import Path
import hashlib,json,os,shlex,shutil,subprocess,sys
PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
TAG='r201ae_MS_strong_staged_20261012';ROOT=REMOTE+'/runs/'+TAG
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')/TAG


def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    closed=load(pf/'r201ad_gpu23_transport_recovery_closed_actual_20261012.json')
    assert closed['status']=='ACTUAL_AD_GPU23_HARD_FULL50_AND_RANDOM_USER_STOP_GT_RAW_CPU_RECOVERED'
    assert closed['all_original_NN_and_audit_children_closed'] and closed['completed_control_updates']==13126 and closed['user_stopped_controls']==2
    assert closed['recovery_exit_code']==0 and closed['recovery_terminal_consumed'] and closed['recovery_original_process_absent']
    reference=load(pf/'msvr_stronger_private_reference_actual_20261012.json')
    assert reference['status']=='ACTUAL_STRONG_TEACHER_PRIVATE_REFERENCE_ALL_OFFICIAL_QUERY_GALLERY_PARITY_NO_TRAINING'
    review=load(pf/'r201ae_MS_strong_staged_source_review_20261012.json');assert review['status']=='PASS' and not review['blocking_findings']
    sources=review['sources_sha256'];reuse=review['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    proof=pf/'r201ae_MS_strong_staged_actual_session_20261012.json';assert not proof.exists() and not ARCHIVE.exists()
    assert shutil.disk_usage(ARCHIVE.parent).free>3*1024**3
    protected=load(pf/'r201ad_capacity_anchors_actual_20261012.json')['protected_weights'];assert len(protected)==4
    code=f'''import hashlib,json,shutil,subprocess
from pathlib import Path
r=Path({REMOTE!r});assert not Path({ROOT!r}).exists()
selected_GPU_compute_processes={{g:subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip() for g in (2,3)}}
assert not any(selected_GPU_compute_processes.values())
assert shutil.disk_usage(r).free>2_300_000_000
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {protected!r}.items())
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {reuse!r}.items() if not n.startswith('results/'))
print(json.dumps(dict(project_free=shutil.disk_usage(r).free,selected_GPUs=[2,3],selected_GPU_compute_processes=selected_GPU_compute_processes,temperature_power_read=False)))'''
    ready=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    (pf/'r201ae_capacity_and_idle_actual_20261012.json').write_text(json.dumps(ready,indent=2)+'\n',encoding='utf-8')
    for n in sources:
        if not n.startswith('results/'):command(['scp',*OPTIONS,str(PROJECT/n),'2026:'+REMOTE+'/'+n])
    code=f'''import hashlib
from pathlib import Path
r=Path({REMOTE!r});assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {(sources|reuse)!r}.items() if not n.startswith('results/'))
print('ACTUAL_AE_REVIEWED_SOURCES_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    argv=[PYTHON,'-u','launch_r201ae_MS_strong_staged.py','--data-root','/data/gaob/Re-ID/dataset',
        '--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
        '--anchor-root',REMOTE+'/runs/full_official_baselines_20261004/training','--output',ROOT+'/MSVR310']
    code=f"""import json,subprocess,time
from pathlib import Path
r=Path({ROOT!r});r.mkdir(exist_ok=False)
with (r/'controller.log').open('w') as log:
 p=subprocess.Popen({argv!r},cwd={REMOTE!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(status='REMOTE_CONTROLLER_STARTED_NATIVE_AND_FORMAL_NOT_YET_CONFIRMED',pid=p.pid,started=time.time(),argv={argv!r},log=str(r/'controller.log'))
(r/'remote_owner.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))"""
    launched=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    state=dict(status='ACTUAL_AE_DETACHED_REMOTE_CONTROLLER_STARTED_NOT_NATIVE_OR_FORMAL_CONFIRMED',started=datetime.now().astimezone().isoformat(timespec='seconds'),
        launcher_pid=os.getpid(),remote_owner=launched,physical_gpus=[2,3],maximum_NN=2,sources_sha256=sources,protected_weights=protected,
        goal='ACTIVE_UNMET',planned_epochs=200,planned_formal_updates=2820,local_ACK_receiver_required=False,
        limits='User permits only physical GPU2/3, with at most two NN workers and two explicit waves. Original local AD receiver loss is actual evidence for detached remote lifecycle. All raw remains remote until a separate verified completion collector. No runtime native/formal/metric success inferred from Popen. No duplicate controller or automatic restart.')
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');print(json.dumps(state),flush=True)


if __name__=='__main__':main()
