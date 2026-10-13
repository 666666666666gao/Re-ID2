"""Admit one matched AS controller after reviewed sources and closed AR/AQ diagnosis."""
from datetime import datetime
from pathlib import Path
import hashlib,json,os,shlex,shutil,sys
PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
TAG='r201as_RGB_query_normalization_20261014';ROOT=REMOTE+'/runs/'+TAG


def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text('utf-8-sig'))
    ap=load(pf/'r201ar_AQ_spatial_attention_20261014_actual_session.json')
    assert ap['status']=='ACTUAL_AR_AQ_FULL_QG_SPATIAL_ATTENTION_COMPLETE'
    assert ap['terminal']['original_children_controller_absent'] and ap['full_output_features_exact']
    review=load(pf/'r201as_RGB_query_normalization_source_review_20261014.json')
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
    argv=[PYTHON,'-u','launch_r201as_RGB_query_normalization.py','--data-root','/data/gaob/Re-ID/dataset',
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
    row=dict(status='ACTUAL_AS_DETACHED_CONTROLLER_STARTED_NATIVE_NOT_YET_CONFIRMED',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        local_deploy_pid=os.getpid(),remote_owner=owner,capacity=ready,admission=admitted,sources_sha256=sources,
        reused_sources_sha256=reuse,protected_weights=protected,physical_gpus=[2,3],maximum_NN=2,
        native_updates_planned=6,formal_updates_planned=13126,epochs_planned=100,goal='ACTIVE_UNMET',
        estimated_training_hours=3.0,first_near_end_observation_after_formal_seconds=11100,
        limits='Popen is not native/formal success. One controller/no automatic retry. Only M projected query normalized; AQ packed QKV decay0 inherited. AR actual full QG closed; K201/teacher/MSstrong/Dunique preserved. No201NN/missing专项/49/seedfishing/T-P.')
    proof.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:row[k] for k in ('status','at','local_deploy_pid','remote_owner','capacity')}),flush=True)


if __name__=='__main__':main()
