"""Admit one reviewed global AV42 cutoff pair after complete F-local closure."""
from datetime import datetime,timezone,timedelta
from pathlib import Path
import importlib.util
import json
import shlex
import shutil

T=Path('C:/Users/gb/.codex_tmp')
spec=importlib.util.spec_from_file_location('transport',T/'publish_RGBNT100_N5_s44_closed_s45_started_20261015.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
P,command,load,OPTIONS,sha=prior.P,prior.command,prior.load,prior.OPTIONS,prior.sha
PF=P/'results/preflight'
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'


def main():
    output=PF/'rgbnt100_band_ranges_actual_session_20261016.json'
    assert not output.exists()
    review=load(PF/'rgbnt100_band_ranges_source_review_20261016.json')
    assert review['verdict']=='PASS' and not review['blocking_findings']
    request=load(PF/'rgbnt100_band_ranges_source_review_request_20261016.json')
    seals=request['sources_sha256']|request['directly_reused_sources_sha256']
    assert all(sha(P/n)==h for n,h in seals.items())
    prep=load(PF/'rgbnt100_band_ranges_preparation_20261016.json')
    sources=prep['neural_sources_sha256']
    assert len(sources)==5 and all(sha(P/n)==h for n,h in sources.items())
    selected=load(P/'results/rgbnt100_expert_N5_AV_20261015/five_normal_analysis/result.json')
    assert selected['status']=='ALL_FIVE_COMPLETE_SELECTED' and selected['recipe']=='AV'
    assert selected['main_best_single']['axis_shared']['seed']==42 and not selected['main_axis_both_plus1']
    audit=load(PF/'rgbnt100_F_local_pair_result_integrity_audit_20261016.json')
    assert audit['verdict'] in ('PASS','WARN') and not audit['blocking_findings']
    intake=load(PF/'rgbnt100_F_local_pair_complete_intake_20261016.json')
    assert intake['status']=='ACTUAL_F_LOCAL_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE'
    assert intake['terminal']['all_children_and_controller_absent']
    retired=load(PF/'rgbnt100_F_local_pair_nonleader_weights_retired_actual_20261016.json')
    assert retired['bytes_retired']==797412018 and all(not Path(r['path']).exists() for r in retired['retired'])
    assert shutil.disk_usage('D:/').free>3_000_000_000
    root=prep['remote_family']
    protected=load(PF/'r201ad_capacity_anchors_actual_20261012.json')['protected_weights']
    previous_sources=load(PF/'rgbnt100_expert_N5_source_review_20261015.json')['sources_sha256'].copy()
    previous_sources.update(load(PF/'rgbnt100_F_local_pair_actual_session_20261016.json')['sources_sha256'])
    old_owner=load(PF/'rgbnt100_F_local_pair_actual_session_20261016.json')['remote_owner']['pid']
    previous_local=P/'results/rgbnt100_F_local_grid2_AV_20261016/RGBNT100/controller_result.json'
    previous_remote=REMOTE+'/runs/rgbnt100_F_local_grid2_AV_20261016/RGBNT100/controller_result.json'
    code=f'''import hashlib,json,shutil,subprocess
from pathlib import Path
p=Path({REMOTE!r});assert not Path({root!r}).exists() and not Path('/proc/{old_owner}').exists()
cards={{g:subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip() for g in (2,3)}}
assert not any(cards.values()) and shutil.disk_usage(p).free>3_000_000_000
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {previous_sources!r}.items())
assert hashlib.sha256(Path({previous_remote!r}).read_bytes()).hexdigest()=={sha(previous_local)!r}
print(json.dumps(dict(project_free=shutil.disk_usage(p).free,selected_GPU_processes=cards,protected_four_exact=True,previous19_sources_exact=True,previous_controller_exact=True,temperature_power_read=False)))'''
    capacity=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    for name in sources:
        command(['scp',*OPTIONS,str(P/name),'2026:'+REMOTE+'/'+name])
    code=f'''import hashlib,json,sys
from pathlib import Path
p=Path({REMOTE!r});sys.path.insert(0,str(p))
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {sources!r}.items())
import torch
import rgbnt100_expert_seed_recipe as selected
runner,parent=selected.configure_runner('AV')
from rgbnt100_frequency_band_ranges import build_ranges
import identity_coordinate_rgbnt100_expert_seeds as factory
import run_r201g_normal_priority as g
import run_r201ac_RGB_hard_identity as sampler
assert g.coordinate_build is factory.build and runner.make_loader is sampler.make_loader
assert runner.make_optimizer is parent.make_optimizer and not torch.cuda.is_initialized()
print(json.dumps(dict(status='ACTUAL_BAND_RANGE_MAIN_ALIASES_CPU_PASS',sources_exact=True,models=0,NN_calls=0,CUDA_initialized=False)))'''
    warm=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    argv=[PYTHON,'-u','launch_rgbnt100_band_range_pair.py','--previous-controller',previous_remote,
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
    result=dict(status='ACTUAL_BAND_RANGE_CONTROLLER_STARTED_NATIVE_NOT_YET_CONFIRMED',at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        remote_root=root,remote_owner=owner,capacity=capacity,CPU_main_aliases=warm,sources_sha256=sources,
        source_review_sha256=sha(PF/'rgbnt100_band_ranges_source_review_20261016.json'),source_request_sha256=sha(PF/'rgbnt100_band_ranges_source_review_request_20261016.json'),
        previous_controller_sha256=sha(previous_local),protected_weights=protected,physical_gpus=[2,3],maximum_NN=2,
        first_native_node_seconds=180,near_end_after_actual_formal_start_seconds=10800,necessary_followup_seconds=300,
        formal_complete=False,new_metrics_claimed=False,band_low_cutoff=.25,band_high_cutoff=.375,
        source_changes='Only existing persistent FFT masks on global AV42, both M/F and both controls; zero added learned parameters. Failed local-grid methods are not installed.',
        goal='ACTIVE_UNMET')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps({k:result[k] for k in ('status','at','remote_owner','capacity')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
