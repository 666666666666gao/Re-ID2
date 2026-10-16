"""Admit one F-rank128 pair on original global AV42 after audited cutoff closure."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import importlib.util
import json
import shlex
import shutil

T = Path('C:/Users/gb/.codex_tmp')
spec = importlib.util.spec_from_file_location('transport', T/'publish_RGBNT100_N5_s44_closed_s45_started_20261015.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
P, command, load, OPTIONS, sha = prior.P, prior.command, prior.load, prior.OPTIONS, prior.sha
PF = P/'results/preflight'
REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'


def main():
    output = PF/'rgbnt100_F_rank_actual_session_20261016.json'
    assert not output.exists()
    review = load(PF/'rgbnt100_F_rank_deployment_source_review_20261016.json')
    assert review['verdict'] == 'PASS' and not review['blocking_findings']
    request = load(PF/'rgbnt100_F_rank_deployment_source_review_request_20261016.json')
    seals = request['sources_sha256'] | request['directly_reused_sources_sha256']
    assert all(sha(P/n) == h for n, h in seals.items())
    preparation = load(PF/'rgbnt100_F_rank_suite_R2_preparation_20261016.json')
    sources = preparation['sources_sha256']
    assert len(sources) == 5 and all(sha(P/n) == h for n, h in sources.items())
    neural = load(PF/'rgbnt100_F_rank_neural_source_review_R2_20261016.json')
    assert neural['status'] == 'PASS_SOURCE_ONLY' and neural['verdict'] == 'PASS' and not neural['blocking_findings']
    policy = load(PF/'reid2_paper_reference_targets_20261016.json')
    assert policy['normal']['RGBNT100']['target'] == dict(mAP=87.2, Rank_1=98.6)
    selected = load(P/'results/rgbnt100_expert_N5_AV_20261015/five_normal_analysis/result.json')
    assert selected['status'] == 'ALL_FIVE_COMPLETE_SELECTED' and selected['recipe'] == 'AV'
    assert selected['main_best_single']['axis_shared']['seed'] == 42
    audit = load(PF/'rgbnt100_band_ranges_result_integrity_audit_20261016.json')
    assert audit['status'] == 'FINAL' and audit['verdict'] in ('PASS', 'WARN') and not audit['blocking_findings']
    intake = load(PF/'rgbnt100_band_ranges_complete_intake_20261016.json')
    assert intake['status'] == 'ACTUAL_BAND_RANGE_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE'
    assert intake['terminal']['all_children_and_controller_absent']
    retired = load(PF/'rgbnt100_band_ranges_nonleader_weights_retired_actual_20261016.json')
    assert retired['bytes_retired'] == 797412018 and all(not Path(row['path']).exists() for row in retired['retired'])
    assert shutil.disk_usage('D:/').free > 3_000_000_000
    root = REMOTE+'/runs/rgbnt100_F_rank128_AV_20261016'
    protected = load(PF/'r201ad_capacity_anchors_actual_20261012.json')['protected_weights']
    previous_sources = load(PF/'rgbnt100_expert_N5_source_review_20261015.json')['sources_sha256'].copy()
    previous_sources.update(load(PF/'rgbnt100_F_local_pair_actual_session_20261016.json')['sources_sha256'])
    previous_sources.update(load(PF/'rgbnt100_band_ranges_actual_session_20261016.json')['sources_sha256'])
    assert len(previous_sources) == 24
    old_owner = load(PF/'rgbnt100_band_ranges_actual_session_20261016.json')['remote_owner']['pid']
    previous_local = P/'results/rgbnt100_band_ranges_AV_20261016/RGBNT100/controller_result.json'
    previous_remote = REMOTE+'/runs/rgbnt100_band_ranges_AV_20261016/RGBNT100/controller_result.json'
    code = f'''import hashlib,json,shutil,subprocess
from pathlib import Path
p=Path({REMOTE!r});assert not Path({root!r}).exists() and not Path('/proc/{old_owner}').exists()
cards={{g:subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip() for g in (2,3)}}
assert not any(cards.values()) and shutil.disk_usage(p).free>3_000_000_000
memory=subprocess.check_output(['nvidia-smi','-i','2,3','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {previous_sources!r}.items())
assert hashlib.sha256(Path({previous_remote!r}).read_bytes()).hexdigest()=={sha(previous_local)!r}
print(json.dumps(dict(project_free=shutil.disk_usage(p).free,selected_GPU_processes=cards,memory_and_utilization_snapshot=memory,protected_four_exact=True,previous24_sources_exact=True,previous_controller_exact=True,temperature_power_read=False)))'''
    capacity = json.loads(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON)+' -'], input=code))
    for name in sources:
        command(['scp', *OPTIONS, str(P/name), '2026:'+REMOTE+'/'+name])
    code = f'''import hashlib,json,sys
from pathlib import Path
p=Path({REMOTE!r});sys.path.insert(0,str(p))
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {sources!r}.items())
import torch
import rgbnt100_expert_seed_recipe as selected
runner,parent=selected.configure_runner('AV')
from rgbnt100_frequency_rank import build_rank,make_optimizer
import identity_coordinate_rgbnt100_expert_seeds as factory
import run_r201g_normal_priority as g
import run_r201ac_RGB_hard_identity as sampler
assert g.coordinate_build is factory.build and runner.make_loader is sampler.make_loader
assert runner.make_optimizer is parent.make_optimizer and not torch.cuda.is_initialized()
print(json.dumps(dict(status='ACTUAL_F_RANK_MAIN_ALIASES_CPU_PASS',sources_exact=True,models=0,NN_calls=0,CUDA_initialized=False)))'''
    warm = json.loads(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON)+' -'], input=code))
    argv = [PYTHON, '-u', 'launch_rgbnt100_frequency_rank_pair.py', '--previous-controller', previous_remote,
            '--data-root', '/data/gaob/Re-ID/dataset', '--pretrained', '/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
            '--anchor-root', REMOTE+'/runs/full_official_baselines_20261004/training', '--output', root+'/RGBNT100']
    code = f'''import json,subprocess,time
from pathlib import Path
r=Path({root!r});r.mkdir(parents=True,exist_ok=False)
with (r/'controller.log').open('w') as log:
 p=subprocess.Popen({argv!r},cwd={REMOTE!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
owner=dict(pid=p.pid,started=time.time(),argv={argv!r})
(r/'remote_owner.json').write_text(json.dumps(owner,indent=2)+'\\n')
print(json.dumps(owner))'''
    owner = json.loads(command(['ssh', *OPTIONS, '2026', shlex.quote(PYTHON)+' -'], input=code))
    result = dict(status='ACTUAL_F_RANK128_CONTROLLER_STARTED_NATIVE_NOT_YET_CONFIRMED',
        at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'), remote_root=root,
        remote_owner=owner, capacity=capacity, CPU_main_aliases=warm, sources_sha256=sources,
        source_review_sha256=sha(PF/'rgbnt100_F_rank_deployment_source_review_20261016.json'),
        source_request_sha256=sha(PF/'rgbnt100_F_rank_deployment_source_review_request_20261016.json'),
        previous_controller_sha256=sha(previous_local), protected_weights=protected, physical_gpus=[2, 3], maximum_NN=2,
        first_native_node_seconds=180, near_end_after_actual_formal_start_seconds=13200, necessary_followup_seconds=300,
        formal_complete=False, new_metrics_claimed=False, frequency_rank=128,
        expected_parameters=99947398, expected_trainable_parameters=2249606,
        paper_target=dict(mAP=87.2, Rank_1=98.6),
        source_changes='Only existing F hidden width64 to128 and existing conditioning Linear output, original global AV42 readers/default FFT and original module responsibilities; same change both controls. F/scorer/pooling installed hooks and non-F RNG retained by R2. No failed cutoff/local stacking or new seed sweep.',
        goal='ACTIVE_UNMET')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'at', 'remote_owner', 'capacity')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
