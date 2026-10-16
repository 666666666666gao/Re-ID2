"""Collect actual completed B128 evidence paths for an independent reviewer; no NN."""
from datetime import datetime,timezone,timedelta
import hashlib,json
from pathlib import Path

P=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
PF=P/'results/preflight'


def main():
    output=PF/'rgbnt100_paper_batch_result_integrity_request_20261016.json'
    assert not output.exists()
    intake_path=PF/'rgbnt100_paper_batch_intake_actual_20261016.json'
    intake=json.loads(intake_path.read_text('utf-8'))
    assert intake['status']=='ACTUAL_RGBNT100_B128_TWO_FULL50_GT_ARCHIVED' and len(intake['archives'])==4
    root=Path(intake['local_text_root'])
    analysis=root/'experiments/normal_analysis/result.json'
    assert json.loads(analysis.read_text('utf-8'))['status']=='ACTUAL_RGBNT100_B128_PAIR_FULL_NORMAL_CPU_READOUT'
    paths=[intake_path]
    paths.extend(f for f in root.rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.jsonl','.log'))
    paths.extend(Path(row['local']) for row in intake['archives'].values())
    for name in ('rgbnt100_paper_batch_source_request_20261016.json','rgbnt100_paper_batch_postprocess_source_request_20261016.json'):
        request=PF/name
        paths.append(request)
        paths.extend(Path(row['path']) for row in json.loads(request.read_text('utf-8'))['sources'])
    paths.extend(PF/name for name in ('rgbnt100_paper_batch_actual_session_20261016.json',
        'rgbnt100_paper_batch_completion_actual_20261016.json','rgbnt100_paper_batch_source_review_20261016.json',
        'rgbnt100_paper_batch_postprocess_source_review_20261016.json','reid2_paper_reference_targets_20261016.json',
        'vehicle_atmoe_head_RGBNT100_result_integrity_audit_20261016.json'))
    sources=['run_rgbnt100_paper_batch_baseline.py','launch_rgbnt100_paper_batch_baselines.py',
        'audit_rgbnt100_paper_batch_normal.py','analyze_rgbnt100_paper_batch_normal.py',
        'analyze_identity_coordinate_three_normal.py','run_full_official_experiment.py',
        'vehicle_atmoe_head_configuration.py','official_training_data.py','full_evaluation.py',
        'experiment_data.py','run_experiment.py','audit_full_official49.py','gpu_thermal_execute.py',
        'modeling/make_model.py','configs/RGBNT100/DeMo.yml']
    paths.extend(P/name for name in sources)
    for folder in [P/'results/full_official_baselines_20261004/training/RGBNT100_demo_s42',
        P/'results/r201av_RGB_frequency_value_decay_20261014/RGBNT100/training/RGBNT100_r201av_axis_shared_s42',
        P/'results/vehicle_atmoe_head_dynamic_amp_20261016/RGBNT100/training/RGBNT100_atmoe32_demo_s42']:
        paths.extend(folder/name for name in ('result.json','best_per_query.json','best_per_query.csv','epochs.csv'))
    inputs=[]
    for file in dict.fromkeys(paths):
        data=file.read_bytes();inputs.append(dict(path=str(file),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    record=dict(status='CLOSED_RESULT_PATHS_COLLECTED_FOR_FRESH_INTEGRITY_REVIEW',
        at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),project=str(P),inputs=inputs,
        selected_text_root=str(root),archives=intake['archives'],source_files=[str(P/name) for name in sources],
        report=str(PF/'rgbnt100_paper_batch_result_integrity_audit_20261016.json'),
        trace_root=str(P/'.aris/traces/experiment-audit/2026-10-16_RGBNT100_B128_full_normal_run01'),
        scope='Two finite original-DeMo B128 identity baselines: heads8 and32,50epochs each from publicCLIP,seed42,P8K16,all8675train/50IDs and1715Q8575G. Not new dual-axis experts or missing results.',
        official_GT_root='/data/gaob/Re-ID/dataset',official_GT_dataset='RGBNT100',
        allowed_readonly_remote_CPU=dict(host_alias='2026',project='/data/gaob/Re-ID/DeMo-DualAxis',
            python='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python',
            transport_helper='C:/Users/gb/.codex_tmp/publish_RGBNT100_N5_s44_closed_s45_started_20261015.py',
            restriction='Only these actually completed selected archives and installedRGBNT100GT. NumPy/pureCPU permitted; noTorch/model/CUDA/NN/GPU/resources/process/currenttraining/log/PID/handlequeries or remote writes/deletion. Capture actual transport stdout/stderr privately; bounded nonsensitive summaries.'),
        local_python='C:/Users/gb/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/python.exe -X utf8 -B -S',
        scope_checks=['A-F experiment-audit checklist. Read sources and actual artifacts directly; GT from installed dataset, not model outputs. Recount full1715x8575 distance cases for BOTH new models, six metrics/CMC50/perquery/groups and same-ID same-camera exclusion.',
            'Read selected full5120D feature arrays and checkpoint tensors where feasible. Verify native strictreload evidence scope, head8/head32 actualexpert shapes, total/trainable counts, finiteness, configuration and actualpublicCLIP origin. Do not claim an independent weight-forward replay.',
            'Audit both full50epoch curves, earliestmAP-best same-checkpoint six, actual3135attempts permodel, pairedsampleorders, loss means and actual successfulupdates/AMPskips. Check forward8675coverage and successful-updatecoverage separately; six native updates are not formal updates.',
            'Verify five-row CPU comparisons and rescue/harm against original/AV/B64head32, all1715 querymetadata. Historical B64 changesP/updatecount, AV has extraexpert50. Onlynewheads8/32 B128 pair freshlymatched forheadfactor; no claim allhistoricalcomparisons singlefactor or equalcompute.',
            'Acceptance paperCLIP RGBNT10087.2mAP/98.6R1; ownreproduction is diagnostic. Chosen baseline performance is not dual-axis/Ours contribution. Fullresults are one seed42 and benchmark-selected epoch/config; no missing49/paper6 evaluation or multisedefficacy claim.'],
        requirements=['Fresh skeptical reviewer, exactrequest and input起终bytes/SHA; FINALPASS/WARN/FAIL withblocking_findings[]. RequestedAstra/max, actualrouteUNATTESTED ifunattested, same-family/provisional.',
            'No inputmutations or newdependency installation. Preserve actual failures/rounding limitations. JSONreport and ignoredtrace only, no extra humanhandoff or unnecessary permission question.'],
        new_neural_calls=0,new_optimizer_updates=0,review_started=False)
    output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(dict(status=record['status'],inputs=len(inputs),input_bytes=sum(r['bytes'] for r in inputs),request=str(output))))


if __name__=='__main__':
    main()
