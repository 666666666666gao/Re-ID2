"""Collect completed-pair evidence for fresh integrity review; never run an NN."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

P = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
PF = P / 'results/preflight'


def main():
    output = PF / 'rgbnt100_B128_reference_result_integrity_request_20261026.json'
    assert not output.exists()
    intake_path = PF / 'rgbnt100_B128_reference_normfix_intake_actual_20261026.json'
    intake = json.loads(intake_path.read_text('utf-8'))
    assert intake['status'] == 'ACTUAL_RGBNT100_B128_REFERENCE_TWO_FULL50_NORMAL_EIGHT_STATES_ARCHIVED'
    assert intake['formal_updates'] == 13126 and intake['native_updates'] == 6 and intake['state_cases'] == 8
    assert len(intake['archives']) == 6 and len(intake['children']) == 10
    assert all(r['exit_code'] == 0 for r in intake['children'])
    root = Path(intake['local_text_root'])
    analysis = json.loads((root / 'experiments/normal_analysis/result.json').read_text('utf-8'))
    assert analysis['status'] == 'ACTUAL_RGBNT100_B128_REFERENCE_EXPERT_PAIR_FULL_NORMAL_CPU_READOUT'
    assert analysis['new_full50_models'] == 2 and analysis['state_cases'] == 8
    paths = [intake_path]
    paths += [f for f in root.rglob('*') if f.is_file() and f.suffix in ('.json', '.csv', '.jsonl', '.log')]
    paths += [Path(r['local']) for r in intake['archives'].values()]
    run_request_path = PF / 'rgbnt100_B128_reference_normfix_source_request_20261026.json'
    post_request_path = PF / 'rgbnt100_B128_reference_postprocess_source_request_20261026.json'
    run_request = json.loads(run_request_path.read_text('utf-8'))
    post_request = json.loads(post_request_path.read_text('utf-8'))
    paths += [run_request_path, post_request_path]
    paths += [Path(r['path']) for r in run_request['sources'] + run_request['additional_private_launchers'] + post_request['sources']]
    paths += [P / n for n in run_request['inherited_sources_sha256']]
    paths += [PF / n for n in ['rgbnt100_B128_reference_experts_normfix_actual_session_20261026.json',
        'rgbnt100_B128_reference_experts_normfix_startup_actual_20261026.json',
        'rgbnt100_B128_reference_normfix_source_review_20261026.json',
        'rgbnt100_B128_reference_postprocess_source_review_20261026.json',
        'rgbnt100_B128_cached_normalization_arithmetic_actual_20261026.json',
        'reid2_paper_reference_targets_20261016.json']]
    teacher_intake = json.loads((PF / 'rgbnt100_paper_batch_intake_actual_20261016.json').read_text('utf-8'))
    paths += [Path(teacher_intake['archives'][n]['local']) for n in ('head8/best', 'head8/raw')]
    bank = json.loads((PF / 'rgbnt100_B128_training_bank_intake_actual_20261017.json').read_text('utf-8'))
    paths.append(PF / 'rgbnt100_B128_training_bank_intake_actual_20261017.json')
    sources = ['analyze_rgbnt100_B128_reference_experts_normal.py',
        'results/preflight/intake_rgbnt100_B128_reference_normfix_20261026.py',
        'launch_rgbnt100_B128_reference_experts_normfix.py', 'evaluate_rgbnt100_B128_reference_normfix.py',
        'evaluate_rgbnt100_B128_reference_states.py', 'run_rgbnt100_B128_reference_experts.py',
        'original_identity_anchor.py', 'official_training_data.py', 'full_evaluation.py',
        'audit_full_official_normal.py', 'audit_full_official49.py', 'experiment_data.py',
        'run_full_official_frozen_anchor_experiment.py', 'analyze_identity_coordinate_three_normal.py']
    paths += [P / n for n in sources]
    inputs = []
    for file in dict.fromkeys(paths):
        data = file.read_bytes()
        inputs.append(dict(path=str(file), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    for v in intake['archives'].values():
        f = Path(v['local'])
        assert f.stat().st_size == v['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest() == v['sha256']
    record = dict(status='CLOSED_B128_REFERENCE_RESULT_PATHS_COLLECTED_FOR_FRESH_REVIEW',
        at_utc=datetime.now(timezone.utc).isoformat(), project=str(P), inputs=inputs, selected_text_root=str(root),
        archives=intake['archives'], teacher_archives={n: teacher_intake['archives'][n] for n in ('head8/best', 'head8/raw')},
        training_bank_intake=bank, source_files=[str(P / n) for n in sources],
        report=str(PF / 'rgbnt100_B128_reference_result_integrity_audit_20261026.json'),
        trace_root=str(P / '.aris/traces/experiment-audit/2026-10-26_RGBNT100_B128_reference_full_normal_run01'),
        scope='Two existing AV ordinary/dual models on one B128/head8 teacher and its train-only bank; each extra50/6563 formal updates, native6 separate; full8675train/50IDs,1715Q8575G and eight frozen states. No new missing/49/201 or multiseed result.',
        official_GT_root='/data/gaob/Re-ID/dataset', official_GT_dataset='RGBNT100',
        allowed_readonly_remote_CPU=dict(host_alias='2026', project='/data/gaob/Re-ID/DeMo-DualAxis',
            python='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python',
            transport_helper='C:/Users/gb/.codex_tmp/publish_RGBNT100_N5_s44_closed_s45_started_20261015.py',
            restriction='Only these already completed selected inputs and installedRGBNT100GT. Pure CPU NumPy and CPU-only torch.load(map_location=cpu,weights_only=True) are allowed with CUDA_VISIBLE_DEVICES empty. No model forward/training/CUDA/GPU/resources/process/current-job/log/PID/handle queries or remote writes/deletion. Capture actual transport privately; no secrets in report.'),
        local_python='C:/Users/gb/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/python.exe -X utf8 -B -S',
        scope_checks=[
            'Read actual artifacts/sources, not producer status alone. Recompute both normal and all eight frozen-state distances against installed full1715Q8575G identity/camera GT; validate same-ID same-camera exclusion, all six metrics,CMC1-50,AP/INP,validity and groups. State11 distance/features must equal selected normal; state00 must equal frozen teacher. No output-as-GT.',
            'Read all current selected raw feature/distance arrays and checkpoint tensors on CPU; finite shapes and actual5120D,selected epoch and archive binding,original teacher identity modules frozen byte-exact as defined in source. Report actual model/parameter count and native/reload evidence scope. Do not claim independent neural weight-forward replay.',
            'Audit both continuous50 curves/earliest best/last and same-checkpoint six;6563 successful updates/control,0formalAMPskip,6native separate,full8675 records and paired epoch/name/partial-set orders. Check losses/coverage and benchmark-selection disclosure.',
            'Verify six main rows, all1715 query cohorts,paper target flags,ordinary-vs-dual and teacher-relative changes,00/10/01/11 rescues/harms and AP changes. Frozen interventions are not retrained ablations. Historical AV changes teacher and its bank, so no pure one-factor or equal totalcompute claim against original50.',
            'PaperRGBNT10087.2mAP/98.6R1 governs acceptance; own reproduction is diagnostic. Separate teacher strength from expert increment and ordinary comparison. K201 untouched,MSnot evaluated here. BestSingle allsix one selected checkpoint; no fabricated missing,SOTA,calibrated utility,multiseed or generalization guarantee.',
            'The original full00 failure was duplicate normalization, proved by fixed-cache arithmetic; new isolated reference fix omits the second normalize while leaving scoring/model/gates unchanged. Preserve original failure and numeric precision limitations. If feature-derived re-scoring differs at FP32 near ties, quantify it separately rather than silently changing scores or declaring independent replay.'
        ],
        requirements=['Fresh skeptical Codex reviewer; verify input bytes before/after, FINALPASS/WARN/FAIL with blocking_findings. RequestedAstra/max is not host attestation; actualrouteUNATTESTED when unavailable,same-family/provisional.',
            'No input mutation/new install/repeated NN/unauthorized messages. JSON report and ignored traces only; no extra human handoff. Adoption and any binary cleanup remain after this actual integrity review.'],
        new_NN=0, new_optimizer_updates=0, review_started=False)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    print(json.dumps(dict(status=record['status'], inputs=len(inputs), input_bytes=sum(v['bytes'] for v in inputs), request=str(output))), flush=True)


if __name__ == '__main__':
    main()
