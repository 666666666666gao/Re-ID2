"""Serialize the independent narrow review; no experiment execution."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_axis_v5_frozen_eval'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_v5_frozen26_deploy_20261003.py')
harness = json.loads((TRACE / '001-harness-results.json').read_text(encoding='utf-8'))
assert harness['status'] == 'PASS'
assert harness['helper_sha256'] == hashlib.sha256(HELPER.read_bytes()).hexdigest()
assert set(harness['checked_source_sha256']) == {'missing_mass_development.py', 'diagnose_mass_axis.py', 'launch_mass_frozen_evaluation.py'}
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in harness['checked_source_sha256'].items())
launch = json.loads((ROOT / 'results/preflight/axis_collaboration_v5_mass_launch.json').read_text(encoding='utf-8'))
assert len(launch['source_sha256']) == 18
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in launch['source_sha256'].items())

source_checks = [
    dict(name='V5_build_hooks_shapes_and_independent_states', status='PASS', mode='source_review', detail='run_mass_experiment.build returns the FP32 CUDA MassAxisCollaborationDeMo and accepts the actual terminal arguments. New extraction hooks see two M reads/two F reads and one normal calibrator call, then two independent calibrator calls. Inherited controlled_states uses independent m0/f0, zeros the disabled expert, and calls the overridden V5 fuse. State10 recomputes relation_score(modality) softmax mass without F conditions, full-state mass or joint psi; state01 has no M input. Full state11 is the original forward result. All four descriptors retain 5632 dimensions; route, gates, prediction and norm arrays keep per-record alignment. Hooks/methods are restored after extraction.'),
    dict(name='frozen_FP32_zero_updates_and_no_test_selection', status='PASS', mode='source_review', detail='Both leaves load only the dev-selected best.pth with weights_only=True/strict=True, call model.eval and assert every module is in eval mode. Their extractors are under torch.no_grad; no optimizer construction/step, backward, model save, checkpoint search or official_records invocation is reachable. Registered tensor versions are compared before/after inference; launcher additionally checks all four frozen inputs after every smoke/full stage. Initial smoke uses exactly dev[:64], matching the inference loader batch size64. Full11/clean features require exact error0 before any saved distance reuse; strict existing metric tolerance remains1e-8.'),
    dict(name='installed_GT_queries_and_two_fixed_references', status='PASS', mode='source_review', detail='split_records reads only installed training directories and fixed identity-disjoint development IDs. Both leaves compare saved query indices, IDs, camera, scene and names to installed ordered records before inference. RGBNT uses same-ID/same-camera junk exclusion; MSVR310 uses same-ID/same-scene. For each of base00 and full11 reference galleries, reference_contributions selects a GT-valid hardest positive and hardest different-ID negative once and shares their fixed reference direction across all four query states. The two galleries may select different pairs, explicitly recorded in separate CSV/NPZ fields. Predictions are diagnostics, never retrieval ground truth.'),
    dict(name='full_metric_and_output_schema', status='PASS', mode='source_review', detail='The reused full_metrics receives query-by-gallery distances with the same ordered GT and outputs mAP, mINP, Rank-1/5/10/20, CMC1..50, per-query CSV and camera/scene/identity groups; it crosschecks the official-style stable-sort evaluate_reid implementation. The development split includes only queries with cross-camera/scene positives, so per-query harm/rescue rows stay valid and aligned. State11 uses saved distances after exact feature parity; states00/10/01 use original full_evaluation.distance. Reports/CSV/NPZ names match the launcher reads; two reference contribution tables and state arrays retain their own fixed indices.'),
    dict(name='six_masks_thirteen_conditions_and_baseline_reuse', status='PASS', mode='source_review', detail='The copied repaired leaf imports the existing six normalized-zero input masks r/n/t/rn/rt/nt and METRICS. Smoke extracts clean plus all six masks, checks shape/finite values and state versions. Full evaluation emits clean plus six both-missing and six query-only conditions; query-only uses each masked query against the clean gallery, while both-missing uses the same mask on both. Clean reuses saved distances only after exact clean-feature parity and retains strict_reload agreement. The launcher schedules only axis_mass_fullref, so no historical DeMo baseline is rerun.'),
    dict(name='physical_paths_quoting_and_authorized_resource_scope', status='PASS', mode='source_review', detail='collect_results.HOSTS2026 resolves /data/gaob/Re-ID/DeMo-DualAxis and the existing tri_reid/bin/python; the V5 launch receipt supplies the matching training root. Helper uses those paths, existing dataset/CLIP paths, and cwd=root so root imports and configs resolve. Its remote_python uses shlex.quote for the interpreter and Python code. Exactly three new leaves are uploaded after read-only guards; old sources are only checked. Controller child stages set CUDA_VISIBLE_DEVICES=3, use the existing idle(3) threshold and240second wait, and never kill or preempt other jobs. Existing exclusive output/log guards are retained without new fallback/retry machinery.')
]
checks = harness['checks'] + source_checks
report = dict(
    status='PASS', blockers=[], non_blocking_findings=[],
    observed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
    helper_sha256=harness['helper_sha256'], checked_source_sha256=harness['checked_source_sha256'],
    checks_count=len(checks), deterministic_groups=len(harness['checks']), source_review_groups=len(source_checks),
    mock_cases=sum(len(check.get('cases', [])) for check in harness['checks']), checks=checks,
    reviewer_agent='/root/review_v5_frozen_eval', requested_reviewer_model='gpt-6-astra',
    requested_reasoning_effort='max', fork_turns='none', provider_internal_identity_inferred=False,
    review_independence='same-family', acceptance_status='provisional',
    scope='Only three new V5 frozen-evaluation leaves and private2026 GPU3 deploy helper, with narrow existing API/schema/source-binding verification.',
    real_remote_calls=0, real_process_spawns=0, torch_imports=0, gpu_forwards=0, optimizer_updates=0,
    source_or_handoff_modified=False, source_binding_rechecked_at_finalization=True,
    preserved_existing_source_count=18, harness_attempts=1, harness_failures_preserved=[],
    harness_status='First stdlib AST/mock harness execution exited0. Initial bare python discovery failed with No pyvenv.cfg; the existing uv interpreter ran the harness without installation.',
    trace_path='.aris/traces/experiment-bridge/2026-10-03_axis_v5_frozen_eval',
    limitations=[
        'PASS covers reviewed source and mocked orchestration. No SSH, upload, actual subprocess child, Torch/model execution, dataset evaluation or GPU measurement was performed by this reviewer.',
        'Mock file contents/results/GPU memory are fixtures for sequencing and failure gates, not actual frozen-feature/metric/accuracy/resource evidence. Runtime smoke and source/input/resource guards remain responsible for those checks.',
        'The18 existing sources were hash-compared and only required APIs were inspected; this is not a new historical model/training/results review.',
        'No deployment, checkpoint reselection, optimizer update, official-test use or baseline rerun was performed. Single-seed frozen interventions are diagnostics, not independently trained ablations or a final acceptance claim.'
    ])
raw = (json.dumps(report, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
(TRACE / 'review-result.json').write_bytes(raw)
target = ROOT / 'results/preflight/axis_collaboration_v5_frozen_review.json'
assert not target.exists()
target.write_bytes(raw)
meta = dict(status='PASS', checks_count=len(checks), helper_sha256=report['helper_sha256'], checked_source_sha256=report['checked_source_sha256'],
            requested_route=dict(model='gpt-6-astra', reasoning_effort='max', fork_turns='none'),
            real_remote_calls=0, torch_imports=0, source_or_handoff_modified=False)
(TRACE / 'run.meta.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=report['status'], checks_count=report['checks_count'], mock_cases=report['mock_cases'], helper_sha256=report['helper_sha256'], checked_source_sha256=report['checked_source_sha256'], report=str(target))))
