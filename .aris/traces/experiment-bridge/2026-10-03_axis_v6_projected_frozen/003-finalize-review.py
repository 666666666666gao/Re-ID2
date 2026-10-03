"""Serialize the narrow V6 frozen-evaluation review without running experiments."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_axis_v6_projected_frozen'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_v6_projected_frozen26_deploy_20261003.py')
OBSERVER = Path('C:/Users/gb/.codex_tmp/demo_axis_v6_projected_frozen26_observe_20261003.py')
harness = json.loads((TRACE / '001-harness-results.json').read_text(encoding='utf-8'))
assert harness['status'] == 'PASS' and harness['checks_count'] == 15
assert harness['helper_sha256'] == hashlib.sha256(HELPER.read_bytes()).hexdigest()
assert harness['observer_sha256'] == hashlib.sha256(OBSERVER.read_bytes()).hexdigest()
assert set(harness['checked_source_sha256']) == {'missing_projected_mass_development.py', 'diagnose_projected_mass_axis.py', 'launch_projected_mass_frozen_evaluation.py'}
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value for name, value in harness['checked_source_sha256'].items())
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value for name, value in harness['prepared_dependency_sha256'].items())
prior = ROOT / 'results/preflight/axis_collaboration_v5_frozen_review.json'
assert harness['prior_review_sha256'] == hashlib.sha256(prior.read_bytes()).hexdigest()

source_checks = [
    dict(name='unchanged_GT_sampler_masks_galleries_and_metric_schema', status='PASS', mode='source_review', detail='Exact leaf equality after the two authorized substitutions preserves V5-reviewed installed training-only identity-heldout split, deterministic evaluation loader with batch64/no shuffle/no dropped records, ordered query/ID/camera/scene/name checks, MSVR same-ID/same-scene exclusion, six normalized-zero masks and clean+6both+6query-only conditions. Both fixed base00 and full11 positive/negative reference protocols, six retrieval metrics, CMC1..50, per-query/group CSV/JSON and closed four-state features are unchanged. No historical DeMo or V5 model is scheduled.'),
    dict(name='frozen_checkpoint_and_independent_controlled_states', status='PASS', mode='source_review', detail='The projected builder has the same arguments/configuration and returns the V6 class in FP32 CUDA. Its inherited controlled_states/fuse retain independent M-only relation mass, zero disabled expert inputs and no F condition/full-state mass/joint psi in state10. Both leaves strictly load only dev-selected best.pth, set eval/no_grad, compare state versions, require exact feature error0 and use saved normal distances only after that parity gate. Other states/missing distances are unchanged; the1e-8 normal metric gate remains. Neither leaf calls optimizer/backward/checkpoint reselection/official test.'),
    dict(name='future_AMPfix_receipt_and_resource_isolation', status='PASS', mode='source_review', detail='Deployment explicitly reads axis_collaboration_v6_projected_ampfix_launch.json and derives the MSVR run from that output, so the preserved failed V6 trial is not substituted. Helper requires that fresh run COMPLETE/50/outer exit0, all22 prior sources plus missing_evaluation, four frozen input proofs, GPU3 memory<500MiB and free disk>200000000 bytes before upload; it rechecks source/input/GPU/disk before one spawn. Only three new leaves are uploaded. Existing HOSTS2026 interpreter/project paths and quoted remote_python transport are reused; no old-source writes, delete, environment change or process termination is added. The reviewed GPU3 launcher retains240second idle waiting.'),
    dict(name='observer_readonly_scope_and_completion_meaning', status='PASS', mode='source_review', detail='Observer binds every launch-receipt source SHA before reading status, inspects only the four MSVR stages, and uses ps only to observe the controller. Raw stage JSON/CSV/log bytes carry size/SHA and are copied once under per-stage intake manifests; NPZ/PTH stay remote. It preserves primary failure logs/snapshots before asserting on failed child or dead incomplete controller. The final console states13 missing and4 four-state conditions and occurs only after controller COMPLETE. This observer and review do not certify a new training or GPU forward success.')
]
checks = harness['checks'] + source_checks
report = dict(
    status='PASS', blockers=[], non_blocking_findings=[],
    observed_at=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
    helper_sha256=harness['helper_sha256'], observer_sha256=harness['observer_sha256'],
    checked_source_sha256=harness['checked_source_sha256'], checks_count=len(checks),
    deterministic_groups=len(harness['checks']), source_review_groups=len(source_checks),
    mock_cases=sum(len(check.get('cases', [])) for check in harness['checks']), checks=checks,
    reviewer_agent='/root/review_v5_frozen_eval', requested_reviewer_model='gpt-6-astra', requested_reasoning_effort='max',
    fork_turns='none at original reviewer creation; V6 follow-up reuses this independent reviewer', provider_internal_identity_inferred=False,
    review_independence='same-family', acceptance_status='provisional',
    scope='Only V6 three frozen-evaluation leaves, private deploy helper and private observer; exact comparison to previously accepted V5 leaves and narrow dependency/API verification.',
    prior_review_path='results/preflight/axis_collaboration_v5_frozen_review.json', prior_review_sha256=harness['prior_review_sha256'],
    prepared_dependency_source_count=22, prepared_dependency_sha256=harness['prepared_dependency_sha256'],
    dependency_binding_basis='Prior failed V6 launch source names, replacing only projected_mass_axis_collaboration.py with current authorized two-cast source hash; mock fixture, not an actual AMPfix launch receipt.',
    ampfix_launch_present_at_review_start=harness['ampfix_launch_present_at_review_start'],
    runtime_readiness_assessed=False, real_remote_calls=0, real_process_spawns=0, torch_imports=0, gpu_forwards=0, optimizer_updates=0,
    source_or_handoff_modified=False, prior_V5_review_modified=False, source_binding_rechecked_at_finalization=True,
    harness_attempts=2,
    harness_failures_preserved=[dict(attempt=1, kind='reviewer_fixture_only', cause='Windows WinError206: generated nested fixture directory exceeded local path length limit.', stdout='001-harness.stdout.log', stderr='001-harness.stderr.log', source='001-narrow-harness.py', correction='Only the reviewer temporary fixture root was shortened to C:/Users/gb/.codex_tmp/v6f_*; production files unchanged.')],
    harness_status='Second harness execution exited0 after the reviewer fixture path correction; all15 deterministic groups and48 mock cases passed.',
    trace_path='.aris/traces/experiment-bridge/2026-10-03_axis_v6_projected_frozen',
    limitations=[
        'This is a code/delta/AST/mock PASS. It is not AMP repair validation, actual frozen-feature/metric parity, GPU resource measurement, fresh50 completion or NN success.',
        'The failed V6 AMP attempt remains historical failed evidence. This review does not alter or promote it and does not bypass the future AMPfix launch receipt or COMPLETE50/exit0 gates.',
        'Mock checkpoints, arrays, results, source fixtures, disk and GPU readings test control flow only. No Torch import, model instantiation, dataset inference, SSH, upload or real subprocess child occurred.',
        'The prior V5 source/report is unchanged. No broader training/model/results re-review or additional experiment was performed.',
        'Fresh deployment must still pass the existing runtime source/input/GPU/disk and smoke/full checks. Frozen single-seed interventions remain diagnostics, not independent trained ablations or official-test acceptance.'
    ])
raw = (json.dumps(report, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
(TRACE / 'review-result.json').write_bytes(raw)
target = ROOT / 'results/preflight/axis_collaboration_v6_projected_frozen_review.json'
assert not target.exists()
target.write_bytes(raw)
(TRACE / 'run.meta.json').write_text(json.dumps(dict(status='PASS', checks_count=len(checks), mock_cases=report['mock_cases'], helper_sha256=report['helper_sha256'], observer_sha256=report['observer_sha256'], checked_source_sha256=report['checked_source_sha256'], harness_attempts=2, real_remote_calls=0, torch_imports=0), indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=report['status'], checks_count=len(checks), mock_cases=report['mock_cases'], helper_sha256=report['helper_sha256'], observer_sha256=report['observer_sha256'], checked_source_sha256=report['checked_source_sha256'], report=str(target))))
