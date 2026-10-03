"""Emit the completed source/stdlib review; no experiment execution."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_axis_v6_projected_aux'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_v6_projected27_deploy_20261003.py')
initial = json.loads((TRACE / 'initial_checks.json').read_text(encoding='utf-8'))
revision = json.loads((TRACE / 'resource_revision_checks.json').read_text(encoding='utf-8'))
assert initial['status'] == revision['status'] == 'PASS_SOURCE_AND_STDLIB_MOCKS'
excluded = {'all_new_sources_and_private_helper_compile', 'controller_mock_all_pass'}
checks = [dict(row, verification='initial_source_or_gate_check_reused_after_final_byte_or_AST_equivalence')
          for row in initial['checks'] if row['check'] not in excluded and not row['check'].startswith('deployment_helper_mock_')]
checks.extend(dict(row, verification='final_host27_resource_revision') for row in revision['checks'])
assert all(row['status'] == 'PASS' for row in checks)
assert len(checks) == 41
sources = revision['checked_source_sha256']
assert set(sources) == {'projected_mass_axis_collaboration.py', 'run_projected_mass_experiment.py',
                        'verify_projected_axis_mass.py', 'launch_projected_mass_trial.py'}
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value for name, value in sources.items())
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == revision['helper_sha256']
now = datetime.now(timezone.utc).isoformat()
result = {
    'status': 'PASS',
    'review_kind': 'SOURCE_AND_STDLIB_MOCKS_ONLY',
    'observed_at': now,
    'blockers': [],
    'non_blocking_findings': [],
    'checks_count': len(checks),
    'checked_source_sha256': sources,
    'helper_path': str(HELPER),
    'helper_sha256': revision['helper_sha256'],
    'reviewer_task': '/root/review_v6_projected_aux',
    'requested_route': {'tool': 'collaboration.spawn_agent', 'fork_turns': 'none',
                        'model': 'gpt-6-astra', 'reasoning_effort': 'max'},
    'serving_identity': 'Requested routing parameters only; actual provider execution identity was not independently introspected.',
    'review_independence': 'same-family',
    'acceptance_status': 'provisional',
    'scope': 'One fresh MSVR310 seed42 B64 50-epoch auxiliary-tap trial on explicitly selected READY2027 GPU0, only after the real tensor and native3AMP smoke gates pass. No other dataset or old experiment rerun.',
    'conclusions': [
        'V6 inherits all V5 parameters/state initialization, projections, 512D auxiliary heads, 5632D retrieval descriptor and loss weights. Entire forward AST is unchanged except the two training auxiliary feature taps; evaluation returns before those taps.',
        'The M tap applies existing PM to independent Bx7x512 pooled evidence in FP32, normalizes each relation direction and averages seven relations into512D. The F tap applies existing PF to its independent512D evidence and normalizes in FP32. Gradients remain attached; no new loss, parameter, reconstruction term, gate or route was introduced.',
        'The full runner AST matches V5 after normalizing only the new class/import/variant and descriptive metadata. Configuration, sampler seeds, optimizer, nativeAMP, CE/Triplet weights, FP32 evaluation and fixed50 earliest-max-dev-mAP selection remain exact.',
        'Evaluation uses actual dataset identity/scene ground truth and fixed identity-disjoint development data. For MSVR, same-identity/same-scene matches are excluded. No official-test selection or execution is added.',
        'The retained real tensor contract checks initial state/capacity,8 dev triplets, six missing masks, closed-expert isolation, uniform-V4 equivalence and one B64 semantic forward with zero optimizer updates. The new auxiliary-CE-only autograd probe requires both final PM/PF weights to receive finite nonzero gradients.',
        'The native smoke requires three effective updates, finite nonzero gradients for every trainable tensor, strict model load, optimizer restoration and exactly equal reloaded evaluation features. The controller preserves logs and exit receipts and prevents training if either prior process or result gate fails.',
        'Final controller changes only three GPU literals to physicalGPU0. Deployment uses the exact READY2027 interpreter, project cwd, dataset root and publicCLIP path; uploads exactly four absent new sources and verifies all15 relevant existing dependencies without overwriting the ongoing2026 work.',
        'The new helper checks READY and exact reviewed bytes, then currentGPU0 memory<500MiB and free disk>3.2GB before upload and before start. Historical measured MSVR smoke/best/last total2,792,932,497B, leaving407,067,503B above those three files at this threshold on the separate host.'
    ],
    'resolved_blockers': [{
        'id': 'initial_host26_checkpoint_disk_budget',
        'initial_helper_sha256': '3c0630a1be8dca9da1f5a6400e632e3eb47cc175176f08389054da5d9fe8130f',
        'issue': 'The initial>2.8GB budget counted last checkpoints as model-only although the runner serializes full Adam state. Initial real26 free4,416,991,232B also had pendingV5 RGBNT100 last and frozen arrays to preserve.',
        'primary_evidence': 'results/preflight/axis_collaboration_v6_actual_disk_probe.json; run_projected_mass_experiment.py last serialization; observed V5 MSVR trainable99,669,638 and Adam config.',
        'resolution': 'Explicitly choose alreadyREADY2027 GPU0 and reviewed new helper27, use>3.2GB for this single trial without the2026 pending writes, correct the budget explanation, retain current-resource checks twice. No automatic fallback, deletion, environment mutation or old2026-source change.',
        'status': 'RESOLVED',
        'original_failure_preserved': '.aris/traces/experiment-bridge/2026-10-03_axis_v6_projected_aux/initial_resource_blocker.json'
    }],
    'runtime_acceptance': {
        'actual_V6_tensor_gate': 'NOT_RUN_BY_REVIEWER; mandatory before smoke/train',
        'actual_V6_native3AMP_smoke': 'NOT_RUN_BY_REVIEWER; mandatory before full50',
        'actual_V6_efficacy': 'NOT_RUN_BY_REVIEWER; no improvement or completed-training claim',
        'real_SSH_calls_by_reviewer': 0,
        'real_GPU_calls_by_reviewer': 0,
        'NN_model_imports_by_reviewer': 0,
        'optimizer_updates_by_reviewer': 0,
        'official_test_uses_by_reviewer': 0
    },
    'interpretation_limits': [
        'The auxiliary-CE gradient probe checks loss connectivity, not optimization benefit or mechanism efficacy.',
        'Moving the two auxiliary taps changes the optimization protocol. Existing V4 ordinary/frequency controls are contextual and cannot establish matched-V6 superiority; matched new controls remain necessary before that claim.',
        'V5 first-two-dataset evidence is already recorded and was not rerun. Weak frozen four-state deltas motivate this hypothesis but do not prove projection supervision is the cause.',
        'A source/mock PASS permits the controller to perform its real gates; it does not replace current capacity checks or assert that tensor/smoke/training has completed.'
    ],
    'checks': checks,
    'trace': str(TRACE),
    'trace_artifacts': ['review_static_mock.py', 'initial_checks.json', 'initial_helper_budget_2p8GB.py',
                       'initial_resource_blocker.json', 'first_check_stdout.txt', 'first_check_stderr.txt',
                       'review_resource_revision.py', 'resource_revision_checks.json',
                       'resource_revision_stdout.txt', 'resource_revision_stderr.txt',
                       'run.meta.json', '001-review.request.json', '002-resource-revision.request.json']
}
text = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
(ROOT / 'results/preflight/axis_collaboration_v6_projected_review.json').write_text(text, encoding='utf-8')
(TRACE / 'review.result.json').write_text(text, encoding='utf-8')
(TRACE / '001-review.response.json').write_text(text, encoding='utf-8')
(TRACE / '001-review.meta.json').write_text(json.dumps({
    'call_number': 1, 'purpose': 'V6 projected auxiliary review with evidenced resource correction',
    'finished_at': now, 'task_name': '/root/review_v6_projected_aux',
    'requested_model': 'gpt-6-astra', 'requested_reasoning_effort': 'max',
    'review_independence': 'same-family', 'acceptance_status': 'provisional',
    'status': 'PASS', 'checks_count': len(checks), 'response_format': 'JSON per task instruction'
}, indent=2) + '\n', encoding='utf-8')
(TRACE / 'final_helper27_reviewed.py').write_bytes(HELPER.read_bytes())
print(json.dumps({'status': result['status'], 'checks_count': len(checks),
                  'helper_sha256': result['helper_sha256'], 'checked_source_sha256': sources}))
