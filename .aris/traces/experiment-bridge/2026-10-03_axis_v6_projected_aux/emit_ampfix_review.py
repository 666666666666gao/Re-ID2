"""Emit a separate AMP-fix review; preserve the original61-check report."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_axis_v6_projected_aux'
checks = json.loads((TRACE / 'ampfix_checks.json').read_text(encoding='utf-8'))
assert checks['status'] == 'PASS_SOURCE_AND_STDLIB_MOCKS' and checks['checks_count'] == 24
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value for name, value in checks['checked_source_sha256'].items())
assert hashlib.sha256(Path(checks['helper_path']).read_bytes()).hexdigest() == checks['helper_sha256']
assert hashlib.sha256(Path(checks['observer_path']).read_bytes()).hexdigest() == checks['observer_sha256']
old_review = ROOT / 'results/preflight/axis_collaboration_v6_projected_review.json'
old_bytes = old_review.read_bytes()
assert old_bytes == (TRACE / 'review61_before_actual_amp.json').read_bytes()
now = datetime.now(timezone.utc).isoformat()
result = {
    'status': 'PASS',
    'review_kind': 'NARROW_SOURCE_AND_STDLIB_MOCKS_ONLY',
    'observed_at': now,
    'blockers': [],
    'non_blocking_findings': [],
    'checks_count': 24,
    'checks_scope': 'New necessary checks only; archived61 checks were not rerun or represented as actual AMP acceptance.',
    'checked_source_sha256': checks['checked_source_sha256'],
    'helper_path': checks['helper_path'],
    'helper_sha256': checks['helper_sha256'],
    'observer_path': checks['observer_path'],
    'observer_sha256': checks['observer_sha256'],
    'reviewer_task': '/root/review_v6_projected_aux',
    'requested_initial_route': {'tool': 'collaboration.spawn_agent', 'fork_turns': 'none', 'model': 'gpt-6-astra', 'reasoning_effort': 'max'},
    'current_invocation': 'Native followup_task to the same reviewer after the captured actual AMP failure.',
    'serving_identity': 'Requested route attribution only; provider execution identity not independently introspected.',
    'review_independence': 'same-family',
    'acceptance_status': 'provisional',
    'scope': 'Only two auxiliary FP32 cast positions, single-file model replacement after failed workers stop, distinct ampfix output and observer namespaces. Unchanged runner/verifier/controller and originalV5 sources are reused.',
    'actual_prior_failure': {
        'status': 'RETAINED_ACTUAL_TENSOR_FAILURE',
        'controller_pid': 3100567,
        'tensor_pid': 3100699,
        'tensor_exit_code': 1,
        'cause': 'With autocast disabled, old code called modality_expert.pool(m0, eligible) while m0 was Half and pool_score Linear weights were FP32. The .float() after pooling could not execute before that Linear failed. The frequency pool had the analogous input-dtype requirement.',
        'primary_log': 'results/axis_collaboration_v6_projected_mass_trial/preflight/MSVR310_tensor.log',
        'snapshot': 'results/preflight/axis_collaboration_v6_projected_latest_snapshot.json',
        'optimizer_updates': 0,
        'smoke_or_fresh50_launched': False,
        'old61_report_preserved': True,
        'failed_four_sources_preserved': '.aris/traces/experiment-bridge/2026-10-03_axis_v6_projected_aux/failed_amp_source',
        'reviewer_learning': 'The preceding static review missed the pool-internal Linear dtype requirement. Its61-check PASS was source/mock evidence only and did not establish actual AMP correctness. This review explicitly inspects and exercises the pool call order with tagged values.'
    },
    'conclusions': [
        'Byte reversal proves exactly two expression changes: m0.float() and f0.float() now execute before the respective independent auxiliary pools, inside the existing torch.autocast(cuda, enabled=False) block. The runner/verifier build the model in FP32, so both pool_score Linear inputs now match their FP32 weights.',
        'Actual pool/tap AST executed with standard-library tagged values reproduces both old late-cast failures before .float(), then confirms new float conversion precedes each pool_score Linear and PM/PF projection. The M relation normalization/mean and F normalization remain unchanged. This is an operation-order witness, not a numerical Torch or AMP test.',
        'No detach, new parameter, head, loss, reconstruction term, gate or route is introduced. Reversing the two cast moves recovers the exact failed model bytes. Evaluation returns fused before the auxiliary block; all retrieval/four-state calculations remain unchanged.',
        'Runner, verifier and controller bytes match the archived61-check revision. Existing actual tensor and native3AMP gates remain mandatory; failed tensor execution still prevents smoke and fresh50. Compact post50 terminal storage, full optimizer smoke, fixed50 seed42/B64 sampling and dev-only earliest-best selection are unchanged.',
        'The helper matches the actual failed22-source launch receipt and both recorded PIDs before mutation, checks they are no longer present, requires a distinct absent ampfix output, GPU1 memory<500MiB and disk free>2.6GB, then uploads only projected_mass_axis_collaboration.py. After upload it checks the new22-source state and currentGPU/disk again before starting one controller with the existing2026 cwd/interpreter/data/publicCLIP.',
        'Mocked success uploads one model file and starts one new controller; either oldPID alive, busyGPU or insufficient disk prevents the dependent operation. GPU/disk failure after upload prevents launch. The old failure directory marker and other21 source files remain intact in these simulations.',
        'The observer changes only its ampfix launch receipt, local output root and latest-snapshot namespace. Actual collect() and generated remote read code pass synthetic failure and completion scenarios: raw failure log/intake is saved before asserting, completed intake can be collected again without duplicate payload writes, and old failed output is untouched. The observer does not launch/retry jobs and retains the240-second floor/estimated-milestone cadence.'
    ],
    'runtime_acceptance': {
        'corrected_actual_AMP_tensor_gate': 'PENDING_ROOT_EXECUTION; not established by this review',
        'corrected_actual_native3AMP_smoke': 'PENDING_AFTER_TENSOR_PASS; not established by this review',
        'corrected_fresh50': 'MUST_WAIT_FOR_BOTH_REAL_GATES',
        'real_SSH_calls_by_reviewer': 0,
        'real_GPU_calls_by_reviewer': 0,
        'NN_model_imports_by_reviewer': 0,
        'optimizer_updates_by_reviewer': 0,
        'official_test_uses_by_reviewer': 0
    },
    'interpretation_limits': [
        'Source and tagged-dtype/mock PASS authorizes only the corrected real preflight workflow; it does not assert actual AMP success, numeric equivalence, nonzero Torch gradients or efficacy.',
        'The auxiliary supervision remains a changed optimization protocol relative to old controls; no matched-V6 superiority or mechanism claim follows from this repair.',
        'The old failed trial is retained and the new ampfix trial is a distinct output. No old failure, review or launch receipt is replaced.'
    ],
    'review_harness_note': {
        'initial_harness_issue': 'Observer synthetic files under the deep trace directory exceeded the local Windows path limit before collect() ran.',
        'resolution': 'Shorten only the owned test temporary root to C:/Users/gb/.codex_tmp; no production-source or observer change.',
        'first_stdout': 'ampfix_first_harness_stdout.txt', 'first_stderr': 'ampfix_first_harness_stderr.txt',
        'final_stdout': 'ampfix_stdout.txt', 'final_stderr': 'ampfix_stderr.txt'
    },
    'checks': checks['checks'],
    'trace': str(TRACE),
    'trace_artifacts': ['review_ampfix.py', 'ampfix_checks.json', 'ampfix_stdout.txt', 'ampfix_stderr.txt',
                       'ampfix_first_harness_stdout.txt', 'ampfix_first_harness_stderr.txt',
                       'ampfix_generated_remote_guard.py', 'ampfix_generated_remote_start.py',
                       '004-ampfix.request.json', '004-ampfix.response.json', '004-ampfix.meta.json']
}
serialized = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
(ROOT / 'results/preflight/axis_collaboration_v6_projected_ampfix_review.json').write_text(serialized, encoding='utf-8')
(TRACE / 'ampfix.review.result.json').write_text(serialized, encoding='utf-8')
(TRACE / '004-ampfix.response.json').write_text(serialized, encoding='utf-8')
(TRACE / '004-ampfix.meta.json').write_text(json.dumps({
    'call_number': 4, 'purpose': 'Captured actual AMP failure: two cast positions and isolated retry workflow',
    'finished_at': now, 'task_name': '/root/review_v6_projected_aux',
    'requested_model': 'gpt-6-astra', 'requested_reasoning_effort': 'max',
    'review_independence': 'same-family', 'acceptance_status': 'provisional',
    'status': 'PASS', 'checks_count': 24, 'actual_corrected_AMP': 'PENDING'
}, indent=2) + '\n', encoding='utf-8')
assert old_review.read_bytes() == old_bytes
print(json.dumps({key: result[key] for key in ('status', 'checks_count', 'helper_sha256', 'observer_sha256', 'checked_source_sha256')}))
