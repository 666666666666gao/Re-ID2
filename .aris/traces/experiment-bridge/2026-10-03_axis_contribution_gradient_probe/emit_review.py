import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_axis_contribution_gradient_probe'
checks = json.loads((TRACE / 'checks.json').read_text(encoding='utf-8'))
assert checks['status'] == 'PASS_SOURCE_AND_STDLIB_MOCKS'
assert checks['checks_count'] == len(checks['checks']) == 26
assert all(item['status'] == 'PASS' for item in checks['checks'])
assert set(checks['checked_source_sha256']) == {
    'probe_contribution_gradients.py', 'launch_contribution_gradient_probe.py'
}
for name, expected in checks['checked_source_sha256'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
for kind in ('helper', 'collector'):
    assert hashlib.sha256(Path(checks[kind + '_path']).read_bytes()).hexdigest() == checks[kind + '_sha256'], kind
now = datetime.now(timezone.utc).isoformat()
route = {
    'requested_model': 'gpt-6-astra',
    'requested_reasoning_effort': 'max',
    'requested_fork_turns': 'none',
    'request_attribution': 'parent task route request',
    'review_independence': 'same-family',
    'acceptance_status': 'provisional',
    'serving_identity_verified': False,
    'actual_serving_identity': None,
    'identity_note': 'Requested route is recorded; actual supplier execution identity is not inferred.'
}
report = {
    'status': 'PASS',
    'blockers': [],
    'reviewed_at': now,
    'scope': checks['scope'],
    'helper_path': checks['helper_path'],
    'helper_sha256': checks['helper_sha256'],
    'checked_source_sha256': checks['checked_source_sha256'],
    'collector_path': checks['collector_path'],
    'collector_sha256': checks['collector_sha256'],
    'checks_count': checks['checks_count'],
    'checks': checks['checks'],
    'review_independence': 'same-family',
    'acceptance_status': 'provisional',
    'review_route': route,
    'runtime_validation': {
        'status': 'PENDING_ACTUAL_GRADIENT_DIAGNOSTIC',
        'NN_model_imports': 0,
        'real_SSH_calls': 0,
        'real_GPU_calls': 0,
        'real_optimizer_updates': 0,
        'claims_actual_autograd_pass': False
    },
    'conclusions': [
        'Task loss equals the original full weighted head-pair sum with final contribution loss excluded; C is the already weighted final returned scalar.',
        'The two predictor-only gradients share one seeded training forward and are unscaled once by512 before global norms and cosine.',
        'Strict existing dev-best loading and before/after proofs protect four original input files per case; temporary training BatchNorm state is discarded.',
        'Both child exit receipts are written before the success assertion; exact names, labels, cameras, scenes and image bytes must agree before COMPLETE.',
        'The deployment helper binds the two new leaves and26 existing source files, checks both GPUs before upload and launch, and uploads only the two new leaves.',
        'Collector static checks confirm raw evidence and intake persistence before failure assertions and240-second read-only polling.'
    ],
    'limitations': [
        'AST, source inspection and stdlib mocks do not establish runtime AMP or autograd success.',
        'The planned single identical training batch is a local gradient diagnostic, not convergence, generalization or causal evidence.',
        'No prior61/24 checks were rerun; no production source or human document was edited.'
    ],
    'trace_directory': str(TRACE)
}

def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

request = {
    'received_from': '/root',
    'task': 'Narrow experiment-bridge CODE_REVIEW of contribution-gradient diagnostic leaves, deployment helper and collector.',
    'source_files': list(checks['checked_source_sha256']),
    'requirements': [
        'Check original-step task loss withoutC, weightedC, gradient scale/norm/cosine, strict dev-best and identical seeded fit batch.',
        'Check no optimizer or checkpoint writes, child exit handling, source/input protection, two-GPU guards and bound deployment helper.',
        'Statically inspect collector evidence preservation and240-second polling.',
        'No SSH/GPU/NN execution, no production source or human document changes, no repeat of prior61/24 checks.'
    ],
    'route_request': route
}
write(TRACE / '001-review.request.json', request)
write(TRACE / '001-review.response.json', report)
write(TRACE / 'review.result.json', report)
write(ROOT / 'results/preflight/axis_contribution_gradient_probe_review.json', report)
write(TRACE / 'run.meta.json', {
    'skill': 'experiment-bridge', 'CODE_REVIEW': True,
    'review_route': route, 'finished_at': now,
    'review_script': 'review_probe.py',
    'checks_artifact': 'checks.json', 'stdout': 'stdout.txt', 'stderr': 'stderr.txt',
    'report': 'results/preflight/axis_contribution_gradient_probe_review.json',
    'status': 'PASS', 'checks_count': 26,
    'helper_sha256': checks['helper_sha256'],
    'checked_source_sha256': checks['checked_source_sha256'],
    'runtime_validation': report['runtime_validation']
})
print(json.dumps({
    'status': 'PASS', 'checks_count': 26,
    'report': 'results/preflight/axis_contribution_gradient_probe_review.json',
    'helper_sha256': checks['helper_sha256'],
    'checked_source_sha256': checks['checked_source_sha256'],
    'acceptance_status': 'provisional',
    'real_runtime_validation': 'PENDING'
}, indent=2))
