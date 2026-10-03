"""Write only the requested machine-readable review and trace metadata."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_axis_routed_metric_frozen'
TARGET = ROOT / 'results/preflight/axis_routed_metric_frozen_review.json'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_routed_metric_frozen26_deploy_20261003.py')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

checks = json.loads((TRACE / 'checks.json').read_text(encoding='utf-8'))
assert checks['status'] == 'PASS_STDLIB_SOURCE_AST_AND_MOCK_CHECKS'
assert len(checks['checks']) == 66 and all(item['status'] == 'PASS' for item in checks['checks'])
assert len(checks['checked_source_sha256']) == 3
assert all(sha(ROOT / name) == value for name, value in checks['checked_source_sha256'].items())
assert sha(HELPER) == checks['helper_sha256']
timestamp = datetime.now().astimezone().isoformat(timespec='seconds')
shared = dict(timestamp=timestamp, model='gpt-6-astra', reasoning_effort='max', reviewer_family='openai',
              agent_id='/root/review_routed_metric_frozen', review_independence='same-family', acceptance_status='provisional')
review = dict(
    status='PASS', blockers=[], non_blocking_issues=[], **shared,
    scope='Fresh secondary experiment-bridge Phase 2.5 review of three exact-derived frozen development evaluation leaves and their one-shot 2026 deployment helper; source/AST/stdlib in-memory mocks only.',
    checked_source_sha256=checks['checked_source_sha256'], helper_path=HELPER.as_posix(), helper_sha256=checks['helper_sha256'],
    reviewed_baseline_sha256={name: sha(ROOT / name) for name in ('diagnose_metric_mass_axis.py', 'missing_metric_mass_development.py', 'launch_metric_mass_frozen_evaluation.py')},
    supporting_receipts={name: dict(path=(ROOT / path).as_posix(), sha256=sha(ROOT / path)) for name, path in (
        ('training_review', 'results/preflight/axis_routed_metric_review.json'),
        ('training_launch', 'results/preflight/axis_routed_metric_launch.json'),
        ('baseline_frozen_review', 'results/preflight/axis_metric_frozen_review.json'),
        ('baseline_actual_frozen_audit', 'results/preflight/axis_metric_msvr_frozen_analysis.json'))},
    findings=[
        'The three leaves are byte-exact declared derivatives of the reviewed, actually completed MetricA frozen protocol: builder/unique variant, script and wait names, one descriptive docstring, and GPU3 to GPU1 only.',
        'Routed supervision changes training loss pairs after the evaluation return. Its evaluation-forward AST equals the MetricA inherited Mass forward; MetricA fuse and independent controlled_states remain inherited. New routed weights still require their own real smoke/parity checks.',
        'The helper binds its own bytes, exactly three reviewed leaves, and current local/remote training-receipt sources. Actual local training receipt names server2026, GPU1, controller3293229 and the unique routed training directory; all training source hashes still match.',
        'The helper rejects absent/nonzero train exit, non-COMPLETE result, a budget other than50, wrong variant or absent checkpoint before copying any new script. It checks GPU1 memory below500MiB and disk above200000000bytes, then rechecks staged source, all four immutable training inputs, capacity, and fresh output/log before one controller spawn.',
        'The controller waits240seconds for parent train exit, requires exit0/COMPLETE50 and the expected MSVR310 variant/dataset, then runs four_state_smoke, four_state_full, missing_smoke, missing_full sequentially on GPU1. It gates every stage on successful exit, zero optimizer updates, unchanged state tensor versions, exact saved-feature parity and unchanged hashes/sizes of best.pth, best_dev_arrays.npz, result.json and train exit receipt.',
        'Both leaves strictly reload the dev-selected best.pth, use the fixed installed training-data identity-heldout split, and compare saved query indices, identity, camera, scene and filename ordering with installed records. Evaluation uses real GT identity with same-ID/same-scene exclusion for MSVR310; no model prediction supplies truth and no official-test routine is called.',
        'Four-state diagnostics retain independent00/10/01 and full11, close disabled condition/psi paths, preserve base and full-reference contribution measurements with shared positive/negative indices across states, and save query-level harm/rescue. Missing diagnostics retain clean plus both/query-only versions of six masks:13 conditions. Both use all six retrieval metrics, CMC1..50 and per-query camera/scene/identity groups.',
    ],
    checks=checks['checks'], checks_passed=66,
    execution=dict(neural_execution=False, torch_imported=False, numpy_imported=False, ssh_calls=0, gpu_queries=0,
                   actual_experiment_launches=0, optimizer_updates=0, package_installations=0, implementation_edits=0,
                   actual_routed_training_completion='NOT_OBSERVED_BY_REVIEWER', actual_frozen_smoke='NOT_RUN',
                   actual_frozen_full='NOT_RUN', live_remote_capacity='NOT_OBSERVED_BY_REVIEWER'),
    next_action='Invoke this exact helper only after observing the original routed parent50 exit0/COMPLETE milestone. Its remote preflight independently rejects early invocation before staging. Then retain the fixed first64 smoke/full sequence and preserve any failure evidence.',
    limits=[
        'Independent context within the same model family; provisional source review, not cross-family acceptance.',
        'Source and mock checks do not establish live training completion, current GPU availability, numerical parity or new routed-weight retrieval benefit.',
        'Single seed42 identity-heldout development, frozen checkpoints and zero updates; neither independent trained ablation nor official-test or multi-seed acceptance.',
    ],
    trace_path=TRACE.as_posix(),
)
write(TARGET, review)
write(TRACE / '001-code-review.response.json', review)
write(TRACE / '001-code-review.request.json', dict(
    purpose='pre-deployment-code-review', requested_model='gpt-6-astra', requested_reasoning_effort='max',
    sources=list(checks['checked_source_sha256']), helper_path=HELPER.as_posix(),
    requirements='Review derivation, training receipt/source/path binding, exit0/COMPLETE50-before-launch gates, GPU1 isolation, immutable four inputs, strict dev-best selection, exact feature parity, 4 states/fullref contributions/harm-rescue, 13 missing conditions, true GT, six metrics/CMC/groups, no optimizer or official test.',
    restrictions='Standard library source/AST/mock only; no NN or torch import, SSH/GPU, installation, implementation edits or human Markdown.',
    review_independence='same-family', acceptance_status='provisional'))
write(TRACE / '001-code-review.meta.json', dict(call_number=1, purpose='pre-deployment-code-review',
    status='ok', verdict='PASS', response_format='JSON per explicit task constraint', **shared))
write(TRACE / 'run.meta.json', dict(skill='experiment-bridge', run_id=TRACE.name, executor='codex',
    project_dir=ROOT.as_posix(), status='ok', response_format='JSON, no new human Markdown', **shared))
write(TRACE / 'review-trace.event.json', dict(event='review_completed', verdict='PASS', checks_passed=66,
    artifact=TARGET.as_posix(), artifact_sha256=sha(TARGET), checks_sha256=sha(TRACE / 'checks.json'),
    reviewed_source_bytes_unchanged=True, **shared))
write(TRACE / 'environment_events.json', dict(events=[
    dict(event='initial_read_only_interpreter_lookup_failed', command='python -', message='No pyvenv.cfg file',
         implementation_executed=False, resolution='Used already-cached uv CPython3.13 with --offline --no-project; no installation.'),
    dict(event='stdlib_review_command_succeeded', exit_code=0,
         command='uv run --offline --no-project --python existing-uv-CPython3.13 python review_checks.py')]))
print(json.dumps(dict(status='PASS', checks=66, artifact=TARGET.as_posix(), artifact_sha256=sha(TARGET), helper_sha256=sha(HELPER))))
