"""Publish the completed incremental review, keeping the previous41-check report."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_axis_v6_projected_aux'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_v6_projected26_compact_deploy_20261003.py')
prior = json.loads((TRACE / 'previous_host27_review_PASS41.json').read_text(encoding='utf-8'))
revision = json.loads((TRACE / 'compact26_revision_checks.json').read_text(encoding='utf-8'))
assert prior['status'] == 'PASS' and prior['checks_count'] == 41
assert revision['status'] == 'PASS_SOURCE_AND_STDLIB_MOCKS' and revision['checks_count'] == 20
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == revision['helper_sha256']
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value for name, value in revision['checked_source_sha256'].items())
now = datetime.now(timezone.utc).isoformat()
result = dict(prior)
result.update({
    'status': 'PASS', 'observed_at': now, 'blockers': [], 'non_blocking_findings': [],
    'checks_count': 61, 'current_revision_checks_count': 20, 'prior_review_checks_count': 41,
    'checks_count_semantics': 'Cumulative passed source/mock review checks:41 archived prior checks plus20 narrow compact26 checks. Historical2027 resource checks remain history, not the active deployment authorization.',
    'checked_source_sha256': revision['checked_source_sha256'],
    'helper_path': str(HELPER), 'helper_sha256': revision['helper_sha256'],
    'scope': 'One fresh MSVR310 seed42 B64 50-epoch projected auxiliary trial on explicitly selected validated2026GPU1, only after actual tensor and native3AMP gates PASS. No additional dataset, old experiment rerun, checkpoint deletion or environment installation.',
    'revision_route': 'Native follow-up to the same original fresh reviewer agent; requested gpt-6-astra/max, same-family/provisional; serving identity not independently verified.',
    'previous_review_archive': str(TRACE / 'previous_host27_review_PASS41.json'),
    'current_revision': 'Post50 terminal last checkpoint omits only optimizer state; explicit storage metadata and return toGPU1. Model and tensor verifier bytes unchanged.',
    'reviewed_terminal_storage': {
        'optimizer_omitted_only_after_full50': True,
        'retained': ['model', 'scheduler', 'scaler', 'epoch', 'steps', 'optimizer_steps'],
        'last_optimizer_resume_supported': False,
        'native_smoke_full_model_and_optimizer_restore_unchanged': True,
        'training_evaluation_best_selection_unchanged': True,
        'terminal_COMPLETE_after_compact_save_and_best_reload_mocked': True
    }
})
result['conclusions'] = prior['conclusions'][:2] + [
    'Reversing only the new terminal-storage metadata line and the removed optimizer entry in last.pth serialization recovers the exact previous runner bytes. The entire50-epoch training/evaluation/best-selection loop and full nativeAMP optimizer smoke AST remain identical to V5.',
    'The compact last save occurs only after range(1,51) finishes. It retains model, scheduler, scaler, epoch50 and step counters. The subsequent strict selected-best reload, fixed FP32 dev equality check and both COMPLETE records remain unchanged; actual post50 statements passed a stdlib sentinel execution.',
] + prior['conclusions'][3:6] + [
    'Launcher bytes exactly match the originally reviewed physicalGPU1 controller, so prior success/failure engineering-gate checks remain applicable. The new helper uses the validated2026 interpreter/cwd/dataset/publicCLIP, uploads exactly4 absent new sources, and protects all18 existing source files.',
    'The former2027GPU0 attempt was blocked by the actual initial read-only guard before SCP:1304MiB occupied, all4 new sources absent and output absent. Primary failure is retained. Explicit compact26 selection does not add automatic fallback or preempt any other task.',
    'Actual V5 controller and all3 run results are COMPLETE/50/exit0; unchanged V5 code saves each last before terminal COMPLETE. Thus no pending V5 model+Adam terminal allocation remains. Measured full smoke1,196,948,078B plus two model-state estimates399,087,701B each gives1,995,123,480B; reserve281,600,000B for pending100 four-state raw features and323,276,520B for remaining outputs at the2.6GB gate.',
    'Before upload and before starting the sole controller, the new helper checks currentGPU1 memory<500MiB and disk free>2.6GB. Both stages reject busyGPU or insufficient disk in mocks, including disk loss after upload. No current remote resource claim is made by the reviewer.'
]
result['resolved_blockers'] = [{
    'id': 'initial_host26_checkpoint_disk_budget',
    'status': 'RESOLVED_BY_EXPLICIT_COMPACT_TERMINAL_STORAGE_AND_COMPLETED_V5_LAST_FILES',
    'original_issue': prior['resolved_blockers'][0]['issue'],
    'initial_failure_preserved': 'initial_resource_blocker.json',
    'intermediate_host27_review_preserved': 'previous_host27_review_PASS41.json',
    'actual_host27_resource_rejection': 'results/preflight/axis_collaboration_v6_guard27_primary.json',
    'final_resolution': 'Return explicitly to validated2026GPU1 after V5three training terminals completed. Omit only duplicated Adam state from the new post50 terminal checkpoint, keep full optimizer smoke and all training untouched, document no optimizer resume from last, budget2.6GB with pending frozen-array and text/metadata margins, recheck current resources twice. No existing artifacts are deleted.'
}]
result['interpretation_limits'] = prior['interpretation_limits'] + [
    'The new last checkpoint is not an optimizer-resumable training checkpoint. Its storage change happens after completion and is explicitly documented; the fresh50 experiment never resumes from it.'
]
result['checks'] = [dict(row, review_revision='archived_prior41', scope_note='Prior source/gate checks are retained; host27 capacity/deployment checks are historical and superseded.') for row in prior['checks']]
result['checks'].extend(dict(row, review_revision='current_compact26') for row in revision['checks'])
assert len(result['checks']) == result['checks_count']
result['trace_artifacts'] += ['previous_host27_review_PASS41.json', '003-compact26.request.json',
                            'review_compact26_revision.py', 'compact26_revision_checks.json',
                            'compact26_stdout.txt', 'compact26_stderr.txt', 'final_compact26_helper_reviewed.py']
serialized = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
(ROOT / 'results/preflight/axis_collaboration_v6_projected_review.json').write_text(serialized, encoding='utf-8')
(TRACE / 'review.result.json').write_text(serialized, encoding='utf-8')
(TRACE / '003-compact26.response.json').write_text(serialized, encoding='utf-8')
(TRACE / '003-compact26.meta.json').write_text(json.dumps({
    'call_number': 3, 'purpose': 'Post50 compact terminal and explicit26 resource revision',
    'finished_at': now, 'task_name': '/root/review_v6_projected_aux',
    'requested_model': 'gpt-6-astra', 'requested_reasoning_effort': 'max',
    'review_independence': 'same-family', 'acceptance_status': 'provisional',
    'status': 'PASS', 'current_revision_checks_count': 20, 'cumulative_checks_count': 61
}, indent=2) + '\n', encoding='utf-8')
(TRACE / 'final_compact26_helper_reviewed.py').write_bytes(HELPER.read_bytes())
print(json.dumps({key: result[key] for key in ('status', 'checks_count', 'current_revision_checks_count', 'helper_sha256', 'checked_source_sha256')}))
