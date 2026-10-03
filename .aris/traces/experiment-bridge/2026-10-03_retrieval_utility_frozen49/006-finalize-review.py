"""Seal the exact reviewed inputs and JSON-only provisional verdict."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE=ROOT/'.aris/traces/experiment-bridge/2026-10-03_retrieval_utility_frozen49'
HELPER=Path('C:/Users/gb/.codex_tmp/demo_retrieval_utility_frozen26_deploy_20261003.py')
FILES=('diagnose_retrieval_utility_axis.py','missing_retrieval_utility_development.py','launch_retrieval_utility_frozen_evaluation.py')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


static=load(TRACE/'001-static-checks.json')
first_mocks=load(TRACE/'002-mock-checks.json')
flow=load(TRACE/'003-dataflow-checks.json')
proof=load(TRACE/'004-binding-proof.json')
fixed=load(TRACE/'005-helper-mock-checks.json')
for result in (static,first_mocks,flow,fixed): assert result['status']=='PASS'
assert proof['status']=='CONFIRMED_BLOCKER'
hashes={name:sha(ROOT/name) for name in FILES}
assert hashes==static['checked_source_sha256']==first_mocks['checked_source_sha256']==fixed['checked_source_sha256']
assert sha(HELPER)==fixed['helper_sha256']
assert len(static['current_training_source_sha256'])==55
assert all(sha(ROOT/name)==digest for name,digest in static['current_training_source_sha256'].items())
assert fixed['deployment_mock_cases']==29 and first_mocks['controller_mock_cases']==24
initial_checks=[check for check in static['checks'] if check['name']!='exact_parent_delta_'+HELPER.name]
controller_checks=[check for check in first_mocks['checks'] if check['name'].startswith('controller_')]
assert len(controller_checks)==24
assert len(fixed['checks'])==29
checks=initial_checks+controller_checks+fixed['checks']+flow['checks']
checks.append(dict(name='final_helper_delta_and_full55_receipt_binding',status='PASS',evidence={
    'exact3_leaves_changed_only_by_names':True,
    'helper_changes':'Name/output substitutions plus deletion of the one inherited redundant missing_evaluation.py hash overwrite.',
    'final_helper_sha256':sha(HELPER),'source55_equal_current_receipt':True,
    'repaired_local_changed_dependency_stops_before_remote':True,
    'repaired_remote_changed_dependency_stops_before_SCP':True,
    'traces':['004-binding-proof.json','005-source.diff','005-helper-mock-checks.json']}))
now=datetime.now().astimezone().isoformat(timespec='seconds')
parent_local=ROOT/'results/axis_collaboration_v10_retrieval_utility_trial/development/MSVR310_axis_retrieval_utility_fullref_s42'
parent=load(parent_local/'result.json')
parent_exit=load(parent_local.parent/(parent_local.name+'_exit.json'))
report=dict(
    status='PASS',
    reviewed_at=now,
    reviewer='fresh native Codex reviewer /root/review_retrieval_utility_frozen49',
    reviewer_model='gpt-6-astra',reasoning_effort='max',fork_turns='none',
    skill='experiment-bridge Phase2.5 CODE_REVIEW',
    review_independence='same-family',acceptance_status='provisional',
    scope='SOURCE_AST_STDLIB_MOCKS_ONLY',
    summary='The exact three leaves and repaired helper implement the intended frozen MSVR310 retrieval-utility four-state and 49-pair development diagnostics. No unresolved deployment blocker found.',
    blockers=[],
    non_blocking=[dict(severity='LOW',file='launch_retrieval_utility_frozen_evaluation.py',lines=[1,28],
        issue='Inherited module docstring and wait-log label still say P1-B/relation-frequency.',
        impact='Cosmetic only: dataset, variant, run folder, child script names and output paths select the retrieval-utility run correctly.')],
    resolved_blockers=[dict(file=str(HELPER),original_line=18,
        issue='Inherited reassignment replaced the parent receipt digest for missing_evaluation.py, one of the actual55 sources.',
        evidence='Original import-stripped local helper AST accepted an in-memory changed dependency; see004-binding-proof.json.',
        resolution='Root deleted the redundant assignment; the 55-source receipt map is now preserved unchanged.',
        verification='29 final helper mocks pass, including local and remote changed missing_evaluation.py cases stopping before staging.',
        original_helper_sha256=proof['original_helper_sha256'],final_helper_sha256=sha(HELPER))],
    checked_source_sha256=hashes,helper_sha256=sha(HELPER),
    parent_receipt='results/preflight/retrieval_utility_launch.json',
    parent_controller_pid=3340643,parent_source_count=55,final_bound_source_count=58,
    current_local_parent_artifacts=dict(status=parent['status'],epochs=parent['epochs'],
        variant=parent['arguments']['variant'],dataset=parent['arguments']['dataset'],exit_code=parent_exit['exit_code'],
        evidence_scope='Already-collected local JSON files only; reviewer made no remote observations.'),
    checks=checks,
    verification_counts=dict(static_and_dataflow_checks=len(initial_checks)+len(flow['checks'])+1,
        controller_mock_cases=24,final_helper_mock_cases=29,total_checks=len(checks)),
    protected_inputs=['best.pth','best_dev_arrays.npz','result.json','parent sibling _exit.json'],
    stage_order=['four_state_smoke','four_state_full','missing_smoke','missing_full'],
    grid=dict(cached_descriptor_banks=7,query_gallery_conditions=49,independent_distance_calls=48,
        saved_normal_distance_uses=1,saved_normal_requires_exact_feature_parity=True,
        per_query=True,metrics=['mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20'],
        CMC='1..50',groups=['camera','scene','identity']),
    runtime_validation=dict(actual_CUDA_smoke='NOT_RUN_BY_REVIEWER',actual_full_four_states='NOT_RUN_BY_REVIEWER',
        actual_full49='NOT_RUN_BY_REVIEWER',numeric_inference_parity='REMAINS_RUNTIME_GATE',
        no_frozen_runtime_or_efficacy_acceptance_inferred=True),
    actual_actions=dict(production_module_imports=0,torch_imports=0,numpy_imports=0,SSH_calls=0,GPU_queries=0,
        neural_forwards=0,optimizer_updates=0,installations=0,production_source_edits_by_reviewer=0,helper_entrypoint_runs=0),
    traces=TRACE.relative_to(ROOT).as_posix(),
    trace_notes=['The initial review harness adaptation targeted a nonexistent stale-source fixture; corrected only the harness and preserved its failed source/stdout/stderr.',
        'Mock helper execution used import-stripped AST with all filesystem, subprocess, remote and staging operations replaced in memory; the deployment helper entrypoint was never run.',
        'The inherited binding defect and root one-line repair are retained separately from the final PASS.',
        'No official-test, independently trained ablation, multiseed acceptance, or global metric improvement is claimed.'])
assert all(check['status']=='PASS' for check in checks)
assert parent['status']=='COMPLETE' and parent['epochs']==50 and parent_exit['exit_code']==0
assert parent['arguments']['variant']=='axis_retrieval_utility_fullref' and parent['arguments']['dataset']=='MSVR310'
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
payload=(json.dumps(report,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
response=TRACE/'006-code-review.response.json'
response.write_bytes(payload)
destination=ROOT/'results/preflight/retrieval_utility_frozen_review.json'
destination.write_bytes(payload)
assert response.read_bytes()==destination.read_bytes()
metadata=dict(call_number=1,purpose='frozen49-code-review-with-one-concrete-repair',timestamp=now,
    agent_id='/root/review_retrieval_utility_frozen49',model='gpt-6-astra',reasoning_effort='max',
    reviewer_family='openai',review_independence='same-family',acceptance_status='provisional',
    status='ok',verdict='PASS',response_path=response.name)
(TRACE/'006-code-review.meta.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
event=dict(event='review_trace',skill='experiment-bridge',purpose='retrieval_utility_frozen49',
    agent_id='/root/review_retrieval_utility_frozen49',trace_path=TRACE.relative_to(ROOT).as_posix(),
    status='ok',verdict='PASS',review_independence='same-family',acceptance_status='provisional')
(TRACE/'review-trace.event.json').write_text(json.dumps(event,indent=2)+'\n',encoding='utf-8')
meta=ROOT/'.aris/meta'
meta.mkdir(exist_ok=True)
with (meta/'events.jsonl').open('a',encoding='utf-8') as handle:
    handle.write(json.dumps(event)+'\n')
assert load(destination)['helper_sha256']==sha(HELPER)
print(json.dumps(dict(status='PASS',review_path=str(destination),helper_sha256=sha(HELPER),
    checked_source_sha256=hashes,total_checks=len(checks),resolved_blockers=1,
    review_independence='same-family',acceptance_status='provisional',actual_neural_forwards=0)))
