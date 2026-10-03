"""Archive the completed narrow review; changes only reviewer artifacts."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

root = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
trace = root / '.aris/traces/experiment-bridge/2026-10-03_axis_metric_frozen'
tmp = Path('C:/Users/gb/.codex_tmp')
now = datetime.now().astimezone().isoformat(timespec='seconds')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
checks = json.loads((trace / '001-check-results.json').read_text())
assert checks['status'] == 'PASS' and not checks['torch_imported']
assert all(sha(root / name) == value for name, value in checks['checked_source_sha256'].items())
for key, name in [('helper_sha256', 'demo_axis_metric_frozen26_deploy_20261003.py'), ('observer_sha256', 'demo_axis_metric_frozen26_observe_20261003.py'), ('cpu_helper_sha256', 'demo_axis_metric_complete_metrics_20261003.py')]:
    assert sha(tmp / name) == checks[key]
launch_path = root / 'results/preflight/axis_metric_interface_launch.json'
launch = json.loads(launch_path.read_text())
assert all(sha(root / name) == value for name, value in launch['source_sha256'].items())
binding = dict(launch['source_sha256'])
binding['missing_evaluation.py'] = sha(root / 'missing_evaluation.py')
binding.update(checks['checked_source_sha256'])
response = ('PASS, no blockers. The three evaluation leaves and private deployment/observer helpers are exact '
            'V6 derivatives after the enumerated import, variant, artifact path and wording replacements. '
            'All 30 stdlib delta/syntax/builder/orchestration checks passed without Torch, NN, SSH, GPU or '
            'production process execution. Existing runtime COMPLETE50/exit0, strict reload, exact feature '
            'parity, unchanged input/source/state and GPU3 idle gates remain mandatory. '
            'The actual training receipt currently binds 37 unchanged sources; training completion and '
            'frozen numerical results are not certified by this review. Same-family, provisional.')
report = {
    'status': 'PASS', 'blockers': [], 'non_blocking_findings': [], 'reviewed_at': now,
    'scope': 'Narrow CODE_REVIEW of three metric frozen-evaluation leaves and private deployment/observer helpers; CPU normal-metrics helper inspected. Training mechanism excluded except builder/configuration/state API compatibility.',
    'reviewer_agent': '/root/review_axis_metric_frozen_eval', 'model': 'gpt-6-astra',
    'reasoning_effort': 'max', 'reviewer_model_attribution': 'Parent task explicitly confirms the actual gpt-6-astra/max spawn; provider internal identity not independently inferred.',
    'review_independence': 'same-family', 'acceptance_status': 'provisional',
    'checked_source_sha256': checks['checked_source_sha256'],
    'helper_sha256': checks['helper_sha256'], 'observer_sha256': checks['observer_sha256'],
    'cpu_helper_sha256': checks['cpu_helper_sha256'],
    'prior_review': 'results/preflight/axis_collaboration_v6_projected_frozen_review.json',
    'prior_review_sha256': checks['prior_review_sha256'],
    'source_delta': 'The prior PASS-bound V6 leaves and both original private helpers still match that review. Exact normalized textual equality holds for all five new files. Only builder imports, variant/run names, receipt/output paths and wording differ; no masks, GT, split, distance, checkpoint selection, state freezing, four-state or13-condition code changes.',
    'checks_count': len(checks['checks']), 'checks': checks['checks'],
    'source_review': [
        'Both evaluation leaves require parent result COMPLETE/50 and outer exit0, invoke the metric builder, strictly reload best.pth, set eval, and verify state tensor versions. Four-state extract_states and missing extract_missing run under no_grad; no optimizer/backward or official-test entry point is called.',
        'Installed training-only identity-heldout development records and saved query/ID/camera/scene/name order are compared exactly. MSVR310 uses same-ID/same-scene junk exclusion. Saved normal distances are used only after exact normal feature parity; the first64 smoke keeps the saved batch shape. Other states and missing conditions use the unchanged distance function.',
        'The metric class directly subclasses V5 MassAxisCollaborationDeMo, overrides only initialization and fuse, and retains inherited controlled_states. State10 sends zero F input to calibration/fuse, derives M-only relation mass independently and has no conditional message, joint psi or interaction. The metric builder returns that class in FP32/CUDA and uses the same configuration/evaluation function AST as V5; no runtime NN claim is made.',
        'Four states00/10/01/11 and clean+six both-missing+six query-only-missing conditions retain the original masks/gallery semantics. full_metrics computes mAP/mINP/Rank1/5/10/20, CMC1..50, per-query CSV, and identity/camera/scene groups against installed GT with the original exclusions.',
        'The launcher runs four sequential GPU3 stages, preserves all four input file size/SHA proofs after each, accepts zero optimizer updates/unchanged state/zero parity, enforces4states and13conditions, and terminates on any child failure. The inherited GPU3 idle check waits240seconds while memory is at least500MiB; no other GPU or process is preempted.',
        'The deployment helper requires the PASS review and its own/three-leaf hashes, reads the actual metric training receipt, verifies all existing receipt sources locally/remotely, and requires parent COMPLETE50/exit0, four frozen inputs, GPU3 memory<500MiB and disk>200000000 bytes. It uploads only the three new leaves and rechecks sources/inputs/GPU/disk before one isolated controller spawn. No retry, deletion, environment change or old-source rewrite is present.',
        'The observer reads only the four MSVR stages, source-checks the launch binding, ingests original JSON/CSV/log bytes once with size/SHA, excludes checkpoint/NPZ binaries, saves primary failure/snapshot before failure assertions, and waits240seconds only while incomplete. Its final completion string requires controller COMPLETE.',
        'The optional normal CPU metrics helper hides CUDA, verifies actual terminal COMPLETE50/exit0 and installed saved-array GT/order, recomputes six metrics/CMC/groups from frozen distances, compares the original four metrics with best and strict_reload, and transfers exact text outputs. It does not run a model, perform selection, update weights or evaluate official test.'
    ],
    'training_receipt_at_finalization': {
        'path': str(launch_path.relative_to(root)).replace('\\', '/'),
        'sha256': sha(launch_path), 'status': launch['status'], 'source_count': len(launch['source_sha256']),
        'all_local_receipt_sources_equal': True, 'frozen_combined_source_count': len(binding),
        'output': launch['output'],
        'meaning': 'Actual launch receipt checked locally. Parent reports fresh50 has started; this review does not certify its future terminal result.'
    },
    'noNN': True, 'torch_imports': 0, 'gpu_forwards': 0, 'optimizer_updates': 0,
    'real_remote_calls': 0, 'real_production_process_spawns': 0, 'package_installs': 0,
    'production_sources_or_helpers_modified': False, 'source_binding_rechecked_at_finalization': True,
    'runtime_readiness_assessed': False,
    'limitations': [
        'Code/delta and stdlib orchestration PASS only. Mock files and child/observer records are reviewer fixtures, not model outputs or successful GPU execution.',
        'The future real frozen run must still satisfy COMPLETE50/exit0, strict checkpoint reload, first64 and full feature parity, source/input/state preservation, actual GPU3 capacity and complete four-state/13-condition outputs.',
        'No broader training mechanism review, checkpoint reselection, new trained ablation or official-test acceptance is implied.',
        'An initial bare python command failed before Python initialization because the shell alias lacks pyvenv.cfg. The check then used the documented existing uv interpreter in offline/no-project mode, with no install. The sole actual narrow-check execution passed30checks.'
    ],
    'trace_path': str(trace.relative_to(root)).replace('\\', '/'), 'response': response
}
prompt = '''按experiment-bridge CODE_REVIEW 对新metric接口的冻结评测3leaf/helper独立same-family/provisional窄审。C:/Users/gb/projects/demo_dual_axis_20261002 新diagnose_metric_mass_axis.py/missing_metric_mass_development.py/launch_metric_mass_frozen_evaluation.py；privatehelper C:/Users/gb/.codex_tmp/demo_axis_metric_frozen26_deploy_20261003.py、observer demo_axis_metric_frozen26_observe_20261003.py。另CPU正常六指标helper demo_axis_metric_complete_metrics_20261003.py可必要检查。三leaf全部从已审且实际完成V6的对应diagnose_projected_mass_axis.py/missing_projected_mass_development.py/launch_projected_mass_frozen_evaluation.py严格派生，仅builder import/run名称/variant axis_metric_fullref/文案替换，GT/split/masks/distance/source冻结/4states13missing完全未改。训练候选另一个freshagent审4sources+真实NN由root仅审查PASS后GPU1tensor→3AMP→fresh50；你不重审训练机制除构建函数指向必要检查。新model metric_mass_axis_collaboration.py复用V5，只F现有scalar logit(.05)，w=sigmoid(Fscalar)*Fgate，tail direction*stopped identitynorm*sqrt(w/(1-w));关闭F=V5原M-only无message/psi/I，保留fixed5632D。冻结评测要求parent真实COMPLETE50 exit0、strictreload、first64smoke原saved features0、4states、13conditions(clean+6bothmissing+6querymissing)六指标/CMC50/perquery/groups、四输入文件SHA/stateversions不变，零optimizer/test，无选点改动。GPU26仅3 idle保护others，原source绑定由待创建metric training launch。静态/stdlib小mock失败门即可，NO NN/SSH/GPU/install，别造巨型fake torch或新增fallback/hash机制/兼容。不要改实施源或helper，不新增人读MD。输出results/preflight/axis_metric_frozen_review.json statusPASS/FAIL/blockers/checked_source_sha256精确3leaf/helper_sha256/observer_sha256/可CPUhelper/noNN/model gpt-6-astra max（root实际spawn确认）/same-family provisional；trace .aris/traces/experiment-bridge/2026-10-03_axis_metric_frozen。blocker立即消息，预计3～5min足够即归档。'''
followup = '训练实际receipt现已存在 axis_metric_interface_launch.json：controller3255284/GPU1，tensor+3AMP smoke exit0，initialstate仅Fscalar不同、identity5120equal、metricweight error5.59e-9，533参数梯度皆有限非零，strictmodeloptimizer reloadtrue；已fresh50开始，尚未终态。可用receipt核对helper预备绑定，无NN/SSH。只新增事实不要求扩审，按必要检查完成即归档。'
metadata = {'skill': 'experiment-bridge', 'run_id': '2026-10-03_axis_metric_frozen', 'timestamp': now,
            'executor': 'codex', 'executor_model': 'gpt-6-astra', 'reasoning_effort': 'max', 'executor_family': 'openai',
            'agent_id': '/root/review_axis_metric_frozen_eval', 'review_independence': 'same-family',
            'acceptance_status': 'provisional', 'project_dir': str(root), 'status': 'ok',
            'response_format': 'JSON, per explicit no-new-human-MD scope'}
request = {'call_number': 1, 'purpose': 'metric-frozen-code-review', 'timestamp': now, 'tool': 'spawn_agent',
           'model': 'gpt-6-astra', 'reasoning_effort': 'max', 'prompt': prompt, 'followup_message': followup,
           'files_referenced': list(checks['checked_source_sha256']) + [str(tmp / n) for n in ('demo_axis_metric_frozen26_deploy_20261003.py', 'demo_axis_metric_frozen26_observe_20261003.py', 'demo_axis_metric_complete_metrics_20261003.py')]}
data = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
version = root / ('results/preflight/axis_metric_frozen_review_' + datetime.now().strftime('%Y%m%dT%H%M%S') + '.json')
version.write_bytes(data)
(root / 'results/preflight/axis_metric_frozen_review.json').write_bytes(data)
(trace / '001-code-review.response.json').write_bytes(data)
(trace / 'review-result.json').write_bytes(data)
for filename, value in [('run.meta.json', metadata), ('001-code-review.request.json', request), ('001-code-review.meta.json', metadata)]:
    (trace / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
events = root / '.aris/meta/events.jsonl'
events.parent.mkdir(parents=True, exist_ok=True)
with events.open('a', encoding='utf-8') as handle:
    handle.write(json.dumps({'event': 'review_trace', 'skill': 'experiment-bridge', 'purpose': 'metric-frozen-code-review', 'agent_id': metadata['agent_id'], 'trace_path': report['trace_path'], 'status': 'ok', 'verdict': 'PASS'}) + '\n')
print(json.dumps({'status': 'PASS', 'checks': report['checks_count'], 'report': 'results/preflight/axis_metric_frozen_review.json', 'helper_sha256': report['helper_sha256'], 'observer_sha256': report['observer_sha256'], 'cpu_helper_sha256': report['cpu_helper_sha256']}))
