import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r'C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = Path(__file__).resolve().parent
report = json.loads((TRACE / '001-deterministic-results.json').read_text(encoding='utf-8'))
narrow = json.loads((TRACE / '003-narrow-check-results.json').read_text(encoding='utf-8'))
assert narrow['status'] == 'PASS'
old_failures = [c for c in report['checks'] if not c['passed']]
assert len(old_failures) == 1 and old_failures[0]['name'] == narrow['corrected_check']['name']
report['checks'] = [narrow['corrected_check'] if c['name'] == narrow['corrected_check']['name'] else c for c in report['checks']]
report['checks'].append(narrow['additional_check'])
assert all(c['passed'] for c in report['checks'])
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
           for name, digest in report['checked_source_sha256'].items())
now = datetime.now(timezone.utc)
report.update(status='PASS', blockers=[], non_blocking_findings=[], checks_count=len(report['checks']),
              observed_at=now.isoformat(), gpu_forwards=0, optimizer_updates=0,
              source_binding_rechecked_at_finalization=True,
              substantive_verdict='The requested single-variable source intervention is implemented correctly; no source blocker was found.',
              review_scope='New mass model and copied runner only; future actual-GPU contract and launcher are outside this review.',
              initialization_status='Static same allocation/state/RNG path established; actual initial tensors remain unexecuted.',
              mass_gradient_status='Differentiable path and independent analytic/finite-difference agreement established; actual Torch autograd/AMP remains unexecuted.',
              original_full_harness_attempts=1, narrow_followup_checks=2,
              harness_failures_preserved=[
                  {'kind': 'interpreter_entry_before_script_execution', 'file': '000-python-entry.failure.json',
                   'cause': 'Default E:/Scripts/python.exe reports No pyvenv.cfg file.',
                   'resolution': 'Used an existing uv-managed Python 3.10 executable directly; no install or environment mutation.'},
                  {'kind': 'false_negative_review_check', 'files': ['001-source-algebra.py', '001-harness.stdout.log', '001-deterministic-results.json'],
                   'cause': 'String check expected for key, value while Python 3.10 ast.unparse emits for (key, value).',
                   'diagnosis': '002-target-harness-diagnosis.json',
                   'resolution': 'One structural AST correction in 003-target-and-output-normalization.py; original full harness was not rerun and source code was unchanged.'}])
report.pop('unexpected_harness_failures', None)
report.pop('harness_attempt', None)
output = ROOT / 'results/preflight/axis_collaboration_v5_mass_model_review.json'
output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

response = '''PASS，25 项本地源码/AST/标准库检查通过，blockers=[]（gpt-6-astra/max，same-family/provisional）。

确认 M/I 在归一化后乘关系 mass，M-only 独立重算且无 full-route/F/psi 泄漏，01/00 与共享 full11 停梯度目标保持原语义；runner 训练和评测 AST 等价。均匀 mass 代数退化 V4，a_S 对多有效关系下的 M/I 及最终归一化描述子存在有效梯度路径；单有效关系时零关系竞争梯度是预期行为。

结果：results/preflight/axis_collaboration_v5_mass_model_review.json。原 Python 入口失败和首次 harness 格式误判均保留于指定 trace；只做窄修正，未改源码或交接、未运行旧套件、GPU 或模型 forward。实际初始张量、Torch/AMP 与性能仍待独立 M0。
'''
(TRACE / '001-model-review.response.md').write_text(response, encoding='utf-8')
run_meta = json.loads((TRACE / 'run.meta.json').read_text(encoding='utf-8'))
meta = {'call_number': 1, 'purpose': 'axis-mass-v5-model-source-review', 'timestamp': now.isoformat(),
        'agent_id': '/root/review_axis_mass_v5_model', 'model': 'gpt-6-astra', 'reasoning_effort': 'max',
        'fork_turns': 'none', 'reviewer_family': 'openai', 'review_independence': 'same-family',
        'acceptance_status': 'provisional', 'status': 'ok', 'substantive_status': 'PASS',
        'checks_count': len(report['checks']),
        'duration_ms': round((now - datetime.fromisoformat(run_meta['started_at'])).total_seconds() * 1000),
        'duration_scope': 'Trace-persistence to finalization; earlier source reads preceded trace initialization.',
        'report_path': str(output.relative_to(ROOT)), 'full_response': '001-model-review.response.md'}
(TRACE / '001-model-review.meta.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
events = ROOT / '.aris/meta/events.jsonl'
events.parent.mkdir(parents=True, exist_ok=True)
with events.open('a', encoding='utf-8') as stream:
    stream.write(json.dumps({'event': 'review_trace', 'skill': 'experiment-bridge',
                             'purpose': 'axis-mass-v5-model-source-review', 'agent_id': '/root/review_axis_mass_v5_model',
                             'trace_path': str(TRACE.relative_to(ROOT)).replace('\\', '/') + '/',
                             'status': 'ok', 'substantive_status': 'PASS', 'timestamp': now.isoformat()}) + '\n')
print(json.dumps({'status': report['status'], 'checks_count': report['checks_count'], 'blockers': report['blockers'],
                  'report': str(output), 'sources_still_unchanged': True}))
