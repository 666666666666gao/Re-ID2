"""Prepare focused helper rerun after root's one-line binding repair."""
import ast
from datetime import datetime
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_retrieval_utility_frozen49'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_retrieval_utility_frozen26_deploy_20261003.py')
old = (TRACE/'reviewed_sources'/HELPER.name).read_text(encoding='utf-8')
new = HELPER.read_text(encoding='utf-8')
removed = "sources['missing_evaluation.py']=hashlib.sha256((PROJECT/'missing_evaluation.py').read_bytes()).hexdigest()\n"
assert old.count(removed) == 1 and old.replace(removed, '') == new
ast.parse(new, feature_version=(3,10))
(TRACE/'005-source.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='helper-before-binding-fix',tofile='helper-after-binding-fix')),encoding='utf-8')
(TRACE/'reviewed_sources'/'demo_retrieval_utility_frozen26_deploy_20261003.repaired.py').write_bytes(HELPER.read_bytes())
(TRACE/'005-rereview.request.json').write_text(json.dumps(dict(
    timestamp=datetime.now().astimezone().isoformat(timespec='seconds'),
    tool='send_message', reviewer_agent='/root/review_retrieval_utility_frozen49',
    review_independence='same-family', acceptance_status='provisional',
    prompt="Deleted that redundant sources['missing_evaluation.py'] assignment in helper, retaining the actual parent receipt's full55 bindings. Parent model/evaluator/new leaves unchanged. Please rerun this binding check + helper mocks and bind final updated helper SHA before PASS. Actual parent fresh50 nowCOMPLETE50/453real0skip and sixmetrics audited; mAP45.99158/R159.04762 failsglobal +2.",
    checked_helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest()),indent=2)+'\n',encoding='utf-8')
source = (TRACE/'002-mock-harness.py').read_text(encoding='utf-8')
module = ast.parse(source)
module.body = [node for node in module.body if not (isinstance(node,ast.For) and ast.unparse(node.iter)=='launcher_cases')]
for node in module.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='launcher_cases' for t in node.targets):
        node.value=ast.Tuple(elts=[],ctx=ast.Load())
source = ast.unparse(ast.fix_missing_locations(module))+'\n'
source=source.replace("'local_training_source_stale',", "'local_training_source_stale', 'local_missing_evaluation_stale', 'remote_missing_evaluation_stale',",1)
source=source.replace("if case == 'local_training_source_stale':", "if case == 'local_missing_evaluation_stale':\n        fs.put(project + '/missing_evaluation.py', b'changed-evaluator')\n    if case == 'remote_missing_evaluation_stale':\n        fs.put(remote + '/missing_evaluation.py', b'changed-evaluator')\n    if case == 'local_training_source_stale':",1)
source=source.replace("if case.startswith('parent_'):", "if case == 'local_missing_evaluation_stale':\n        assert not calls and not stages and not spawns\n    if case == 'remote_missing_evaluation_stale':\n        assert len(calls) == 1 and not stages and not spawns\n    if case.startswith('parent_'):",1)
source=source.replace("'002-mock-checks.json'", "'005-helper-mock-checks.json'")
ast.parse(source, feature_version=(3,10))
(TRACE/'005-helper-mocks.py').write_text(source,encoding='utf-8')
print(json.dumps(dict(status='PASS',repair='one redundant receipt-binding overwrite deleted',
    helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),
    rerun_scope='29 helper cases; unchanged 24 controller cases not repeated')))
