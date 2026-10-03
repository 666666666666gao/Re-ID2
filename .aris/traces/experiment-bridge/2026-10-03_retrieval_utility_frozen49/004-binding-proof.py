"""Isolated proof for the existing helper's parent-source binding check."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_retrieval_utility_frozen49'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_retrieval_utility_frozen26_deploy_20261003.py')
original = TRACE/'reviewed_sources'/HELPER.name
launch = json.loads((TRACE/'reviewed_sources/retrieval_utility_launch.json').read_text(encoding='utf-8'))
assert 'missing_evaluation.py' in launch['source_sha256']
files = ('missing_retrieval_utility_development.py','diagnose_retrieval_utility_axis.py','launch_retrieval_utility_frozen_evaluation.py')
hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
helper_hash = hashlib.sha256(original.read_bytes()).hexdigest()
storage = {str(ROOT/name): (ROOT/name).read_bytes() for name in set(launch['source_sha256']) | set(files)}
storage[str(HELPER)] = original.read_bytes()
storage[str(ROOT/'results/preflight/retrieval_utility_launch.json')] = json.dumps(launch).encode()
storage[str(ROOT/'results/preflight/retrieval_utility_frozen_review.json')] = json.dumps(dict(status='PASS',blockers=[],helper_sha256=helper_hash,checked_source_sha256=hashes)).encode()
changed_path = str(ROOT/'missing_evaluation.py')
storage[changed_path] += b'\n# Review-only in-memory changed-source fixture\n'


class MockPath:
    def __init__(self, path): self.path=str(path)
    def __truediv__(self, name): return MockPath(Path(self.path)/name)
    def read_bytes(self): return storage[self.path]
    def read_text(self, encoding='utf-8'): return self.read_bytes().decode(encoding)


def local_prefix(source):
    module = ast.parse(source, feature_version=(3,10))
    statements = []
    for node in module.body:
        if isinstance(node, (ast.Import,ast.ImportFrom)): continue
        if isinstance(node, ast.Expr) and ast.unparse(node).startswith('sys.path.insert('): continue
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0]) == '(root, python)': break
        statements.append(node)
    assert all('remote_python' not in ast.unparse(node) for node in statements)
    return compile(ast.fix_missing_locations(ast.Module(body=statements,type_ignores=[])),'local_binding_AST_prefix','exec')


namespace = dict(hashlib=hashlib,json=json,Path=MockPath,PROJECT=MockPath(ROOT),__file__=str(HELPER))
exec(local_prefix(original.read_text(encoding='utf-8')),namespace)
assert namespace['sources']['missing_evaluation.py'] != launch['source_sha256']['missing_evaluation.py']
result = dict(status='CONFIRMED_BLOCKER',file=str(HELPER),line=18,
    original_helper_sha256=helper_hash,
    parent_source_count=len(launch['source_sha256']),
    parent_missing_evaluation_sha256=launch['source_sha256']['missing_evaluation.py'],
    accepted_changed_fixture_sha256=namespace['sources']['missing_evaluation.py'],
    local_gate_accepted_changed_source=True, actual_remote_calls=0, actual_source_files_changed=0,
    required_fix='Delete redundant sources[missing_evaluation.py] reassignment; preserve the actual 55-source receipt map unchanged.',
    scope='Cooperating-operator source consistency invariant; no malicious actor or new digest design assumed.')
(TRACE/'004-binding-proof.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print(json.dumps(result))
