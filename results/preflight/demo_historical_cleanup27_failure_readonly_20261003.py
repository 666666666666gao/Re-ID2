import ast
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS
HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
plan=json.loads((PROJECT/'results/preflight/historical_nonbest_weight_cleanup_plan.json').read_text())
tree=ast.parse(Path('C:/Users/gb/.codex_tmp/demo_historical_nonbest_execute_20261003.py').read_text())
loop=next(node for node in tree.body if isinstance(node,ast.For))
code=eval(compile(ast.Expression(body=loop.body[0].value),'<guard>','eval'),{'inventory':plan['hosts']['2027']})
code=code[:code.index("for row in inventory['candidates']:Path(row['path']).unlink()")]
code+="\nprint('ALL_PREDELETION_GUARDS_PASS')"
process=subprocess.run(['ssh',*OPTIONS,'2027',shlex.quote(HOSTS['2027'][1])+' -c '+shlex.quote(code)],capture_output=True,text=True,encoding='utf-8',timeout=60)
record=dict(returncode=process.returncode,stdout=process.stdout,stderr=process.stderr,scope='Exact prior predeletion guards only; no unlink or file mutation')
target=PROJECT/'results/preflight/historical_nonbest27_failure_diagnostic.json';assert not target.exists();target.write_bytes(json.dumps(record,indent=2).encode('utf-8'))
print(json.dumps(record))
