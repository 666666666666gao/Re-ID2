import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS

root,python=HOSTS['2026']
code=f'''import json,os,subprocess
from pathlib import Path
r=Path({root!r});c=r/'runs/identity_alignment_m2b_20261004'
p=subprocess.run([{python!r},'-c','from gpu_thermal_execute import telemetry'],text=True,capture_output=True)
print(json.dumps(dict(cwd=os.getcwd(),import_returncode=p.returncode,import_stdout=p.stdout,import_stderr=p.stderr,
 output_exists=c.exists(),root_log_exists=Path(str(c)+'.log').exists(),launch_exists=Path(str(c)+'_launch.json').exists())))
'''
result=subprocess.run(['ssh',*OPTIONS,'2026',shlex.quote(python)+' -c '+shlex.quote(code)],text=True,capture_output=True,encoding='utf-8',check=True)
record=json.loads(result.stdout)
target=PROJECT/'results/preflight/identity_alignment_m2b_deploy_failure.json';assert not target.exists()
record.update(status='CPU_GATE_IMPORT_FAILED_BEFORE_ANY_NEURAL_LAUNCH',
 failed_deployer_sha256=hashlib.sha256(Path('C:/Users/gb/.codex_tmp/demo_identity_alignment_deploy_20261004.py').read_bytes()).hexdigest(),
 completed_boundary='76sources+plan/review copied and checked on25/26/27; no remoteCPU gate record or NN launch')
target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record),flush=True)
