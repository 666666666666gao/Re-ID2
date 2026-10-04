"""Launch one reviewed diagnostic after the failed clean preflight is terminal."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

root, python = HOSTS['2026']
pf = PROJECT / 'results/preflight'
proof = pf / 'full_official_availability_graph_adam_diagnostic_launch_20261005.json'
assert not proof.exists()
review = json.loads((pf / 'full_official_availability_graph_adam_diagnostic_review_20261005.json').read_text(encoding='utf-8'))
assert review['status'] == 'PASS' and not review['blocking_findings']
assert review['source_sha256'] == hashlib.sha256((PROJECT / 'diagnose_graph_disabled_adam.py').read_bytes()).hexdigest()
assert review['deployer_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
launch = json.loads((pf / 'full_official_availability_graph_clean_launch_20261005.json').read_text(encoding='utf-8'))['launch']
output = root + '/runs/full_official_availability_graph_adam_diagnostic_20261005'
prerequisite = json.loads(remote_python('2026', f'''import json,subprocess
from pathlib import Path
failed=Path({launch['output']!r})
def terminal(pid):
 p=Path('/proc')/str(pid)/'stat'
 return not p.exists() or p.read_text().split(') ',1)[1].split()[0]=='Z'
assert terminal({launch['pid']})
for phase in ('contract','preflight'):
 for p in (failed/phase).glob('*_launch.json'):assert terminal(json.loads(p.read_text())['pid'])
assert not list((failed/'training').glob('*_launch.json'))
assert not Path({output!r}).exists()
memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
value=next(int(row.split(',')[1]) for row in memory.splitlines() if int(row.split(',')[0])==3)
assert value<500
print(json.dumps(dict(failed_controller_and_children_terminal=True,failed_fresh50_runs=0,gpu3_memory_mib=value)))
'''))
command(['scp', *OPTIONS, str(PROJECT / 'diagnose_graph_disabled_adam.py'), '2026:' + root + '/diagnose_graph_disabled_adam.py'])
argv = [python, '-u', 'diagnose_graph_disabled_adam.py', '--data-root', '/data/gaob/Re-ID/dataset',
    '--pretrained', '/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt', '--output', output + '/run']
actual = json.loads(remote_python('2026', f'''import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path({root!r});out=Path({output!r});out.mkdir(exist_ok=False)
assert hashlib.sha256((root/'diagnose_graph_disabled_adam.py').read_bytes()).hexdigest()=={review['source_sha256']!r}
with (out/'stdout.log').open('x') as handle:
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='3')
 child=subprocess.Popen({argv!r},cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
row=dict(pid=child.pid,output=str(out),command={argv!r},physical_gpu=3,started=time.time(),temperature_power_control=False)
(out/'launch.json').write_text(json.dumps(row,indent=2)+'\\n');print(json.dumps(row))
'''))
proof.write_bytes((json.dumps(dict(status='ONE_DIAGNOSTIC_LAUNCHED_NO_FRESH50',launch=actual,prerequisite=prerequisite,source_sha256=review['source_sha256']),indent=2)+'\n').encode('utf-8'))
print('ONE_ADAM_DIAGNOSTIC_LAUNCHED', json.dumps(actual), flush=True)
