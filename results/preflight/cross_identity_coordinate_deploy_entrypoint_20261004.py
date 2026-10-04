from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
p = PROJECT / 'results/preflight'
plan = json.loads((p / 'cross_identity_coordinate_plan.json').read_text(encoding='utf-8'))
review = json.loads((p / 'cross_identity_coordinate_review.json').read_text(encoding='utf-8'))
assert review['verdict'] == 'PASS_SOURCE_REVIEW' and not review['blocking_issues']
assert review['sources'] == plan['sources']
assert review['deployer_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
sources = {**plan['previous_sources'], **plan['sources']}
assert len(sources) == 69
assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == sha for name, sha in sources.items())
target = p / 'cross_identity_coordinate_2026_launch.json'
assert not target.exists()
for host, (root, _) in HOSTS.items():
    command(['scp', *OPTIONS, *[str(PROJECT / name) for name in plan['sources']], host + ':' + root + '/'])
    command(['scp', *OPTIONS, str(p / 'cross_identity_coordinate_plan.json'), str(p / 'cross_identity_coordinate_review.json'), host + ':' + root + '/results/preflight/'])
    code = 'import hashlib,json;from pathlib import Path;r=Path(' + repr(root) + ');print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ' + repr(list(sources)) + '}))'
    assert json.loads(remote_python(host, code)) == sources
root, python = HOSTS['2026']
campaign = root + '/runs/' + plan['input_campaign']
output = root + '/runs/' + plan['campaign']
code = f'''import hashlib,json,os,subprocess
from pathlib import Path
r=Path({root!r});c=Path({campaign!r});o=Path({output!r})
assert not o.exists() and not Path(str(o)+'.log').exists()
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {sources!r}.items())
assert json.loads((c/'controller_result.json').read_text())['status']=='COMPLETE'
assert json.loads((c/'independent_cpu_audit.json').read_text())['status']=='PASS'
memory={{}}
for gpu in (2,3):
 values=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
 assert int(values[0])==gpu
 memory[str(gpu)]=float(values[1]);assert memory[str(gpu)]<500
command=[{python!r},'-u','launch_cross_identity_coordinates.py','--campaign',str(c),'--output',str(o),'--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
with Path(str(o)+'.log').open('x') as log:
 child=subprocess.Popen(command,cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,output=str(o),log=str(o)+'.log',command=command,selected_gpu_memory_mib=memory,temperature_power_control=False)))
'''
launch = json.loads(remote_python('2026', code))
launch['recorded_at'] = datetime.now().isoformat(timespec='seconds')
launch['review_independence'] = 'same-family'
launch['acceptance_status'] = 'provisional'
target.write_text(json.dumps(launch, indent=2) + '\n', encoding='utf-8')
print('CROSS_COORDINATE_LAUNCHED', json.dumps(launch), flush=True)
