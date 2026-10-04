from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
p = PROJECT/'results/preflight'
plan = json.loads((p/'measurement_gate_m3a_plan.json').read_text(encoding='utf-8'))
review = json.loads((p/'measurement_gate_m3a_review.json').read_text(encoding='utf-8'))
assert review['verdict'] == 'PASS_SOURCE_REVIEW' and not review['blockers']
assert review['checked_source_sha256'] == plan['sources']
assert review['checked_previous_source_sha256'] == plan['previous_sources']
assert review['checked_deployer_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
previous = plan['previous_sources']; sources = previous | plan['sources']
assert len(sources) == 83
assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest() == sha for n, sha in sources.items())
assert not (p/'measurement_gate_m3a_2026_launch.json').exists()
for host, (root, _) in HOSTS.items():
    code = 'import hashlib,json;from pathlib import Path;r=Path('+repr(root)+');print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in '+repr(list(previous))+'}))'
    assert json.loads(remote_python(host, code)) == previous
    command(['scp', *OPTIONS, *[str(PROJECT/n) for n in plan['sources']], host+':'+root+'/'])
    command(['scp', *OPTIONS, str(p/'measurement_gate_m3a_plan.json'), str(p/'measurement_gate_m3a_review.json'), host+':'+root+'/results/preflight/'])
    code = 'import hashlib,json;from pathlib import Path;r=Path('+repr(root)+');print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in '+repr(list(sources))+'}))'
    assert json.loads(remote_python(host, code)) == sources
    print('M3A_SOURCE83_EXACT', host, flush=True)
root, python = HOSTS['2026']; output = root+'/runs/'+plan['campaign']
reference = root+'/runs/'+plan['input_reference']
data = '/data/gaob/Re-ID/dataset'; pretrained = '/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt'
code = f'''import ast,hashlib,json,shutil,sys
from pathlib import Path
r=Path({root!r});sys.path.insert(0,str(r))
for n in {list(plan['sources'])!r}:ast.parse((r/n).read_text(encoding='utf-8'),filename=n)
from gpu_thermal_execute import telemetry
selected=[telemetry(g) for g in (2,3)]
assert all(v['memory_used_mib']<500 for v in selected)
c=Path({reference!r});a=c/'independent_cpu_audit.json'
assert json.loads((c/'controller_result.json').read_text())['status']=='COMPLETE'
audit=json.loads(a.read_text());assert audit['status']=='PASS' and audit['cases']==3087 and audit['perquery_count']==648270
assert hashlib.sha256(a.read_bytes()).hexdigest()=={plan['reference_audit_sha256']!r}
assert len(list(c.rglob('*.pth')))==3
assert Path({data!r}).is_dir() and Path({pretrained!r}).is_file() and shutil.disk_usage(r).free>2_000_000_000
assert not Path({output!r}).exists() and not Path({output!r}+'.log').exists()
print(json.dumps(dict(status='PASS_REMOTE_CPU_AST_SELECTED_CAPACITY_REFERENCE',selected_gpus=selected,
 temperature_power_control=False,disk_free_bytes=shutil.disk_usage(r).free,neural_jobs=0)))
'''
checks = json.loads(remote_python('2026', code))
target = p/'measurement_gate_m3a_remote_cpu_checks.json'; assert not target.exists()
target.write_text(json.dumps(checks, indent=2)+'\n', encoding='utf-8')
argv = [python, '-u', 'launch_measurement_gate_trial.py', '--output', output, '--data-root', data, '--pretrained', pretrained]
code = f'''import hashlib,json,os,subprocess,time
from pathlib import Path
r=Path({root!r})
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {sources!r}.items())
assert not Path({output!r}).exists() and not Path({output!r}+'.log').exists()
with Path({output!r}+'.log').open('x') as log:
 child=subprocess.Popen({argv!r},cwd=r,start_new_session=True,stdout=log,stderr=subprocess.STDOUT,
  env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
record=dict(pid=child.pid,host='2026',command={argv!r},output={output!r},log={output!r}+'.log',started=time.time(),
 selected_gpus=[2,3],max_parallel=2,temperature_power_control=False,source_sha256={sources!r},
 status='M3A_DISPATCHED_CONTRACT_AND_SMOKES_NOT_ACCEPTED_FRESH50_NOT_YET_STARTED')
Path({output!r}+'_launch.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''
launch = json.loads(remote_python('2026', code))
launch['observed_at'] = datetime.now().isoformat(timespec='seconds')
target = p/'measurement_gate_m3a_2026_launch.json'; assert not target.exists()
target.write_text(json.dumps(launch, indent=2)+'\n', encoding='utf-8')
print('M3A_DISPATCHED', json.dumps(dict(pid=launch['pid'], output=output)), flush=True)
