from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS,command,remote_python

HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
p=PROJECT/'results/preflight'
plan=json.loads((p/'trained_outlet_utility_plan.json').read_text(encoding='utf-8'))
review=json.loads((p/'trained_outlet_utility_review.json').read_text(encoding='utf-8'))
assert review['verdict']=='PASS_SOURCE_REVIEW' and not review['blockers']
assert review['checked_source_sha256']==plan['sources']
assert review['checked_previous_source_sha256']==plan['old57_sources_unchanged']
assert review['checked_deployer_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
previous=plan['old57_sources_unchanged'];sources={**previous,**plan['sources']}
assert len(sources)==60 and all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in sources.items())
for host,(root,_) in HOSTS.items():
    code='import hashlib,json;from pathlib import Path;r=Path('+repr(root)+');print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in '+repr(list(previous))+'}))'
    assert json.loads(remote_python(host,code))==previous
    command(['scp',*OPTIONS,*[str(PROJECT/n) for n in plan['sources']],host+':'+root+'/'])
    code='import hashlib,json;from pathlib import Path;r=Path('+repr(root)+');print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in '+repr(list(sources))+'}))'
    assert json.loads(remote_python(host,code))==sources
    print('TRAINED_OUTLET_SOURCE60_EXACT',host,flush=True)
root,python=HOSTS['2026'];campaign=root+'/runs/'+plan['input_campaign'];output=root+'/runs/'+plan['campaign']
data='/data/gaob/Re-ID/dataset';pretrained='/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt'
code=f'''import ast,json,shutil,sys
from pathlib import Path
r=Path({root!r});sys.path.insert(0,str(r))
for n in {list(plan['sources'])!r}:ast.parse((r/n).read_text(encoding='utf-8'),filename=n)
from gpu_thermal_execute import telemetry
selected=[telemetry(g) for g in (2,3)]
assert all(row['memory_used_mib']<500 for row in selected)
c=Path({campaign!r});end=json.loads((c/'controller_result.json').read_text())
assert end['status']=='COMPLETE' and len(end['runs'])==10
assert json.loads((c/'independent_cpu_audit.json').read_text())['status']=='PASS'
assert Path({data!r}).is_dir() and Path({pretrained!r}).is_file()
assert shutil.disk_usage(r).free>2_000_000_000
assert not Path({output!r}).exists() and not Path({output!r}+'.log').exists()
print(json.dumps(dict(status='PASS_REMOTE_SOURCE_AST_AND_INPUTS_SELECTED_CAPACITY',AST_files=3,
 selected_gpus=selected,python=sys.version,disk_free_bytes=shutil.disk_usage(r).free,
 temperature_power_control=False,neural_jobs=0)))
'''
checks=json.loads(remote_python('2026',code))
target=p/'trained_outlet_utility_remote_cpu_checks.json';assert not target.exists()
target.write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
argv=[python,'-u','launch_trained_outlet_utility.py','--campaign',campaign,'--output',output,
      '--data-root',data,'--pretrained',pretrained]
code=f'''import hashlib,json,os,subprocess,time
from pathlib import Path
r=Path({root!r})
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {sources!r}.items())
assert not Path({output!r}).exists() and not Path({output!r}+'.log').exists()
with Path({output!r}+'.log').open('x') as handle:
 child=subprocess.Popen({argv!r},cwd=r,start_new_session=True,stdout=handle,stderr=subprocess.STDOUT,
  env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
record=dict(host='2026',pid=child.pid,output={output!r},log={output!r}+'.log',command={argv!r},started=time.time(),
 selected_gpus=[2,3],max_parallel=2,temperature_power_control=False,source_sha256={sources!r},
 status='FROZEN_DIAGNOSTIC_DISPATCHED_SIX_SMOKES_NOT_YET_ACCEPTED',optimizer_updates=0,new_weights=0)
Path({output!r}+'_launch.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
launch=json.loads(remote_python('2026',code));launch['observed_at']=datetime.now().isoformat(timespec='seconds')
target=p/'trained_outlet_utility_2026_launch.json';assert not target.exists()
target.write_text(json.dumps(launch,indent=2)+'\n',encoding='utf-8')
print('TRAINED_OUTLET_FROZEN_DIAGNOSTIC_DISPATCHED',json.dumps(dict(pid=launch['pid'],output=output)),flush=True)
