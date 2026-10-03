import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, remote_python

launch=json.loads((PROJECT/'results/preflight/common_outlet_scale_2026_launch.json').read_text())
review=json.loads((PROJECT/'results/preflight/common_outlet_scale_review.json').read_text())
assert review['status']=='PASS' and not review['blockers']
name='audit_common_outlet_scale.py'
expected=review['checked_source_sha256'][name]
assert hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==expected
root,python=HOSTS['2026']
output=launch['output']
argv=[python,'-u',name,'--root',output,'--data-root','/data/gaob/Re-ID/dataset']
code=f'''import hashlib,json,os,subprocess
from pathlib import Path
root=Path({output!r});process=Path('/proc/{launch['pid']}')
assert not process.exists() or (process/'stat').read_text().split()[2]=='Z'
assert json.loads((root/'controller_result.json').read_text())['status']=='COMPLETE'
assert hashlib.sha256((Path({root!r})/{name!r}).read_bytes()).hexdigest()=={expected!r}
assert not (root/'independent_cpu_audit.json').exists() and not (root/'independent_cpu_audit_exit.json').exists()
with (root/'independent_cpu_audit.log').open('x') as handle:
 result=subprocess.run({argv!r},cwd={root!r},stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
(root/'independent_cpu_audit_exit.json').write_text(json.dumps(dict(exit_code=result.returncode,command={argv!r})))
assert result.returncode==0,(result.returncode,'read preserved independent_cpu_audit.log')
report=json.loads((root/'independent_cpu_audit.json').read_text())
assert report['status']=='PASS' and report['cases']==882
print(json.dumps(dict(status=report['status'],cases=report['cases'],queries=report['queries'],max_error={{v:r['max_sixmetric_error_pp'] for v,r in report['runs'].items()}})))
'''
print('OUTLET_INDEPENDENT_CPU',remote_python('2026',code).strip(),flush=True)
