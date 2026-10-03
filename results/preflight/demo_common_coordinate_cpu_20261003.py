import hashlib
import json
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

name = 'audit_shared_identity_states.py'
review = json.loads((PROJECT / '.aris/traces/experiment-bridge/2026-10-03_shared_states/cpu_audit_source_review.json').read_text())
assert review['status'] == 'PASS_SOURCE_ONLY' and not review['blockers']
assert hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == review['source_sha256']
launch = json.loads((PROJECT / 'results/preflight/common_coordinate_2026_launch.json').read_text())
root, python = HOSTS['2026']
output = launch['output'] + '/controlled_states'
guard = f'''import json
from pathlib import Path
root=Path({output!r})
assert json.loads((root/'controller_result.json').read_text())['status']=='COMPLETE'
assert json.loads((root.parent/'controller_result.json').read_text())['status']=='COMPLETE'
process=Path('/proc/{launch['pid']}')
assert not process.exists() or (process/'stat').read_text().split()[2]=='Z'
assert not (root/'independent_cpu_audit.json').exists()
assert not (root/'independent_cpu_audit.log').exists()
print('FOUR_FROZEN_RUNS_TERMINAL_CPU_READY')
'''
print(remote_python('2026', guard).strip(), flush=True)
command(['scp', *OPTIONS, str(PROJECT / name), '2026:' + root + '/' + name])
argv = [python, '-u', name, '--root', output, '--data-root', '/data/gaob/Re-ID/dataset']
code = f'''import hashlib,json,os,subprocess
from pathlib import Path
source=Path({root!r})/{name!r}
assert hashlib.sha256(source.read_bytes()).hexdigest()=={review['source_sha256']!r}
with (Path({output!r})/'independent_cpu_audit.log').open('x') as handle:
 result=subprocess.run({argv!r},cwd={root!r},stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
exit_path=Path({output!r})/'independent_cpu_audit_exit.json'
assert not exit_path.exists()
exit_path.write_text(json.dumps(dict(exit_code=result.returncode,command={argv!r})))
assert result.returncode==0,(result.returncode,'read preserved independent_cpu_audit.log')
report=json.loads((Path({output!r})/'independent_cpu_audit.json').read_text())
assert report['status']=='PASS' and report['cases']==1176
print(json.dumps(dict(status=report['status'],cases=report['cases'],queries=report['queries'],gallery=report['gallery'],max_error={{variant:value['max_sixmetric_recount_error_pp'] for variant,value in report['runs'].items()}})))
'''
print('ACTUAL_INDEPENDENT_CPU', remote_python('2026', code).strip(), flush=True)
