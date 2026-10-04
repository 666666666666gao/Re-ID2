"""Deploy only the reviewed frozen readout probe; no new training or weights."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

root, python = HOSTS['2026']
source = root + '/runs/full_official_control_m3b_20261004'
output = root + '/runs/full_official_control_readout_20261004'
pf = PROJECT / 'results/preflight'
proof = pf / 'full_official_control_readout_launch_20261004.json'
assert not proof.exists()
plan_file = pf / 'full_official_control_readout_plan_20261004.json'
review_file = pf / 'full_official_control_readout_source_review_20261004.json'
plan = json.loads(plan_file.read_text(encoding='utf-8'))
review = json.loads(review_file.read_text(encoding='utf-8'))
assert review['status'] == 'PASS' and not review['blocking_findings']
assert plan['source_files_sha256'] == review['sources_sha256']
assert review['deployer_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == value for name, value in plan['source_files_sha256'].items())
parents = json.loads((pf / 'full_official_control_m3b_launch_20261004.json').read_text(encoding='utf-8'))['source_sha256']
assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == value for name, value in parents.items())
prerequisite = json.loads(remote_python('2026', f'''import json,os
from pathlib import Path
source=Path({source!r});done=json.loads((source/'controller_result.json').read_text())
assert done['status']=='COMPLETE' and len(done['runs'])==6 and done['frozen_state_metric_cases']==1764
assert done['all_frozen_state_cpu_audits_passed'] and done['paired_identity_and_partial_sampling_exact']
assert not Path({output!r}).exists() and not Path({(output+'_launch.json')!r}).exists()
usage=os.statvfs(source);free=usage.f_bavail*usage.f_frsize
assert free>3500*1024*1024
print(json.dumps(dict(status='SIX_ORIGINAL_RUNS_TERMINAL_1764_CPU_PASS',free_bytes=free)))
'''))
for name in plan['source_files_sha256']:
    command(['scp', *OPTIONS, str(PROJECT / name), '2026:' + root + '/' + name])
for path in (plan_file, review_file):
    command(['scp', *OPTIONS, str(path), '2026:' + root + '/results/preflight/' + path.name])
expected = parents | plan['source_files_sha256']
actual = json.loads(remote_python('2026', 'import hashlib,json;from pathlib import Path;root=Path(' + repr(root) + ');names=' + repr(list(expected)) + ';print(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}))'))
assert actual == expected
argv = [python, '-u', 'launch_full_official_control_readout.py', '--source-root', source, '--output', output]
launch = json.loads(remote_python('2026', f'''import json,os,subprocess,time
from pathlib import Path
root=Path({root!r});output=Path({output!r});receipt=Path({(output+'_launch.json')!r})
assert not output.exists() and not receipt.exists()
with Path({(output+'_controller.log')!r}).open('x') as handle:
 child=subprocess.Popen({argv!r},cwd=root,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
assert os.getpgid(child.pid)==child.pid
row=dict(pid=child.pid,process_group=child.pid,output=str(output),command={argv!r},started=time.time(),physical_gpus=[2,3],temperature_power_control=False)
receipt.write_text(json.dumps(row,indent=2)+chr(10));print(json.dumps(row))
'''))
proof.write_text(json.dumps(dict(status='FROZEN_READOUT_CONTROLLER_LAUNCHED_REAL_SMOKES_PENDING',
    launched_at=datetime.now().isoformat(timespec='seconds'), prerequisite=prerequisite, launch=launch,
    source_sha256=expected, preflight_pass=False, actual_training_updates=0, new_weights=0,
    planned_metric_cases=1176, goal_complete=False), indent=2) + '\n', encoding='utf-8')
command(['scp', *OPTIONS, str(proof), '2026:' + root + '/results/preflight/' + proof.name])
print('FULL_OFFICIAL_READOUT_CONTROLLER_LAUNCHED', json.dumps(launch), flush=True)
