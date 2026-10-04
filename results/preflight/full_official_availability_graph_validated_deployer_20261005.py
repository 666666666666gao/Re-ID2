"""Deploy only reviewed graph additions after the completed negative M4 decision."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

root, python = HOSTS['2026']
prior = root + '/runs/full_official_modality_outlet_m4_20261005'
baseline = root + '/runs/full_official_baselines_20261004'
output = root + '/runs/full_official_availability_graph_m5_validated_20261005'
pf = PROJECT / 'results/preflight'
proof = pf / 'full_official_availability_graph_validated_launch_20261005.json'
assert not proof.exists()
plan_file = pf / 'full_official_availability_graph_validated_plan_20261005.json'
review_file = pf / 'full_official_availability_graph_validated_execution_review_20261005.json'
training_review_file = pf / 'full_official_availability_graph_training_review_20261005.json'
plan = json.loads(plan_file.read_text(encoding='utf-8'))
review = json.loads(review_file.read_text(encoding='utf-8'))
training_review = json.loads(training_review_file.read_text(encoding='utf-8'))
assert training_review['status'] == 'PASS' and not training_review['blocking_issues']
assert review['status'] == 'PASS' and not review['blocking_findings']
assert plan['source_sha256'] == review['sources_sha256']
assert all(hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() == digest for file, digest in plan['source_sha256'].items())
assert review['deployer_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
parents = plan['protected_parent_sources_sha256']
assert all(hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() == digest for file, digest in parents.items())
analysis = json.loads((PROJECT / plan['decision']['analysis']).read_text(encoding='utf-8'))
assert analysis['status'] == 'THREE_FULL_OFFICIAL_M4_RUNS_AND882_STATE_CASES_ANALYZED'
assert not analysis['comparisons']['demo -> M4/axis_shared']['normal']['both_metrics_plus2']
precheck = json.loads(remote_python('2026', f'''import hashlib,json,os,subprocess
from pathlib import Path
root=Path({root!r});prior=Path({prior!r})
done=json.loads((prior/'controller_result.json').read_text())
assert done['status']=='COMPLETE' and done['frozen_state_metric_cases']==882 and done['all_frozen_state_cpu_audits_passed']
assert done['paired_identity_and_partial_sampling_exact']
base=json.loads((Path({baseline!r})/'controller_result.json').read_text())
assert base['status']=='COMPLETE' and base['paired_identity_sampling_exact']
failed=root/'runs/full_official_availability_graph_m5_clean_20261005'
receipt=json.loads(Path(str(failed)+'_launch.json').read_text())
p=Path('/proc')/str(receipt['pid'])/'stat'
assert not p.exists() or p.read_text().split(') ',1)[1].split()[0]=='Z'
for phase in ('contract','preflight'):
 for entry in (failed/phase).glob('*_launch.json'):
  row=json.loads(entry.read_text());p=Path('/proc')/str(row['pid'])/'stat'
  assert not p.exists() or p.read_text().split(') ',1)[1].split()[0]=='Z'
assert not list((failed/'training').glob('*_launch.json'))
diagnostic=json.loads((root/'runs/full_official_availability_graph_adam_diagnostic_20261005/launch.json').read_text())
p=Path('/proc')/str(diagnostic['pid'])/'stat'
assert not p.exists() or p.read_text().split(') ',1)[1].split()[0]=='Z'
diag=json.loads((Path(diagnostic['output'])/'run/result.json').read_text())
assert diag['status']=='FOUR_ACTUAL_ADAM_DIAGNOSTIC_UPDATES_COMPLETE'
assert not Path({output!r}).exists() and not Path({(output+'_launch.json')!r}).exists()
expected={parents!r}
assert {{file:hashlib.sha256((root/file).read_bytes()).hexdigest() for file in expected}}==expected
usage=os.statvfs(root);free=usage.f_bavail*usage.f_frsize
assert free>4500*1024*1024
memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
selected={{int(row.split(',')[0]):int(row.split(',')[1]) for row in memory.strip().splitlines() if int(row.split(',')[0]) in (2,3)}}
assert set(selected)=={{2,3}} and all(value<500 for value in selected.values())
print(json.dumps(dict(status='M4_COMPLETE882_AND_FULL_BASELINES_SOURCE_FIXED',free_bytes=free,gpu_memory_mib=selected,temperature_power_control=False)))
'''))
for file in plan['source_sha256']:
    command(['scp', *OPTIONS, str(PROJECT / file), '2026:' + root + '/' + file])
for file in (plan_file, review_file, training_review_file, pf / 'full_official_availability_graph_adam_diagnostic_intake_20261005.json'):
    command(['scp', *OPTIONS, str(file), '2026:' + root + '/results/preflight/' + file.name])
verified = json.loads(remote_python('2026', 'import hashlib,json;from pathlib import Path;root=' + repr(root) + ';files=' + repr(list(plan['source_sha256'])) + ';print(json.dumps({file:hashlib.sha256((Path(root)/file).read_bytes()).hexdigest() for file in files}))'))
assert verified == plan['source_sha256']
argv = [python, '-u', 'launch_full_official_availability_graph_trial.py',
    '--data-root', '/data/gaob/Re-ID/dataset',
    '--pretrained', '/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
    '--previous-campaign', prior, '--baseline-campaign', baseline, '--output', output]
launch = json.loads(remote_python('2026', f'''import json,os,subprocess,time
from pathlib import Path
root=Path({root!r});output=Path({output!r});receipt=Path({(output+'_launch.json')!r})
assert not output.exists() and not receipt.exists()
with Path({(output+'_controller.log')!r}).open('x') as handle:
 child=subprocess.Popen({argv!r},cwd=root,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
assert os.getpgid(child.pid)==child.pid
row=dict(pid=child.pid,process_group=child.pid,output=str(output),command={argv!r},started=time.time(),physical_gpus=[2,3],temperature_power_control=False)
receipt.write_text(json.dumps(row,indent=2)+'\\n');print(json.dumps(row))
'''))
proof.write_text(json.dumps(dict(status='GRAPH_CONTROLLER_LAUNCHED_REAL_PARENT_AMP_CONTRACTS_AND_TWELVE_SMOKE_UPDATES_PENDING',
    launched_at=datetime.now().isoformat(timespec='seconds'), launch=launch, prerequisite=precheck,
    source_sha256=plan['source_sha256'], unchanged_parent_source_sha256=parents,
    actual_fresh50_runs=0, actual_contracts_passed=0, actual_smoke_updates=0,
    transfer_overlap_policy='Independent preparation/preflight/training proceed without waiting for local raw distance transfer.',
    goal_complete=False), indent=2) + '\n', encoding='utf-8')
command(['scp', *OPTIONS, str(proof), '2026:' + root + '/results/preflight/' + proof.name])
print('FULL_OFFICIAL_AVAILABILITY_GRAPH_CONTROLLER_LAUNCHED', json.dumps(launch), flush=True)
