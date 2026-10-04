"""Deploy reviewed public-identity additions after the complete negative M5 decision."""
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
closed_graph = root + '/runs/full_official_availability_graph_m5_validated_20261005'
output = root + '/runs/full_official_public_outlet_identity_m6_20261005'
pf = PROJECT / 'results/preflight'
proof = pf / 'full_official_public_outlet_identity_launch_20261005.json'
assert not proof.exists()
plan_file = pf / 'full_official_public_outlet_identity_plan_20261005.json'
review_file = pf / 'full_official_public_outlet_identity_execution_review_20261005.json'
training_review_file = pf / 'full_official_public_outlet_identity_training_review_20261005.json'
plan = json.loads(plan_file.read_text(encoding='utf-8'))
review = json.loads(review_file.read_text(encoding='utf-8'))
training_review = json.loads(training_review_file.read_text(encoding='utf-8'))
assert training_review['status'] == 'PASS' and not training_review['blocking_findings']
assert review['status'] == 'PASS' and not review['blocking_findings']
assert plan['sources_sha256'] == review['sources_sha256']
assert all(hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() == digest for file, digest in plan['sources_sha256'].items())
assert review['deployer_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
assert all(plan['sources_sha256'][file] == digest for file, digest in training_review['sources_sha256'].items())
parents = plan['protected_parent_sources_sha256']
assert all(hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() == digest for file, digest in parents.items())
analysis = json.loads((PROJECT / plan['decision']['analysis']).read_text(encoding='utf-8'))
assert analysis['status'] == 'FOUR_FULL_OFFICIAL_GRAPH_RUNS196_FROZEN_AND882_STATE_CASES_ANALYZED'
assert not analysis['comparisons']['demo -> M5/axis_shared']['normal']['both_metrics_plus2']
assert analysis['models']['axis_shared']['inference_effects']['full11_minus00']['normal']['delta_pp']['mAP'] < 0
precheck = json.loads(remote_python('2026', f'''import hashlib,json,os,subprocess
from pathlib import Path
root=Path({root!r});prior=Path({prior!r});graph=Path({closed_graph!r})
done=json.loads((prior/'controller_result.json').read_text())
assert done['status']=='COMPLETE' and done['frozen_state_metric_cases']==882 and done['all_frozen_state_cpu_audits_passed']
assert done['paired_identity_and_partial_sampling_exact']
base=json.loads((Path({baseline!r})/'controller_result.json').read_text())
assert base['status']=='COMPLETE' and base['paired_identity_sampling_exact']
finished=json.loads((graph/'controller_result.json').read_text())
assert finished['status']=='COMPLETE' and finished['frozen_metric_cases']==196 and finished['enhanced_state_metric_cases']==882
assert finished['all_frozen_and_state_cpu_audits_passed'] and finished['paired_identity_and_partial_sampling_exact']
entries=[Path(str(graph)+'_launch.json')]
for phase in ('contract','preflight','training','frozen49','audit','diagnosis','diagnosis_audit'):
 entries.extend((graph/phase).glob('*_launch.json'))
for entry in entries:
 child=json.loads(entry.read_text());p=Path('/proc')/str(child['pid'])/'stat'
 assert not p.exists() or p.read_text().split(') ',1)[1].split()[0]=='Z',str(entry)
assert not Path({output!r}).exists() and not Path({(output+'_launch.json')!r}).exists()
expected={parents!r}
assert {{file:hashlib.sha256((root/file).read_bytes()).hexdigest() for file in expected}}==expected
usage=os.statvfs(root);free=usage.f_bavail*usage.f_frsize
assert free>{plan['execution']['formal_space_gate_bytes']!r}
memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
selected={{int(row.split(',')[0]):int(row.split(',')[1]) for row in memory.strip().splitlines() if int(row.split(',')[0]) in (2,3)}}
assert set(selected)=={{2,3}} and all(value<500 for value in selected.values())
print(json.dumps(dict(status='M5_CLOSED196_AND882_M4_AND_BASELINES_FIXED',free_bytes=free,gpu_memory_mib=selected,temperature_power_control=False)))
'''))
for file in plan['sources_sha256']:
    command(['scp', *OPTIONS, str(PROJECT / file), '2026:' + root + '/' + file])
for file in (plan_file, review_file, training_review_file, pf / 'full_official_availability_graph_adam_diagnostic_intake_20261005.json'):
    command(['scp', *OPTIONS, str(file), '2026:' + root + '/results/preflight/' + file.name])
verified = json.loads(remote_python('2026', 'import hashlib,json;from pathlib import Path;root=' + repr(root) + ';files=' + repr(list(plan['sources_sha256'])) + ';print(json.dumps({file:hashlib.sha256((Path(root)/file).read_bytes()).hexdigest() for file in files}))'))
assert verified == plan['sources_sha256']
argv = [python, '-u', 'launch_full_official_public_outlet_identity_trial.py',
    '--data-root', '/data/gaob/Re-ID/dataset',
    '--pretrained', '/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt',
    '--previous-campaign', prior, '--baseline-campaign', baseline,
    '--closed-graph-campaign', closed_graph, '--output', output]
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
proof.write_text(json.dumps(dict(status='PUBLIC_IDENTITY_CONTROLLER_LAUNCHED_NATIVE_CONTRACTS_AND_TWELVE_SMOKE_UPDATES_PENDING',
    launched_at=datetime.now().isoformat(timespec='seconds'), launch=launch, prerequisite=precheck,
    source_sha256=plan['sources_sha256'], unchanged_parent_source_sha256=parents,
    actual_fresh50_runs=0, actual_contracts_passed=0, actual_smoke_updates=0,
    transfer_overlap_policy='Prepare, preflight and train independently of raw archive completion when reviewed and resource-eligible.',
    goal_complete=False), indent=2) + '\n', encoding='utf-8')
command(['scp', *OPTIONS, str(proof), '2026:' + root + '/results/preflight/' + proof.name])
print('FULL_OFFICIAL_PUBLIC_IDENTITY_CONTROLLER_LAUNCHED', json.dumps(launch), flush=True)
