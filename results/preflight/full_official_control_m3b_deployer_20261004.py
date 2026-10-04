"""One deployment/launch only after all six full official baseline audits complete."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

root, python = HOSTS['2026']
baseline = root + '/runs/full_official_baselines_20261004'
output = root + '/runs/full_official_control_m3b_20261004'
pf = PROJECT / 'results/preflight'
proof = pf / 'full_official_control_m3b_launch_20261004.json'
assert not proof.exists()
plan_file = pf / 'full_official_control_m3b_plan_20261004.json'
review_file = pf / 'full_official_control_m3b_source_review_20261004.json'
diagnostics_review_file = pf / 'full_official_control_diagnostics_source_review_20261004.json'
plan = json.loads(plan_file.read_text(encoding='utf-8'))
review = json.loads(review_file.read_text(encoding='utf-8'))
diagnostics_review = json.loads(diagnostics_review_file.read_text(encoding='utf-8'))
assert plan['status'] == 'SOURCE_REVIEW_PASS_NOT_DEPLOYED_NOT_LAUNCHED_CUDA_PENDING'
assert review['status'] == 'PASS' and not review['blocking_findings']
assert diagnostics_review['status'] == 'PASS' and not diagnostics_review['blocking_findings']
assert plan['sources'] == (review['sources_sha256'] | diagnostics_review['sources_sha256'])
assert all(hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() == sha for file, sha in plan['sources'].items())
deploy_review_file = pf / 'full_official_control_m3b_diagnostics_deployer_review_20261004.json'
deploy_review = json.loads(deploy_review_file.read_text(encoding='utf-8'))
assert deploy_review['status'] == 'PASS' and deploy_review['deployer_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
precheck = f'''import json
from pathlib import Path
root=Path({baseline!r});done=json.loads((root/'controller_result.json').read_text())
assert done['status']=='COMPLETE' and len(done['runs'])==6 and done['paired_identity_sampling_exact']
assert not Path({output!r}).exists() and not Path({(output+'_launch.json')!r}).exists()
for row in done['runs']:
 name=row['dataset']+'_'+row['variant']+'_s42'
 train=json.loads((root/'training'/name/'result.json').read_text());audit=json.loads((root/'frozen49'/name/'independent_cpu_audit.json').read_text())
 assert train['status']=='COMPLETE' and train['epochs']==50 and train['training_heldout_identities']==0
 assert not train['training_coverage']['unvisited'] and audit['status']=='PASS' and audit['cases']==49
 for phase in ('training','frozen49','audit'):assert json.loads((root/phase/(name+'_exit.json')).read_text())['exit_code']==0
print(json.dumps(dict(status='SIX_FULL_OFFICIAL_BASELINES_AND294_GT_CASES_COMPLETE',runs=6,cases=294)))
'''
prerequisite = json.loads(remote_python('2026', precheck))
# Only after the entire live baseline campaign has finished may shared executables change.
parents = ('measurement_gate_axis.py', 'identity_alignment_axis.py', 'frequency_relation_axis.py',
    'common_outlet_axis.py', 'shared_identity_axis.py', 'run_shared_identity_experiment.py',
    'axis_collaboration.py', 'official_training_data.py', 'experiment_data.py', 'run_experiment.py',
    'common_coordinate_axis.py', 'scaled_axis_collaboration.py', 'gpu_thermal_execute.py',
    'audit_shared_identity_states.py', 'full_evaluation.py', 'missing_evaluation.py')
source_files = [*plan['sources'], *parents]
source_hashes = {file: hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() for file in source_files}
assert {file: source_hashes[file] for file in parents} == deploy_review['parents_sha256']
for file in source_hashes:
    command(['scp', *OPTIONS, str(PROJECT / file), '2026:' + root + '/' + file])
for file in (plan_file, review_file, diagnostics_review_file, deploy_review_file):
    command(['scp', *OPTIONS, str(file), '2026:' + root + '/results/preflight/' + file.name])
verified = json.loads(remote_python('2026', 'import hashlib,json;from pathlib import Path;root=Path(' + repr(root) + ');files=' + repr(list(source_hashes)) + ';print(json.dumps({f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}))'))
assert verified == source_hashes
argv = [python, '-u', 'launch_full_official_control_trial.py', '--data-root', '/data/gaob/Re-ID/dataset',
    '--pretrained', '/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt', '--baseline-campaign', baseline, '--output', output]
launch_code = f'''import json,os,subprocess,time
from pathlib import Path
root=Path({root!r});output=Path({output!r});receipt=Path({(output+'_launch.json')!r})
assert not output.exists() and not receipt.exists()
log=Path({(output+'_controller.log')!r})
with log.open('x') as handle:
 child=subprocess.Popen({argv!r},cwd=root,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
assert os.getpgid(child.pid)==child.pid
row=dict(pid=child.pid,process_group=child.pid,output=str(output),command={argv!r},started=time.time(),physical_gpus=[2,3],temperature_power_control=False)
receipt.write_text(json.dumps(row,indent=2)+'\\n');print(json.dumps(row))
'''
launch = json.loads(remote_python('2026', launch_code))
proof.write_text(json.dumps(dict(status='CONTROLLER_LAUNCHED_CUDA_CONTRACT_AND_SIX_SMOKES_PENDING',
    launched_at=datetime.now().isoformat(timespec='seconds'), prerequisite=prerequisite,
    launch=launch, source_sha256=source_hashes, preflight_pass=False, actual_fresh50_runs=0, goal_complete=False),
    indent=2) + '\n', encoding='utf-8')
command(['scp', *OPTIONS, str(proof), '2026:' + root + '/results/preflight/' + proof.name])
print('FULL_OFFICIAL_CONTROL_CONTROLLER_LAUNCHED', json.dumps(launch), flush=True)
