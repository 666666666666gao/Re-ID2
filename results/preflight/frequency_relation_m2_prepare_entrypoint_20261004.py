import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path

r=Path('C:/Users/gb/projects/demo_dual_axis_20261002');p=r/'results/preflight'
old=json.loads((p/'trained_outlet_utility_plan.json').read_text(encoding='utf-8'))
previous={**old['old57_sources_unchanged'],**old['sources']}
assert len(previous)==60 and all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in previous.items())
names=('frequency_relation_axis.py','run_frequency_relation_experiment.py','verify_frequency_relation_axis.py',
 'diagnose_frequency_relation_utility.py','launch_frequency_relation_trial.py','audit_frequency_relation_trial.py')
for n in names:ast.parse((r/n).read_text(encoding='utf-8'),filename=n)
audit=r/'results/trained_outlet_utility_20261004/independent_cpu_audit.json'
assert json.loads(audit.read_text(encoding='utf-8'))['status']=='PASS'
plan=dict(status='IMPLEMENTED_FRESH_SOURCE_REVIEW_PENDING_NO_NEURAL_EXECUTION',at=datetime.now().isoformat(timespec='seconds'),
 campaign='frequency_relation_m2_20261004',input_reference='common_outlet_m1_trial_20261003_v3',
 hypothesis='Can preserving same-observation already-routed F identity distances at the actual PF retrieval outlet restore independent frequency utility and provide real joint retrieval gains?',
 evidence='Frozen six M1best,2352stageconditions/493920queryrows independent CPU PASS. All6 PF-post versus PF-pre mAP drops; full-input F increments weak. This does not prove causality or prescribe a larger gate.',
 evidence_audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
 source_teacher='Actual full-view routed F_pre entering PF, stopped gradient. Existing source-available observation; no complete-modality teacher for a missing input.',
 objective='weight0.1 SmoothL1 between student PF-output and stopped actual F_pre normalized off-diagonal pair distances, full-view only. Identical reference and prediction normalize by their own mean distances. No additional classification/heads, no demand that all joint states outperform all single states.',
 held_fixed=['original_mean','publicCLIP','FFT','expert_access','conditional_update','joint_router','contribution_predictor','gates','old_losses','partial_query_full_gallery_training','sampler','5632D','parameter_count'],
 limits='Relation distance preservation is rotation invariant and does not itself align identity directions or calibrate gates. Partial training retains prior protocol with no new teacher loss. Any regained PF standalone accuracy is not sufficient: require real11minus00/harm/rescue and fair controls before expansion.',
 controls=dict(no_relation='Already completed original_mean M1 axis/frequency/twins; no rerun. Exact fresh initialization/eval/zero-weight training parity required. Batches/partial sets compared at final CPU audit.',
 matched_relation=['axis_shared','frequency_shared','twins_shared']),
 resources=dict(host='2026',physical_gpus=[2,3],max_parallel=2,temperature_power_control=False,poll_seconds=240,text_only_hosts=['2025','2027']),
 schedule={'2':['axis_shared','twins_shared'],'3':['frequency_shared']},
 preflight='Actual CUDA state/parameter parity, all7availability×4states exact, zero-weight AMP training output parity, detached teacher and realPF loss gradient; then all3smokes with3actualupdates/strict reload/nonzero active gradients before anyfresh50.',
 training=dict(epochs=50,seed=42,batch=64,K=4,initialization='fresh publicCLIP',checkpoint='highest devmAP; earliest tie',AMP='nativeGradScaler512; count skips',official_test_uses=0),
 evaluation=dict(full_model_conditions=147,six_state_conditions=882,eight_stage_conditions=1176,
 independent_CPU_cases=2058,condition_query_rows=432180,metrics=['mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20'],CMC='1..50',groups_perquery=True),
 retention='Only3best weights plus necessary existing references. Smokes in-memory strict reload; no new initial/last/smoke checkpoints. No deletion of still-required dependencies.',
 next_decision='If actual PF utility recovers but11doesnot improve00, separately investigate common identity orientation or controller/estimator coupling. If relation objective hurts fair-control complete/missing metrics, retain negative evidence and revise one factor; do not expand toall3/multiseed before a clear candidate.',
 previous_sources=previous,sources={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},
 review_independence='same-family',acceptance_status='provisional',goal_complete=False)
target=p/'frequency_relation_m2_plan.json';assert not target.exists()
target.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
print('M2_SOURCES_PREPARED_NO_NEURAL',json.dumps(plan['sources']),flush=True)
