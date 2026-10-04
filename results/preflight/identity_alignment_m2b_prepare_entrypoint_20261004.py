import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, remote_python

p = PROJECT/'results/preflight'
prior = json.loads((p/'cross_identity_coordinate_plan.json').read_text(encoding='utf-8'))
previous = {**prior['previous_sources'], **prior['sources']}
assert len(previous) == 69 and all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in previous.items())
names = ('identity_alignment_axis.py','run_identity_alignment_experiment.py','verify_identity_alignment_axis.py',
    'launch_identity_alignment_trial.py','diagnose_identity_alignment_utility.py','diagnose_identity_alignment_coordinates.py','audit_identity_alignment_trial.py')
for n in names: ast.parse((PROJECT/n).read_text(encoding='utf-8'),filename=n)
sources = {n:hashlib.sha256((PROJECT/n).read_bytes()).hexdigest() for n in names}
root,_ = HOSTS['2026']
code = f'''import hashlib,json,subprocess
from pathlib import Path
r=Path({root!r});m=r/'runs/frequency_relation_m2_20261004';c=r/'runs/cross_identity_coordinates_20261004'
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {previous!r}.items())
assert json.loads((m/'controller_result.json').read_text())['status']=='COMPLETE'
a=json.loads((m/'independent_cpu_audit.json').read_text());b=json.loads((c/'independent_cpu_audit.json').read_text())
assert a['status']=='PASS' and a['cases']==2058 and b['status']=='PASS' and b['cases']==1029
weights=list(m.rglob('*.pth'));assert len(weights)==3 and all(f.name=='best.pth' for f in weights)
memory={{}}
for gpu in (2,3):
 v=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
 assert int(v[0])==gpu;memory[str(gpu)]=float(v[1])
print(json.dumps(dict(status='ACTUAL_M2_AND_CROSS_COMPLETE_AUDITED_SOURCE69_EXACT',
 m2_audit_sha256=hashlib.sha256((m/'independent_cpu_audit.json').read_bytes()).hexdigest(),
 cross_audit_sha256=hashlib.sha256((c/'independent_cpu_audit.json').read_bytes()).hexdigest(),
 selected_gpu_memory_mib=memory,retained_m2_best=len(weights))))
'''
external = json.loads(remote_python('2026',code))
target = p/'identity_alignment_m2b_plan.json'; assert not target.exists()
plan = dict(status='IMPLEMENTED_FRESH_SOURCE_REVIEW_PENDING_NO_NEURAL_EXECUTION',at=datetime.now().isoformat(timespec='seconds'),
    campaign='identity_alignment_m2b_20261004',input_reference='frequency_relation_m2_20261004',
    evidence_cross='cross_identity_coordinates_20261004',external_current=external,
    previous_sources=previous,sources=sources,
    hypothesis='Can training actual routed PF identity compatibility with the stopped public gallery convert regained standalone utility into real joint retrieval gains?',
    objective='Existing M2 relation0.1 retained; only add0.1 mean positive log-probability over actual PF query vs stopped normalized base_common gallery at temperature0.07. Positives: same training identity and different observation name; same observation is removed from the denominator. Anchors with no distinct positive excluded and loss averaged over valid anchors.',
    observed_sampling_evidence='Actual original M2 all3 have identical453orders:436batches contain repeated observations,120identity groups have only1distinct observation. Excluding duplicate positives is necessary for the stated other-observation target, without changing sampling or other losses.',
    reference='Current same full-view observable base_common; stopped per forward, not a separately frozen whole teacher. No missing-view reconstruction or feature-copy target.',
    models=['axis_shared','frequency_shared','twins_shared'],controls='Completed M2 all3 without alignment, same observed batches/augmentation/partial sets; no rerun.',
    held_fixed=['M2relation0.1','original_mean','CLIP','FFT','expert_access','router','calibrator','gates','residual_scales','old_losses','partial_sampling','active_params','5632D','best_rule'],
    resources=dict(host='2026',physical_gpus=[2,3],max_parallel=2,poll_seconds=240,temperature_power_control=False,
        gpu2=['axis_shared','twins_shared'],gpu3=['frequency_shared'],text_only_hosts=['2025','2027']),
    preflight='Actual fresh CUDA same-state/all7availabilityx4states and zero-alignment AMP output equality to M2, synthetic detached-reference/duplicate-positive exact CE/actualPF gradients; three smokes each3realupdates/allactivegradients/strictreload barrier before any fresh50.',
    training=dict(epochs=50,seed=42,batch=64,K=4,relation_weight=.1,alignment_weight=.1,alignment_temperature=.07,
        initialization='fresh publicCLIP',AMP='nativeGradScaler512,countskips',checkpoint='devmAPbest/earliesttie',official_test_uses=0),
    evaluation=dict(full_conditions=147,six_state_conditions=882,eight_stage_conditions=1176,cross_coordinate_cases=1029,
        independent_CPU_cases=3087,repeated_condition_query_rows=648270,metrics=['mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20'],
        CMC='1..50',perquery_and_identity_camera_scene=True),
    retention='Only3newbest and necessary prior references; no initial/last/smoke weights, no old dependency cleanup.',
    decision='Need compatibility and independentPF identity utility plus actual11minus00/harm-rescue/fair-control gains. Improved alignment loss/crossretrieval alone is insufficient; no all3dataset/multiseed expansion without clear candidate.',
    goal_complete=False,estimated_from_previous_campaign=True,
    review_independence='same-family',acceptance_status='provisional',publication='ExistingONEhandoffrepo/Desktop/25/26/27 samebytes; onlycode/texttoRe-ID2; NPZ/weights26only')
target.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('IDENTITY_ALIGNMENT_PLAN_PREPARED',json.dumps(dict(sources=sources,external=external)),flush=True)
