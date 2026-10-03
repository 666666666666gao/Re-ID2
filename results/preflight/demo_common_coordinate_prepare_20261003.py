import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT

previous = json.loads((PROJECT / 'results/preflight/shared_states_2026_launch.json').read_text())['source_sha256']
new = ['common_coordinate_axis.py','run_common_coordinate_experiment.py','missing_common_coordinate_development.py',
       'verify_common_coordinate_axis.py','diagnose_common_coordinate_states.py','launch_common_coordinate_trial.py']
sources = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in [*previous, *new]}
for name in sources:
    if name.endswith('.py'):
        ast.parse((PROJECT / name).read_text(encoding='utf-8'))
helper = Path('C:/Users/gb/.codex_tmp/demo_common_coordinate_deploy_20261003.py')
ast.parse(helper.read_text())
now = datetime.now().isoformat(timespec='seconds')
plan = dict(status='PREPARED_REQUIRES_FRESH_SOURCE_REVIEW_NO_NEURAL_RUN', created=now,
    objective='Original all3 datasets and comprehensive missing-modality mAP/Rank1 >= same-protocol DeMo +2pp remains active, unmet.',
    evidence=dict(commit='343be26ac69d7ba23f00c25c89bed0981ced09d8', cases=1176, raw_GT_recount_error_pp=0,
        normal_axis_full_vs_base_mAP=-.3579768860555248, normal_axis_full_vs_base_Rank1=-.9523809523809632,
        missing_mean_axis_vs_ordinary_frequency_mAP=-1.549433464302206,
        disjoint_private_conditions=12, interpretation='Experts can affect retrieval but complete-input fusion harms; private disjoint source dot products zero; substantial partial-set base-representation gap already exists. Coordinate intervention is a hypothesis, not a complete explanation.'),
    single_change='Keep V11 encoder, DeMo private identity anchor, FFT, modality/frequency access, conditions, joint routing, PM/PF/PI, residual scales, gates, losses and partial-query/full-gallery training. Move existing delta_M and delta_I to common coordinates by the SAME relation-mean operation already used by V11 ordinary-frequency; no additions or hyperparameter sweep. All three augmented variants now use identical fusion geometry.',
    architecture='5632D: sqrt(.75)*normalized unchanged private5120 + sqrt(.25)*normalized common512 with M/F/I; seven relations/three bands are a route table, no21 independent experts.',
    unchanged_controls='V12 ordinary-frequency and DeMo are mathematical V11 controls; ordinary-frequency exact initial four-state parity across all7 availability sets must pass CUDA contract. Fresh50 repeats provide matched campaign evidence rather than combining historical best versions.',
    runs=[dict(dataset='MSVR310',variant=variant,seed=42,gpu=2 if index%2==0 else 3,queue_position=index//2+1)
          for index,variant in enumerate(('axis_shared','frequency_shared','twins_shared','demo_shared'))],
    resources='User latest directive: host2026 GPU2 andGPU3 ONLY. Two per-GPU queues, all4smoke barrier then sequential per-GPU fresh50+frozen49+controlled6x49. No25/27 training, no GPU0/1 scheduling, no preemption.',
    protocol='Fixed identity-heldout development, same initial publicCLIP/freshheads, exact augmented states and counts, shared DeMo commonhead init; Adam/nativeFP16 GradScaler512/B64/K4/50epochs; same real batch orders and6partial-set sequence; earliest tied bestdevmAP; no official-test tuning.',
    sanity='GPU2 actual28availability contracts, allaug same initialstate/params, invalidcoords0/BNuntouched/.25commonmass, stopped GT gallery, allstate private coordinates EXACT00 and ordinaryfrequency allstates exactlegacy; GPU2/3 each two3stepfull+partial smokes, alltrainable nonzero finite grads+12realsteps+memory strictreload before fresh50.',
    evaluation='Each train strict devbest reload sixmetrics/CMC50/groups/perquery; each best frozen49 cleanfeatures maxerror0/protectedinputs and versionsunchanged; each best sevenbank controlledsmoke then49x6=294cases, previous49full11 exact, rawdistanceNPZprivate. Four models1176cases and independent installedGT CPUrecount via existing audit_shared_identity_states.py required after terminal.',
    criteria='No goal success from one dataset/seed. First this exact single-change variant must improve over same-protocol DeMo and matched ordinary-frequency/twins with actual beneficial11−00, without selective omission of negative missing conditions. Only then consider common recipe all3/multiseed/finaltest. Full all3+missing +2 threshold unchanged.',
    retention='Each run only best.pth overwritten. No last/initial/smoke weights. Keep baseline and capacity controls; no new permission to clean authored nonbest needed.',
    estimated_minutes=35, poll_seconds=240, global_max_parallel_runs=2, selected_gpus=[2,3],
    sources=sources, previous_remote_sources=previous, new_sources=new,
    helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest())
path = PROJECT / 'results/preflight/common_coordinate_plan.json'
assert not path.exists()
path.write_text(json.dumps(plan, indent=2) + '\n')
trace = PROJECT / '.aris/traces/experiment-bridge/2026-10-03_common_coordinate'
trace.mkdir(exist_ok=False)
for name in new + ['run_shared_identity_experiment.py','missing_shared_identity_development.py','verify_shared_identity_axis.py','diagnose_shared_identity_states.py']:
    (trace / Path(name).name).write_bytes((PROJECT / name).read_bytes())
(trace / path.name).write_bytes(path.read_bytes())
(trace / helper.name).write_bytes(helper.read_bytes())
print('COMMON_COORDINATE_PREPARED_NO_NEURAL_RUN', json.dumps(dict(sources=len(sources),selected_gpus=[2,3],plan=str(path))), flush=True)
