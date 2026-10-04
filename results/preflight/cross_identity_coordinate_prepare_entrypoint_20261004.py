import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, remote_python

p = PROJECT / 'results/preflight'
previous_plan = json.loads((p / 'frequency_relation_m2_plan.json').read_text(encoding='utf-8'))
previous_sources = {**previous_plan['previous_sources'], **previous_plan['sources']}
assert len(previous_sources) == 66
assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == sha for name, sha in previous_sources.items())
names = ('diagnose_cross_identity_coordinates.py', 'launch_cross_identity_coordinates.py', 'audit_cross_identity_coordinates.py')
for name in names:
    ast.parse((PROJECT / name).read_text(encoding='utf-8'))
sources = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in names}
root, _ = HOSTS['2026']
campaign = root + '/runs/frequency_relation_m2_20261004'
code = f'''import hashlib,json,subprocess
from pathlib import Path
r=Path({root!r});c=Path({campaign!r})
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {previous_sources!r}.items())
terminal=json.loads((c/'controller_result.json').read_text());audit=json.loads((c/'independent_cpu_audit.json').read_text())
assert terminal['status']=='COMPLETE' and len(terminal['runs'])==3
assert audit['status']=='PASS' and audit['cases']==2058 and audit['perquery_count']==432180
weights=list(c.rglob('*.pth'));assert len(weights)==3 and all(f.name=='best.pth' for f in weights)
for variant in ('axis_shared','frequency_shared','twins_shared'):
 assert (c/'original_mean/utility'/('MSVR310_'+variant+'_s42')/'full/result.json').exists()
memory={{}}
for gpu in (2,3):
 values=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
 assert int(values[0])==gpu
 memory[str(gpu)]=float(values[1])
print(json.dumps(dict(status='PREVIOUS_M2_ACTUAL_COMPLETE_AUDITED_SOURCE66_EXACT',audit_sha256=hashlib.sha256((c/'independent_cpu_audit.json').read_bytes()).hexdigest(),weights=[str(f.relative_to(c)) for f in weights],selected_gpu_memory_mib=memory)))
'''
external = json.loads(remote_python('2026', code))
plan_path = p / 'cross_identity_coordinate_plan.json'
assert not plan_path.exists()
plan = dict(status='IMPLEMENTED_FRESH_SOURCE_REVIEW_PENDING_NO_NEURAL_EXECUTION', at=datetime.now().isoformat(timespec='seconds'),
    campaign='cross_identity_coordinates_20261004', input_campaign='frequency_relation_m2_20261004',
    hypothesis='Are actual routed F_pre and projected F_post in coordinates compatible with base_common identity features, independently of within-frequency retrieval ability and gate amplitude?',
    evidence='M2 installed-GT2058/432180 PASS. Axis F_post normal32.687296mAP versus M1+3.869874, F_pre40.509623 versus M1+1.931238. Axis11minus00+0.000938mAP/0R1. All49 versus matched twins-2.666757mAP/-4.023324R1. No valid final candidate.',
    external_current=external, previous_sources=previous_sources, sources=sources,
    models=['axis_shared', 'frequency_shared', 'twins_shared'],
    checkpoint='Existing original M2 dev-mAP best; no weight selection, training, gates, residual scaling or new descriptor.',
    probe_pairs=['common_common','F_pre_F_pre','F_post_F_post','F_pre_common','common_F_pre','F_post_common','common_F_post'],
    protocol='Three normalized512D actual stages, seven availability banks, all49 query/gallery conditions per7 directed stage pairs. Same-ID/same-scene junk excluded from installed MSVR GT. Three within-stage controls must exactly reproduce previous M2 metrics/perquery.',
    evaluations=1029, condition_query_rows=216090,
    verification='Three64-record all7-bank smoke barrier, strict best reload/eval, original fuse exactly reconstructed, zero RNT deployed feature difference, state versions and five existing input files unchanged. Independently reconstruct cross-stage matrices from saved normalized features within2e-6, and independently rank exact raw matrices/recount six metrics,CMC1..50,perquery and identity/camera/scene groups.',
    limits='Cross-space retrieval is not a new deployable method or proof of causal information loss. Poor compatibility would motivate a single identity-coordinate alignment intervention; good compatibility with negligible deployed benefit would motivate separate estimation/control or amplitude diagnostics. No simultaneous cross-availability objective changes.',
    resources=dict(host='2026',physical_gpus=[2,3],max_parallel=2,poll_seconds=240,temperature_power_control=False,
        gpu2=['axis_shared','twins_shared'],gpu3=['frequency_shared'],text_only_hosts=['2025','2027']),
    estimated_gpu_minutes=5, estimated_cpu_minutes=2,
    optimizer_updates=0,new_weights=0,official_test_uses=0,goal_complete=False,
    review_independence='same-family',acceptance_status='provisional',
    publication='Existing ONE docs/实验交接.md, Desktop/document and25/26/27 same bytes; code/text results to Re-ID2, raw NPZ remain26.')
plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('CROSS_COORDINATE_PLAN_PREPARED', json.dumps(dict(sources=sources, external=external)), flush=True)
