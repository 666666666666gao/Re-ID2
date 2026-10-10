from datetime import datetime
import hashlib,json
from pathlib import Path

p=Path('C:/Users/gb/projects/demo_dual_axis_20261002');pf=p/'results/preflight'
load=lambda path:json.loads(path.read_text(encoding='utf-8'))
legacy=pf/'r201m_MSVR_sampling_diversity_actual_20261009.json';previous=load(legacy)
assert previous['status']=='ACTUAL_CLOSED_MSVR_PAIRED_PK8_TRAINING_FILENAME_DIVERSITY'
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==sha for n,sha in (previous['input_sha256']|previous['tables_sha256']).items())
o_closed=load(pf/'r201o_same_state_contribution_MSVR310_actual_session_20261010.json')
assert o_closed['status']=='ACTUAL_O_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW' and o_closed['exit_code']==0
inputs={};orders={}
for family,tag in (('M','r201m_primary_margin06_20261008'),('N','r201n_bounded_identity_shift_20261009'),('O','r201o_same_state_contribution_20261010')):
    for variant in ('frequency_shared','axis_shared'):
        folder=p/'results'/tag/'MSVR310/training'/('MSVR310_r201'+family.lower()+'_'+variant+'_s42')
        result=load(folder/'result.json');assert result['status']=='COMPLETE' and result['epochs']==50 and result['optimizer_steps']==1000
        path=folder/'batch_orders.jsonl';rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        assert len(rows)==1000 and all(r['optimizer_updated'] and r['sampling_identities']==r['sampling_instances']==8 and len(r['names'])==64 for r in rows)
        name=family+'_'+variant;orders[name]=[(r['epoch'],r['step'],r['names']) for r in rows]
        inputs[name]=dict(path=path.relative_to(p).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),batches=1000)
assert all(order==orders['M_frequency_shared'] for order in orders.values())
first=orders['O_axis_shared'];repeats=0;seen=set()
for epoch,step,names in first:
    seen.update(names)
    for index in range(0,64,8):
        group=names[index:index+8]
        assert len({n.split('_')[0] for n in group})==1
        repeats+=8-len(set(group))
assert len(seen)==1032 and repeats==previous['summary']['repeated_filename_slots']==22022
record=dict(status='ACTUAL_CLOSED_O_MSVR_SAMPLING_SCHEDULE_EQUIVALENT_TO_COMPLETED_M_AND_N',at=datetime.now().astimezone().isoformat(timespec='seconds'),
    closed_models=6,batches_per_model=1000,sample_slots_per_model=64000,unique_training_records=1032,repeated_filename_slots=22022,repeated_filename_slot_fraction=22022/64000,
    original_sampling_diagnostic=legacy.relative_to(p).as_posix(),original_sampling_diagnostic_sha256=hashlib.sha256(legacy.read_bytes()).hexdigest(),
    inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    new_neural_calls=0,new_optimizer_updates=0,current_RGBNT100_progress_reads=0,remote_queries=0,current_training_changed=False,
    limits='Exact equality is epoch/step/filename scheduling across closed MSVR M/N/O ordinary/axis only, not tensor augmentation, weights, gradients, predictions or metrics. Repeated filenames are permitted PK sampling and may have distinct augmentations. Existing M source scene statistics apply to the same filename groups; no new cause, invalid-positive claim, sampler trial or root-cause determination. Historical K4/K8 comparison exists; restoring K4 alone is not treated as an untried fix. Future choice remains gated on current O normal/paper-six/full-primary gradient outcomes.')
out=pf/'r201o_closed_MSVR_sampling_equivalence_actual_20261010.json';assert not out.exists()
out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k not in ('inputs','limits')},ensure_ascii=False),flush=True)
