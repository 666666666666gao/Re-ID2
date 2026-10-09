"""Describe actual closed P8K8 filename diversity; no neural or sampler replay."""
from collections import Counter,defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

p=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
root=p/'results/r201m_primary_margin06_20261008/MSVR310'
proof=p/'results/preflight/r201m_MSVR_sampling_diversity_actual_20261009.json'
assert not proof.exists()
load=lambda path:json.loads(path.read_text(encoding='utf-8'))
actual=load(p/'results/preflight/r201m_primary_margin06_MSVR310_actual_session_20261008.json')
assert actual['exit_code']==0 and actual['successful_updates']==2000
lanes={};hashes={}
for variant in ('frequency_shared','axis_shared'):
    folder=root/'training'/('MSVR310_r201m_'+variant+'_s42')
    trained=load(folder/'result.json')
    rows=[json.loads(line) for line in (folder/'batch_orders.jsonl').read_text(encoding='utf-8').splitlines()]
    assert trained['status']=='COMPLETE' and trained['train_records']==1032 and trained['classes']==155
    assert len(rows)==trained['optimizer_steps']==1000 and trained['amp_skipped_steps']==0
    assert all(b['sampling_identities']==b['sampling_instances']==8 and b['sampling_batch']==64 for b in rows)
    lanes[variant]=rows
    for name in ('result.json','batch_orders.jsonl','official_split_manifest.json'):
        path=folder/name;hashes[str(path.relative_to(p).as_posix())]=hashlib.sha256(path.read_bytes()).hexdigest()
first=lanes['frequency_shared'];second=lanes['axis_shared']
assert [(b['epoch'],b['step'],b['names']) for b in first]==[(b['epoch'],b['step'],b['names']) for b in second]
observations=defaultdict(set)
def fields(name):
    assert name.endswith('.jpg') and name[4:6]=='_s' and name[9:11]=='_v' and name[12]=='_'
    pid,scene,view=int(name[:4]),int(name[6:9]),int(name[11]);assert 0<=view<=7
    return pid,scene,view
for b in first:
    for name in b['names']:observations[fields(name)[0]].add(name)
assert len(observations)==155 and sum(map(len,observations.values()))==1032
identities=[]
for pid,names in sorted(observations.items()):
    identities.append(dict(pid=pid,unique_records=len(names),unique_scenes=len({fields(n)[1] for n in names}),
        unique_views=len({fields(n)[2] for n in names}),requires_replacement_at_K8=len(names)<8))
series=[]
for epoch in range(1,51):
    group=[b for b in first if b['epoch']==epoch];assert len(group)==20
    chunks=[]
    for b in group:
        chunked=[b['names'][i:i+8] for i in range(0,64,8)]
        assert len({fields(c[0])[0] for c in chunked})==8
        assert all(len({fields(n)[0] for n in c})==1 for c in chunked)
        chunks.extend(chunked)
    unique=[len(set(c)) for c in chunks]
    series.append(dict(epoch=epoch,batches=20,identity_chunks=len(chunks),sample_slots=1280,
        unique_filename_slots=sum(unique),repeated_filename_slots=1280-sum(unique),
        repeated_filename_slot_fraction=1-sum(unique)/1280,
        mean_unique_records_per_identity_chunk=sum(unique)/len(unique),
        one_unique_record_identity_chunks=sum(n==1 for n in unique),
        one_scene_identity_chunks=sum(len({fields(n)[1] for n in c})==1 for c in chunks),
        one_view_identity_chunks=sum(len({fields(n)[2] for n in c})==1 for c in chunks)))
identity_table=root/'normal_analysis/training_identity_diversity.csv'
epoch_table=root/'normal_analysis/training_sampling_diversity.csv'
for path,rows in ((identity_table,identities),(epoch_table,series)):
    with path.open('x',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
for name in ('data/datasets/msvr310.py','data/datasets/sampler.py','configs/MSVR310/DeMo.yml','run_r201l_uniform_k8.py','run_r201m_primary_margin06.py'):
    hashes[name]=hashlib.sha256((p/name).read_bytes()).hexdigest()
summary=dict(training_identities=155,training_records=1032,
    record_count_histogram=dict(sorted(Counter(r['unique_records'] for r in identities).items())),
    identities_below_K8=sum(r['unique_records']<8 for r in identities),
    identities_below_original_K4=sum(r['unique_records']<4 for r in identities),
    one_record_identities=sum(r['unique_records']==1 for r in identities),
    one_scene_identities=sum(r['unique_scenes']==1 for r in identities),
    original_MSVR_config_PK='P16K4 at B64',current_M_PK='P8K8 at B64',
    batch_rows=1000,identity_chunks=sum(r['identity_chunks'] for r in series),
    sample_slots=sum(r['sample_slots'] for r in series),
    repeated_filename_slots=sum(r['repeated_filename_slots'] for r in series),
    repeated_filename_slot_fraction=sum(r['repeated_filename_slots'] for r in series)/sum(r['sample_slots'] for r in series),
    mean_unique_records_per_identity_chunk=sum(r['unique_filename_slots'] for r in series)/sum(r['identity_chunks'] for r in series),
    one_record_chunk_fraction=sum(r['one_unique_record_identity_chunks'] for r in series)/sum(r['identity_chunks'] for r in series),
    one_scene_chunk_fraction=sum(r['one_scene_identity_chunks'] for r in series)/sum(r['identity_chunks'] for r in series),
    selected_axis_best_epoch3=series[2],late_epoch50=series[-1])
tables={str(path.relative_to(p).as_posix()):hashlib.sha256(path.read_bytes()).hexdigest() for path in (identity_table,epoch_table)}
record=dict(status='ACTUAL_CLOSED_MSVR_PAIRED_PK8_TRAINING_FILENAME_DIVERSITY',
    at=datetime.now().astimezone().isoformat(timespec='seconds'),summary=summary,
    paired_all_epoch_step_filename_orders=True,input_sha256=hashes,tables_sha256=tables,
    new_neural_calls=0,new_optimizer_updates=0,
    limits='Only filenames observed in the closed full-training logs; observed union matches verified official counts, not a new directory scan. Same logged filename may have distinct random augmentation, so repetitions are not identical tensors, illegal positives or dataset corruption. Scene/view are parsed exactly as the installed MSVR source. Repetition is a property of source coverage and K8 sampling, not by itself a cause of late degradation. Both controls have identical repeated-record schedules; this does not identify a dual-axis-specific cause. The old K4 and current K8 sampler comparison is already completed historical evidence; do not pretend restoring K4 alone is untried. No new sampler simulation, training/GT inference or holdout.')
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==sha for n,sha in hashes.items())
proof.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=record['status'],summary=summary)))
