"""CPU-only schedule preflight using the complete closed X50 batch sequence."""
from pathlib import Path
from datetime import datetime
import hashlib,json,sys

repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo))
from neighbor_identity_schedule import rewrite_epoch
pf=repo/'results/preflight';out=pf/'r201ac_RGB_neighbors_schedule_CPU_actual_20261012.json'
assert not out.exists()
bank_path=repo/'results/r201ab_RGB_training_neighbors_20261012/result.json'
bank=json.loads(bank_path.read_text());assert bank['status']=='COMPLETE_RGBNT100_TRAIN_ONLY_NEIGHBOR_BANK'
labels={row['name']:bank['training_labels'][str(row['identity'])] for row in bank['train_records_metadata']}
neighbors={int(pid):row for pid,row in bank['nearest_identity_labels'].items()}
source=repo/'results/r201x_RGB_auxiliary_fp32_20261012/RGBNT100/training/RGBNT100_r201x_axis_shared_s42/batch_orders.jsonl'
data=source.read_bytes();rows=[json.loads(line) for line in data.decode('utf-8').splitlines()]
assert len(rows)==6563 and {r['epoch'] for r in rows}==set(range(1,51))
epochs=[];planned=[]
for epoch in range(1,51):
    original=[r for r in rows if r['epoch']==epoch]
    names=[n for row in original for n in row['names']]
    rewritten,stats=rewrite_epoch(names,labels,neighbors)
    assert stats['batches']==len(original)
    epochs.append(dict(epoch=epoch,**stats))
    for i,row in enumerate(original):
        planned.append(dict(epoch=epoch,step=row['step'],names=rewritten[i*64:(i+1)*64]))
assert len(planned)==6563
root=repo/'results/r201ac_RGB_neighbors_schedule_CPU_20261012';root.mkdir(exist_ok=False)
with (root/'planned_batch_orders.jsonl').open('w',encoding='utf-8') as f:
    for row in planned:f.write(json.dumps(row,ensure_ascii=False)+'\n')
anchors=sum(r['identity_anchors'] for r in epochs)
before=sum(r['top3_covered_before'] for r in epochs);after=sum(r['top3_covered_after'] for r in epochs)
assert anchors==52504 and after>before
result=dict(status='COMPLETE_CPU_CLOSED_X_P8K8_TRAIN_NEIGHBOR_SWAP_REPLAY',at=datetime.now().astimezone().isoformat(timespec='seconds'),
    epochs=epochs,total_batches=6563,total_record_positions=6563*64,identity_anchors=anchors,
    top3_coverage_before=before/anchors,top3_coverage_after=after/anchors,
    swaps=sum(r['swaps'] for r in epochs),official_training_labels_only=True,
    each_epoch_record_multiset_exact=True,P=8,K=8,new_random_draws=0,
    new_neural_calls=0,new_optimizer_updates=0,
    original_orders_sha256=hashlib.sha256(data).hexdigest(),bank_result_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),
    limits='Only CPU schedule feasibility on closed real data. Runtime sampler integration, fresh source review, actual three-step updates and50 training not implemented or launched. Coverage is not retrieval gain. Future runtime must consume original sampler RNG before deterministic reordering; same epochs, update counts, record positions and P8K8. No test labels select negatives, no new loss/architecture/K201/missing专项/49.')
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(root/'result.json').write_bytes(out.read_bytes())
print(json.dumps({k:result[k] for k in ('status','total_batches','identity_anchors','top3_coverage_before','top3_coverage_after','swaps','new_neural_calls')}),flush=True)
