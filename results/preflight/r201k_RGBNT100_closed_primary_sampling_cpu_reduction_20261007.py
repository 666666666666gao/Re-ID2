"""Reduce two already-complete K100 training logs; stdlib only, no model or GPU."""
import csv
from collections import Counter
from datetime import datetime
import hashlib,json,math
from pathlib import Path

project=Path(__file__).resolve().parents[2]
output=project/'results/r201k_relation_local_pi_20261007/RGBNT100/closed_training_diagnostic'
assert not output.exists()
fields=('loss','primary_triplet_loss','primary_margin_violations','primary_positive_distance_mean','primary_negative_distance_mean')
metrics=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
summaries=[];inputs={};epoch_rows=[];window_rows=[];paired_orders=[]
for variant in ('frequency_shared','axis_shared'):
    folder=project/'results/r201k_relation_local_pi_20261007/RGBNT100/training'/('RGBNT100_r201k_'+variant+'_s42')
    paths={name:folder/name for name in ('result.json','epochs.csv','batch_orders.jsonl','official_split_manifest.json')}
    data=json.loads(paths['result.json'].read_text(encoding='utf-8'))
    assert data['status']=='COMPLETE' and data['epochs']==50 and data['arguments']['seed']==42
    assert data['steps']==data['optimizer_steps'] and data['amp_skipped_steps']==0
    assert (data['train_records'],data['query_records'],data['gallery_records'])==(8675,1715,8575)
    assert data['training_coverage']==dict(eligible=8675,visited=8675,unvisited=[]) and data['training_heldout_identities']==0
    assert '  NUM_INSTANCE: 16' in data['config'].splitlines()
    manifest=json.loads(paths['official_split_manifest.json'].read_text(encoding='utf-8'))
    name_to_identity={r['name']:r['identity'] for r in manifest['train']}
    identities=set(name_to_identity.values());assert len(name_to_identity)==8675 and len(identities)==50
    epochs=list(csv.DictReader(paths['epochs.csv'].open(encoding='utf-8',newline='')))
    assert [int(r['epoch']) for r in epochs]==list(range(1,51))
    by_epoch={ep:[] for ep in range(1,51)};orders=[];all_seen=set();all_pairs=set();coidentity_batches=Counter()
    with paths['batch_orders.jsonl'].open(encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line)
            assert row['optimizer_updated'] and row['primary_margin']==.3 and row['primary_metric_coefficient']==1.
            assert row['loss']==row['full_loss'] and row['partial_loss_effective']==0 and row['auxiliary_original_soft_triplet']
            assert 0<=row['primary_margin_violations']<=64 and all(math.isfinite(row[k]) for k in fields)
            names=row['names'];counts=Counter(name_to_identity[n] for n in names)
            assert len(names)==64 and len(counts)==4 and sorted(counts.values())==[16]*4
            ids=sorted(counts);all_seen.update(names)
            for i,left in enumerate(ids):
                for right in ids[i+1:]:
                    all_pairs.add((left,right));coidentity_batches[(left,right)]+=1
            by_epoch[row['epoch']].append({k:row[k] for k in fields})
            orders.append((row['epoch'],row['step'],names,row['partial_set']))
    assert len(orders)==data['steps']==sum(map(len,by_epoch.values()))
    assert set(all_seen)==set(name_to_identity)
    paired_orders.append(orders)
    windows={}
    for label,lo,hi in (('early_1_5',1,5),('middle_6_25',6,25),('late_26_50',26,50)):
        members=[r for ep in range(lo,hi+1) for r in by_epoch[ep]]
        full=math.fsum(r['loss'] for r in members)/len(members)
        primary=math.fsum(r['primary_triplet_loss'] for r in members)/len(members)
        values=dict(batches=len(members),full_loss_mean=full,primary_triplet_mean=primary,
            primary_fraction_of_total_loss=primary/full,
            violated_sample_fraction=math.fsum(r['primary_margin_violations'] for r in members)/(64*len(members)),
            zero_primary_batch_fraction=sum(r['primary_margin_violations']==0 for r in members)/len(members),
            positive_distance_mean=math.fsum(r['primary_positive_distance_mean'] for r in members)/len(members),
            negative_distance_mean=math.fsum(r['primary_negative_distance_mean'] for r in members)/len(members))
        windows[label]=values;window_rows.append(dict(variant=variant,window=label,**values))
    previous_steps=0
    for ep in epochs:
        number=int(ep['epoch']);members=by_epoch[number]
        assert len(members)==int(ep['steps'])-previous_steps
        previous_steps=int(ep['steps'])
        means={k:math.fsum(r[k] for r in members)/len(members) for k in fields}
        assert abs(means['loss']-float(ep['loss']))<1e-8
        epoch_rows.append(dict(variant=variant,epoch=number,batches=len(members),**means,
            zero_primary_batch_fraction=sum(r['primary_margin_violations']==0 for r in members)/len(members),
            violated_sample_fraction=means['primary_margin_violations']/64,
            remaining_loss_aggregate=means['loss']-means['primary_triplet_loss'],**{k:float(ep[k]) for k in metrics}))
    best=max(epochs,key=lambda r:float(r['mAP']));assert int(best['epoch'])==data['best']['epoch']
    assert all(abs(float(best[k])-data['best'][k])<1e-8 for k in metrics)
    summaries.append(dict(variant=variant,steps=data['steps'],best_epoch=int(best['epoch']),
        best={k:float(best[k]) for k in metrics},epoch50={k:float(epochs[-1][k]) for k in metrics},windows=windows,
        sample_coverage=dict(batch_size=64,identities_per_batch=4,instances_per_identity=16,negative_identity_candidates_per_query_in_batch=3,
            training_identities=len(identities),visited_training_records=len(all_seen),
            possible_unordered_identity_pairs=len(identities)*(len(identities)-1)//2,
            observed_unordered_identity_pairs=len(all_pairs),
            coidentity_pair_batch_count_min=min(coidentity_batches.values()),coidentity_pair_batch_count_max=max(coidentity_batches.values()))))
    inputs.update({p.relative_to(project).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths.values()})
assert paired_orders[0]==paired_orders[1] and len(epoch_rows)==100
parent=project/'results/preflight/r201ik_closed_primary_mining_readout_actual_20261007.json'
previous=json.loads(parent.read_text(encoding='utf-8'));assert previous['cases']==10 and previous['epochs']==500
inputs[parent.relative_to(project).as_posix()]=dict(bytes=parent.stat().st_size,sha256=hashlib.sha256(parent.read_bytes()).hexdigest())
comparators=[{k:r[k] for k in ('revision','dataset','variant','best_mAP','final_mAP','windows')} for r in previous['summaries'] if r['revision']=='k' and r['variant']=='axis_shared']
output.mkdir()
for name,rows in (('epoch_means.csv',epoch_rows),('window_means.csv',window_rows)):
    with (output/name).open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
value=dict(status='ACTUAL_CLOSED_K100_PRIMARY_HINGE_AND_OFFICIAL_BATCH_SAMPLING_CPU_COMPLETE',completed_at=datetime.now().isoformat(timespec='seconds'),
    cases=2,epochs=100,successful_updates=sum(r['steps'] for r in summaries),paired_sampling_exact=True,inputs=inputs,
    summaries=summaries,earlier_closed_K201_MSVR_axis_comparators=comparators,new_neural_calls=0,new_optimizer_updates=0,
    limits='Existing complete K42 official training text only, independent consistency with epoch losses/steps and official filename identities; no RAW, image, model or gradient replay. Primary loss small or inactive is not proof of zero whole-model gradient, nor sole cause of retrieval decline. Remaining loss aggregate is not separately measured CE/auxiliary/contribution. P4K16 per-batch negative identity count does not imply missing global pair coverage. Earlier comparator windows use the prior audit definition violated_sample_fraction_mean. No new sampling/loss change, no running N5 query/source mutation or +2 completion inferred.')
(output/'result.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:value[k] for k in ('status','completed_at','cases','epochs','successful_updates','new_neural_calls','new_optimizer_updates')},ensure_ascii=False))
for r in summaries:print(json.dumps(dict(variant=r['variant'],best_epoch=r['best_epoch'],best_mAP=r['best']['mAP'],epoch50_mAP=r['epoch50']['mAP'],late=r['windows']['late_26_50'],sample_coverage=r['sample_coverage']),ensure_ascii=False))
