"""CPU-only reduction of existing complete training logs; no RAW or model load."""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

project=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
output=project/'results/preflight/r201ik_closed_primary_mining_readout_actual_20261007.json'
assert not output.exists()
summaries=[];inputs={}
fields=('loss','full_loss','primary_triplet_loss','primary_margin_violations','primary_positive_distance_mean','primary_negative_distance_mean')
for revision,datasets in (('i',('RGBNT201','MSVR310','RGBNT100')),('k',('RGBNT201','MSVR310'))):
    tag={'i':'r201i_primary_margin_20261006','k':'r201k_relation_local_pi_20261007'}[revision]
    for dataset in datasets:
        for variant in ('frequency_shared','axis_shared'):
            folder=project/'results'/tag/dataset/'training'/(dataset+'_r201'+revision+'_'+variant+'_s42')
            paths={name:folder/name for name in ('result.json','epochs.csv','batch_orders.jsonl')}
            data=json.loads(paths['result.json'].read_text(encoding='utf-8'))
            assert data['status']=='COMPLETE' and data['epochs']==50 and data['arguments']['seed']==42
            epochs=list(csv.DictReader(paths['epochs.csv'].open(encoding='utf-8',newline='')))
            assert [int(row['epoch']) for row in epochs]==list(range(1,51))
            by_epoch={i:[] for i in range(1,51)}
            with paths['batch_orders.jsonl'].open(encoding='utf-8') as handle:
                for line in handle:
                    row=json.loads(line)
                    assert row['optimizer_updated'] and row['primary_margin']==.3 and row['primary_metric_coefficient']==1.
                    assert 0<=row['primary_margin_violations']<=64 and row['loss']==row['full_loss']
                    by_epoch[row['epoch']].append({key:row[key] for key in fields})
            assert sum(map(len,by_epoch.values()))==data['optimizer_steps']==data['steps']
            means=[]
            for ep in epochs:
                members=by_epoch[int(ep['epoch'])];assert members
                mean={key:math.fsum(row[key] for row in members)/len(members) for key in fields}
                assert abs(mean['loss']-float(ep['loss']))<1e-8
                means.append(dict(epoch=int(ep['epoch']),batches=len(members),**mean,mAP=float(ep['mAP']),Rank1=float(ep['Rank-1']),
                    zero_primary_batch_fraction=sum(row['primary_margin_violations']==0 for row in members)/len(members),
                    mean_primary_violated_sample_fraction=mean['primary_margin_violations']/64,
                    remaining_loss_aggregate=mean['loss']-mean['primary_triplet_loss']))
            windows={}
            for label,lo,hi in (('early_1_5',1,5),('middle_6_25',6,25),('late_26_50',26,50)):
                members=[row for epoch in range(lo,hi+1) for row in by_epoch[epoch]]
                windows[label]=dict(batches=len(members),primary_triplet_mean=math.fsum(r['primary_triplet_loss'] for r in members)/len(members),
                    violated_sample_fraction_mean=math.fsum(r['primary_margin_violations'] for r in members)/(len(members)*64),
                    zero_primary_batch_fraction=sum(r['primary_margin_violations']==0 for r in members)/len(members))
            best=max(epochs,key=lambda row:float(row['mAP']))
            assert int(best['epoch'])==data['best']['epoch']
            summaries.append(dict(revision=revision,dataset=dataset,variant=variant,steps=data['steps'],
                best_epoch=int(best['epoch']),best_mAP=float(best['mAP']),best_Rank1=float(best['Rank-1']),
                final_mAP=means[-1]['mAP'],final_Rank1=means[-1]['Rank1'],windows=windows,epoch_means=means))
            inputs.update({str(p.relative_to(project)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths.values()})
assert len(summaries)==10 and sum(len(row['epoch_means']) for row in summaries)==500
value=dict(status='ACTUAL_CLOSED_I3_K2_PRIMARY_HINGE_MINING_LOG_CPU_REDUCTION',completed_at=datetime.now().isoformat(timespec='seconds'),
    cases=10,epochs=500,inputs=inputs,summaries=summaries,new_neural_calls=0,new_optimizer_updates=0,
    limits='Existing training logs, one expert seed42 and benchmark-selected epoch; no feature or gradient replay, no current K100 data. Primary hinge violations are hard-mined training samples out of64. Remaining aggregate includes CE/auxiliary/contribution, not a measured single loss or gradient. Low violation counts do not by themselves prove overfit, negative transfer or the sole cause of late retrieval degradation.')
output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for row in summaries:
    print(json.dumps({k:row[k] for k in ('revision','dataset','variant','best_epoch','best_mAP','final_mAP','windows')},ensure_ascii=False))
