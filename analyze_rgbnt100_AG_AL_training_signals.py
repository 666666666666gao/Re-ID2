"""Read actual full-training telemetry; separate it from selected Q/G routing."""
import csv
import json
import math
from pathlib import Path

P = Path(__file__).resolve().parent
METRICS = ('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
RELATIONS = ('R','N','T','RN','RT','NT','RNT')


def main():
    rows=[]
    summaries=[]
    sources=[]
    for label,tag in (('AG','r201ag_RGB_coordinate_metric_20261012'),('AL','r201al_RGB_route_temperature_20261013')):
        for variant in ('frequency_shared','axis_shared'):
            root=P/'results'/tag/'RGBNT100/training'/('RGBNT100_r201'+label.lower()+'_'+variant+'_s42')
            result=json.loads((root/'result.json').read_text(encoding='utf-8'))
            assert result['status']=='COMPLETE' and result['epochs']==50 and result['optimizer_steps']==6563 and result['amp_skipped_steps']==0
            assert (result['train_records'],result['query_records'],result['gallery_records'],result['descriptor_dim'])==(8675,1715,8575,5120)
            with (root/'epochs.csv').open(encoding='utf-8',newline='') as f:
                epochs=list(csv.DictReader(f))
            assert [int(r['epoch']) for r in epochs]==list(range(1,51))
            by_epoch={i:[] for i in range(1,51)}
            count=0
            with (root/'batch_orders.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    r=json.loads(line);count+=1
                    assert r['step']==count and r['optimizer_updated'] and r['partial_loss_effective']==0
                    assert r['sampling_batch']==64 and r['freeze_identity_encoder']
                    assert r['primary_full_metric']=='unit5120_margin06_batch_hard_triplet' and r['primary_margin']==.6
                    assert 0<=r['primary_margin_violations']<=64
                    assert abs(r['main_CE_effective']-.25*r['main_CE_unweighted'])<1e-8
                    by_epoch[r['epoch']].append(r)
            assert count==6563
            model=label+'_'+variant
            series=[]
            for ep in epochs:
                epoch=int(ep['epoch']);batch=by_epoch[epoch]
                mean=lambda key:math.fsum(r[key] for r in batch)/len(batch)
                bound=lambda key:math.fsum(r['final_descriptor_bound_full'][key] for r in batch)/len(batch)
                vector=lambda key,size:[math.fsum(r[key][i] for r in batch)/len(batch) for i in range(size)]
                assert abs(mean('loss')-float(ep['loss']))<1e-8 and mean('loss')==mean('full_loss')
                bands=vector('route_band_mass_mean',3)
                relations=vector('route_relation_mass_mean',7)
                gates=vector('full_gate_mean',3)
                scales=vector('residual_scales',3)
                ce=mean('main_CE_effective');triplet=mean('primary_triplet_loss')
                assert ce+triplet>0
                row=dict(model=model,epoch=epoch,batches=len(batch),loss=mean('loss'),
                    main_CE_unweighted=mean('main_CE_unweighted'),main_CE_effective=ce,
                    primary_triplet_loss=triplet,primary_margin_violation_fraction=mean('primary_margin_violations')/64,
                    zero_primary_triplet_batch_fraction=sum(r['primary_triplet_loss']==0 for r in batch)/len(batch),
                    CE_share_of_primary_loss=ce/(ce+triplet),
                    positive_distance_mean=mean('primary_positive_distance_mean'),negative_distance_mean=mean('primary_negative_distance_mean'),
                    full_training_route_entropy_mean=mean('route_entropy_mean'),
                    full_training_route_joint_product_KL_mean=mean('route_nonindependence_mean'),
                    low_mass=bands[0],mid_mass=bands[1],high_mass=bands[2],
                    dominant_mean_relation=RELATIONS[max(range(7),key=relations.__getitem__)],
                    dominant_mean_relation_mass=max(relations),gate_M=gates[0],gate_F=gates[1],gate_I=gates[2],
                    scale_M=scales[0],scale_F=scales[1],scale_I=scales[2],
                    contribution_regression_loss=mean('contribution_loss_raw'),contribution_zero_loss=mean('contribution_zero_loss_raw'),
                    contribution_beats_zero_batch_fraction=sum(r['contribution_loss_raw']<r['contribution_zero_loss_raw'] for r in batch)/len(batch),
                    bound_clipped_fraction=bound('clipped_fraction'),candidate_shift_mean=bound('candidate_shift_mean'),
                    bounded_shift_mean=bound('bounded_shift_mean'),
                    bounded_shift_max=max(r['final_descriptor_bound_full']['bounded_shift_max'] for r in batch),
                    **{key:float(ep[key]) for key in METRICS})
                assert row['bounded_shift_max']<=.1+1e-6
                rows.append(row);series.append(row)
            selected=result['best']['epoch']
            assert max(epochs,key=lambda r:float(r['mAP']))['epoch']==str(selected)
            summaries.append(dict(model=model,selected_epoch=selected,
                first=series[0],selected=series[selected-1],last=series[-1],
                actual_batches=count,
                zero_primary_triplet_batches=sum(r['primary_triplet_loss']==0 for group in by_epoch.values() for r in group),
                contribution_beats_zero_batches=sum(r['contribution_loss_raw']<r['contribution_zero_loss_raw'] for group in by_epoch.values() for r in group)))
            sources.append(dict(model=model,batches=str((root/'batch_orders.jsonl').relative_to(P)),epochs=str((root/'epochs.csv').relative_to(P))))
    out=P/'results/rgbnt100_AG_AL_training_signals_CPU_20261013';out.mkdir(exist_ok=False)
    with (out/'full_training_signals.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    proof=dict(status='ACTUAL_AG_AL_FULL_TRAINING_TELEMETRY_CPU_COMPLETE',models=4,epoch_rows=len(rows),batch_rows=26252,
        sources=sources,summaries=summaries,new_neural_calls=0,new_optimizer_updates=0,
        trace_scope='run_r201m_primary_margin06.step.losses records routing/gates during the first full-view primary-loss callback, before partial forward and Adam. The inherited N callback separately reads descriptor_bound_reports[(11,True)].',
        limits='Training full-view batch telemetry, not every-epoch official Q/G inference or per-query routing. '
        'Batch routing nonindependence is the saved KL expression; selected-Q/G reports instead store L1. '
        'Loss proportions are not gradient proportions; zero-loss counts do not prove absent auxiliary gradients. '
        'Prediction versus zero is the same stopped-gradient batch contribution target, not AP or calibrated test utility. '
        'Single paired seed42; training dynamics do not establish causal failure or a successful new initialization. '
        'Corrects the earlier broad no-routing-curve wording: saved full-training curves can be reconstructed, while per-epoch full-Q/G routing was not stored.')
    (out/'result.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fields=('model','epoch','primary_triplet_loss','primary_margin_violation_fraction','main_CE_effective',
        'contribution_beats_zero_batch_fraction','bound_clipped_fraction','bounded_shift_mean','full_training_route_entropy_mean','mAP','Rank-1')
    print(json.dumps([{key:r[key] for key in fields} for s in summaries for r in (s['first'],s['selected'],s['last'])],ensure_ascii=False))


if __name__=='__main__':
    main()
