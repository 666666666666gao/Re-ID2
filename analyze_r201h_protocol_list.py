"""Reduce four closed H references against the paired G/unit/original controls."""
import csv
from datetime import datetime
import json
import math
from pathlib import Path

from analyze_identity_coordinate_three_normal import METRICS,load,query_rows,table

PROJECT=Path(__file__).resolve().parent


def main():
    root=PROJECT/'results/r201h_protocol_list_20261006'
    intake=load(PROJECT/'results/preflight/r201h_protocol_list_terminal_text_intake_20261006.json')
    assert intake['status']=='ACTUAL_H201_FOUR_FULL50_NORMAL_GT_TEXT_VERIFIED'
    controller=load(root/'controller_result.json')
    assert controller['status']=='COMPLETE' and controller['controls']==4 and controller['successful_updates']==10588
    assert controller['additional_epochs']==200 and controller['native_updates']==12 and controller['paired_sampling_exact']
    folders={'original_DeMo':PROJECT/'results/full_official_baselines_20261004/training/RGBNT201_demo_s42'}
    for variant in ('frequency_shared','axis_shared'):
        folders['G_'+variant]=PROJECT/'results/r201g_normal_priority_20261006/training'/('RGBNT201_r201g_'+variant+'_s42')
        folders['unit_'+variant]=PROJECT/'results/rgbnt201_identity_outlet_r201c_20261005/training'/('RGBNT201_identity_'+variant+'_narrow_s42')
        for reference in ('label','camera'):
            folders['H_'+reference+'_'+variant]=root/'training'/('RGBNT201_r201h_'+reference+'_'+variant+'_s42')
    manifest=load(folders['original_DeMo']/'official_split_manifest.json')
    reads,training,orders={}, {}, {}
    metrics,groups,curves,rank_coverage=[],[],[],[]
    for model,folder in folders.items():
        data=load(folder/'result.json')
        assert data['status']=='COMPLETE' and data['epochs']==50 and data['steps']==data['optimizer_steps']==2647
        assert data['amp_skipped_steps']==data['training_heldout_identities']==0 and data['descriptor_dim']==5120
        assert (data['train_records'],data['query_records'],data['gallery_records'])==(3951,836,836)
        assert data['training_coverage']==dict(eligible=3951,visited=3951,unvisited=[])
        assert load(folder/'official_split_manifest.json')==manifest
        rows,summary=query_rows(folder,'RGBNT201')
        epochs=list(csv.DictReader((folder/'epochs.csv').open(encoding='utf-8',newline='')))
        assert [int(row['epoch']) for row in epochs]==list(range(1,51))
        selected=max(epochs,key=lambda row:float(row['mAP']))
        assert int(selected['epoch'])==data['best']['epoch']
        assert all(abs(float(selected[key])-summary[key])<1e-8 and abs(data['full_metrics'][key]-summary[key])<1e-8 for key in METRICS)
        reads[model]=(rows,summary)
        metrics.append(dict(model=model,selected_epoch=data['best']['epoch'],**{key:summary[key] for key in METRICS}))
        groups.extend(dict(model=model,grouping=axis,**value) for axis,values in summary['groups'].items() for value in values)
        training[model]=dict(selected_epoch=data['best']['epoch'],parameters=data['parameters'],trainable_parameters=data['trainable_parameters'],
            descriptor_dim=5120,total_budget_epochs=50 if model=='original_DeMo' else 100,
            final_epoch={key:float(epochs[-1][key]) for key in METRICS},runtime=data['runtime'],peak_memory=data['peak_memory'])
        if model!='original_DeMo':
            batches=[json.loads(line) for line in (folder/'batch_orders.jsonl').read_text(encoding='utf-8').splitlines()]
            assert len(batches)==2647 and all(row['optimizer_updated'] for row in batches)
            orders[model]=[(r['epoch'],r['step'],r['names'],r['partial_set']) for r in batches]
        if model.startswith('H_'):
            reference='camera' if model.startswith('H_camera_') else 'label'
            audit=load(folder/'normal_cpu_audit.json')
            assert audit['status']=='PASS' and audit['max_metric_error']<1e-8 and audit['full_training_coverage'] and audit['CMC50_and_per_query_and_groups']
            assert load(folder/'normal_local_archive.json')['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
            assert all(r['reference_mask']==reference and r['primary_full_metric']=='protocol_smooth_AP_unit5120'
                and r['loss']==r['full_loss'] and r['partial_loss_effective']==r['partial_CE_weight']==r['partial_triplet_weight']==0
                and r['temperature']==.01 and r['metric_coefficient']==1. and r['all64_observations_in_classification'] for r in batches)
            if reference=='label':assert all(r['valid_AP_queries']==64 and not r['zero_valid_batch'] for r in batches)
            for epoch in epochs:
                batch=[r for r in batches if r['epoch']==int(epoch['epoch'])]
                means={key:math.fsum(r[key] for r in batch)/len(batch) for key in ('loss','full_loss','full_AP_loss','partial_ce','cross_triplet')}
                assert abs(means['loss']-float(epoch['loss']))<1e-8
                curves.append(dict(model=model,epoch=int(epoch['epoch']),batches=len(batch),**means,**{key:float(epoch[key]) for key in METRICS}))
                rank_coverage.append(dict(model=model,reference=reference,epoch=int(epoch['epoch']),batches=len(batch),observations=64*len(batch),
                    valid_AP_queries=sum(r['valid_AP_queries'] for r in batch),invalid_AP_queries=sum(64-r['valid_AP_queries'] for r in batch),
                    zero_valid_batches=sum(r['zero_valid_batch'] for r in batch),used_positive_pairs=sum(r['used_positive_pairs'] for r in batch)))
    assert all(order==orders['G_axis_shared'] for order in orders.values())
    assert len({(v['parameters'],v['trainable_parameters']) for k,v in training.items() if k!='original_DeMo'})==1
    definitions=[('H_camera_axis_shared','H_label_axis_shared'),('H_camera_frequency_shared','H_label_frequency_shared'),
        ('H_camera_axis_shared','H_camera_frequency_shared'),('H_label_axis_shared','H_label_frequency_shared')]
    for reference in ('label','camera'):
        for variant in ('frequency_shared','axis_shared'):
            definitions.extend([('H_'+reference+'_'+variant,'G_'+variant),('H_'+reference+'_'+variant,'original_DeMo')])
        definitions.append(('H_'+reference+'_axis_shared','unit_axis_shared'))
    comparisons,pairs=[],[]
    for improved,reference in definitions:
        arows,a=reads[improved];brows,b=reads[reference]
        assert all(all(x[key]==y[key] for key in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')) for x,y in zip(arows,brows))
        delta={key:a[key]-b[key] for key in METRICS}
        comparisons.append(dict(improved=improved,reference=reference,**delta,both_plus2=delta['mAP']>=2 and delta['Rank-1']>=2,
            rank1_rescued=sum(int(x['Rank-1'])==1 and int(y['Rank-1'])==0 for x,y in zip(arows,brows)),
            rank1_harmed=sum(int(x['Rank-1'])==0 and int(y['Rank-1'])==1 for x,y in zip(arows,brows)),
            AP_improved=sum(float(x['AP'])>float(y['AP']) for x,y in zip(arows,brows)),AP_worsened=sum(float(x['AP'])<float(y['AP']) for x,y in zip(arows,brows))))
        pairs.extend(dict(improved=improved,reference=reference,**{key:x[key] for key in ('query_index','name','identity','camera','scene')},
            delta_AP_pp=100*(float(x['AP'])-float(y['AP'])),delta_INP_pp=100*(float(x['INP'])-float(y['INP'])),
            delta_rank1=int(x['Rank-1'])-int(y['Rank-1'])) for x,y in zip(arows,brows))
    assert len(metrics)==9 and len(comparisons)==14 and len(pairs)==11704 and len(curves)==len(rank_coverage)==200
    output=root/'normal_analysis';output.mkdir(exist_ok=False)
    for name,rows in (('six_metrics',metrics),('group_metrics',groups),('comparisons',comparisons),('paired_query_changes',pairs),
                      ('training_loss_curves',curves),('training_AP_reference_coverage',rank_coverage)):
        table(output/(name+'.csv'),rows)
    result=dict(status='ACTUAL_H201_FOUR_FULL50_PROTOCOL_REFERENCE_CPU_READOUT',completed_at=datetime.now().isoformat(timespec='seconds'),
        model_results=9,comparisons=comparisons,training=training,paired_query_rows=11704,new_epochs=200,new_batches=10588,
        six_metrics_CMC50_identity_camera_scene_verified=True,all_eight_augmented_training_orders_equal=True,
        new_neural_calls=0,new_optimizer_updates=0,
        limits='RGBNT201 seed42 normal only, original50 versus newexperts original50+additional50; benchmark mAP-best/earliesttie, not independent untouched test. Camera-vs-label AP control changes eligibility only; no novelty/causal/collaboration/three-dataset/missing/multiseed guarantee. All training observations remain CE/aux and potential galleries despite excluded AP query rows. Current ordinary control is mixed-source expert, not literal one-frequency-branch.')
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','completed_at','model_results','paired_query_rows','new_batches')}))


if __name__=='__main__':main()
