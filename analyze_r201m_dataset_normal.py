"""CPU reduction of M primary margin0.6 versus original DeMo and exact uniform-L recipe."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from analyze_identity_coordinate_three_normal import COUNTS,METRICS,load,query_rows,table

PROJECT=Path(__file__).resolve().parent
SAMPLER_K={'RGBNT201':8,'MSVR310':4,'RGBNT100':16}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=('MSVR310','RGBNT100'),required=True)
    dataset=parser.parse_args().dataset;pf=PROJECT/'results/preflight'
    review=load(pf/'r201m_primary_margin06_source_review_20261008.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    assert {'results/preflight/r201m_dataset_normal_complete_intake_20261008.py','analyze_r201m_dataset_normal.py'} <= set(review['sources_sha256'])
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
    intake=load(pf/('r201m_'+dataset+'_normal_terminal_text_intake_20261008.json'))
    assert intake['status']=='ACTUAL_M_DATASET_TWO_FULL50_NORMAL_GT_TEXT_VERIFIED' and intake['dataset']==dataset
    root=PROJECT/'results/r201m_primary_margin06_20261008'/dataset
    controller=load(root/'controller_result.json');train,count,gallery,old_steps=COUNTS[dataset]
    steps=load(pf/'r201l_uniform_K8_sampler_cpu_replay_20261007.json')['datasets'][dataset]['8']['total_steps']
    assert controller['status']=='COMPLETE' and controller['dataset']==dataset and controller['controls']==2
    assert controller['successful_updates']==2*steps and controller['additional_epochs']==100 and controller['native_updates']==6
    assert controller['paired_sampling_exact'] and controller['normal_archives_local_verified']==2
    parent_family='L'
    parent_tag='r201l_uniform_k8_20261007'
    groot=PROJECT/'results'/parent_tag/dataset
    folders={'original_DeMo':PROJECT/'results/full_official_baselines_20261004/training'/(dataset+'_demo_s42')}
    for variant in ('frequency_shared','axis_shared'):
        folders['L_recipe_'+variant]=groot/'training'/(dataset+'_r201'+parent_family.lower()+'_'+variant+'_s42')
        folders['M_'+variant]=root/'training'/(dataset+'_r201m_'+variant+'_s42')
    metrics,groups,comparisons,pairs,curves,training,reads,orders=[],[],[],[],[],{},{},{}
    for model,folder in folders.items():
        data=load(folder/'result.json')
        assert data['status']=='COMPLETE' and data['epochs']==50 and data['steps']==data['optimizer_steps']==(old_steps if model=='original_DeMo' else steps) and data['amp_skipped_steps']==0
        assert (data['train_records'],data['query_records'],data['gallery_records'])==(train,count,gallery)
        assert data['training_coverage']==dict(eligible=train,visited=train,unvisited=[])
        assert data['training_heldout_identities']==0 and data['descriptor_dim']==5120
        assert '  NUM_INSTANCE: '+str(SAMPLER_K[dataset] if model=='original_DeMo' else 8) in data['config'].splitlines()
        rows,summary=query_rows(folder,dataset)
        epochs=list(csv.DictReader((folder/'epochs.csv').open(encoding='utf-8',newline='')))
        assert [int(row['epoch']) for row in epochs]==list(range(1,51))
        selected=max(epochs,key=lambda row:float(row['mAP']))
        assert int(selected['epoch'])==data['best']['epoch']
        assert all(abs(float(selected[key])-summary[key])<1e-8 and abs(data['full_metrics'][key]-summary[key])<1e-8 for key in METRICS)
        metrics.append(dict(dataset=dataset,model=model,selected_epoch=data['best']['epoch'],**{key:summary[key] for key in METRICS}))
        groups.extend(dict(dataset=dataset,model=model,grouping=axis,**value) for axis,values in summary['groups'].items() for value in values)
        training[model]=dict(selected_epoch=data['best']['epoch'],parameters=data['parameters'],trainable_parameters=data['trainable_parameters'],descriptor_dim=5120,
            budget_epochs=50 if model=='original_DeMo' else 100,final_epoch={key:float(epochs[-1][key]) for key in METRICS},runtime=data['runtime'],peak_memory=data['peak_memory'])
        reads[model]=rows,summary
        if model!='original_DeMo':
            batches=[json.loads(line) for line in (folder/'batch_orders.jsonl').read_text(encoding='utf-8').splitlines()]
            assert len(batches)==(old_steps if model=='original_DeMo' else steps) and all(row['optimizer_updated'] for row in batches)
            orders[model]=[(row['epoch'],row['step'],row['names'],row['partial_set']) for row in batches]
            assert all(row['loss']==row['full_loss'] and row['partial_CE_weight']==row['partial_triplet_weight']==row['partial_loss_effective']==0 for row in batches)
            audit=load(folder/'normal_cpu_audit.json')
            assert audit['status']=='PASS' and audit['max_metric_error']<1e-8 and audit['CMC50_and_per_query_and_groups'] and audit['full_training_coverage']
            assert load(folder/'normal_local_archive.json')['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
            if model.startswith('M_'):
                identities={r['name']:r['identity'] for r in load(folder/'official_split_manifest.json')['train']}
                assert all(row['sampling_identities']==row['sampling_instances']==8 and row['sampling_batch']==64
                    and len(Counter(identities[n] for n in row['names']))==8
                    and set(Counter(identities[n] for n in row['names']).values())=={8} for row in batches)
                assert all(row['primary_full_metric']=='unit5120_margin06_batch_hard_triplet' and row['primary_margin']==.6 and row['primary_metric_coefficient']==1. and row['auxiliary_original_soft_triplet'] and row['PI_outlet']=='shared_relation_local' and 0<=row['primary_margin_violations']<=64 for row in batches)
                for epoch in epochs:
                    members=[row for row in batches if row['epoch']==int(epoch['epoch'])]
                    means={key:math.fsum(row[key] for row in members)/len(members) for key in ('loss','full_loss','primary_triplet_loss','primary_margin_violations','primary_positive_distance_mean','primary_negative_distance_mean','primary03_same_feature_loss','primary03_same_feature_violations','partial_ce','cross_triplet','partial_loss_effective')}
                    assert abs(means['loss']-float(epoch['loss']))<1e-8 and means['loss']==means['full_loss'] and means['partial_loss_effective']==0
                    curves.append(dict(dataset=dataset,model=model,epoch=int(epoch['epoch']),batches=len(members),**means,**{key:float(epoch[key]) for key in METRICS}))
            else:
                assert all(row['primary_full_metric']=='unit5120_margin03_batch_hard_triplet' and row['auxiliary_original_soft_triplet'] for row in batches)
    assert orders['M_axis_shared']==orders['M_frequency_shared'] and orders['L_recipe_axis_shared']==orders['L_recipe_frequency_shared']
    assert orders['M_axis_shared']==orders['L_recipe_axis_shared']
    assert len({(v['parameters'],v['trainable_parameters']) for model,v in training.items() if model!='original_DeMo'})==1
    for improved,reference in (('M_axis_shared','original_DeMo'),('M_frequency_shared','original_DeMo'),('M_axis_shared','M_frequency_shared'),('M_axis_shared','L_recipe_axis_shared'),('M_frequency_shared','L_recipe_frequency_shared'),('M_axis_shared','L_recipe_frequency_shared')):
        arows,a=reads[improved];brows,b=reads[reference]
        assert all(all(x[key]==y[key] for key in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')) for x,y in zip(arows,brows))
        delta={key:a[key]-b[key] for key in METRICS}
        rescue=sum(int(x['Rank-1'])==1 and int(y['Rank-1'])==0 for x,y in zip(arows,brows));harm=sum(int(x['Rank-1'])==0 and int(y['Rank-1'])==1 for x,y in zip(arows,brows))
        assert abs(delta['Rank-1']-100*(rescue-harm)/count)<1e-8
        comparisons.append(dict(dataset=dataset,improved=improved,reference=reference,**delta,both_plus1=delta['mAP']>=1 and delta['Rank-1']>=1,both_plus2=delta['mAP']>=2 and delta['Rank-1']>=2,
            rank1_rescued=rescue,rank1_harmed=harm,AP_improved=sum(float(x['AP'])>float(y['AP']) for x,y in zip(arows,brows)),AP_worsened=sum(float(x['AP'])<float(y['AP']) for x,y in zip(arows,brows))))
        pairs.extend(dict(dataset=dataset,improved=improved,reference=reference,**{key:x[key] for key in ('query_index','name','identity','camera','scene')},delta_AP_pp=100*(float(x['AP'])-float(y['AP'])),delta_INP_pp=100*(float(x['INP'])-float(y['INP'])),delta_rank1=int(x['Rank-1'])-int(y['Rank-1'])) for x,y in zip(arows,brows))
    assert len(metrics)==5 and len(comparisons)==6 and len(pairs)==6*count and len(curves)==100
    output=root/'normal_analysis';output.mkdir(exist_ok=False)
    for name,rows in (('six_metrics',metrics),('group_metrics',groups),('comparisons',comparisons),('paired_query_changes',pairs),('training_curves',curves)):table(output/(name+'.csv'),rows)
    result=dict(status='ACTUAL_M_DATASET_TWO_FULL50_NORMAL_CPU_READOUT',completed_at=datetime.now().isoformat(timespec='seconds'),dataset=dataset,model_results=5,comparisons=comparisons,paired_query_rows=len(pairs),new_epoch_rows=100,new_stage_formal_updates=2*steps,training=training,
        original_both_plus1=next(r['both_plus1'] for r in comparisons if r['improved']=='M_axis_shared' and r['reference']=='original_DeMo'),
        original_both_plus2=next(r['both_plus2'] for r in comparisons if r['improved']=='M_axis_shared' and r['reference']=='original_DeMo'),official_GT_six_CMC50_perquery_groups=True,paired_M_batch_orders_and_partial_sets=True,parent_L_recipe_orders_exact=True,new_neural_calls=0,new_optimizer_updates=0,
        limits='Weak-dataset/expertseed42 on frozen originalDeMo42. Only final primary hinge margin.3->.6 relative to L; original/auxiliary soft unchanged, B64P8K8/orders/steps/K graph/params/5120/partial0 match. RGBNT201 K42 margin.3 best preserved, dataset-specific training parameters explicit. All50 epochs, full official/benchmark-selected earliestmAP/same6/paper-six masks later, no new49; original50 versus expert-stage100 disclosed. Native gradients and detached route statistics are diagnostics, not causal proof. No guaranteed +1, missing or wholepipeline repeat claim.')
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','dataset','model_results','paired_query_rows','original_both_plus1')}),flush=True)


if __name__=='__main__':main()
