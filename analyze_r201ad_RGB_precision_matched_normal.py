"""CPU readout of train-only hard identity grouping against complete matched random AD controls."""
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
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=('RGBNT100',),required=True)
    dataset=parser.parse_args().dataset;pf=PROJECT/'results/preflight'
    review=load(pf/'r201ad_RGB_precision_matched_source_review_20261012.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    assert {'results/preflight/r201ad_RGB_normal_complete_intake_20261012.py','analyze_r201ad_RGB_precision_matched_normal.py'} <= set(review['sources_sha256'])
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
    intake=load(pf/('r201ad_'+dataset+'_normal_terminal_text_intake_20261012.json'))
    assert intake['status']=='ACTUAL_AD_DATASET_FOUR_FULL50_NORMAL_GT_TEXT_VERIFIED' and intake['dataset']==dataset
    root=PROJECT/'results/r201ad_RGB_precision_matched_20261012'/dataset
    controller=load(root/'controller_result.json');train,count,gallery,old_steps=COUNTS[dataset]
    steps=load(pf/'r201l_uniform_K8_sampler_cpu_replay_20261007.json')['datasets'][dataset]['8']['total_steps']
    assert controller['status']=='COMPLETE' and controller['dataset']==dataset and controller['controls']==4
    assert controller['successful_updates']==4*steps and controller['additional_epochs']==200 and controller['native_updates']==12
    assert controller['paired_sampling_exact'] and controller['normal_archives_local_verified']==4
    parent_family='AD_random'
    parent_tag='r201ad_RGB_precision_matched_20261012'
    groot=PROJECT/'results'/parent_tag/dataset
    folders={'original_DeMo':PROJECT/'results/full_official_baselines_20261004/training'/(dataset+'_demo_s42')}
    for variant in ('frequency_shared','axis_shared'):
        folders['AD_random_'+variant]=groot/'training'/(dataset+'_r201'+parent_family.lower()+'_'+variant+'_s42')
        folders['AD_hard_'+variant]=root/'training'/(dataset+'_r201ad_hard_'+variant+'_s42')
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
            if model.startswith('AD_'):
                coefficient=.001
                previous_coefficient={'MSVR310':.01,'RGBNT100':.05}[dataset]
                outlet={'MSVR310':'actual_first11_projected_unit_relation_mean512','RGBNT100':'actual_first11_projected_unit_relation_score_pool512'}[dataset]
                assert data['anchor']['M_F_auxiliary_weight_previous']==previous_coefficient and data['anchor']['M_F_auxiliary_weight']==coefficient
                assert all(row['M_F_auxiliary_weight_previous']==previous_coefficient and row['M_F_auxiliary_weight']==coefficient for row in batches)
                assert data['anchor']['parent_main_CE_weight']=={'MSVR310':.05,'RGBNT100':.25}[dataset] and data['anchor']['parent_contribution_regression_weight']=={'MSVR310':0.,'RGBNT100':.05}[dataset]
                assert data['anchor']['M_F_auxiliary_outlet']['parameters_added']==0
                treatment='hard' if model.startswith('AD_hard_') else 'random'
                assert data['anchor']['M_auxiliary_scorer_precision']['precision']=='FP32' and data['anchor']['M_auxiliary_scorer_precision']['parameters_added']==0
                assert all(r['sampling_treatment']==treatment and r['M_auxiliary_scorer_fp32']['input_shape']==[64,7,512] and r['M_auxiliary_scorer_fp32']['input_dtype']==r['M_auxiliary_scorer_fp32']['output_dtype']=='torch.float32' and r['M_auxiliary_scorer_fp32']['output_requires_grad'] for r in batches)
                assert data['anchor']['F_auxiliary_scorer_precision']['precision']=='FP32' and data['anchor']['F_auxiliary_scorer_precision']['parameters_added']==0
                assert all(r['F_auxiliary_scorer_fp32']['input_shape']==[64,7,512] and r['F_auxiliary_scorer_fp32']['input_dtype']==r['F_auxiliary_scorer_fp32']['output_dtype']=='torch.float32' and r['F_auxiliary_scorer_fp32']['output_requires_grad'] for r in batches)
                if dataset=='MSVR310':
                    assert data['anchor']['M_F_auxiliary_outlet']['frozen_parameter_names']==['modality_expert.pool_score.weight','frequency_expert.pool_score.weight'] and data['anchor']['M_F_auxiliary_outlet']['frozen_parameter_count']==1024
                training[model]['frozen_auxiliary_parameter_count']=1024 if dataset=='MSVR310' else 0
                for row in batches:
                    c=row['projected_auxiliary_forward_contract']
                    assert row['M_F_auxiliary_outlet']==outlet and c['projection_calls']==dict(M=2,F=2) and c['pool_calls']==dict(M=2,F=2) and c['neck_calls']==dict(M=1,F=1) and c['feature_shape']==[64,512] and c['extra_neck_projection_calls']==0
                    assert row['parent_main_CE_weight']=={'MSVR310':.05,'RGBNT100':.25}[dataset] and row['parent_contribution_regression_weight']=={'MSVR310':0.,'RGBNT100':.05}[dataset] and row['xent_counted_calls']==2
                assert data['anchor']['contribution_reference']=='Stopped-gradient corresponding gallery state; fixed00 positive/negative indices'
                assert all(row['contribution_reference']=='same_state_gallery_fixed00_indices' and row['contribution_loss_raw']>=0 and row['contribution_zero_loss_raw']>=0 for row in batches)
                assert data['anchor']['final_unit_descriptor_bound']['epsilon']==.10
                assert data['anchor']['final_unit_descriptor_bound']['parameter_change']==0
                reports=[row['final_descriptor_bound_full'] for row in batches]
                assert all(r['epsilon']==.10 and 0<=r['clipped_fraction']<=1 and r['bounded_shift_max']<=.10+1e-6 for r in reports)
                identities={r['name']:r['identity'] for r in load(folder/'official_split_manifest.json')['train']}
                assert all(row['sampling_identities']==row['sampling_instances']==8 and row['sampling_batch']==64
                    and len(Counter(identities[n] for n in row['names']))==8
                    and set(Counter(identities[n] for n in row['names']).values())=={8} for row in batches)
                assert all(row['primary_full_metric']=='unit5120_margin06_batch_hard_triplet' and row['primary_margin']==.6 and row['primary_metric_coefficient']==1. and row['auxiliary_original_soft_triplet'] and row['PI_outlet']=='shared_relation_local' and 0<=row['primary_margin_violations']<=64 for row in batches)
                for epoch in epochs:
                    members=[row for row in batches if row['epoch']==int(epoch['epoch'])]
                    means={key:math.fsum(row[key] for row in members)/len(members) for key in ('loss','full_loss','primary_triplet_loss','primary_margin_violations','primary_positive_distance_mean','primary_negative_distance_mean','primary03_same_feature_loss','primary03_same_feature_violations','partial_ce','cross_triplet','partial_loss_effective','contribution_loss_raw','contribution_zero_loss_raw')}
                    assert abs(means['loss']-float(epoch['loss']))<1e-8 and means['loss']==means['full_loss'] and means['partial_loss_effective']==0
                    bound_means={key:math.fsum(row['final_descriptor_bound_full'][key] for row in members)/len(members)
                        for key in ('clipped_fraction','candidate_shift_mean','bounded_shift_mean')}
                    curves.append(dict(dataset=dataset,model=model,epoch=int(epoch['epoch']),batches=len(members),**means,
                        **{'bound_'+k:v for k,v in bound_means.items()},
                        bounded_shift_max=max(row['final_descriptor_bound_full']['bounded_shift_max'] for row in members),
                        **{key:float(epoch[key]) for key in METRICS}))
            else:
                assert all(row['primary_full_metric']=='unit5120_margin06_batch_hard_triplet' and row['auxiliary_original_soft_triplet'] for row in batches)
    assert orders['AD_hard_axis_shared']==orders['AD_hard_frequency_shared'] and orders['AD_random_axis_shared']==orders['AD_random_frequency_shared']
    replay=load(pf/'r201ac_RGB_neighbors_schedule_CPU_actual_20261012.json')
    expected=[json.loads(line) for line in (PROJECT/'results/r201ac_RGB_neighbors_schedule_CPU_20261012/planned_batch_orders.jsonl').read_text().splitlines()]
    hard=orders['AD_hard_axis_shared'];random_orders=orders['AD_random_axis_shared']
    assert len(expected)==len(hard)==len(random_orders)==6563
    assert all((r['epoch'],r['step'],r['names'])==(actual[0],actual[1],actual[2]) for r,actual in zip(expected,hard))
    assert [(r[0],r[1],r[3]) for r in hard]==[(r[0],r[1],r[3]) for r in random_orders]
    for epoch in range(1,51):
        assert Counter(n for r in hard if r[0]==epoch for n in r[2])==Counter(n for r in random_orders if r[0]==epoch for n in r[2])
    for variant in ('frequency_shared','axis_shared'):
        rows=[json.loads(line) for line in (folders['AD_hard_'+variant]/'batch_orders.jsonl').read_text().splitlines()]
        for epoch,stats in enumerate(replay['epochs'],1):
            members=[r for r in rows if r['epoch']==epoch]
            assert [r['hard_identity_sampling']['batch_index'] for r in members]==list(range(len(members)))
            assert all(r['hard_identity_sampling']==dict(seed=42+epoch,batch_index=i,post_sampler_RNG_exact=True,
                **{k:v for k,v in stats.items() if k!='epoch'}) for i,r in enumerate(members))
        assert load(folders['AD_hard_'+variant]/'result.json')['anchor']['hard_identity_sampling']['bank_sha256']==replay['bank_result_sha256']
    assert len({v['parameters'] for model,v in training.items() if model!='original_DeMo'})==1
    assert training['AD_hard_axis_shared']['trainable_parameters']==training['AD_hard_frequency_shared']['trainable_parameters']
    for variant in ('axis_shared','frequency_shared'):
        assert training['AD_random_'+variant]['trainable_parameters']==training['AD_hard_'+variant]['trainable_parameters']
    for improved,reference in (('AD_hard_axis_shared','original_DeMo'),('AD_hard_frequency_shared','original_DeMo'),('AD_hard_axis_shared','AD_hard_frequency_shared'),('AD_hard_axis_shared','AD_random_axis_shared'),('AD_hard_frequency_shared','AD_random_frequency_shared'),('AD_hard_axis_shared','AD_random_frequency_shared')):
        arows,a=reads[improved];brows,b=reads[reference]
        assert all(all(x[key]==y[key] for key in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')) for x,y in zip(arows,brows))
        delta={key:a[key]-b[key] for key in METRICS}
        rescue=sum(int(x['Rank-1'])==1 and int(y['Rank-1'])==0 for x,y in zip(arows,brows));harm=sum(int(x['Rank-1'])==0 and int(y['Rank-1'])==1 for x,y in zip(arows,brows))
        assert abs(delta['Rank-1']-100*(rescue-harm)/count)<1e-8
        comparisons.append(dict(dataset=dataset,improved=improved,reference=reference,**delta,both_plus1=delta['mAP']>=1 and delta['Rank-1']>=1,both_plus2=delta['mAP']>=2 and delta['Rank-1']>=2,
            rank1_rescued=rescue,rank1_harmed=harm,AP_improved=sum(float(x['AP'])>float(y['AP']) for x,y in zip(arows,brows)),AP_worsened=sum(float(x['AP'])<float(y['AP']) for x,y in zip(arows,brows))))
        pairs.extend(dict(dataset=dataset,improved=improved,reference=reference,**{key:x[key] for key in ('query_index','name','identity','camera','scene')},delta_AP_pp=100*(float(x['AP'])-float(y['AP'])),delta_INP_pp=100*(float(x['INP'])-float(y['INP'])),delta_rank1=int(x['Rank-1'])-int(y['Rank-1'])) for x,y in zip(arows,brows))
    assert len(metrics)==5 and len(comparisons)==6 and len(pairs)==6*count and len(curves)==200
    output=root/'normal_analysis';output.mkdir(exist_ok=False)
    for name,rows in (('six_metrics',metrics),('group_metrics',groups),('comparisons',comparisons),('paired_query_changes',pairs),('training_curves',curves)):table(output/(name+'.csv'),rows)
    result=dict(status='ACTUAL_AD_DATASET_FOUR_FULL50_NORMAL_CPU_READOUT',completed_at=datetime.now().isoformat(timespec='seconds'),dataset=dataset,model_results=5,comparisons=comparisons,paired_query_rows=len(pairs),new_epoch_rows=200,new_stage_formal_updates=4*steps,training=training,
        original_both_plus1=next(r['both_plus1'] for r in comparisons if r['improved']=='AD_hard_axis_shared' and r['reference']=='original_DeMo'),
        original_both_plus2=next(r['both_plus2'] for r in comparisons if r['improved']=='AD_hard_axis_shared' and r['reference']=='original_DeMo'),official_GT_six_CMC50_perquery_groups=True,paired_AD_hard_batch_orders_and_partial_sets=True,parent_epoch_record_multiset_and_partial_sets_exact=True,hard_orders_match_CPU_replay=True,new_neural_calls=0,new_optimizer_updates=0,
        limits='AD matches actual secondM/FscorerFP32 in all four newly trained random/hard ordinary/axis controls; only training identity co-occurrence differs between matching AD random/hard, same original post-sampler RNG, exact per-epoch record multiset/P8K8/partial set/update budget and runtime matches closed CPU replay. All other X model/loss/teacher50+expert50/heads42/newAdam/BN2/cap.10/5120 held. OldX MAMP differs precision and is not sampling-only evidence. Benchmark mAP-best earliest tie, onecheckpoint sixmetrics/best-last/fullcurves. No missing专项/201NN/49/seed hunt, inherited teacher gains disclosed; exposure increase alone does not prove retrieval gain.')
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','dataset','model_results','paired_query_rows','original_both_plus1')}),flush=True)


if __name__=='__main__':main()
