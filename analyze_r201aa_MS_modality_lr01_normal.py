"""CPU comparison of AA GF update LR.1 versus original DeMo and completed Z."""
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
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=('MSVR310',),required=True)
    dataset=parser.parse_args().dataset;pf=PROJECT/'results/preflight'
    review=load(pf/'r201aa_MS_modality_lr01_source_review_20261012.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    assert {'results/preflight/r201aa_MS_modality_lr01_normal_complete_intake_20261012.py','analyze_r201aa_MS_modality_lr01_normal.py'} <= set(review['sources_sha256'])
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
    intake=load(pf/('r201aa_'+dataset+'_normal_terminal_text_intake_20261012.json'))
    assert intake['status']=='ACTUAL_AA_DATASET_TWO_FULL50_NORMAL_GT_TEXT_VERIFIED' and intake['dataset']==dataset
    root=PROJECT/'results/r201aa_MS_modality_lr01_20261012'/dataset
    controller=load(root/'controller_result.json');train,count,gallery,old_steps=COUNTS[dataset]
    steps=load(pf/'r201l_uniform_K8_sampler_cpu_replay_20261007.json')['datasets'][dataset]['4']['total_steps']
    assert controller['status']=='COMPLETE' and controller['dataset']==dataset and controller['controls']==2
    assert controller['successful_updates']==2*steps and controller['additional_epochs']==100 and controller['native_updates']==6
    assert controller['paired_sampling_exact'] and controller['normal_archives_local_verified']==2
    parent_family='Z'
    parent_tag='r201z_MS_modality_adapt_20261012'
    groot=PROJECT/'results'/parent_tag/dataset
    folders={'original_DeMo':PROJECT/'results/full_official_baselines_20261004/training'/(dataset+'_demo_s42')}
    for variant in ('frequency_shared','axis_shared'):
        folders['parent_recipe_'+variant]=groot/'training'/(dataset+'_r201'+parent_family.lower()+'_'+variant+'_s42')
        folders['AA_'+variant]=root/'training'/(dataset+'_r201aa_'+variant+'_s42')
    metrics,groups,comparisons,pairs,curves,training,reads,orders=[],[],[],[],[],{},{},{}
    for model,folder in folders.items():
        data=load(folder/'result.json')
        assert data['status']=='COMPLETE' and data['epochs']==50 and data['steps']==data['optimizer_steps']==(old_steps if model=='original_DeMo' else steps) and data['amp_skipped_steps']==0
        assert (data['train_records'],data['query_records'],data['gallery_records'])==(train,count,gallery)
        assert data['training_coverage']==dict(eligible=train,visited=train,unvisited=[])
        assert data['training_heldout_identities']==0 and data['descriptor_dim']==5120
        assert '  NUM_INSTANCE: '+str(SAMPLER_K[dataset] if model=='original_DeMo' else 4) in data['config'].splitlines()
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
            if model.startswith('AA_'):
                coefficient=.001
                previous_coefficient=.01
                outlet={'MSVR310':'actual_first11_projected_unit_relation_mean512','RGBNT100':'actual_first11_projected_unit_relation_score_pool512'}[dataset]
                assert data['anchor']['M_F_auxiliary_weight_previous']==previous_coefficient and data['anchor']['M_F_auxiliary_weight']==coefficient
                assert all(row['M_F_auxiliary_weight_previous']==previous_coefficient and row['M_F_auxiliary_weight']==coefficient for row in batches)
                assert data['anchor']['parent_main_CE_weight']=={'MSVR310':.05,'RGBNT100':.25}[dataset] and data['anchor']['parent_contribution_regression_weight']=={'MSVR310':0.,'RGBNT100':.05}[dataset]
                assert data['anchor']['M_F_auxiliary_outlet']['parameters_added']==0
                assert data['anchor']['freeze_identity_encoder'] is False and data['anchor']['partial_identity_adaptation']['adapted_parameters']==9958912
                assert data['trainable_parameters']==13081734 and data['parameters']==100821638
                assert data['anchor']['generalFusion_learning_rate_control']['factor']==.1
                for row in batches:
                    lr=row['generalFusion_learning_rate_control']
                    assert lr['factor']==.1 and lr['parameter_elements']==9958912 and lr['effective_min']==.1*lr['nominal_min'] and lr['effective_max']==.1*lr['nominal_max'] and lr['nonGF_LR_unchanged'] and lr['nominal_group_LR_restored_after_update']
                assert all(r['generalFusion_parameter_adaptation'] and r['generalFusion_eval_buffers_and_frozen4_retained'] and r['generalFusion_trainable_parameters']==9958912 for r in batches)
                assert data['anchor']['sampler_control']==dict(parent_K=8,current_K=4,batch_size=64,identities_per_batch=16,old_updates_per_arm=1000,new_updates_per_arm=705,same_epochs_not_equal_updates=True)
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
                assert all(row['sampling_identities']==16 and row['sampling_instances']==4 and row['sampling_batch']==64
                    and len(Counter(identities[n] for n in row['names']))==16
                    and set(Counter(identities[n] for n in row['names']).values())=={4} for row in batches)
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
    assert orders['AA_axis_shared']==orders['AA_frequency_shared'] and orders['parent_recipe_axis_shared']==orders['parent_recipe_frequency_shared']
    assert len(orders['AA_axis_shared'])==705 and orders['AA_axis_shared']==orders['parent_recipe_axis_shared']
    assert len({v['parameters'] for model,v in training.items() if model!='original_DeMo'})==1
    assert training['AA_axis_shared']['trainable_parameters']==training['AA_frequency_shared']['trainable_parameters']
    for variant in ('axis_shared','frequency_shared'):
        assert training['AA_'+variant]['trainable_parameters']==training['parent_recipe_'+variant]['trainable_parameters']
    for improved,reference in (('AA_axis_shared','original_DeMo'),('AA_frequency_shared','original_DeMo'),('AA_axis_shared','AA_frequency_shared'),('AA_axis_shared','parent_recipe_axis_shared'),('AA_frequency_shared','parent_recipe_frequency_shared'),('AA_axis_shared','parent_recipe_frequency_shared')):
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
    result=dict(status='ACTUAL_AA_DATASET_TWO_FULL50_NORMAL_CPU_READOUT',completed_at=datetime.now().isoformat(timespec='seconds'),dataset=dataset,model_results=5,comparisons=comparisons,paired_query_rows=len(pairs),new_epoch_rows=100,new_stage_formal_updates=2*steps,training=training,
        original_both_plus1=next(r['both_plus1'] for r in comparisons if r['improved']=='AA_axis_shared' and r['reference']=='original_DeMo'),
        original_both_plus2=next(r['both_plus2'] for r in comparisons if r['improved']=='AA_axis_shared' and r['reference']=='original_DeMo'),official_GT_six_CMC50_perquery_groups=True,paired_Z_batch_orders_and_partial_sets=True,paired_parent_recipe_orders_exact=True,AA_vs_Z_orders_equal=True,new_neural_calls=0,new_optimizer_updates=0,
        limits='AA versus completed Z changes only existing generalFusion actual optimizer-update LR to .1 of its nominal scheduled value, then restores nominal LR. Scheduler/nonGF groups/gradients/objective/forwards/BN/initialization/architecture unchanged. Equal total100821638 and active13081734 parameters; no added trainables. Same705updates/P16K4/50epochs andparentexactorders. Currentbase adapts; 00 notorigDeMo/cap.10 not bound tofixedteacher. AllnormalGT/noholdout/benchmarkmAP-best earliestties/samebest six/best-last/curves/groups/perquery, teacher50+additional50. No201NN/new49/missing checkpoint/failedseedsearch/guaranteedgain.')
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','dataset','model_results','paired_query_rows','original_both_plus1')}),flush=True)


if __name__=='__main__':main()
