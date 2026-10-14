"""Reduce completed BC normal states against original and same-budget AV controls."""
import csv,json,math
from pathlib import Path
from analyze_identity_coordinate_three_normal import METRICS,load,query_rows,table

PROJECT=Path(__file__).resolve().parent
TAG='r201bc_RGB_reference_interaction_20261015'


def main():
    root=PROJECT/'results'/TAG/'RGBNT100'
    controller=load(root/'controller_result.json')
    assert controller['status']=='COMPLETE' and controller['successful_updates']==13126
    assert controller['AG_training_record_order_exact']
    original=PROJECT/'results/full_official_baselines_20261004/training/RGBNT100_demo_s42'
    folders={'original_DeMo':original}
    for variant in ('frequency_shared','axis_shared'):
        folders['AK_'+variant]=PROJECT/'results/r201ak_RGB_lastblock_staged_20261013/RGBNT100/training'/('RGBNT100_r201ak_'+variant+'_s42')
        folders['AG_'+variant]=PROJECT/'results/r201ag_RGB_coordinate_metric_20261012/RGBNT100/training'/('RGBNT100_r201ag_'+variant+'_s42')
        folders['AL_'+variant]=PROJECT/'results/r201al_RGB_route_temperature_20261013/RGBNT100/training'/('RGBNT100_r201al_'+variant+'_s42')
        folders['AQ_'+variant]=PROJECT/'results/r201aq_RGB_attention_decay_20261013/RGBNT100/training'/('RGBNT100_r201aq_'+variant+'_s42')
        folders['AS_'+variant]=PROJECT/'results/r201as_RGB_query_normalization_20261014/RGBNT100/training'/('RGBNT100_r201as_'+variant+'_s42')
        folders['AV_'+variant]=PROJECT/'results/r201av_RGB_frequency_value_decay_20261014/RGBNT100/training'/('RGBNT100_r201av_'+variant+'_s42')
        folders['BC_'+variant]=root/'training'/('RGBNT100_r201bc_'+variant+'_s42')
    reads={};metrics=[];groups=[];training={};curves=[];orders={}
    for model,folder in folders.items():
        result=load(folder/'result.json');rows,summary=query_rows(folder,'RGBNT100')
        assert result['status']=='COMPLETE' and result['epochs']==50 and result['amp_skipped_steps']==0
        assert (result['train_records'],result['query_records'],result['gallery_records'])==(8675,1715,8575)
        assert result['training_heldout_identities']==0 and result['training_coverage']==dict(eligible=8675,visited=8675,unvisited=[])
        assert result['descriptor_dim']==5120
        epochs=list(csv.DictReader((folder/'epochs.csv').open(encoding='utf-8',newline='')))
        assert [int(row['epoch']) for row in epochs]==list(range(1,51))
        selected=max(epochs,key=lambda row:float(row['mAP']))
        assert int(selected['epoch'])==result['best']['epoch']
        assert all(abs(float(selected[k])-summary[k])<1e-8 and abs(result['full_metrics'][k]-summary[k])<1e-8 for k in METRICS)
        metrics.append(dict(model=model,selected_epoch=result['best']['epoch'],**{k:summary[k] for k in METRICS}))
        groups.extend(dict(model=model,grouping=axis,**value) for axis,values in summary['groups'].items() for value in values)
        training[model]=dict(parameters=result['parameters'],trainable_parameters=result['trainable_parameters'],descriptor_dim=5120,
            total_epochs=50 if model=='original_DeMo' else 100,selected_epoch=result['best']['epoch'],
            final_epoch={k:float(epochs[-1][k]) for k in METRICS},runtime=result['runtime'],peak_memory=result['peak_memory'])
        reads[model]=rows,summary
        if model!='original_DeMo':
            batches=[json.loads(line) for line in (folder/'batch_orders.jsonl').read_text().splitlines()]
            assert result['steps']==result['optimizer_steps']==len(batches)==6563
            assert all(r['optimizer_updated'] and r['loss']==r['full_loss'] and r['partial_loss_effective']==0 for r in batches)
            orders[model]=[(r['epoch'],r['step'],r['names'],r['partial_set']) for r in batches]
            assert load(folder/'normal_cpu_audit.json')['status']=='PASS'
            if model.startswith('BC_'):
                assert (result['parameters'],result['trainable_parameters'])==(99617926,1920134)
                assert all(r['M_F_auxiliary_Triplet_outlet']=='actual_first11_per512_unit_relation_coordinates3584' and
                    r['M_F_auxiliary_CE_outlet']=='unchanged_existing_score_FP32_pool512' and r['auxiliary_metric_feature_dimension']==3584 for r in batches)
                assert all(r['freeze_identity_encoder'] and r['routing_temperature']==2. and r['routing_parameters_added']==0 for r in batches)
                assert all(r['attention_projection_decay']['old']==1e-4 and r['attention_projection_decay']['new']==0. and r['attention_projection_decay']['packed_includes_V'] and r['final_descriptor_bound_full']['epsilon']==.10 and r['final_descriptor_bound_full']['bounded_shift_max']<=.10+1e-6 for r in batches)
                assert all(r['modality_projected_query_normalization']['parameters_added']==0 and r['modality_projected_query_normalization']['attention_calls']==84 for r in batches)
                assert all(r['frequency_value_input_decay']['old']==1e-4 and r['frequency_value_input_decay']['new']==0. and r['frequency_value_input_decay']['targets']==['frequency_expert.values.1.1.weight','frequency_expert.values.2.1.weight'] for r in batches)
                training[model]['F_mid_high_value_Linear_weight_decay']=0.
                training[model]['correction_budget']=.10
                training[model]['packed_attention_QKV_decay']=0.
                training[model]['routing_temperature']=2.
                for epoch in epochs:
                    members=[r for r in batches if r['epoch']==int(epoch['epoch'])]
                    loss=math.fsum(r['loss'] for r in members)/len(members)
                    assert abs(loss-float(epoch['loss']))<1e-8
                    curves.append(dict(model=model,epoch=int(epoch['epoch']),loss=loss,batches=len(members),**{k:float(epoch[k]) for k in METRICS}))
    assert orders['BC_axis_shared']==orders['BC_frequency_shared']
    for variant in ('frequency_shared','axis_shared'):assert orders['BC_'+variant]==orders['AV_'+variant]==orders['AS_'+variant]==orders['AQ_'+variant]==orders['AL_'+variant]==orders['AG_'+variant]==orders['AK_'+variant]
    comparisons=[];pairs=[]
    for improved,reference in [('BC_axis_shared','original_DeMo'),('BC_frequency_shared','original_DeMo'),('BC_axis_shared','AV_axis_shared'),('BC_frequency_shared','AV_frequency_shared'),
        ('BC_axis_shared','BC_frequency_shared'),('BC_axis_shared','AS_axis_shared'),('BC_frequency_shared','AS_frequency_shared'),('BC_axis_shared','AQ_axis_shared'),('BC_frequency_shared','AQ_frequency_shared'),('BC_axis_shared','AL_axis_shared'),('BC_frequency_shared','AL_frequency_shared'),('BC_axis_shared','AG_axis_shared')]:
        arows,a=reads[improved];brows,b=reads[reference]
        assert all(tuple(x[k] for k in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery'))==
            tuple(y[k] for k in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')) for x,y in zip(arows,brows))
        delta={k:a[k]-b[k] for k in METRICS}
        rescue=sum(int(x['Rank-1'])==1 and int(y['Rank-1'])==0 for x,y in zip(arows,brows))
        harm=sum(int(x['Rank-1'])==0 and int(y['Rank-1'])==1 for x,y in zip(arows,brows))
        assert abs(delta['Rank-1']-100*(rescue-harm)/1715)<1e-8
        comparisons.append(dict(improved=improved,reference=reference,**delta,rank1_rescued=rescue,rank1_harmed=harm,
            both_plus1=delta['mAP']>=1 and delta['Rank-1']>=1,both_plus2=delta['mAP']>=2 and delta['Rank-1']>=2,
            AP_improved=sum(float(x['AP'])-float(y['AP'])>1e-12 for x,y in zip(arows,brows)),AP_worsened=sum(float(y['AP'])-float(x['AP'])>1e-12 for x,y in zip(arows,brows))))
        pairs.extend(dict(improved=improved,reference=reference,**{k:x[k] for k in ('query_index','name','identity','camera','scene')},
            delta_AP_pp=100*(float(x['AP'])-float(y['AP'])),delta_INP_pp=100*(float(x['INP'])-float(y['INP'])),delta_rank1=int(x['Rank-1'])-int(y['Rank-1'])) for x,y in zip(arows,brows))
    state_metrics=[];state_changes=[]
    for variant in ('frequency_shared','axis_shared'):
        name='RGBNT100_r201bc_'+variant+'_s42';folder=root/'states'/name
        record=load(folder/'result.json');assert record['status']=='COMPLETE' and record['state_cases']==4
        assert record['full11_features_distance_equal_selected_normal'] and record['state_tensor_versions_unchanged']
        rows_by_state={}
        for state in ('00','10','01','11'):
            rows_by_state[state]=list(csv.DictReader((folder/('state_'+state+'.csv')).open(encoding='utf-8',newline='')))
            assert len(rows_by_state[state])==1715
            state_metrics.append(dict(model='BC_'+variant,state=state,**{k:record['measurements'][state][k] for k in METRICS}))
        for reference in ('00','10','01'):
            after=rows_by_state['11'];before=rows_by_state[reference]
            state_changes.append(dict(model='BC_'+variant,comparison='11-'+reference,
                **{k:record['measurements']['11'][k]-record['measurements'][reference][k] for k in METRICS},
                rank1_rescued=sum(int(a['Rank-1'])==1 and int(b['Rank-1'])==0 for a,b in zip(after,before)),
                rank1_harmed=sum(int(a['Rank-1'])==0 and int(b['Rank-1'])==1 for a,b in zip(after,before))))
    output=root/'normal_analysis';output.mkdir(exist_ok=False)
    selection_curves=[]
    for variant in ('frequency_shared','axis_shared'):
        folder=root/'training'/('RGBNT100_r201bc_'+variant+'_s42')
        batches=[json.loads(line) for line in (folder/'batch_orders.jsonl').read_text().splitlines()]
        for epoch in range(1,51):
            policies=[r['attention_projection_decay'] for r in batches if r['epoch']==epoch]
            assert policies and all(p['targets']==policies[0]['targets'] for p in policies)
            for i,name in enumerate(policies[0]['targets']):
                for j,part in enumerate(('Q','K','V')):
                    index=3*i+j
                    selection_curves.append(dict(model='BC_'+variant,epoch=epoch,parameter=name,part=part,
                        first_weight_norm=policies[0]['weight_norm_before'][index],last_weight_norm=policies[-1]['weight_norm_after'][index],
                        mean_unscaled_task_gradient_norm=math.fsum(p['unscaled_task_gradient_norm'][index] for p in policies)/len(policies),
                        mean_counterfactual_old_L2_gradient_norm=math.fsum(p['counterfactual_old_L2_gradient_norm'][index] for p in policies)/len(policies)))
    table(output/'attention_selection_training_curves.csv',selection_curves)
    value_curves=[]
    for variant in ('frequency_shared','axis_shared'):
        folder=root/'training'/('RGBNT100_r201bc_'+variant+'_s42')
        batches=[json.loads(line) for line in (folder/'batch_orders.jsonl').read_text().splitlines()]
        for epoch in range(1,51):
            policies=[r['frequency_value_input_decay'] for r in batches if r['epoch']==epoch]
            assert policies and all(p['targets']==policies[0]['targets'] for p in policies)
            for i,name in enumerate(policies[0]['targets']):
                value_curves.append(dict(model='BC_'+variant,epoch=epoch,parameter=name,
                    first_weight_norm=policies[0]['weight_norm_before'][i],last_weight_norm=policies[-1]['weight_norm_after'][i],
                    mean_unscaled_task_gradient_norm=math.fsum(p['unscaled_task_gradient_norm'][i] for p in policies)/len(policies),
                    mean_counterfactual_old_L2_gradient_norm=math.fsum(p['counterfactual_old_L2_gradient_norm'][i] for p in policies)/len(policies)))
    assert len(value_curves)==200
    table(output/'frequency_value_input_training_curves.csv',value_curves)
    for name,values in [('six_metrics',metrics),('group_metrics',groups),('comparisons',comparisons),('paired_query_changes',pairs),('training_curves',curves),('state_metrics',state_metrics),('state_comparisons',state_changes)]:table(output/(name+'.csv'),values)
    result=dict(status='ACTUAL_BC_TWO_FULL50_NORMAL_EIGHT_STATES_CPU_READOUT',model_results=len(metrics),state_cases=8,state_comparisons=state_changes,comparisons=comparisons,training=training,
        paired_query_rows=len(pairs),new_epoch_rows=len(curves),new_stage_formal_updates=13126,official_GT_six_CMC50_perquery_groups=True,
        AG_training_record_order_exact=True,new_neural_calls=0,new_optimizer_updates=0,
        limits='Single seed42; benchmark mAP used for earliest-tie checkpoint selection; teacher50+extra50 versus original50. BC returns to AV and changes only existing relation-local PI hidden64 via 2sigmoid(Wb unit(stopgrad(original base relation512))). One shared zero-initialized64x512 weight32768 parameters, identical addition in ordinary and dual-axis models. Initial AV values/existing tensors/RNG retained; PM/PF/auxiliary/conditions/routing/5120/cap.10 unchanged. Known conditional modulation, not new information, calibrated utility or guaranteed gain. BA prototype initialization and AZ corrected-state metric not retained. Full official normal only, teacher50 plus fresh expert42/newAdam/extra50, benchmarkmAP earliesttie. Paired ordinary/axis and completed AS/AQ have equal additional budget and exact record order; AL/AG/AK are historical references. Reported peak_memory is whole-run/process maximum. Frozen base00 retains original teacher; frozen intervention is not a retrained ablation. Normal only, no missing or multiseed claim; all metrics from one selected weight.')
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],comparisons=comparisons)),flush=True)


if __name__=='__main__':main()
