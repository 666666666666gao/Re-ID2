"""Reduce completed AO normal/actualstates versus original and closed AG/AD references."""
import csv,json,math
from pathlib import Path
from analyze_identity_coordinate_three_normal import METRICS,load,query_rows,table

PROJECT=Path(__file__).resolve().parent
TAG='r201ao_RGB_hdm_query_guidance_20261013'


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
        folders['AO_'+variant]=root/'training'/('RGBNT100_r201ao_'+variant+'_s42')
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
            if model.startswith('AO_'):
                assert (result['parameters'],result['trainable_parameters'])==(99585158,1887366)
                assert all(r['M_F_auxiliary_Triplet_outlet']=='actual_first11_per512_unit_relation_coordinates3584' and
                    r['M_F_auxiliary_CE_outlet']=='unchanged_existing_score_FP32_pool512' and r['auxiliary_metric_feature_dimension']==3584 for r in batches)
                assert all(r['freeze_identity_encoder'] and r['routing_temperature']==2. and r['routing_parameters_added']==0 for r in batches)
                assert all(r['HDM_query_guidance'] and r['HDM_query_guidance_parameters_added']==0 and r['HDM_query_guidance_coefficient']==.1 and r['final_descriptor_bound_full']['epsilon']==.10 and r['final_descriptor_bound_full']['bounded_shift_max']<=.10+1e-6 for r in batches)
                training[model]['correction_budget']=.10
                training[model]['HDM_query_guidance']=True
                training[model]['routing_temperature']=2.
                for epoch in epochs:
                    members=[r for r in batches if r['epoch']==int(epoch['epoch'])]
                    loss=math.fsum(r['loss'] for r in members)/len(members)
                    assert abs(loss-float(epoch['loss']))<1e-8
                    curves.append(dict(model=model,epoch=int(epoch['epoch']),loss=loss,batches=len(members),**{k:float(epoch[k]) for k in METRICS}))
    assert orders['AO_axis_shared']==orders['AO_frequency_shared']
    for variant in ('frequency_shared','axis_shared'):assert orders['AO_'+variant]==orders['AL_'+variant]==orders['AG_'+variant]==orders['AK_'+variant]
    comparisons=[];pairs=[]
    for improved,reference in [('AO_axis_shared','original_DeMo'),('AO_frequency_shared','original_DeMo'),
        ('AO_axis_shared','AO_frequency_shared'),('AO_axis_shared','AL_axis_shared'),('AO_frequency_shared','AL_frequency_shared'),('AO_axis_shared','AG_axis_shared')]:
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
        name='RGBNT100_r201ao_'+variant+'_s42';folder=root/'states'/name
        record=load(folder/'result.json');assert record['status']=='COMPLETE' and record['state_cases']==4
        assert record['full11_features_distance_equal_selected_normal'] and record['state_tensor_versions_unchanged']
        rows_by_state={}
        for state in ('00','10','01','11'):
            rows_by_state[state]=list(csv.DictReader((folder/('state_'+state+'.csv')).open(encoding='utf-8',newline='')))
            assert len(rows_by_state[state])==1715
            state_metrics.append(dict(model='AO_'+variant,state=state,**{k:record['measurements'][state][k] for k in METRICS}))
        for reference in ('00','10','01'):
            after=rows_by_state['11'];before=rows_by_state[reference]
            state_changes.append(dict(model='AO_'+variant,comparison='11-'+reference,
                **{k:record['measurements']['11'][k]-record['measurements'][reference][k] for k in METRICS},
                rank1_rescued=sum(int(a['Rank-1'])==1 and int(b['Rank-1'])==0 for a,b in zip(after,before)),
                rank1_harmed=sum(int(a['Rank-1'])==0 and int(b['Rank-1'])==1 for a,b in zip(after,before))))
    output=root/'normal_analysis';output.mkdir(exist_ok=False)
    for name,values in [('six_metrics',metrics),('group_metrics',groups),('comparisons',comparisons),('paired_query_changes',pairs),('training_curves',curves),('state_metrics',state_metrics),('state_comparisons',state_changes)]:table(output/(name+'.csv'),values)
    result=dict(status='ACTUAL_AO_TWO_FULL50_NORMAL_EIGHT_STATES_CPU_READOUT',model_results=9,state_cases=8,state_comparisons=state_changes,comparisons=comparisons,training=training,
        paired_query_rows=len(pairs),new_epoch_rows=len(curves),new_stage_formal_updates=13126,official_GT_six_CMC50_perquery_groups=True,
        AG_training_record_order_exact=True,new_neural_calls=0,new_optimizer_updates=0,
        limits='Single seed42; benchmark mAP used for earliest-tie checkpoint selection; teacher50+extra50 versus original50. Matched AG/AO and ordinary/axis have equal budget and exact record order; AO adds only source-specific frozen pre-ATMoE HDM query conditions versus AL T2/cap.10; AG and AK are historical references. ALbase00 retains original teacher; frozen intervention is not a retrained ablation. Normal only, no missing or multiseed claim; all metrics from one selected weight.')
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],comparisons=comparisons)),flush=True)


if __name__=='__main__':main()
