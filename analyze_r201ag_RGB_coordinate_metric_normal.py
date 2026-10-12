"""Reduce two completed AG normal controls and the matched AD hard controls."""
import csv,json,math
from pathlib import Path
from analyze_identity_coordinate_three_normal import METRICS,load,query_rows,table

PROJECT=Path(__file__).resolve().parent
TAG='r201ag_RGB_coordinate_metric_20261012'


def main():
    root=PROJECT/'results'/TAG/'RGBNT100'
    controller=load(root/'controller_result.json')
    assert controller['status']=='COMPLETE' and controller['successful_updates']==13126
    assert controller['AD_hard_training_record_order_exact']
    original=PROJECT/'results/full_official_baselines_20261004/training/RGBNT100_demo_s42'
    folders={'original_DeMo':original}
    for variant in ('frequency_shared','axis_shared'):
        folders['AD_hard_'+variant]=PROJECT/'results/r201ad_RGB_precision_matched_20261012/RGBNT100/training'/('RGBNT100_r201ad_hard_'+variant+'_s42')
        folders['AG_'+variant]=root/'training'/('RGBNT100_r201ag_'+variant+'_s42')
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
            if model.startswith('AG_'):
                assert (result['parameters'],result['trainable_parameters'])==(99585158,1887366)
                assert all(r['M_F_auxiliary_Triplet_outlet']=='actual_first11_per512_unit_relation_coordinates3584' and
                    r['M_F_auxiliary_CE_outlet']=='unchanged_existing_score_FP32_pool512' and r['auxiliary_metric_feature_dimension']==3584 for r in batches)
                for epoch in epochs:
                    members=[r for r in batches if r['epoch']==int(epoch['epoch'])]
                    loss=math.fsum(r['loss'] for r in members)/len(members)
                    assert abs(loss-float(epoch['loss']))<1e-8
                    curves.append(dict(model=model,epoch=int(epoch['epoch']),loss=loss,batches=len(members),**{k:float(epoch[k]) for k in METRICS}))
    assert orders['AG_axis_shared']==orders['AG_frequency_shared']
    for variant in ('frequency_shared','axis_shared'):assert orders['AG_'+variant]==orders['AD_hard_'+variant]
    comparisons=[];pairs=[]
    for improved,reference in [('AG_axis_shared','original_DeMo'),('AG_frequency_shared','original_DeMo'),
        ('AG_axis_shared','AG_frequency_shared'),('AG_axis_shared','AD_hard_axis_shared'),('AG_frequency_shared','AD_hard_frequency_shared')]:
        arows,a=reads[improved];brows,b=reads[reference]
        assert all(tuple(x[k] for k in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery'))==
            tuple(y[k] for k in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')) for x,y in zip(arows,brows))
        delta={k:a[k]-b[k] for k in METRICS}
        rescue=sum(int(x['Rank-1'])==1 and int(y['Rank-1'])==0 for x,y in zip(arows,brows))
        harm=sum(int(x['Rank-1'])==0 and int(y['Rank-1'])==1 for x,y in zip(arows,brows))
        assert abs(delta['Rank-1']-100*(rescue-harm)/1715)<1e-8
        comparisons.append(dict(improved=improved,reference=reference,**delta,rank1_rescued=rescue,rank1_harmed=harm,
            both_plus1=delta['mAP']>=1 and delta['Rank-1']>=1,both_plus2=delta['mAP']>=2 and delta['Rank-1']>=2,
            AP_improved=sum(float(x['AP'])>float(y['AP']) for x,y in zip(arows,brows)),AP_worsened=sum(float(x['AP'])<float(y['AP']) for x,y in zip(arows,brows))))
        pairs.extend(dict(improved=improved,reference=reference,**{k:x[k] for k in ('query_index','name','identity','camera','scene')},
            delta_AP_pp=100*(float(x['AP'])-float(y['AP'])),delta_INP_pp=100*(float(x['INP'])-float(y['INP'])),delta_rank1=int(x['Rank-1'])-int(y['Rank-1'])) for x,y in zip(arows,brows))
    output=root/'normal_analysis';output.mkdir(exist_ok=False)
    for name,values in [('six_metrics',metrics),('group_metrics',groups),('comparisons',comparisons),('paired_query_changes',pairs),('training_curves',curves)]:table(output/(name+'.csv'),values)
    result=dict(status='ACTUAL_AG_TWO_FULL50_NORMAL_CPU_READOUT',model_results=5,comparisons=comparisons,training=training,
        paired_query_rows=len(pairs),new_epoch_rows=len(curves),new_stage_formal_updates=13126,official_GT_six_CMC50_perquery_groups=True,
        AD_hard_training_record_order_exact=True,new_neural_calls=0,new_optimizer_updates=0,
        limits='Single seed42; benchmark mAP used for earliest-tie checkpoint selection; teacher50+extra50 versus original50. Matched AD/AG and ordinary/axis have equal budget and exact record order. Normal only, no missing or multiseed claim; all metrics from one selected weight.')
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],comparisons=comparisons)),flush=True)


if __name__=='__main__':main()
