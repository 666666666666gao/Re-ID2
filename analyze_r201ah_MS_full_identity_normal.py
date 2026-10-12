"""AH normal six metrics, fixed four states, paired query changes and full curves."""
import csv,json,math
from pathlib import Path
from analyze_identity_coordinate_three_normal import METRICS,load,query_rows,six,table

PROJECT=Path(__file__).resolve().parent;TAG='r201ah_MS_full_identity_20261012'


def main():
    root=PROJECT/'results'/TAG/'MSVR310';controller=load(root/'controller_result.json')
    assert controller['status']=='COMPLETE' and controller['successful_updates']==1410 and controller['state_cases']==8
    assert controller['AE_frozen_training_record_order_exact']
    folders={'original_DeMo':PROJECT/'results/full_official_baselines_20261004/training/MSVR310_demo_s42',
        'strong_full5632':PROJECT/'results/full_official_baselines_20261004/training/MSVR310_demo_shared_s42'}
    folders.update({'AH_'+v:root/'training'/('MSVR310_r201ah_'+v+'_s42') for v in ('frequency_shared','axis_shared')})
    metrics=[];groups=[];reads={};training={};curves=[];orders={};states={};state_metrics=[]
    for label,folder in folders.items():
        d=load(folder/'result.json');rows,summary=query_rows(folder,'MSVR310')
        assert d['status']=='COMPLETE' and d['epochs']==50 and d['steps']==d['optimizer_steps']==705
        assert d['training_heldout_identities']==d['amp_skipped_steps']==0 and d['training_coverage']==dict(eligible=1032,visited=1032,unvisited=[])
        expected_dim=5120 if label=='original_DeMo' else 5632;assert d['descriptor_dim']==expected_dim
        epochs=list(csv.DictReader((folder/'epochs.csv').open(encoding='utf-8',newline='')))
        assert [int(r['epoch']) for r in epochs]==list(range(1,51))
        selected=max(epochs,key=lambda r:float(r['mAP']));assert int(selected['epoch'])==d['best']['epoch']
        assert all(abs(float(selected[k])-summary[k])<1e-8 and abs(d['full_metrics'][k]-summary[k])<1e-8 for k in METRICS)
        metrics.append(dict(model=label,dimension=expected_dim,selected_epoch=d['best']['epoch'],**{k:summary[k] for k in METRICS}))
        groups.extend(dict(model=label,grouping=axis,**value) for axis,values in summary['groups'].items() for value in values)
        reads[label]=rows,summary
        training[label]=dict(parameters=d['parameters'],trainable_parameters=d['trainable_parameters'],dimension=expected_dim,
            total_training_epochs=100 if label.startswith('AH_') else 50,selected_epoch=d['best']['epoch'],
            last={k:float(epochs[-1][k]) for k in METRICS},runtime=d['runtime'],peak_memory=d['peak_memory'])
        if not label.startswith('AH_'):continue
        assert (d['parameters'],d['trainable_parameters'])==(100902022,2674822)
        assert load(folder/'normal_cpu_audit.json')['status']=='PASS'
        batches=[json.loads(line) for line in (folder/'batch_orders.jsonl').read_text().splitlines()]
        assert len(batches)==705 and all(r['optimizer_updated'] and r['loss']==r['full_loss'] and r['partial_loss_effective']==0 for r in batches)
        assert all(r['main_loss_deployed_dimension']==5632 and r['retained_public_projection_frozen'] for r in batches)
        orders[label]=[(r['epoch'],r['step'],r['names'],r['partial_set']) for r in batches]
        for epoch in epochs:
            members=[r for r in batches if r['epoch']==int(epoch['epoch'])]
            loss=math.fsum(r['loss'] for r in members)/len(members);assert abs(loss-float(epoch['loss']))<1e-8
            curves.append(dict(model=label,epoch=int(epoch['epoch']),batches=len(members),loss=loss,
                primary_triplet=math.fsum(r['primary_triplet_loss'] for r in members)/len(members),**{k:float(epoch[k]) for k in METRICS}))
        state_dir=root/'states'/folder.name;report=load(state_dir/'result.json')
        assert report['status']=='COMPLETE' and report['state_cases']==4 and report['full11_features_distance_exact_selected_normal']
        assert report['frozen00_full_teacher_six_per_query_equal'] and report['state_tensor_versions_unchanged']
        states[label]={}
        for key in ('00','10','01','11'):
            state_rows=list(csv.DictReader((state_dir/('state_'+key+'.csv')).open(encoding='utf-8',newline='')))
            state_summary=load(state_dir/('state_'+key+'.json'))
            assert len(state_rows)==591 and all(r['valid']=='True' for r in state_rows)
            assert all(abs(six(state_rows)[m]-state_summary[m])<1e-8 and abs(state_summary[m]-report['measurements'][key][m])<1e-8 for m in METRICS)
            assert report['installed_GT_audit'][key]['max_metric_error']<1e-8
            states[label][key]=state_rows,state_summary
            state_metrics.append(dict(model=label,state=key,**{m:state_summary[m] for m in METRICS}))
        assert states[label]['11'][0]==rows and states[label]['00'][0]==reads['strong_full5632'][0]
    assert orders['AH_axis_shared']==orders['AH_frequency_shared']
    prior=PROJECT/'results/r201ae_MS_strong_staged_20261012/MSVR310/training/MSVR310_r201ae_axis_frozen_s42/batch_orders.jsonl'
    old=[json.loads(line) for line in prior.read_text().splitlines()]
    assert orders['AH_axis_shared']==[(r['epoch'],r['step'],r['names'],r['partial_set']) for r in old]
    comparisons=[];pairs=[]
    def compare(a_label,a,b_label,b):
        arows,am=a;brows,bm=b
        assert all(tuple(x[k] for k in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery'))==
            tuple(y[k] for k in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')) for x,y in zip(arows,brows))
        delta={k:am[k]-bm[k] for k in METRICS}
        rescue=sum(int(x['Rank-1'])>int(y['Rank-1']) for x,y in zip(arows,brows));harm=sum(int(x['Rank-1'])<int(y['Rank-1']) for x,y in zip(arows,brows))
        assert abs(delta['Rank-1']-100*(rescue-harm)/591)<1e-8
        comparisons.append(dict(model=a_label,reference=b_label,**delta,rank1_rescued=rescue,rank1_harmed=harm,
            both_plus1=delta['mAP']>=1 and delta['Rank-1']>=1,AP_improved=sum(float(x['AP'])>float(y['AP']) for x,y in zip(arows,brows)),AP_worsened=sum(float(x['AP'])<float(y['AP']) for x,y in zip(arows,brows))))
        pairs.extend(dict(model=a_label,reference=b_label,**{k:x[k] for k in ('query_index','name','identity','camera','scene')},
            delta_AP_pp=100*(float(x['AP'])-float(y['AP'])),delta_INP_pp=100*(float(x['INP'])-float(y['INP'])),delta_rank1=int(x['Rank-1'])-int(y['Rank-1'])) for x,y in zip(arows,brows))
    for label in ('AH_frequency_shared','AH_axis_shared'):
        for reference in ('original_DeMo','strong_full5632'):compare(label,reads[label],reference,reads[reference])
        for key in ('10','01','11'):compare(label+'_'+key,states[label][key],label+'_00',states[label]['00'])
    compare('AH_axis_shared',reads['AH_axis_shared'],'AH_frequency_shared',reads['AH_frequency_shared'])
    cohort_flags=list(csv.DictReader((PROJECT/'results/msvr_strong_full_retention_cohorts_CPU_20261012/query_cohort_flags.csv').open(encoding='utf-8',newline='')))
    retained_indices=[i for i,r in enumerate(cohort_flags) if int(r['public_block_rescues'])==1];assert len(retained_indices)==25
    retention={label:sum(int(reads[label][0][i]['Rank-1']) for i in retained_indices) for label in ('strong_full5632','AH_frequency_shared','AH_axis_shared')}
    out=root/'normal_analysis';out.mkdir(exist_ok=False)
    for name,values in [('six_metrics',metrics),('state_metrics',state_metrics),('group_metrics',groups),('comparisons',comparisons),('paired_query_changes',pairs),('training_curves',curves)]:table(out/(name+'.csv'),values)
    result=dict(status='ACTUAL_AH_TWO_FULL50_NORMAL_GT_EIGHT_STATES_CPU_READOUT',models=4,formal_updates=1410,native_updates=6,
        state_cases=8,comparisons=comparisons,training=training,public_block_25_rescues_retained_correct=retention,AE_frozen_order_exact=True,
        new_neural_calls=0,new_optimizer_updates=0,missing_evaluated=False,
        limits='One benchmark-selected normal best percontrol, earliestmAPtie; single seed42/teacher50+expert50. Strong full baseline gains inherited. Frozen interventions are not retrained ablations. AH ordinary/axis matched, AE differs head width/publicfreeze/interface. Complete weaknormal goal and later paper-six missing goal remain separate.')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],comparisons=comparisons,retention=retention)),flush=True)


if __name__=='__main__':main()
