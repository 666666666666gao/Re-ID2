"""Reduce the two remaining G fixed-best streams and join the closed three-dataset readout."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path

from analyze_identity_coordinate_missing49 import SETS, STATES, availability_groups, load, verified_rows
from analyze_identity_coordinate_three_normal import METRICS, table

PROJECT=Path(__file__).resolve().parent
COUNTS={'MSVR310':(1032,591,1055), 'RGBNT100':(8675,1715,8575)}


def reduce_dataset(stream, DATASET, train, count, gallery):
    actual=load(PROJECT/'results/preflight/r201g_other_two_missing49_stream_full_actual_session_20261006.json')
    assert actual['status']=='ACTUAL_G_OTHER_TWO_FOUR_FULL49_INFERENCE_CONTROLS_ALL784_GT_AND_LOCAL_RAW'
    assert actual['exit_code']==0 and len(actual['raw'])==196 and actual['restored_copies_cleared']
    controller=load(stream/'controller_full_result.json')
    assert controller['status']=='COMPLETE' and controller['controls']==4 and controller['missing_GT_state_cases']==784
    assert controller['raw_conditions']==196 and controller['all_raw_local_verified']
    assert controller['optimizer_updates']==controller['new_weights']==0
    original=PROJECT/'results/full_official_baselines_20261004/frozen49'/(DATASET+'_demo_s42')
    original_audit=load(original/'independent_cpu_audit.json')
    assert original_audit['status']=='PASS' and original_audit['cases']==49
    models={'original_DeMo':(load(original/'result.json')['measurements'],original,'baseline')}
    disjoint=[]
    for family in ('unit','normal'):
        for variant in ('axis_shared','frequency_shared'):
            name=DATASET+('_identity_'+variant+'_narrow_s42' if family=='unit' else '_r201g_'+variant+'_s42')
            folder=PROJECT/'results/identity_coordinate_missing49_stream_full_completed_20261005/full'/name if family=='unit' else stream/'full'/name
            result=load(folder/'result.json')
            audit=load(folder/'independent_cpu_audit.json')
            assert result['status']=='COMPLETE' and audit['status']=='PASS' and result['state_cases']==audit['state_cases']==196
            trainroot=PROJECT/'results/identity_coordinate_three_normal_completed_20261005' if family=='unit' else PROJECT/'results/r201g_other_two_normal_20261006'/DATASET
            trained=load(trainroot/'training'/name/'result.json')
            assert result['selected_epoch']==trained['best']['epoch']
            assert (trained['train_records'],trained['query_records'],trained['gallery_records'])==(train,count,gallery)
            values=result['measurements']
            assert len(values)==49 and all(abs(values['q_RNT_g_RNT']['11'][key]-trained['full_metrics'][key])<1e-8 for key in METRICS)
            for state in STATES:
                models[family+'_'+variant+'/'+state]=({name:item[state] for name,item in values.items()},folder,family)
            if family=='normal':
                assert result['normal_features_and_distance_exact'] and result['training_heldout_identities']==0
                assert result['optimizer_updates']==result['new_weights']==0
                for condition,entry in audit['measured'].items():
                    if 'disjoint_private_distance' in entry:
                        assert entry['cross_source_identity_comparison_not_established']
                        disjoint.extend(dict(variant=variant,condition=condition,state=state,**value) for state,value in entry['disjoint_private_distance'].items())
    assert len(models)==17 and len(disjoint)==96
    pairs=[('normal_axis_shared/11',ref) for ref in ('original_DeMo','normal_axis_shared/00','normal_axis_shared/10','normal_axis_shared/01','normal_frequency_shared/11','unit_axis_shared/11')]
    pairs.extend(('normal_frequency_shared/11',ref) for ref in ('original_DeMo','normal_frequency_shared/00','normal_frequency_shared/10','normal_frequency_shared/01','unit_frequency_shared/11'))
    pairs.extend((family+'/00',old+'/00') for family,old in (('normal_axis_shared','unit_axis_shared'),('normal_frequency_shared','unit_frequency_shared')))
    metric_rows,groups,deltas,group_deltas,paired=[],[],[],[],[]
    checked=0
    for q in SETS:
        for g in SETS:
            condition='q_'+q+'_g_'+g
            rows={}
            for model,(values,folder,family) in models.items():
                state=model.rsplit('/',1)[-1]
                path=folder/(condition+'.csv') if family=='baseline' else folder/condition/('state_'+state+'.csv')
                current=verified_rows(path,values[condition],count,gallery)
                checked+=len(current)
                rows[model]=current
                metric_rows.append(dict(model=model,condition=condition,**{key:values[condition][key] for key in METRICS}))
                groups.extend(dict(model=model,condition=condition,grouping=grouping,**entry) for grouping,entries in values[condition]['groups'].items() for entry in entries)
            assert rows['normal_axis_shared/00']==rows['normal_frequency_shared/00']
            for improved,reference in pairs:
                left,right=rows[improved],rows[reference]
                assert all(all(x[key]==y[key] for key in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')) for x,y in zip(left,right))
                a,b=models[improved][0][condition],models[reference][0][condition]
                delta={key:a[key]-b[key] for key in METRICS}
                deltas.append(dict(condition=condition,improved=improved,reference=reference,**delta,
                    rescued=sum(int(x['Rank-1'])==1 and int(y['Rank-1'])==0 for x,y in zip(left,right)),
                    harmed=sum(int(x['Rank-1'])==0 and int(y['Rank-1'])==1 for x,y in zip(left,right)),
                    AP_improved=sum(float(x['AP'])>float(y['AP']) for x,y in zip(left,right)),
                    AP_worsened=sum(float(x['AP'])<float(y['AP']) for x,y in zip(left,right)),
                    both_plus2=delta['mAP']>=2 and delta['Rank-1']>=2))
                if improved=='normal_axis_shared/11' and reference in ('normal_axis_shared/00','normal_frequency_shared/11','unit_axis_shared/11'):
                    paired.extend(dict(condition=condition,improved=improved,reference=reference,**{key:x[key] for key in ('query_index','name','identity','camera','scene')},
                        delta_AP_pp=100*(float(x['AP'])-float(y['AP'])),delta_INP_pp=100*(float(x['INP'])-float(y['INP'])),delta_rank1=int(x['Rank-1'])-int(y['Rank-1'])) for x,y in zip(left,right))
    sizes={'all49':49,'same_availability':7,'overlap_mismatch':30,'source_disjoint':12,'partial_query_full_gallery':6,'full_query_partial_gallery':6,'both_partial':36}
    for improved,reference in pairs:
        for group,size in sizes.items():
            selected=[row for row in deltas if row['improved']==improved and row['reference']==reference and group in availability_groups(*row['condition'][2:].split('_g_'))]
            assert len(selected)==size
            group_deltas.append(dict(improved=improved,reference=reference,group=group,conditions=size,
                **{key:math.fsum(row[key] for row in selected)/size for key in METRICS},
                rescued=sum(row['rescued'] for row in selected),harmed=sum(row['harmed'] for row in selected),both_plus2_conditions=sum(row['both_plus2'] for row in selected)))
    assert len(metric_rows)==833 and len(deltas)==637 and checked==833*count and len(paired)==147*count
    output=stream/(DATASET+'_missing_analysis')
    output.mkdir(exist_ok=False)
    for name,items in (('six_metrics',metric_rows),('group_metrics',groups),('comparisons',deltas),('availability_group_deltas',group_deltas),('paired_query_changes',paired),('disjoint_distance_geometry',disjoint)):
        table(output/(name+'.csv'),items)
    result=dict(status='ACTUAL_G_OTHER_FIXED_NORMAL_BEST_ALL49_FOURSTATE_CPU_READOUT',completed_at=datetime.now().isoformat(timespec='seconds'),dataset=DATASET,
        model_state_metric_rows=833,comparisons=637,checked_condition_query_rows=checked,paired_query_rows=len(paired),unique_query_records=count,seed=42,
        all_six_CMC50_identity_camera_scene_verified=True,new_neural_calls=0,new_optimizer_updates=0,
        limits='One declared dataset only, fixed normal mAP-best/earliesttie, benchmark-selected seed42. Original50 versus G/unit original50+additional50; normal and unit experts use matched sampling/params/5120D. Baseline missing processing differs from correctly masked00, so baseline missing gains are not unique dual-axis evidence. Equal-condition group means are not official pooled mAP; repeated condition-query rows not independent samples. Disjoint private coordinates may yield near-constant distances and do not establish meaningful cross-source identification. Current ordinary control is mixed-source expert, not literal single-frequency branch. Not allthree missing success, calibrated contribution, psi ablation, pairedseeds or finalgoal.')
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stream',required=True)
    stream=Path(parser.parse_args().stream)
    results={dataset:reduce_dataset(stream,dataset,*counts) for dataset,counts in COUNTS.items()}
    prior=PROJECT/'results/r201g_201_missing49_stream_full_completed_20261006/missing_analysis'
    old=load(prior/'result.json')
    assert old['status']=='ACTUAL_G201_FIXED_NORMAL_BEST_ALL49_FOURSTATE_CPU_READOUT'
    assert old['new_neural_calls']==old['new_optimizer_updates']==0
    assert old['model_state_metric_rows']==833 and old['comparisons']==637 and old['paired_query_rows']==122892
    folders={'RGBNT201':prior, **{dataset:stream/(dataset+'_missing_analysis') for dataset in COUNTS}}
    output=stream/'three_dataset_missing_analysis'
    output.mkdir(exist_ok=False)
    totals={}
    for filename in ('six_metrics','group_metrics','comparisons','availability_group_deltas','paired_query_changes','disjoint_distance_geometry'):
        combined=[]
        for dataset,folder in folders.items():
            with (folder/(filename+'.csv')).open(encoding='utf-8',newline='') as handle:
                combined.extend(dict(dataset=dataset,**row) for row in csv.DictReader(handle))
        table(output/(filename+'.csv'),combined)
        totals[filename]=len(combined)
    assert totals['six_metrics']==2499 and totals['comparisons']==1911 and totals['availability_group_deltas']==273
    assert totals['paired_query_changes']==461874 and totals['disjoint_distance_geometry']==288
    result=dict(status='ACTUAL_G_THREE_FULL_OFFICIAL_FIXED_BEST_ALL49_CPU_READOUT',
        completed_at=datetime.now().isoformat(timespec='seconds'),new_dataset_results=results,
        existing_201_report=str(prior/'result.json'),existing_201_neural_and_CPU_not_replayed=True,
        dataset_count=3,model_state_metric_rows=2499,comparisons=1911,paired_query_rows=461874,
        checked_condition_query_rows=old['checked_condition_query_rows']+sum(item['checked_condition_query_rows'] for item in results.values()),
        unique_query_records=3142,seed=42,all_six_CMC50_identity_camera_scene_verified=True,
        new_neural_calls=0,new_optimizer_updates=0,
        limits='Fixed normal benchmark mAP-best seed42; no missing checkpoint selection. Original50 versus unit/G total100 budget disclosed. Repeated condition-query counts are not independent observations. Group means are not official pooled mAP; baseline missing processing differs from valid masked00. Disjoint private near-constant distance is not established cross-source identification. Current ordinary control is mixed-source expert, not literal single-frequency branch. No unified +2, calibrated contribution, psi retrain, multiple seeds or finalgoal claim.')
    assert result['checked_condition_query_rows']==2617286
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:value for key,value in result.items() if key not in ('new_dataset_results','limits')}))


if __name__=='__main__':
    main()
