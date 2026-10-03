import csv
import json
from pathlib import Path
import statistics

project=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
root=project/'results/common_coordinate_v12_trial_20261003'
metrics=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
variants=('axis_shared','frequency_shared','twins_shared','demo_shared')
assert json.loads((root/'controller_result.json').read_text())['status']=='COMPLETE'
report={'status':'ACTUAL_MATCHED_COMMON_COORDINATE_WAVE_AUDIT','runs':{},'comparisons':{},'goal':'ACTIVE_UNMET'}
orders=[]
measurements={}

def records(path):
    return list(csv.DictReader(path.open(encoding='utf-8')))

def recount(path, expected):
    rows=records(path)
    valid=[r for r in rows if r['valid']=='True']
    calculated={'mAP':100*statistics.mean(float(r['AP']) for r in valid),
                'mINP':100*statistics.mean(float(r['INP']) for r in valid),
                **{f'Rank-{k}':100*statistics.mean(int(r[f'Rank-{k}']) for r in valid) for k in (1,5,10,20)}}
    assert len(valid)==expected['valid_queries'] and len(rows)==expected['query_count']
    assert all(abs(calculated[key]-expected[key])<1e-8 for key in metrics)
    assert all(abs(100*statistics.mean(int(r['first_match'])<=rank for r in valid)-expected['CMC_1_to_50'][rank-1])<1e-8 for rank in range(1,51))
    for key in ('camera','scene','identity'):
        for group in expected['groups'][key]:
            subset=[r for r in valid if int(r[key])==group['value']]
            assert len(subset)==group['queries']
            assert abs(100*statistics.mean(float(r['AP']) for r in subset)-group['mAP'])<1e-8
            assert abs(100*statistics.mean(float(r['INP']) for r in subset)-group['mINP'])<1e-8
            assert all(abs(100*statistics.mean(int(r[f'Rank-{rank}']) for r in subset)-group[f'Rank-{rank}'])<1e-8 for rank in (1,5,10,20))
    return valid

def pair(before,after):
    fields=('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')
    assert len(before)==len(after) and all(all(a[k]==b[k] for k in fields) for a,b in zip(before,after))
    return dict(queries=len(before),GT_and_order_exact=True,
                rank1_rescued=sum(int(b['Rank-1'])>int(a['Rank-1']) for a,b in zip(before,after)),
                rank1_harmed=sum(int(b['Rank-1'])<int(a['Rank-1']) for a,b in zip(before,after)),
                AP_improved=sum(float(b['AP'])>float(a['AP']) for a,b in zip(before,after)),
                AP_worsened=sum(float(b['AP'])<float(a['AP']) for a,b in zip(before,after)),
                AP_unchanged=sum(float(b['AP'])==float(a['AP']) for a,b in zip(before,after)))

banks={}
table=[]
for variant in variants:
    name='MSVR310_'+variant+'_s42'
    run=root/'development'/name
    terminal=json.loads((run/'result.json').read_text())
    epochs=records(run/'epochs.csv')
    batches=[json.loads(line) for line in (run/'batch_orders.jsonl').read_text().splitlines()]
    assert len(epochs)==50 and [int(r['epoch']) for r in epochs]==list(range(1,51))
    assert terminal['epochs']==50 and terminal['status']=='COMPLETE'
    assert terminal['steps']==len(batches)==int(epochs[-1]['steps'])
    assert terminal['optimizer_steps']==sum(b['optimizer_updated'] for b in batches)
    assert terminal['amp_skipped_steps']==len(batches)-terminal['optimizer_steps']
    assert all(not b['reference_requires_grad'] for b in batches)
    assert all(b['partial_set'] in ('R','N','T','RN','RT','NT') for b in batches)
    for batch in batches:
        count=len(batch['names'])
        assert count==64 and len(batch['positive_indices'])==len(batch['negative_indices'])==count
        assert all(i!=positive and 0<=positive<count and 0<=negative<count
                   for i,(positive,negative) in enumerate(zip(batch['positive_indices'],batch['negative_indices'])))
    chosen=max(epochs,key=lambda r:float(r['mAP']))
    assert int(chosen['epoch'])==terminal['best']['epoch']
    assert all(abs(float(chosen[key])-terminal['best'][key])<1e-8 for key in ('mAP','Rank-1','Rank-5','Rank-10'))
    normal=json.loads((run/'development_metrics.json').read_text())
    recount(run/'development_metrics.csv',normal)
    evaluation=root/'frozen49'/name
    frozen=json.loads((evaluation/'result.json').read_text())
    assert frozen['status']=='COMPLETE' and len(frozen['measurements'])==49
    assert frozen['normal_feature_max_error']==0 and frozen['state_tensor_versions_unchanged'] and frozen['optimizer_updates']==0
    measurements[variant]=frozen['measurements']
    banks[variant]={}
    for condition,value in frozen['measurements'].items():
        banks[variant][condition]=recount(evaluation/(condition+'.csv'),value['metrics'])
        table.append(dict(dataset='MSVR310',variant=variant,condition=condition,**{key:value['metrics'][key] for key in metrics}))
    orders.append([(b['epoch'],b['step'],b['names'],b['partial_set']) for b in batches])
    report['runs'][variant]=dict(epochs=50,steps=len(batches),optimizer_steps=terminal['optimizer_steps'],
          amp_skipped_steps=terminal['amp_skipped_steps'],best_epoch=terminal['best']['epoch'],
          full_metrics=terminal['full_metrics'],parameters=terminal['parameters'],trainable_parameters=terminal['trainable_parameters'],
          descriptor_dim=terminal['descriptor_dim'],all49_recount_exact=True,CMC_and_groups_exact=True,
          partial_counts={code:sum(b['partial_set']==code for b in batches) for code in ('R','N','T','RN','RT','NT')},
          peak_memory_bytes=terminal['peak_memory'],epoch_seconds_sum=sum(float(r['seconds']) for r in epochs))
assert all(order==orders[0] for order in orders)
report['sampling_and_partial_masks_exactly_matched']=True
assert len({(report['runs'][v]['parameters'],report['runs'][v]['trainable_parameters'],report['runs'][v]['descriptor_dim']) for v in variants[:3]})==1
baseline=project/'results/anytoany49_demo_20261003/MSVR310_demo_s42/full'
old=json.loads((baseline/'result.json').read_text())['measurements']
measurements['original_demo']=old
banks['original_demo']={condition:recount(baseline/(condition+'.csv'),value['metrics']) for condition,value in old.items()}
for reference in ('original_demo','demo_shared','frequency_shared','twins_shared'):
    comparison={}
    for condition in measurements['axis_shared']:
        delta={key:measurements['axis_shared'][condition]['metrics'][key]-measurements[reference][condition]['metrics'][key] for key in metrics}
        comparison[condition]=dict(delta_pp=delta,double_plus2=delta['mAP']>=2 and delta['Rank-1']>=2,
                                   paired=pair(banks[reference][condition],banks['axis_shared'][condition]))
    report['comparisons'][reference]=dict(conditions=comparison,double_plus2_conditions=sum(v['double_plus2'] for v in comparison.values()),
                                        normal=comparison['q_RNT_g_RNT'])
report['limits']='Single dataset seed42 development mechanism wave; three datasets, multiseed and official final evaluation remain required. Source and GPU contracts do not prove final method success.'
target=project/'results/preflight/common_coordinate_complete_analysis.json'
assert not target.exists()
target.write_text(json.dumps(report,indent=2)+'\n')
with (root/'MSVR310_all4_full49_metrics_196.csv').open('w',newline='',encoding='utf-8') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
print(json.dumps(dict(status=report['status'],runs=report['runs'],paired_counts={key:v['double_plus2_conditions'] for key,v in report['comparisons'].items()})),flush=True)
