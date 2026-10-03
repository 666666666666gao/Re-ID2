import csv
import json
from pathlib import Path
import statistics

project=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
root=project/'results/common_outlet_scale_v12_frozen_20261003'
audit=json.loads((root/'independent_cpu_audit.json').read_text())
assert audit['status']=='PASS' and audit['cases']==882
stages=('deployed','base_common','M_pre','M_post','F_pre','F_post')
metrics=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
groups={'all49':lambda q,g:True,'normal':lambda q,g:q==g=='RNT',
        'source_disjoint':lambda q,g:set(q).isdisjoint(g),
        'same_availability':lambda q,g:q==g,
        'partial_query_full_gallery':lambda q,g:q!='RNT' and g=='RNT',
        'both_partial':lambda q,g:q!='RNT' and g!='RNT',
        'overlap_mismatch':lambda q,g:q!=g and not set(q).isdisjoint(g)}
runs,table={},[]
for variant,run in audit['runs'].items():
    folder=root/('MSVR310_'+variant+'_s42')/'full'
    scales={}
    for availability in ('RNT','R','N','T','RN','RT','NT'):
        with (folder/('scale_'+availability+'.csv')).open() as handle:
            rows=list(csv.DictReader(handle))
        numeric=[k for k in rows[0] if k not in ('row','availability','names')]
        scales[availability]={}
        for key in numeric:
            values=[float(r[key]) for r in rows]
            scales[availability][key]=dict(mean=statistics.mean(values),median=statistics.median(values),
                std=statistics.pstdev(values),minimum=min(values),maximum=max(values))
    grouped={}
    for group,select in groups.items():
        conditions={k:v for k,v in run['conditions'].items() if select(k.split('_')[1],k.split('_')[3])}
        grouped[group]=dict(conditions=len(conditions),
            equal_condition_metrics={stage:{key:statistics.mean(v['metrics'][stage][key] for v in conditions.values()) for key in metrics} for stage in stages},
            projection_changes={stage:dict(mean_delta_pp={key:statistics.mean(v[stage]['delta_pp'][key] for v in conditions.values()) for key in metrics},
                rank1_rescue=sum(v[stage]['Rank1_rescue_queries'] for v in conditions.values()),
                rank1_harm=sum(v[stage]['Rank1_harm_queries'] for v in conditions.values()),
                AP_improved=sum(v[stage]['AP_improved_queries'] for v in conditions.values()),
                AP_worsened=sum(v[stage]['AP_worsened_queries'] for v in conditions.values())) for stage in ('M_projection','F_projection')})
    for condition,values in run['conditions'].items():
        for stage,value in values['metrics'].items():
            table.append(dict(variant=variant,condition=condition,stage=stage,selected_epoch=run['selected_epoch'],**value))
    runs[variant]=dict(selected_epoch=run['selected_epoch'],scales=scales,groups=grouped)
target=project/'results/preflight/common_outlet_scale_analysis.json'
assert not target.exists()
target.write_text(json.dumps(dict(status='ACTUAL_FROZEN_OUTLET_MEASUREMENT',cases=882,runs=runs,
    independent_raw_GT_recount='PASS',limits='Descriptive same-checkpoint measurements. Groups overlap; equal-condition means are not official mAP. Projection stage retrieval is not a retrained ablation. No denominator/pooling intervention was run.'),indent=2)+'\n')
with (root/'MSVR310_all3_sixstages_full49_metrics_882.csv').open('x',newline='',encoding='utf-8') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
print('ACTUAL_OUTLET_SCALES',json.dumps({v:{a:{k:s['mean'] for k,s in d.items() if k in ('legal_relations','weighted_anchor_relative_to_shared','M_common_increment_relative_to_shared','F_common_increment_relative_to_shared','I_common_increment_relative_to_shared','total_increment_relative_to_shared','gate_M','gate_F','gate_I','M_cancellation_ratio')} for a,d in r['scales'].items()} for v,r in runs.items()}),flush=True)
print('ACTUAL_OUTLET_PROJECTION_UTILITY',json.dumps({v:{g:r['groups'][g] for g in ('normal','source_disjoint','all49')} for v,r in runs.items()}),flush=True)
