import json
from pathlib import Path
import statistics

project=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
path=project/'results/preflight/shared_identity_complete_analysis.json'
value=json.loads(path.read_text())
summary={}
for name,group in value['comparisons'].items():
    rows=group['conditions']
    metrics=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
    group['equal_condition_mean_delta_pp']={key:statistics.mean(v['delta_pp'][key] for v in rows.values()) for key in metrics}
    group['median_condition_delta_pp']={key:statistics.median(v['delta_pp'][key] for v in rows.values()) for key in metrics}
    group['improved_condition_counts']={key:sum(v['delta_pp'][key]>0 for v in rows.values()) for key in metrics}
    group['worst5_mAP_conditions']=[dict(condition=k,**v['delta_pp']) for k,v in sorted(rows.items(),key=lambda item:item[1]['delta_pp']['mAP'])[:5]]
    summary[name]=dict(mean=group['equal_condition_mean_delta_pp'],wins=group['improved_condition_counts'],
                      double_plus2=group['double_plus2_conditions'],worst5=group['worst5_mAP_conditions'])
value['condition_mean_limits']='Arithmetic mean giving each of49 conditions equal weight; diagnostic summary, not an official dataset mAP or newly selected deployment score.'
path.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(summary),flush=True)
