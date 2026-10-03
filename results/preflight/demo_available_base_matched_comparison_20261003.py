import csv
import json
from pathlib import Path

project=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
source=project/'results/availability_base_comparison_20261003/all3_demo_V5_original_and_available_base_full49_metrics_588.csv'
rows=list(csv.DictReader(source.open(encoding='utf-8')))
metrics=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
lookup={(r['dataset'],r['variant'],r['state'],r['query_available'],r['gallery_available']):r for r in rows}
paired=[]
summary={}
for dataset in ('RGBNT201','RGBNT100','MSVR310'):
    current=[]
    for row in rows:
        if row['dataset']!=dataset or row['variant']=='demo' or row['state']!='available_base':
            continue
        baseline=lookup[(dataset,'demo','available_base',row['query_available'],row['gallery_available'])]
        delta={key:float(row[key])-float(baseline[key]) for key in metrics}
        item=dict(dataset=dataset,query_available=row['query_available'],gallery_available=row['gallery_available'],
                  **{key+'_delta_pp':delta[key] for key in metrics},
                  mAP_and_Rank1_each_at_least_plus2=delta['mAP']>=2 and delta['Rank-1']>=2)
        paired.append(item);current.append(item)
    assert len(current)==49
    summary[dataset]=dict(conditions=49,double_plus2_conditions=sum(r['mAP_and_Rank1_each_at_least_plus2'] for r in current),
                          normal=next(r for r in current if r['query_available']==r['gallery_available']=='RNT'),
                          RGB_query_full_gallery=next(r for r in current if r['query_available']=='R' and r['gallery_available']=='RNT'))
target=source.parent/'all3_masked_V5_minus_masked_DeMo_147.csv'
assert not target.exists()
with target.open('w',newline='',encoding='utf-8') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
report=project/'results/preflight/availability_base_matched_comparison.json'
assert not report.exists()
report.write_text(json.dumps(dict(status='ACTUAL_EXISTING_FULL49_CSV_PAIRED_SUMMARY',datasets=summary,
     source=str(source.relative_to(project)),rows=147,limits='Frozen V5 and DeMo both receive identical source-mask intervention; no new model forward, training or final method success.'),indent=2)+'\n')
print(json.dumps(summary),flush=True)
