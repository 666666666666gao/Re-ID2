"""Reduce complete AR/AP fixed selected spatial reads; no neural calls or live queries."""
import csv,json,math
from pathlib import Path

P=Path(__file__).resolve().parent
ROOT=P/'results/r201ar_AQ_spatial_attention_20261014'


def main():
    paths={'AL':P/'results/r201ap_AL_spatial_attention_20261013/full/AL_axis_shared','AQ':ROOT/'full/AQ_axis_shared'}
    records={};proofs={}
    for name,path in paths.items():
        d=json.loads((path/'result.json').read_text('utf-8-sig'))
        assert d['status']=='COMPLETE' and d['records']==10290 and d['query_records']==1715 and d['gallery_records']==8575
        assert d['output_features_exact_saved_selected_cache'] and d['optimizer_updates']==d['checkpoint_writes']==0
        with (path/'spatial_attention_records.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        assert len(rows)==10290 and [int(r['index']) for r in rows]==list(range(10290))
        assert sum(r['side']=='query' for r in rows)==1715
        records[name]=rows;proofs[name]=d
    metadata=('index','side','name','identity','camera','scene')
    assert all(tuple(a[k] for k in metadata)==tuple(b[k] for k in metadata) for a,b in zip(records['AL'],records['AQ']))
    measurements={'teacher_mean_sources':['teacher_'+str(i) for i in range(3)]}
    measurements.update({f'M{stage}_band{b}':[f'M_{stage}_band{b}'] for stage in range(2) for b in range(3)})
    measurements.update({f'F_mean_sources_band{b}':[f'F_{i}_band{b}' for i in range(3)] for b in range(3)})
    rows=[]
    for name,proof in proofs.items():
        for side in ('query','gallery'):
            groups=[g for role,g in proof['summaries'].items() if (role=='gallery')==(side=='gallery')]
            total=sum(g['records'] for g in groups);assert total==(1715 if side=='query' else 8575)
            for label,prefixes in measurements.items():
                values={}
                for metric in ('normalized_entropy','effective_support_fraction','maximum_lift'):
                    fields=[prefix+'_head_mean_probability_'+metric for prefix in prefixes]
                    values[metric]=math.fsum(g['records']*math.fsum(g['statistics'][k]['mean'] for k in fields)/len(fields) for g in groups)/total
                rows.append(dict(model=name,side=side,records=total,measurement=label,**values))
    extras=('input_query_norm','projected_query_norm','input_key_spatial_variance','projected_key_spatial_variance','logit_centered_spatial_std')
    prefixes=[f'M_{s}_band{b}' for s in range(2) for b in range(3)]+[f'F_{i}_band{b}' for i in range(3) for b in range(3)]
    reading=[]
    for side in ('query','gallery'):
        members=[r for r in records['AQ'] if r['side']==side]
        for prefix in prefixes:
            values={k:math.fsum(float(r[prefix+'_'+k]) for r in members)/len(members) for k in extras}
            reading.append(dict(model='AQ',side=side,records=len(members),operator=prefix,**values))
    output=ROOT/'analysis';output.mkdir(exist_ok=False)
    for filename,values in (('head_mean_spatial_AL_AQ_comparison.csv',rows),('AQ_query_key_logit_conditions.csv',reading)):
        with (output/filename).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    d=proofs['AQ'];roles={role:value['records'] for role,value in d['summaries'].items()}
    result=dict(status='ACTUAL_AR_FULL_AQ_SPATIAL_READS_CPU_COMPARISON_COMPLETE',full_queries=1715,full_gallery=8575,
        summaries=rows,reading_conditions=reading,roles=roles,new_neural_calls=0,new_optimizer_updates=0,
        source_epochs=dict(AL=proofs['AL']['selected_epoch'],AQ=d['selected_epoch']),
        limits='Two benchmark-selected snapshots of the same frozen identity anchor, not paired retraining causal ablation. Mean probability entropy/support/lift weighted from unrounded subgroup summaries; query/key/logit conditions from recorded 8-significant-digit CSV. M averages equal per-relation/source stats; F grouped by source. Extra conditions only AQ; key variance is after existing value processing, not raw CLIP patch variance. Roles differ between selected models, metadata and complete Q/G order match. Entropy/JS/norms do not establish foreground, identity relevance or retrieval improvement.')
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],AQ_all_query=[r for r in rows if r['model']=='AQ' and r['side']=='query'],roles=roles)),flush=True)


if __name__=='__main__':main()
