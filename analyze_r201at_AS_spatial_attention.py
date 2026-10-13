"""CPU summaries of closed AS full-Q/G probes; no inferred retrieval gains."""
import csv,json,statistics
from pathlib import Path

TAG='r201at_AS_spatial_attention_20261014'
ROOT=Path('results')/TAG


def main():
    controller=json.loads((ROOT/'controller_result.json').read_text());assert controller['status']=='COMPLETE'
    assert controller['optimizer_updates']==controller['checkpoint_writes']==0 and controller['output_features_exact']
    summaries=[];conditions=[];primary={}
    for variant in ('frequency_shared','axis_shared'):
        path=ROOT/'full'/('AS_'+variant);result=json.loads((path/'result.json').read_text())
        assert result['status']=='COMPLETE' and result['records']==10290 and result['attention_fields']==275
        assert result['query_records']==1715 and result['gallery_records']==8575 and result['output_features_exact_saved_selected_cache']
        with (path/'spatial_attention_records.csv').open(newline='') as f:records=list(csv.DictReader(f))
        assert len(records)==10290 and all(int(x['index'])==i for i,x in enumerate(records))
        primary[variant]=dict(selected_epoch=result['selected_epoch'],full_six_metrics=result['full_six_metrics_equal_selected'],query_roles={k:v['records'] for k,v in result['summaries'].items() if k!='gallery'})
        groups={**result['summaries']}
        query_groups=[v for k,v in groups.items() if k!='gallery'];assert sum(v['records'] for v in query_groups)==1715
        keys=list(groups['gallery']['statistics'])
        # Query means come from unrounded role summaries, not the CSV's .8g display.
        for side,members in [('query',query_groups),('gallery',[groups['gallery']])]:
            count=sum(v['records'] for v in members)
            means={k:sum(v['records']*v['statistics'][k]['mean'] for v in members)/count for k in keys}
            summaries.extend(dict(model='AS_'+variant,group=side,records=count,field=k,mean=value) for k,value in means.items())
            for band in range(3):
                for source in range(3):
                    prefix=f'F_source{source}_band{band}_'
                    raw=means[prefix+'input_spatial_variance'];assert raw>0
                    conditions.append(dict(model='AS_'+variant,side=side,source=source,band=band,
                        input_variance=raw,LN_variance=means[prefix+'after_LN_spatial_variance'],linear_variance=means[prefix+'after_linear_spatial_variance'],GELU_variance=means[prefix+'after_GELU_spatial_variance'],
                        input_mean_square=means[prefix+'input_mean_square'],GELU_mean_square=means[prefix+'after_GELU_mean_square'],
                        GELU_to_input_variance_ratio_of_group_means=means[prefix+'after_GELU_spatial_variance']/raw))
        for role,group in groups.items():
            if role!='gallery':summaries.extend(dict(model='AS_'+variant,group=role,records=group['records'],field=k,mean=v['mean']) for k,v in group['statistics'].items())
        meta=[{k:x[k] for k in ('index','side','name','identity','camera','scene')} for x in records]
        if variant=='frequency_shared':reference=meta
        else:assert meta==reference
    out=ROOT/'CPU_analysis';out.mkdir(exist_ok=False)
    for name,rows in [('attention_group_means.csv',summaries),('frequency_processing_group_means.csv',conditions)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    record=dict(status='ACTUAL_AT_AS_BOTH_FULL_QG_READER_DIAGNOSTICS_CPU',records_per_control=10290,query_per_control=1715,gallery_per_control=8575,
        measurements_per_record=275,controls=primary,optimizer_updates=0,checkpoint_writes=0,new_missing49=False,new_seed=False,
        limits='This summarizes fixed selected AS full normal inference, not a new method result. All output features equal the old selected cache, so metrics are inherited verified old metrics. Query error roles are diagnostic only. Mean-head spatial probability and per-head statistics have separate labels. Ordinary F slots read mixed sources, not physical-source experts. LayerNorm/Linear/GELU source statistics are before source embeddings and source mixing. Ratio of group means is not mean per-record ratio; no foreground, semantic identity, causality or absence of raw frequency information is inferred from entropy or variance.')
    (out/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':main()
