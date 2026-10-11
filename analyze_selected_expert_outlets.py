"""CPU-only stage comparisons; retain all GT metrics and query changes."""
import csv
import json
from pathlib import Path
from analyze_identity_coordinate_three_normal import METRICS, six, table

ROOT=Path(__file__).resolve().parent
STAGES=('base00','normal11','M_raw_pool','F_raw_pool','M_routed_pool','F_routed_pool',
        'M_projection_input','F_projection_input','M_projection_output','F_projection_output',
        'M_unit_direction','F_unit_direction','common_context')


def main():
    root=ROOT/'results/selected_expert_outlets_20261011'
    registry=json.loads((root/'jobs.json').read_text())
    metrics,comparisons,pairs=[],[],[]
    for job in registry['jobs']:
        folder=root/job['name']
        result=json.loads((folder/'result.json').read_text())
        audit=json.loads((folder/'independent_cpu_audit.json').read_text())
        assert result['status']=='COMPLETE' and result['stages']==list(STAGES) and result['normal_features_and_distance_exact']
        assert result['optimizer_updates']==result['new_weights']==0 and audit['status']=='PASS' and audit['max_sixmetric_error_pp']<1e-8
        rows={}
        for stage in STAGES:
            with (folder/(stage+'.csv')).open(encoding='utf-8-sig',newline='') as f:rows[stage]=list(csv.DictReader(f))
            value=audit['measured'][stage]
            assert len(rows[stage])==result['query_records'] and all(abs(six(rows[stage])[k]-value[k])<1e-8 for k in METRICS)
            metrics.append(dict(dataset=job['dataset'],model=job['name'],stage=stage,dimension=result['dimensions'][stage],selected_epoch=result['selected_epoch'],**{k:value[k] for k in METRICS}))
        assert all(audit['measured']['normal11'][k]==job['full_metrics'][k] for k in METRICS)
        selected_pairs=[(stage,'base00') for stage in STAGES if stage!='base00']
        selected_pairs.extend((e+'_projection_output',e+'_projection_input') for e in ('M','F'))
        selected_pairs.extend((e+'_routed_pool',e+'_raw_pool') for e in ('M','F'))
        for improved,reference in selected_pairs:
            after,before=rows[improved],rows[reference]
            assert all({k:x[k] for k in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')}=={k:y[k] for k in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')} for x,y in zip(after,before))
            rescue=sum(int(x['Rank-1'])>int(y['Rank-1']) for x,y in zip(after,before))
            harm=sum(int(x['Rank-1'])<int(y['Rank-1']) for x,y in zip(after,before))
            delta={k:audit['measured'][improved][k]-audit['measured'][reference][k] for k in METRICS}
            assert abs(delta['Rank-1']-100*(rescue-harm)/len(after))<1e-8
            comparisons.append(dict(dataset=job['dataset'],model=job['name'],improved=improved,reference=reference,**delta,rank1_rescued=rescue,rank1_harmed=harm,
                AP_improved=sum(float(x['AP'])>float(y['AP']) for x,y in zip(after,before)),AP_worsened=sum(float(x['AP'])<float(y['AP']) for x,y in zip(after,before))))
            pairs.extend(dict(dataset=job['dataset'],model=job['name'],improved=improved,reference=reference,query_index=x['query_index'],name=x['name'],identity=x['identity'],delta_AP_pp=100*(float(x['AP'])-float(y['AP'])),delta_Rank1=int(x['Rank-1'])-int(y['Rank-1'])) for x,y in zip(after,before))
    assert len(metrics)==52 and len(comparisons)==64
    out=root/'outlet_analysis';out.mkdir(exist_ok=False)
    table(out/'six_metrics.csv',metrics);table(out/'comparisons.csv',comparisons);table(out/'paired_query_changes.csv',pairs)
    (out/'result.json').write_text(json.dumps(dict(status='ACTUAL_FOUR_FIXED_SELECTED_P_R_FULL_NORMAL_OUTLET_GT_CPU_COMPLETE',models=4,stage_cases=52,comparisons=64,paired_query_rows=len(pairs),optimizer_updates=0,new_weights=0,new49=False,
        limits='Source and projection stage retrieval diagnose independent evidence, not a new trained method or causal loss. Fixed saved normal-best, no test-driven checkpoint change. M/F output/input both3584D, raw/routed pools both512D; against5120 base dimensions differ and are disclosed. K201 cached best untouched.'),indent=2)+'\n',encoding='utf-8')
    print('ACTUAL_FULL_NORMAL_OUTLET_CPU_COMPLETE',len(metrics),len(comparisons),len(pairs),flush=True)


if __name__=='__main__':main()
