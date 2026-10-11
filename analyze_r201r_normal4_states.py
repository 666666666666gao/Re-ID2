"""CPU-only complete normal R quartet against installed-GT audited states."""
from datetime import datetime
import csv,json,math
from pathlib import Path
from analyze_identity_coordinate_three_normal import METRICS,query_rows,table
ROOT=Path(__file__).resolve().parent

def main():
    root=ROOT/'results/r201r_normal4_states_20261011';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    actual=load(ROOT/'results/preflight/r201r_normal4_stream_actual_20261011.json');assert actual['status']=='ACTUAL_R_TWO_NORMAL_FOUR_STATES_GT_RAW_TEXT_COMPLETE' and actual['exit_code']==0
    assert actual['normal_conditions']==2 and actual['state_cases']==8 and actual['optimizer_updates']==0
    metrics,comparisons,pairs,anchor_drift=[],[],[],[]
    base_rows=[]
    original_rows,original=query_rows(ROOT/'results/full_official_baselines_20261004/training/MSVR310_demo_s42','MSVR310')
    for variant in ('frequency_shared','axis_shared'):
        model='MSVR310_r201r_'+variant+'_s42';folder=root/model;result=load(folder/'result.json');audit=load(folder/'independent_cpu_audit.json')
        assert result['status']=='COMPLETE' and result['conditions']==1 and result['state_cases']==4 and result['optimizer_updates']==result['new_weights']==0 and result['normal_features_and_distance_exact']
        assert audit['status']=='PASS' and audit['max_sixmetric_error_pp']<1e-8 and audit['installed_split_counts']=={'train':1032,'query':591,'gallery':1055}
        condition=load(folder/'q_RNT_g_RNT/condition_cpu_audit.json')['measured'];rows={}
        for state in ('00','10','01','11'):
            value=condition['metrics'][state]
            metrics.append(dict(model=model,state=state,selected_epoch=result['selected_epoch'],**{k:value[k] for k in METRICS}))
            with (folder/'q_RNT_g_RNT'/('state_'+state+'.csv')).open(encoding='utf-8-sig',newline='') as f:rows[state]=list(csv.DictReader(f))
            assert len(rows[state])==591
        base_rows.append(rows['00'])
        for a,b in zip(rows['00'],original_rows):
            assert set(a)==set(b)
            assert all(a[k]==b[k] for k in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery'))
            for key in a:
                if a[key]!=b[key]:
                    anchor_drift.append(dict(model=model,query_index=a['query_index'],name=a['name'],field=key,original_value=b[key],state00_value=a[key]))
        for reference in ('00','10','01'):
            a,b=condition['metrics']['11'],condition['metrics'][reference]
            rescue=sum(int(x['Rank-1'])>int(y['Rank-1']) for x,y in zip(rows['11'],rows[reference]));harm=sum(int(x['Rank-1'])<int(y['Rank-1']) for x,y in zip(rows['11'],rows[reference]))
            delta={k:a[k]-b[k] for k in METRICS};assert abs(delta['Rank-1']-100*(rescue-harm)/591)<1e-8
            comparisons.append(dict(model=model,improved='11',reference=reference,**delta,rank1_rescued=rescue,rank1_harmed=harm,AP_improved=sum(float(x['AP'])>float(y['AP']) for x,y in zip(rows['11'],rows[reference])),AP_worsened=sum(float(x['AP'])<float(y['AP']) for x,y in zip(rows['11'],rows[reference]))))
            pairs.extend(dict(model=model,reference=reference,query_index=x['query_index'],name=x['name'],identity=x['identity'],delta_AP_pp=100*(float(x['AP'])-float(y['AP'])),delta_Rank1=int(x['Rank-1'])-int(y['Rank-1'])) for x,y in zip(rows['11'],rows[reference]))
    assert len(metrics)==8 and len(comparisons)==6 and len(pairs)==3546
    assert base_rows[0]==base_rows[1]
    out=root/'normal4_analysis';out.mkdir(exist_ok=False)
    for name,records in (('six_metrics',metrics),('comparisons',comparisons),('paired_query_changes',pairs)):table(out/(name+'.csv'),records)
    table(out/'original_anchor_perquery_drift.csv',anchor_drift)
    result=dict(status='ACTUAL_R_COMPLETE_NORMAL_QUARTET_GT_CPU_READOUT',at=datetime.now().astimezone().isoformat(timespec='seconds'),metrics=metrics,comparisons=comparisons,paired_query_rows=len(pairs),normal11_equal_selected=True,normal00_perquery_equal_original=not anchor_drift,normal00_equal_between_controls=True,normal00_original_drift=anchor_drift,normal00_original_sixmetric_delta_pp={key:metrics[0][key]-original[key] for key in METRICS},missing_conditions=0,new_neural_calls=0,new_optimizer_updates=0,limits='Two fixed selected models/one teacher/expertseed42. GT recount and samequery/gallery fourstates. Original normal00 exact-equivalence assertion failed; all exported-field differences are retained rather than relaxed into equality. Numerical cause not established. No new49/missing selection/causal synergy or significance claim.')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],comparisons=comparisons),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
