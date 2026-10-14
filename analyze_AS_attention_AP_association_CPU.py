"""Associate saved, output-exact AS attention reads with labelled AP changes; CPU only."""
import csv, json, math, hashlib
from pathlib import Path
from analyze_identity_coordinate_three_normal import query_rows, table

P=Path(__file__).resolve().parent
AT=P/'results/r201at_AS_spatial_attention_20261014'
AS=P/'results/r201as_RGB_query_normalization_20261014/RGBNT100'
OUT=P/'results/rgbnt100_AS_attention_AP_association_CPU_20261014'

def correlation(x,y):
    mx=math.fsum(x)/len(x);my=math.fsum(y)/len(y)
    xx=[v-mx for v in x];yy=[v-my for v in y]
    a=math.fsum(v*v for v in xx);b=math.fsum(v*v for v in yy)
    assert a>0 and b>0
    return math.fsum(a*b for a,b in zip(xx,yy))/math.sqrt(a*b)

def main():
    assert not OUT.exists()
    baseline=P/'results/full_official_baselines_20261004/training/RGBNT100_demo_s42'
    original,base_metrics=query_rows(baseline,'RGBNT100')
    inputs=[baseline/'best_per_query.csv',baseline/'best_per_query.json']
    measurements=[];associations=[];groups=[];scores=[]
    keys=('index','name','identity','camera','scene')
    for variant in ('frequency_shared','axis_shared'):
        model='AS_'+variant;folder=AS/'training'/('RGBNT100_r201as_'+variant+'_s42')
        current,metrics=query_rows(folder,'RGBNT100')
        diagnostic=AT/'full'/model
        result=json.loads((diagnostic/'result.json').read_text('utf-8'))
        assert result['status']=='COMPLETE' and result['output_features_exact_saved_selected_cache']
        assert result['optimizer_updates']==result['checkpoint_writes']==0
        assert all(abs(result['full_six_metrics_equal_selected'][k]-metrics[k])<1e-8 for k in ('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20'))
        with (diagnostic/'spatial_attention_records.csv').open(encoding='utf-8',newline='') as f:
            reads=[r for r in csv.DictReader(f) if r['side']=='query']
        assert len(reads)==len(current)==len(original)==1715
        inputs.extend([folder/'best_per_query.csv',folder/'best_per_query.json',diagnostic/'spatial_attention_records.csv',diagnostic/'result.json'])
        values=[]
        for attention,now,before in zip(reads,current,original):
            assert int(attention['index'])==int(now['query_index'])==int(before['query_index'])
            assert all(attention[k]==now[k]==before[k] for k in keys[1:])
            role=('correct_preserved' if int(before['Rank-1']) else 'rescued') if int(now['Rank-1']) else ('newly_harmed' if int(before['Rank-1']) else 'remaining_original_error')
            assert role==attention['role']
            row=dict(model=model,query_index=int(now['query_index']),name=now['name'],identity=int(now['identity']),camera=int(now['camera']),role=role,
                original_AP=float(before['AP']),selected_AP=float(now['AP']),delta_AP_pp=100*(float(now['AP'])-float(before['AP'])))
            for band in range(3):
                row['M_conditioned_band'+str(band)+'_max_lift']=float(attention[f'M_1_band{band}_head_mean_probability_maximum_lift'])
                row['F_slots_average_band'+str(band)+'_max_lift']=math.fsum(float(attention[f'F_{s}_band{band}_head_mean_probability_maximum_lift']) for s in range(3))/3
            values.append(row)
        features=[k for k in values[0] if k.endswith('_max_lift')]
        ids={r['identity'] for r in values};assert len(ids)==50
        by_identity={i:[r for r in values if r['identity']==i] for i in ids}
        identity_AP={i:math.fsum(r['delta_AP_pp'] for r in rows)/len(rows) for i,rows in by_identity.items()}
        for feature in features:
            x=[r[feature] for r in values];y=[r['delta_AP_pp'] for r in values]
            identity_x={i:math.fsum(r[feature] for r in rows)/len(rows) for i,rows in by_identity.items()}
            associations.append(dict(model=model,feature=feature,queries=1715,identities=50,minimum=min(x),maximum=max(x),mean=math.fsum(x)/len(x),
                Pearson_query_weighted=correlation(x,y),Pearson_within_identity=correlation([r[feature]-identity_x[r['identity']] for r in values],[r['delta_AP_pp']-identity_AP[r['identity']] for r in values])))
        for role in ('correct_preserved','rescued','newly_harmed','remaining_original_error'):
            rows=[r for r in values if r['role']==role];assert rows
            groups.append(dict(model=model,role=role,queries=len(rows),identities=len({r['identity'] for r in rows}),
                mean_delta_AP_pp=math.fsum(r['delta_AP_pp'] for r in rows)/len(rows),**{feature:math.fsum(r[feature] for r in rows)/len(rows) for feature in features}))
        scores.append(dict(model=model,mAP=metrics['mAP'],Rank1=metrics['Rank-1'],delta_mAP=metrics['mAP']-base_metrics['mAP'],delta_Rank1=metrics['Rank-1']-base_metrics['Rank-1']))
        measurements.extend(values)
    OUT.mkdir()
    table(OUT/'paired_query_measurements.csv',measurements)
    table(OUT/'attention_AP_associations.csv',associations)
    table(OUT/'rank1_role_means.csv',groups)
    result=dict(status='ACTUAL_SAVED_AS_ATTENTION_AP_ASSOCIATION_CPU_COMPLETE',scores=scores,associations=associations,role_groups=groups,paired_query_rows=len(measurements),new_neural_forwards=0,new_optimizer_updates=0,remote_training_queries=0,
        inputs={p.relative_to(P).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs},
        limits='Exploratory association at the same benchmark-selected AS weights, not causation, routing supervision, independent test or seed variance. Query-weighted Pearson and within-identity centered Pearson retain query weights; neither is an identity bootstrap or an independent sample confidence interval. F slots in ordinary experts are not physical modality labels. Max-lift is conditioned within each source before averaging, not foreground or identity relevance. F deviations near1 may be dominated by tiny FP32 fluctuations and do not imply meaningful selective reading. No AV progress, extra neural run, new score, missing protocol or training-source change.')
    (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps({k:result[k] for k in ('status','scores','associations','role_groups')},ensure_ascii=False))

if __name__=='__main__':main()
