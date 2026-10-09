"""Stdlib CPU reduction of paper-six fixed-normal checkpoints, including all negatives."""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from analyze_identity_coordinate_missing49 import STATES,load,verified_rows
from analyze_identity_coordinate_three_normal import COUNTS,METRICS,table

PROJECT=Path(__file__).resolve().parent
ROOT=PROJECT/'results/r201n_paper_missing6_20261009'
AVAILABLE=('NT','RT','RN','T','N','R')
MISSING=('R','N','T','RN','RT','NT')


def main():
    review=load(PROJECT/'results/preflight/r201n_paper_missing6_source_review_20261009.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
    controller=load(ROOT/'controller_full_result.json');registry=load(ROOT/'jobs.json')
    assert controller['status']=='COMPLETE' and controller['controls']==4
    assert controller['raw_conditions']==24 and controller['missing_GT_state_cases']==96
    assert controller['optimizer_updates']==controller['new_weights']==0 and controller['all_raw_local_verified']
    assert len(registry['jobs'])==4 and not registry['missing_reselection']
    metrics,groups,deltas,means,pairs=[],[],[],[],[]
    checked=0
    for dataset,(_,count,gallery,_) in COUNTS.items():
        original=PROJECT/'results/full_official_baselines_20261004/frozen49'/(dataset+'_demo_s42')
        assert load(original/'independent_cpu_audit.json')['status']=='PASS'
        original_result=load(original/'result.json')
        assert original_result['status']=='COMPLETE'
        models={'original_DeMo':(original_result['measurements'],original,'baseline')}
        base=PROJECT/'results/r201k_missing49_20261007' if dataset=='RGBNT201' else ROOT
        jobs=[j for j in load(base/'jobs.json')['jobs'] if j['dataset']==dataset and j['seed']==42]
        assert len(jobs)==2 and {j['variant'] for j in jobs}=={'axis_shared','frequency_shared'}
        family='K' if dataset=='RGBNT201' else 'N'
        for job in jobs:
            folder=base/'full'/job['name'];result=load(folder/'result.json');audit=load(folder/'independent_cpu_audit.json')
            native=load(base/'native'/job['name']/'result.json')
            assert result['status']=='COMPLETE' and audit['status']=='PASS'
            assert result['conditions']==audit['conditions']==(49 if dataset=='RGBNT201' else 6)
            assert result['state_cases']==audit['state_cases']==4*result['conditions']
            assert result['optimizer_updates']==result['new_weights']==0 and result['normal_features_and_distance_exact']
            assert result['selected_epoch']==job['selected_epoch'] and result['variant']==job['variant']
            assert native['original_inputs']==result['original_inputs']
            assert set(native['native_small_batch_checks'])=={'R','N','T','RN','RT','NT','RNT'}
            if dataset=='RGBNT201':
                assert all(result['measurements']['q_RNT_g_RNT']['11'][k]==job['full_metrics'][k] for k in METRICS)
                if job['variant']=='axis_shared':assert result['selected_epoch']==30
            else:
                assert result['final_unit_shift_epsilon']==.10
                assert all(b['epsilon']==.10 and b['all_samples_budget_pass'] and max(b['maximum_shift'].values())<=.10+1e-6 for b in result['final_descriptor_budgets'].values())
                assert result['state_tensor_versions_unchanged'] and set(result['measurements'])=={'q_'+a+'_g_'+a for a in AVAILABLE}
                assert all(native['measurements']['q_RNT_g_RNT']['11'][k]==job['full_metrics'][k] for k in METRICS)
            label=family+'_'+job['variant']+'_s42'
            for state in STATES:models[label+'/'+state]=({c:v[state] for c,v in result['measurements'].items()},folder,state)
        axis=family+'_axis_shared_s42';ordinary=family+'_frequency_shared_s42'
        comparisons=[(axis+'/11',r) for r in ('original_DeMo',axis+'/00',axis+'/10',axis+'/01',ordinary+'/11')]
        comparisons.extend((ordinary+'/11',r) for r in ('original_DeMo',ordinary+'/00',ordinary+'/10',ordinary+'/01'))
        if dataset!='RGBNT201':
            for variant in ('axis_shared','frequency_shared'):
                folder=PROJECT/'results/r201m_paper_missing6_20261009/full'/(dataset+'_r201m_'+variant+'_s42')
                result=load(folder/'result.json');audit=load(folder/'independent_cpu_audit.json')
                assert result['status']=='COMPLETE' and audit['status']=='PASS' and result['conditions']==6
                assert result['optimizer_updates']==result['new_weights']==0 and result['normal_features_and_distance_exact']
                label='M_'+variant+'_s42/11'
                models[label]=({c:v['11'] for c,v in result['measurements'].items()},folder,'11')
                comparisons.append((axis+'/11',label))
            comparisons.append((ordinary+'/11','M_frequency_shared_s42/11'))
        local=[]
        for available,missing in zip(AVAILABLE,MISSING):
            condition='q_'+available+'_g_'+available;reads={}
            for model,(values,folder,state) in models.items():
                summary=values[condition]
                path=folder/(condition+'.csv') if state=='baseline' else folder/condition/('state_'+state+'.csv')
                rows=verified_rows(path,summary,count,gallery);reads[model]=rows;checked+=len(rows)
                metrics.append(dict(dataset=dataset,model=model,missing=missing,condition=condition,**{k:summary[k] for k in METRICS}))
                for group in ('identity','camera','scene'):
                    groups.extend(dict(dataset=dataset,model=model,condition=condition,axis=group,value=g['value'],queries=g['queries'],**{k:g[k] for k in METRICS}) for g in summary['groups'][group])
            for improved,reference in comparisons:
                left,right=reads[improved],reads[reference]
                assert all(all(a[k]==b[k] for k in ('query_index','name','identity','camera','scene','valid','kept_gallery','relevant_gallery')) for a,b in zip(left,right))
                delta={k:models[improved][0][condition][k]-models[reference][0][condition][k] for k in METRICS}
                rescued=sum(int(a['Rank-1'])==1 and int(b['Rank-1'])==0 for a,b in zip(left,right))
                harmed=sum(int(a['Rank-1'])==0 and int(b['Rank-1'])==1 for a,b in zip(left,right))
                assert abs(delta['Rank-1']-100*(rescued-harmed)/count)<1e-8
                row=dict(dataset=dataset,missing=missing,condition=condition,improved=improved,reference=reference,**delta,
                    rescued=rescued,harmed=harmed,AP_improved=sum(float(a['AP'])>float(b['AP']) for a,b in zip(left,right)),
                    AP_worsened=sum(float(a['AP'])<float(b['AP']) for a,b in zip(left,right)),both_plus1=delta['mAP']>=1 and delta['Rank-1']>=1)
                deltas.append(row);local.append(row)
                if (improved,reference) in ((axis+'/11',axis+'/00'),(axis+'/11',ordinary+'/11'),(axis+'/11','M_axis_shared_s42/11')):
                    pairs.extend(dict(dataset=dataset,condition=condition,improved=improved,reference=reference,
                        **{k:a[k] for k in ('query_index','name','identity','camera','scene')},
                        delta_AP_pp=100*(float(a['AP'])-float(b['AP'])),delta_INP_pp=100*(float(a['INP'])-float(b['INP'])),
                        delta_rank1=int(a['Rank-1'])-int(b['Rank-1'])) for a,b in zip(left,right))
        for model in models:
            rows=[r for r in metrics if r['dataset']==dataset and r['model']==model]
            assert len(rows)==6
            means.append(dict(dataset=dataset,model=model,missing='paper_six_mean',conditions=6,**{k:math.fsum(r[k] for r in rows)/6 for k in METRICS}))
        for improved,reference in comparisons:
            rows=[r for r in local if r['improved']==improved and r['reference']==reference];assert len(rows)==6
            delta={k:math.fsum(r[k] for r in rows)/6 for k in METRICS}
            deltas.append(dict(dataset=dataset,missing='paper_six_mean',condition='six_equal_conditions',improved=improved,reference=reference,**delta,
                rescued=sum(r['rescued'] for r in rows),harmed=sum(r['harmed'] for r in rows),
                AP_improved=sum(r['AP_improved'] for r in rows),AP_worsened=sum(r['AP_worsened'] for r in rows),
                both_plus1=delta['mAP']>=1 and delta['Rank-1']>=1))
    assert len(metrics)==186 and len(means)==31 and len(deltas)==231 and checked==54*3142+12*(591+1715) and len(pairs)==12*3142+6*(591+1715)
    out=ROOT/'missing_analysis';out.mkdir(exist_ok=False)
    for name,rows in (('six_metrics',metrics),('paper_six_means',means),('group_metrics',groups),('comparisons',deltas),('paired_query_changes',pairs)):table(out/(name+'.csv'),rows)
    result=dict(status='ACTUAL_N_THREE_DATASET_PAPER_SIX_CPU_COMPLETE_K201_RETAINED',completed_at=datetime.now().isoformat(timespec='seconds'),
        new_controls=4,new_missing_conditions=24,new_state_cases=96,reused_RGBNT201_conditions=12,reused_RGBNT201_state_cases=48,
        checked_condition_query_rows=checked,unique_queries=3142,paired_query_rows=len(pairs),
        six_metric_rows=len(metrics),paper_six_mean_rows=len(means),all_six_CMC50_perquery_identity_camera_scene=True,
        mean_comparisons=[r for r in deltas if r['missing']=='paper_six_mean'],new_neural_calls=0,new_optimizer_updates=0,new49=False,
        limits='Six DeMo paper symmetric masks/mean only. Fixed normal benchmark-selected best; all official records and GT filters. MSVR is our extension. K201 .3 unchanged, Nweak epsilon.10 on unchanged M .6 graph, same budget; closed M11 references reused CPU only. Original50 versus expert50+50, mixed-source ordinary. Repeated query/state/conditions are not independent. Completed computation does not imply +1, causality, calibrated contributions or retrained-module necessity; every negative retained.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','new_missing_conditions','checked_condition_query_rows','paired_query_rows')}),flush=True)


if __name__=='__main__':main()
