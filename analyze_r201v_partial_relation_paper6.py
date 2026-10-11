"""Stdlib CPU reduction of paper-six fixed-normal checkpoints, including all negatives."""
from datetime import datetime
import csv
import hashlib
import json
import math
from pathlib import Path

from analyze_identity_coordinate_missing49 import STATES,load,verified_rows
from analyze_identity_coordinate_three_normal import COUNTS,METRICS,table

PROJECT=Path(__file__).resolve().parent
ROOT=PROJECT/'results/r201v_partial_relation_paper6_20261011'
AVAILABLE=('NT','RT','RN','T','N','R')
MISSING=('R','N','T','RN','RT','NT')


def main():
    review=load(PROJECT/'results/preflight/r201v_partial_relation_paper6_source_review_20261011.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
    controller=load(ROOT/'controller_full_result.json');registry=load(ROOT/'jobs.json')
    assert controller['status']=='COMPLETE' and controller['controls']==4
    assert controller['raw_conditions']==24 and controller['missing_GT_state_cases']==96
    assert controller['optimizer_updates']==controller['new_weights']==0 and controller['all_raw_local_verified']
    assert len(registry['jobs'])==4 and not registry['missing_reselection']
    metrics,groups,deltas,means,pairs=[],[],[],[],[]
    normal_metrics,normal_groups,normal_deltas,normal_pairs=[],[],[],[]
    checked=0
    normal_checked=0
    for dataset,(_,count,gallery,_) in COUNTS.items():
        original=PROJECT/'results/full_official_baselines_20261004/frozen49'/(dataset+'_demo_s42')
        assert load(original/'independent_cpu_audit.json')['status']=='PASS'
        original_result=load(original/'result.json')
        assert original_result['status']=='COMPLETE'
        models={'original_DeMo':(original_result['measurements'],original,'baseline')}
        normal_models={'original_DeMo':(original_result['measurements']['q_RNT_g_RNT'],original,'baseline')}
        base=PROJECT/'results/r201k_missing49_20261007' if dataset=='RGBNT201' else ROOT
        jobs=[j for j in load(base/'jobs.json')['jobs'] if j['dataset']==dataset and j['seed']==42]
        assert len(jobs)==2 and {j['variant'] for j in jobs}=={'axis_shared','frequency_shared'}
        family={'RGBNT201':'K','MSVR310':'T','RGBNT100':'S'}[dataset]
        for job in jobs:
            folder=base/'full'/job['name'];result=load(folder/'result.json');audit=load(folder/'independent_cpu_audit.json')
            native=load(base/'native'/job['name']/'result.json')
            native_folder=base/'native'/job['name']
            native_audit=load(native_folder/'independent_cpu_audit.json')
            assert native['status']=='COMPLETE' and native_audit['status']=='PASS'
            assert native['conditions']==native_audit['conditions']==1 and native['state_cases']==native_audit['state_cases']==4
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
                assert result['base_inference_policy']==native['base_inference_policy']=='base_any_observed_new_experts_all_required'
                assert result['parameters_unchanged'] and native['parameters_unchanged']
                assert result['final_unit_shift_epsilon']==.10
                assert all(b['epsilon']==.10 and b['all_samples_budget_pass'] and max(b['maximum_shift'].values())<=.10+1e-6 for b in result['final_descriptor_budgets'].values())
                assert result['state_tensor_versions_unchanged'] and set(result['measurements'])=={'q_'+a+'_g_'+a for a in AVAILABLE}
                assert all(native['measurements']['q_RNT_g_RNT']['11'][k]==job['full_metrics'][k] for k in METRICS)
            label=family+'_'+job['variant']+'_s42'
            for state in STATES:
                models[label+'/'+state]=({c:v[state] for c,v in result['measurements'].items()},folder,state)
                normal_models[label+'/'+state]=(native['measurements']['q_RNT_g_RNT'][state],native_folder,state)
        axis=family+'_axis_shared_s42';ordinary=family+'_frequency_shared_s42'
        comparisons=[(axis+'/11',r) for r in ('original_DeMo',axis+'/00',axis+'/10',axis+'/01',ordinary+'/11')]
        comparisons.extend((ordinary+'/11',r) for r in ('original_DeMo',ordinary+'/00',ordinary+'/10',ordinary+'/01'))
        normal_reads={}
        for model,(summary,folder,state) in normal_models.items():
            condition='q_RNT_g_RNT'
            path=folder/(condition+'.csv') if state=='baseline' else folder/condition/('state_'+state+'.csv')
            normal_reads[model]=verified_rows(path,summary,count,gallery)
            normal_checked+=len(normal_reads[model])
            normal_metrics.append(dict(dataset=dataset,model=model,condition=condition,**{k:summary[k] for k in METRICS}))
            for group in ('identity','camera','scene'):
                normal_groups.extend(dict(dataset=dataset,model=model,condition=condition,axis=group,value=g['value'],queries=g['queries'],**{k:g[k] for k in METRICS}) for g in summary['groups'][group])
        for improved,reference in comparisons:
            left,right=normal_reads[improved],normal_reads[reference]
            assert all(all(a[k]==b[k] for k in ('query_index','name','identity','camera','scene','valid','kept_gallery','relevant_gallery')) for a,b in zip(left,right))
            delta={k:normal_models[improved][0][k]-normal_models[reference][0][k] for k in METRICS}
            rescued=sum(int(a['Rank-1'])==1 and int(b['Rank-1'])==0 for a,b in zip(left,right))
            harmed=sum(int(a['Rank-1'])==0 and int(b['Rank-1'])==1 for a,b in zip(left,right))
            assert abs(delta['Rank-1']-100*(rescued-harmed)/count)<1e-8
            normal_deltas.append(dict(dataset=dataset,condition='q_RNT_g_RNT',improved=improved,reference=reference,**delta,
                rescued=rescued,harmed=harmed,AP_improved=sum(float(a['AP'])>float(b['AP']) for a,b in zip(left,right)),
                AP_worsened=sum(float(a['AP'])<float(b['AP']) for a,b in zip(left,right)),both_plus1=delta['mAP']>=1 and delta['Rank-1']>=1))
            if (improved,reference) in ((axis+'/11','original_DeMo'),(axis+'/11',axis+'/00'),(axis+'/11',ordinary+'/11')):
                normal_pairs.extend(dict(dataset=dataset,condition='q_RNT_g_RNT',improved=improved,reference=reference,
                    **{k:a[k] for k in ('query_index','name','identity','camera','scene')},
                    delta_AP_pp=100*(float(a['AP'])-float(b['AP'])),delta_INP_pp=100*(float(a['INP'])-float(b['INP'])),
                    delta_rank1=int(a['Rank-1'])-int(b['Rank-1'])) for a,b in zip(left,right))
        if dataset!='RGBNT201':
            for variant in ('axis_shared','frequency_shared'):
                folder=PROJECT/'results/r201p_paper_missing6_20261010/full'/(dataset+'_r201p_'+variant+'_s42')
                result=load(folder/'result.json');audit=load(folder/'independent_cpu_audit.json')
                assert result['status']=='COMPLETE' and audit['status']=='PASS' and result['conditions']==6
                assert result['optimizer_updates']==result['new_weights']==0 and result['normal_features_and_distance_exact']
                label='P_'+variant+'_s42/11'
                models[label]=({c:v['11'] for c,v in result['measurements'].items()},folder,'11')
                comparisons.append((axis+'/11',label))
            comparisons.append((ordinary+'/11','P_frequency_shared_s42/11'))
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
                if (improved,reference) in ((axis+'/11',axis+'/00'),(axis+'/11',ordinary+'/11'),(axis+'/11','P_axis_shared_s42/11')):
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
    assert len(normal_metrics)==len(normal_deltas)==27 and normal_checked==9*3142 and len(normal_pairs)==3*3142
    out=ROOT/'missing_analysis';out.mkdir(exist_ok=False)
    for name,rows in (('six_metrics',metrics),('paper_six_means',means),('group_metrics',groups),('comparisons',deltas),('paired_query_changes',pairs)):table(out/(name+'.csv'),rows)
    for name,rows in (('normal_four_state_six_metrics',normal_metrics),('normal_four_state_comparisons',normal_deltas),('normal_four_state_group_metrics',normal_groups),('normal_four_state_paired_query_changes',normal_pairs)):table(out/(name+'.csv'),rows)
    result=dict(status='ACTUAL_V_THREE_DATASET_PAPER_SIX_CPU_COMPLETE_K201_RETAINED',completed_at=datetime.now().isoformat(timespec='seconds'),
        new_controls=4,new_missing_conditions=24,new_state_cases=96,reused_RGBNT201_conditions=12,reused_RGBNT201_state_cases=48,
        checked_condition_query_rows=checked,unique_queries=3142,paired_query_rows=len(pairs),
        six_metric_rows=len(metrics),paper_six_mean_rows=len(means),all_six_CMC50_perquery_identity_camera_scene=True,
        normal_four_state_metric_rows=len(normal_metrics),normal_four_state_comparison_rows=len(normal_deltas),
        normal_checked_query_rows=normal_checked,normal_paired_query_rows=len(normal_pairs),normal_comparisons=normal_deltas,
        mean_comparisons=[r for r in deltas if r['missing']=='paper_six_mean'],new_neural_calls=0,new_optimizer_updates=0,new49=False,
        limits='Six DeMo paper symmetric masks/mean only. Fixed normal benchmark-selected best; all official records and GT filters. MSVR is our extension. K201 .3 unchanged; TMS mean+obsolete scorer freeze1024 and S RGB scored-relation auxiliary outlet use different training recipes on the same5120 framework; .10 bound/.6 mainmargin held, P11 references reusedCPU only as context, not a single-factor training comparison. Original50 versus expert50+50, mixed-source ordinary. Repeated query/state/conditions are not independent. Completed computation does not imply +1, causality, calibrated contributions or retrained-module necessity; every negative retained.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','new_missing_conditions','checked_condition_query_rows','paired_query_rows')}),flush=True)

def verify_normal_parent(root,parent,jobs):
    checked,cases=0,0
    for job in jobs:
        name=job['name'];new_folder=root/'native'/name;old_folder=parent/'native'/name
        new=load(new_folder/'result.json');old=load(old_folder/'result.json')
        assert new['status']==old['status']=='COMPLETE' and new['mode']==old['mode']=='native'
        assert load(new_folder/'independent_cpu_audit.json')['status']==load(old_folder/'independent_cpu_audit.json')['status']=='PASS'
        assert new['selected_epoch']==old['selected_epoch']==job['selected_epoch']
        assert new['model_arguments']==old['model_arguments'] and new['variant']==old['variant']==job['variant']
        assert {n.replace('restore_V_partial_paper6.npz','restore_ST_paper6.npz'):sha for n,sha in new['original_inputs'].items()}==old['original_inputs']
        assert new['normal_features_and_distance_exact'] and new['state_tensor_versions_unchanged']
        assert new['optimizer_updates']==new['new_weights']==0 and new['conditions']==1 and new['state_cases']==4
        count={'MSVR310':591,'RGBNT100':1715}[job['dataset']]
        assert new['query_records']==old['query_records']==count
        for state in STATES:
            assert new['measurements']['q_RNT_g_RNT'][state]==old['measurements']['q_RNT_g_RNT'][state],'Full-normal state summary changed'
            def read(folder):
                with (folder/'q_RNT_g_RNT'/('state_'+state+'.csv')).open(encoding='utf-8') as stream:
                    return list(csv.DictReader(stream))
            nr,ore=read(new_folder),read(old_folder)
            assert len(nr)==len(ore)==count and nr==ore,'Full-normal per-query state changed'
            checked+=count;cases+=1
    assert len(jobs)==4 and cases==16 and checked==18448
    return dict(status='ACTUAL_V_NATIVE_ALL4_NORMAL_ST_PARENT_PARITY_PASS',controls=4,state_cases=16,
        checked_per_query_rows=checked,normal_summaries_and_all_query_fields_exact=True,
        new_neural_calls=0,new_optimizer_updates=0,missing_started=False,
        limits='CPU equality against already completed ST native text on the same fixed best weights; each new native state separately audited against installed GT. No parent neural replay or raw bundle copy.')


def compare_parent_base_policy():
    parent=PROJECT/'results/r201st_paper_missing6_20261011'
    assert load(parent/'controller_full_result.json')['status']=='COMPLETE'
    parity=load(ROOT/'native_parent_policy_parity.json')
    assert parity['status']=='ACTUAL_V_NATIVE_ALL4_NORMAL_ST_PARENT_PARITY_PASS' and parity['checked_per_query_rows']==18448 and parity['state_cases']==16
    rows,pairs,means=[],[],[]
    normal_checked=0
    for dataset,(_,count,gallery,_) in COUNTS.items():
        if dataset=='RGBNT201':continue
        family={'MSVR310':'T','RGBNT100':'S'}[dataset]
        for variant in ('axis_shared','frequency_shared'):
            name=dataset+'_r201'+family.lower()+'_'+variant+'_s42'
            new_folder=ROOT/'full'/name;old_folder=parent/'full'/name
            new=load(new_folder/'result.json');old=load(old_folder/'result.json')
            assert new['selected_epoch']==old['selected_epoch'] and new['model_arguments']==old['model_arguments']
            assert new['variant']==old['variant']==variant
            assert {n.replace('restore_V_partial_paper6.npz','restore_ST_paper6.npz'):sha for n,sha in new['original_inputs'].items()}==old['original_inputs']
            new_native_folder=ROOT/'native'/name;old_native_folder=parent/'native'/name
            new_native=load(new_native_folder/'result.json');old_native=load(old_native_folder/'result.json')
            for state in STATES:
                ns=new_native['measurements']['q_RNT_g_RNT'][state];os=old_native['measurements']['q_RNT_g_RNT'][state]
                assert all(ns[k]==os[k] for k in METRICS),'Full-normal state metric changed'
                nr=verified_rows(new_native_folder/'q_RNT_g_RNT'/('state_'+state+'.csv'),ns,count,gallery)
                ore=verified_rows(old_native_folder/'q_RNT_g_RNT'/('state_'+state+'.csv'),os,count,gallery)
                assert nr==ore,'Full-normal per-query state changed'
                normal_checked+=count
            for state in ('00','11'):
                members=[]
                for available,missing in zip(AVAILABLE,MISSING):
                    condition='q_'+available+'_g_'+available
                    ns=new['measurements'][condition][state];os=old['measurements'][condition][state]
                    nr=verified_rows(new_folder/condition/('state_'+state+'.csv'),ns,count,gallery)
                    ore=verified_rows(old_folder/condition/('state_'+state+'.csv'),os,count,gallery)
                    assert all(all(n[k]==o[k] for k in ('query_index','name','identity','camera','scene','valid','kept_gallery','relevant_gallery')) for n,o in zip(nr,ore))
                    d={k:ns[k]-os[k] for k in METRICS}
                    rescued=sum(int(n['Rank-1'])==1 and int(o['Rank-1'])==0 for n,o in zip(nr,ore))
                    harmed=sum(int(n['Rank-1'])==0 and int(o['Rank-1'])==1 for n,o in zip(nr,ore))
                    assert abs(d['Rank-1']-100*(rescued-harmed)/count)<1e-8
                    row=dict(dataset=dataset,variant=variant,state=state,missing=missing,condition=condition,**d,rescued=rescued,harmed=harmed)
                    rows.append(row);members.append(row)
                    pairs.extend(dict(dataset=dataset,variant=variant,state=state,condition=condition,**{k:n[k] for k in ('query_index','name','identity','camera','scene')},
                        delta_AP_pp=100*(float(n['AP'])-float(o['AP'])),delta_INP_pp=100*(float(n['INP'])-float(o['INP'])),delta_rank1=int(n['Rank-1'])-int(o['Rank-1'])) for n,o in zip(nr,ore))
                means.append(dict(dataset=dataset,variant=variant,state=state,conditions=6,**{k:math.fsum(r[k] for r in members)/6 for k in METRICS}))
    assert len(rows)==48 and len(means)==8 and len(pairs)==24*(591+1715) and normal_checked==8*(591+1715)
    out=ROOT/'missing_analysis'
    for name,value in (('base_policy_condition_deltas',rows),('base_policy_six_mean_deltas',means),('base_policy_paired_query_changes',pairs)):
        table(out/(name+'.csv'),value)
    result=dict(status='ACTUAL_V_FIXED_BEST_BASE_POLICY_VERSUS_ST_CPU_COMPLETE',condition_comparisons=48,six_mean_comparisons=8,paired_query_rows=len(pairs),normal_per_query_state_parity_checked=normal_checked,
        source_factor='Frozen DeMo base all-required versus any-observed relation support; all current weights and strict new-expert source masks unchanged. Normal all4 states per-query identical.',
        limits='Posthoc inference control, not retraining/new model weights or a proved research gain. Base changes downstream residual anchors/gates, so only weights/first-stage inputs remain fixed. OriginalDeMo50 versus frozenexpert50+50 budget and benchmark-selected weights disclosed. No new201NN/49/seedsearch.',mean_deltas=means)
    (out/'base_policy_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','condition_comparisons','paired_query_rows','normal_per_query_state_parity_checked')}),flush=True)



if __name__=='__main__':
    main()
    compare_parent_base_policy()
