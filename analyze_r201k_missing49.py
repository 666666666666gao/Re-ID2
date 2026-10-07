"""All K42 conditions/states against fixed I42 and original DeMo; CPU only."""
from datetime import datetime
import hashlib,json,math
from pathlib import Path

from analyze_identity_coordinate_missing49 import SETS,STATES,availability_groups,load,verified_rows
from analyze_identity_coordinate_three_normal import COUNTS,METRICS,table

PROJECT=Path(__file__).resolve().parent
ROOT=PROJECT/'results/r201k_missing49_20261007'


def main():
    review=load(PROJECT/'results/preflight/r201k_missing49_source_review_20261007.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
    controller=load(ROOT/'controller_full_result.json');registry=load(ROOT/'jobs.json')
    jobs=registry['jobs'];assert controller['status']=='COMPLETE' and controller['controls']==len(jobs)
    assert controller['raw_conditions']==49*len(jobs) and controller['missing_GT_state_cases']==196*len(jobs)
    assert controller['optimizer_updates']==controller['new_weights']==0 and controller['all_raw_local_verified']
    metrics,groups,deltas,group_deltas,pairs,geometry=[],[],[],[],[],[]
    checked=0
    for dataset,(_,count,gallery,_) in COUNTS.items():
        original=PROJECT/'results/full_official_baselines_20261004/frozen49'/(dataset+'_demo_s42')
        assert load(original/'independent_cpu_audit.json')['status']=='PASS'
        models={'original_DeMo':(load(original/'result.json')['measurements'],original,'baseline')}
        for family in ('I','K'):
            base=PROJECT/'results/r201i_missing49_20261007' if family=='I' else ROOT
            entries=[j for j in load(base/'jobs.json')['jobs'] if j['dataset']==dataset and j['seed']==42]
            assert len(entries)==2 and {j['variant'] for j in entries}=={'frequency_shared','axis_shared'}
            for job in entries:
                folder=base/'full'/job['name'];result=load(folder/'result.json');audit=load(folder/'independent_cpu_audit.json')
                assert result['status']=='COMPLETE' and result['conditions']==49 and result['state_cases']==196
                assert audit['status']=='PASS' and audit['conditions']==49 and audit['state_cases']==196
                assert result['optimizer_updates']==result['new_weights']==0 and result['normal_features_and_distance_exact']
                assert result['model_arguments']['seed']==job['seed'] and result['variant']==job['variant']
                if family in ('I','K'):
                    assert result['selected_epoch']==job['selected_epoch']
                    assert all(result['measurements']['q_RNT_g_RNT']['11'][k]==job['full_metrics'][k] for k in METRICS)
                    native=load(base/'native'/job['name']/'result.json')
                    assert native['original_inputs']==result['original_inputs']
                if family=='K':
                    assert job['forward_graph']=='relation_local_shared_PI'
                    launch=load(base/'full'/(job['name']+'_launch.json'))
                    assert launch['command'][2]=='evaluate_r201k_missing49_stream.py'
                label=family+'_'+job['variant']+'_s'+str(job['seed'])
                for state in STATES:
                    models[label+'/'+state]=({condition:v[state] for condition,v in result['measurements'].items()},folder,state)
                for condition,item in audit['measured'].items():
                    if 'disjoint_private_distance' in item:
                        assert item['cross_source_identity_comparison_not_established']
                        geometry.extend(dict(dataset=dataset,model=label,condition=condition,state=s,**v) for s,v in item['disjoint_private_distance'].items())
        axis='K_axis_shared_s42';freq='K_frequency_shared_s42'
        comparisons=[(axis+'/11',r) for r in ('original_DeMo',axis+'/00',axis+'/10',axis+'/01',freq+'/11','I_axis_shared_s42/11')]
        comparisons += [(freq+'/11',r) for r in ('original_DeMo',freq+'/00',freq+'/10',freq+'/01','I_frequency_shared_s42/11')]
        exported={(axis+'/11',axis+'/00'),(axis+'/11',freq+'/11')}
        local_deltas=[]
        for qs in SETS:
            for gs in SETS:
                condition='q_'+qs+'_g_'+gs;rows={}
                for model,(values,folder,kind) in models.items():
                    file=folder/(condition+'.csv') if kind=='baseline' else folder/condition/('state_'+kind+'.csv')
                    current=verified_rows(file,values[condition],count,gallery)
                    checked+=len(current);rows[model]=current
                    metrics.append(dict(dataset=dataset,model=model,condition=condition,**{k:values[condition][k] for k in METRICS}))
                    groups.extend(dict(dataset=dataset,model=model,condition=condition,grouping=kind,**v) for kind,gs in values[condition]['groups'].items() for v in gs)
                assert rows[axis+'/00']==rows[freq+'/00']==rows['I_axis_shared_s42/00']==rows['I_frequency_shared_s42/00']
                for improved,reference in comparisons:
                    left,right=rows[improved],rows[reference]
                    assert all(all(a[k]==b[k] for k in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')) for a,b in zip(left,right))
                    av,bv=models[improved][0][condition],models[reference][0][condition]
                    diff={k:av[k]-bv[k] for k in METRICS}
                    rescued=sum(int(a['Rank-1'])==1 and int(b['Rank-1'])==0 for a,b in zip(left,right))
                    harmed=sum(int(a['Rank-1'])==0 and int(b['Rank-1'])==1 for a,b in zip(left,right))
                    assert abs(diff['Rank-1']-100*(rescued-harmed)/count)<1e-8
                    row=dict(dataset=dataset,condition=condition,improved=improved,reference=reference,**diff,rescued=rescued,harmed=harmed,AP_improved=sum(float(a['AP'])>float(b['AP']) for a,b in zip(left,right)),AP_worsened=sum(float(a['AP'])<float(b['AP']) for a,b in zip(left,right)),both_plus2=diff['mAP']>=2 and diff['Rank-1']>=2)
                    deltas.append(row);local_deltas.append(row)
                    if (improved,reference) in exported:
                        pairs.extend(dict(dataset=dataset,condition=condition,improved=improved,reference=reference,**{k:a[k] for k in ('query_index','name','identity','camera','scene')},delta_AP_pp=100*(float(a['AP'])-float(b['AP'])),delta_INP_pp=100*(float(a['INP'])-float(b['INP'])),delta_rank1=int(a['Rank-1'])-int(b['Rank-1'])) for a,b in zip(left,right))
        for improved,reference in comparisons:
            for group,size in dict(all49=49,same_availability=7,overlap_mismatch=30,source_disjoint=12,partial_query_full_gallery=6,full_query_partial_gallery=6,both_partial=36).items():
                chosen=[r for r in local_deltas if r['improved']==improved and r['reference']==reference and group in availability_groups(*r['condition'][2:].split('_g_'))]
                assert len(chosen)==size
                group_deltas.append(dict(dataset=dataset,improved=improved,reference=reference,group=group,conditions=size,**{k:math.fsum(r[k] for r in chosen)/size for k in METRICS},rescued=sum(r['rescued'] for r in chosen),harmed=sum(r['harmed'] for r in chosen),both_plus2_conditions=sum(r['both_plus2'] for r in chosen)))
    assert len(jobs)==6 and len(metrics)==2499 and len(deltas)==1617 and checked==2617286 and len(pairs)==307916
    out=ROOT/'missing_analysis';out.mkdir(exist_ok=False)
    for name,values in (('six_metrics',metrics),('group_metrics',groups),('comparisons',deltas),('availability_group_deltas',group_deltas),('disjoint_distance_geometry',geometry)):
        table(out/(name+'.csv'),values)
    for dataset in COUNTS:
        table(out/(dataset+'_paired_query_changes.csv'),[r for r in pairs if r['dataset']==dataset])
    result=dict(status='ACTUAL_K_THREE_DATASET_FIXED_NORMAL_BEST_ALL49_CPU_COMPLETE',completed_at=datetime.now().isoformat(timespec='seconds'),controls=len(jobs),datasets=3,conditions=49*len(jobs),state_cases=196*len(jobs),model_state_metric_rows=len(metrics),comparisons=len(deltas),checked_condition_query_rows=checked,unique_queries=3142,paired_query_rows=len(pairs),all_six_CMC50_perquery_identity_camera_scene=True,new_neural_calls=0,new_optimizer_updates=0,
        limits='Unified K42 against fixed I42/DeMo. Every metric and missing state uses the same normal mAP-best earliest-tie checkpoint. Benchmark-selected, original50 versus two-stage100 budgets disclosed; ordinary control is mixed-source. Current K is one learner seed on original DeMo42, not whole-pipeline variance. Equal-condition means/repeatedqueries are diagnostic, not independent. The12 disjoint private distances do not establish cross-source identity comparison. Direct PI input locality is not end-to-end source isolation, calibrated/causal or retrained psi necessity. All negatives/rescue/harm remain; no +2 success inferred from completed execution.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','controls','conditions','state_cases','paired_query_rows')}),flush=True)


if __name__=='__main__':main()
