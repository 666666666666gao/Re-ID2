import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT

keys=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
sets=('R','N','T','RN','RT','NT','RNT')
identity=('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')
removed={'r':'NT','n':'RT','t':'RN','rn':'T','rt':'N','nt':'R'}
old_conditions={'q_RNT_g_RNT':'clean'}
for missing,retained in removed.items():
    old_conditions['q_'+retained+'_g_'+retained]='both_missing_'+missing
    old_conditions['q_'+retained+'_g_RNT']='query_missing_'+missing

def load(path):return json.loads((PROJECT/path).read_text(encoding='utf-8'))
def read_csv(path):
    with (PROJECT/path).open(encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))

def check_rows(path,metrics):
    rows=read_csv(path);valid=[row for row in rows if row['valid']=='True']
    assert len(rows)==metrics['query_count'] and len(valid)==metrics['valid_queries']
    for key,column in [('mAP','AP'),('mINP','INP'),*[(key,key) for key in keys[2:]]]:
        assert abs(100*sum(float(row[column]) for row in valid)/len(valid)-metrics[key])<1e-8,(path,key)
    assert len(metrics['CMC_1_to_50'])==50
    for rank,score in enumerate(metrics['CMC_1_to_50'],1):
        assert abs(100*sum(int(row['first_match'])<=rank for row in valid)/len(valid)-score)<1e-8
    for group_key,groups in metrics['groups'].items():
        assert sum(group['queries'] for group in groups)==len(valid)
        for group in groups:
            selected=[row for row in valid if int(row[group_key])==group['value']]
            assert len(selected)==group['queries']
            for key,column in [('mAP','AP'),('mINP','INP'),*[(key,key) for key in keys[2:]]]:
                assert abs(100*sum(float(row[column]) for row in selected)/len(selected)-group[key])<1e-8
    return rows

all_rows=[];paired_rows=[];report={}
for variant in ('demo','axis_mass_fullref'):
    campaign='results/anytoany49_'+variant+'_20261003'
    intake=load(campaign+'/intake.json')
    for relative,proof in intake['extracted_files'].items():
        path=PROJECT/campaign/relative
        assert path.stat().st_size==proof['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==proof['sha256']
    controller=load(campaign+'/controller_result.json');assert controller['status']=='COMPLETE' and len(controller['runs'])==3
    for dataset in ('MSVR310','RGBNT201','RGBNT100'):
        name=dataset+'_'+variant+'_s42';root=campaign+'/'+name
        if variant=='demo' and dataset=='RGBNT100':
            recheck='results/anytoany49_demo_RGBNT100_samehost_recheck_20261003'
            recheck_intake=load(recheck+'/intake.json')
            for relative,proof in recheck_intake['extracted_files'].items():
                path=PROJECT/recheck/relative
                assert path.stat().st_size==proof['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==proof['sha256']
            assert load(recheck+'/controller_result.json')['status']=='COMPLETE'
            root=recheck+'/'+name
        single=load(root+'/controller_result.json');assert single['status']=='PASS'
        assert len(single['stages'])==2 and all(stage['exit_code']==0 for stage in single['stages'])
        for stage in ('smoke','full'):assert load(root+'/'+stage+'_exit.json')['exit_code']==0
        smoke=load(root+'/smoke/smoke.json');assert smoke['status']=='PASS' and smoke['normal_feature_max_error']==0
        result=load(root+'/full/result.json')
        assert result['status']=='COMPLETE' and result['variant']==variant and result['dataset']==dataset
        assert result['normal_feature_max_error']==0 and result['optimizer_updates']==0 and result['state_tensor_versions_unchanged']
        expected={'q_'+q+'_g_'+g for q in sets for g in sets};assert set(result['measurements'])==expected
        if variant=='demo':
            oldroot=('results/axis_collaboration_v4_missing_development26_repaired/' if dataset=='RGBNT100' else 'results/axis_collaboration_v4_missing_development27/')+name+'/full'
        else:oldroot='results/axis_collaboration_v5_mass_frozen_development/'+name+'/missing_full'
        old=load(oldroot+'/result.json')
        conditions={}
        for condition,item in result['measurements'].items():
            q,g=condition[2:].split('_g_')
            metrics=item['metrics'];assert metrics==load(root+'/full/'+condition+'.json')
            rows=check_rows(root+'/full/'+condition+'.csv',metrics)
            if condition in old_conditions:
                previous=old['measurements'][old_conditions[condition]]['metrics']
                assert all(abs(metrics[key]-previous[key])<1e-8 for key in keys),(dataset,variant,condition)
                original=read_csv(oldroot+'/'+old_conditions[condition]+'.csv')
                assert len(rows)==len(original) and all(all(a[key]==b[key] for key in identity) for a,b in zip(rows,original))
                assert all(abs(float(a['AP'])-float(b['AP']))<1e-12 and int(a['first_match'])==int(b['first_match']) for a,b in zip(rows,original) if a['valid']=='True')
            all_rows.append(dict(dataset=dataset,model='DeMo' if variant=='demo' else 'V5',query_available=q,gallery_available=g,**{key:metrics[key] for key in keys}))
            conditions[condition]=dict(metrics=metrics,rows=rows)
        report[(dataset,variant)]=conditions

summaries={}
for dataset in ('MSVR310','RGBNT201','RGBNT100'):
    original=report[(dataset,'demo')];augmented=report[(dataset,'axis_mass_fullref')]
    pairs={}
    for condition in original:
        before,after=original[condition],augmented[condition]
        assert len(before['rows'])==len(after['rows']) and all(all(a[key]==b[key] for key in identity) for a,b in zip(before['rows'],after['rows']))
        valid=[(a,b) for a,b in zip(before['rows'],after['rows']) if a['valid']=='True']
        harmed=sum(int(a['Rank-1'])==1 and int(b['Rank-1'])==0 for a,b in valid)
        rescued=sum(int(a['Rank-1'])==0 and int(b['Rank-1'])==1 for a,b in valid)
        delta={key:after['metrics'][key]-before['metrics'][key] for key in keys}
        assert abs(100*sum(float(b['AP'])-float(a['AP']) for a,b in valid)/len(valid)-delta['mAP'])<1e-8
        assert abs(100*(rescued-harmed)/len(valid)-delta['Rank-1'])<1e-8
        record=dict(all_six_delta_pp=delta,valid_queries=len(valid),rank1_harmed_count=harmed,rank1_rescued_count=rescued,
            AP_improved_count=sum(float(b['AP'])>float(a['AP']) for a,b in valid),AP_worse_count=sum(float(b['AP'])<float(a['AP']) for a,b in valid),
            double_two_point_gate=delta['mAP']>=2 and delta['Rank-1']>=2)
        pairs[condition]=record
        q,g=condition[2:].split('_g_')
        paired_rows.append(dict(dataset=dataset,query_available=q,gallery_available=g,**{key:delta[key] for key in keys},
            rank1_harmed=harmed,rank1_rescued=rescued,double_two_point_gate=record['double_two_point_gate']))
    summaries[dataset]=dict(pairs=pairs,pairs_passing_double_two_point_gate=sum(row['double_two_point_gate'] for row in pairs.values()),
        all49_double_two_point_gate=all(row['double_two_point_gate'] for row in pairs.values()),
        worst_delta_mAP_pair=min(pairs,key=lambda name:pairs[name]['all_six_delta_pp']['mAP']),
        worst_delta_Rank1_pair=min(pairs,key=lambda name:pairs[name]['all_six_delta_pp']['Rank-1']))
output=PROJECT/'results/anytoany49_comparison_20261003';output.mkdir(exist_ok=False)
for filename,rows in [('all3_models_full49_metrics_294.csv',all_rows),('all3_full49_V5_minus_DeMo_147.csv',paired_rows)]:
    with (output/filename).open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
audit=dict(status='PASS_ACTUAL_FULL49_EVIDENCE_AUDIT',observed_at=datetime.now().isoformat(timespec='seconds'),datasets=summaries,
    metric_rows=294,paired_conditions=147,all_six_and_CMC50_and_groups_recomputed=True,all_prior13_metrics_and_queries_equal=True,
    all_model_pairs_same_GT_query_order=True,optimizer_updates=0,official_test_uses=0,
    limits='Frozen seed42 development diagnostic with fixed checkpoints. No missing-input retraining, official-test or multiseed claim; fullmatrix coverage does not imply improvement.')
audit['RGBNT100_DeMo_canonical_recheck']=dict(host='2026',reason='Observed tiny cross-host AP/INP discrepancy required same-host verification under unchanged exact audit tolerances; baseline and V5 comparison now share2026.',
    selected_campaign='results/anytoany49_demo_RGBNT100_samehost_recheck_20261003',original2027_campaign_retained='results/anytoany49_demo_20261003',
    diagnostic='results/preflight/anytoany49_previous13_difference_diagnostic.json',cause='Unproven',acceptance_tolerances_unchanged=True)
path=PROJECT/'results/preflight/anytoany49_complete_analysis.json';assert not path.exists()
path.write_bytes(json.dumps(audit,indent=2).encode('utf-8'))
print('ANYTOANY49_COMPLETE_AUDIT',json.dumps({dataset:{key:value for key,value in data.items() if key!='pairs'} for dataset,data in summaries.items()}),flush=True)
