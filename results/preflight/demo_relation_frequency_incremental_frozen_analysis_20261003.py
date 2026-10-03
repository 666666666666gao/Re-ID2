import argparse
import ast
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

project=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
parser=argparse.ArgumentParser()
parser.add_argument('--datasets',nargs='+',required=True)
parser.add_argument('--output-name',required=True)
args=parser.parse_args()
helper=Path('C:/Users/gb/.codex_tmp/demo_axis_v4_first2_analysis_20261003.py')
tree=ast.parse(helper.read_text(encoding='utf-8'))
namespace={'project':project,'sources':{}}
definitions=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef))]
exec(compile(ast.Module(body=definitions,type_ignores=[]),str(helper),'exec'),namespace)
comparison_definition=next(node for node in definitions if isinstance(node,ast.FunctionDef) and node.name=='compare_queries')
comparison_source=ast.unparse(comparison_definition)
comparison_source=comparison_source.replace('harmed / reference_correct', 'harmed / reference_correct if reference_correct else None')
comparison_source=comparison_source.replace('rescued / reference_wrong', 'rescued / reference_wrong if reference_wrong else None')
exec(compile(comparison_source,str(helper)+'::explicit_undefined_ratio','exec'),namespace)
load,csv_rows,compare=[namespace[key] for key in ('load','csv_rows','compare_queries')]
keys=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
expected=['clean']+[kind+'_'+mask for mask in ('r','n','t','rn','rt','nt') for kind in ('both_missing','query_missing')]
identity=('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')
report=dict(status='PASS_ACTUAL_FROZEN_EVIDENCE_AUDIT',observed_at=datetime.now().isoformat(timespec='seconds'),datasets={},optimizer_updates=0,official_test_uses=0,scope='Frozen source-specific frequency interface B seed42 identity-heldout development; no trained ablation or official-test inference')


def metric_rows(path,metrics):
    rows=csv_rows(path);valid=[row for row in rows if row['valid']=='True']
    assert len(rows)==metrics['query_count'] and len(valid)==metrics['valid_queries']
    assert len(metrics['CMC_1_to_50'])==50
    for key,column in [('mAP','AP'),('mINP','INP'),*[(key,key) for key in keys[2:]]]:
        assert abs(100*sum(float(row[column]) for row in valid)/len(valid)-metrics[key])<1e-8
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


for dataset in args.datasets:
    assert dataset in ('RGBNT201','RGBNT100','MSVR310')
    name=dataset+'_axis_relation_frequency_fullref_s42'
    root='results/axis_collaboration_v9_relation_frequency_frozen_trial/'+name
    controller=load(root+'/controller_result.json')
    assert controller['status']=='PASS' and len(controller['stages'])==4 and all(stage['exit_code']==0 for stage in controller['stages'])
    for stage in ('four_state_smoke','four_state_full','missing_smoke','missing_full'):
        intake=load(root+'/'+stage+'/intake.json')
        assert intake['exit_code']==0
        for relative,proof in intake['files'].items():
            path=project/'results/axis_collaboration_v9_relation_frequency_frozen_trial'/relative
            assert path.stat().st_size==proof['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==proof['sha256']
    normal=load('results/axis_collaboration_v9_relation_frequency_trial/development/'+name+'/development_metrics/full_metrics.json')
    diagnostic=load(root+'/four_state_full/diagnostic.json')
    assert diagnostic['variant']=='axis_relation_frequency_fullref' and diagnostic['normal_inference_feature_max_error']==0
    assert diagnostic['optimizer_updates']==0 and diagnostic['state_tensor_versions_unchanged'] and diagnostic['installed_ground_truth_order_equal']
    assert diagnostic['normal_distance_source'].startswith('saved best_dev_arrays.npz/distances')
    assert all(abs(diagnostic['metrics']['11'][key]-normal[key])<1e-8 for key in keys)
    states={key:metric_rows(root+'/four_state_full/state_'+key+'.csv',value) for key,value in diagnostic['metrics'].items()}
    changes={}
    for state in ('00','10','01'):
        assert len(states[state])==len(states['11']) and all(all(a[key]==b[key] for key in identity) for a,b in zip(states['11'],states[state]))
        paired=compare(states['11'],states[state])
        paired['limit']='Paired four states of one frozen interface B checkpoint; this is an intervention diagnostic, not a separately trained ablation or a causal effect.'
        delta={key:diagnostic['metrics']['11'][key]-diagnostic['metrics'][state][key] for key in keys}
        assert abs(paired['mAP_delta_recomputed_from_per_query_pp']-delta['mAP'])<1e-8
        assert abs(paired['Rank1_delta_recomputed_from_per_query_pp']-delta['Rank-1'])<1e-8
        harm=diagnostic['harm_rescue'][state+'_to_11']
        assert paired['rank1_harmed_count']==harm['harmed_count'] and paired['rank1_rescued_count']==harm['rescued_count']
        changes[state+'_to_11']=dict(all_six_delta_pp=delta,paired_queries=paired)
    calibrations={}
    for reference,filename,calibration_key in [('base00','contributions.csv','contribution_calibration'),('full11','full_reference_contributions.csv','full_reference_contribution_calibration')]:
        contributions=csv_rows(root+'/four_state_full/'+filename)
        assert len(contributions)==len(states['11']) and all(row['name']==query['name'] for row,query in zip(contributions,states['11']))
        calibration={}
        for key,values in diagnostic[calibration_key].items():
            target=[float(row[key+'_target']) for row in contributions];prediction=[float(row[key+'_prediction']) for row in contributions]
            mt,mp=sum(target)/len(target),sum(prediction)/len(prediction)
            vt=sum((value-mt)**2 for value in target)/len(target);vp=sum((value-mp)**2 for value in prediction)/len(prediction)
            assert vt>0 and vp>0
            covariance=sum((t-mt)*(p-mp) for t,p in zip(target,prediction))/len(target)
            assert abs(mt-values['target_mean'])<1e-7 and abs(mp-values['prediction_mean'])<1e-7
            assert abs(sum(abs(t-p) for t,p in zip(target,prediction))/len(target)-values['MAE'])<1e-7
            calibration[key]=dict(**values,Pearson_correlation_recomputed=covariance/math.sqrt(vt*vp))
        for row in contributions:
            scores={key:float(row['R'+key]) for key in ('00','10','01','11')}
            assert abs(float(row['delta_M_given_F_target'])-(scores['11']-scores['01']))<1e-7
            assert abs(float(row['delta_F_given_M_target'])-(scores['11']-scores['10']))<1e-7
            assert abs(float(row['empirical_interaction_target'])-(scores['11']-scores['10']-scores['01']+scores['00']))<1e-7
        calibrations[reference]=calibration
    assert dataset == 'MSVR310'
    missing=load(root+'/missing_full/result.json')
    assert missing['status']=='COMPLETE' and missing['variant']=='axis_relation_frequency_fullref'
    assert missing['normal_feature_max_error']==0 and missing['optimizer_updates']==0 and missing['state_tensor_versions_unchanged']
    sets=('R','N','T','RN','RT','NT','RNT')
    expected={'q_'+q+'_g_'+g for q in sets for g in sets}
    assert set(missing['measurements'])==expected
    assert all(abs(missing['measurements']['q_RNT_g_RNT']['metrics'][key]-normal[key])<1e-8 for key in keys)
    baselines={}
    for label,variant in [('DeMo','demo'),('V5','axis_mass_fullref')]:
        campaign='results/anytoany49_'+variant+'_20261003'
        intake=load(campaign+'/intake.json')
        for relative,proof in intake['extracted_files'].items():
            path=project/campaign/relative
            assert path.stat().st_size==proof['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==proof['sha256']
        folder=campaign+'/'+dataset+'_'+variant+'_s42'
        result=load(folder+'/full/result.json')
        assert load(folder+'/controller_result.json')['status']=='PASS'
        assert load(folder+'/full_exit.json')['exit_code']==0
        assert result['status']=='COMPLETE' and result['normal_feature_max_error']==0 and result['optimizer_updates']==0 and result['state_tensor_versions_unchanged']
        assert set(result['measurements'])==expected
        baselines[label]=(folder+'/full',result)
    comparisons={};flat=[]
    for condition,item in missing['measurements'].items():
        current=item['metrics'];assert current==load(root+'/missing_full/'+condition+'.json')
        new_rows=metric_rows(root+'/missing_full/'+condition+'.csv',current)
        comparison=dict(metrics={'InterfaceB':{key:current[key] for key in keys}})
        for label,(folder,result) in baselines.items():
            original=result['measurements'][condition]['metrics']
            assert original==load(folder+'/'+condition+'.json')
            original_rows=metric_rows(folder+'/'+condition+'.csv',original)
            assert len(new_rows)==len(original_rows) and all(all(a[key]==b[key] for key in identity) for a,b in zip(new_rows,original_rows))
            paired=compare(new_rows,original_rows)
            paired['limit']='Separately trained frozen development checkpoints, not a trained ablation or causal effect.'
            delta={key:current[key]-original[key] for key in keys}
            assert abs(paired['mAP_delta_recomputed_from_per_query_pp']-delta['mAP'])<1e-8
            assert abs(paired['Rank1_delta_recomputed_from_per_query_pp']-delta['Rank-1'])<1e-8
            comparison['metrics'][label]={key:original[key] for key in keys}
            comparison['InterfaceB_minus_'+label+'_pp']=delta
            comparison['paired_queries_'+label]=paired
        delta=comparison['InterfaceB_minus_DeMo_pp']
        comparison['mAP_Rank1_both_at_least_2pp']=delta['mAP']>=2 and delta['Rank-1']>=2
        comparisons[condition]=comparison
        q,g=condition[2:].split('_g_')
        for label,metrics in comparison['metrics'].items():
            flat.append(dict(dataset=dataset,model=label,query_available=q,gallery_available=g,**metrics))
    report['datasets'][dataset]=dict(best_epoch=diagnostic['best_epoch'],four_state_metrics={state:{key:value[key] for key in keys} for state,value in diagnostic['metrics'].items()},four_state_changes=changes,contribution_calibration=calibrations,mean_gates=diagnostic['mean_gates'],mean_norms=diagnostic['mean_norms_base_M_residual_F_residual_full_residual'],joint_route_vs_marginal_product_L1=diagnostic['mean_joint_minus_marginal_product_L1'],missing_conditions=comparisons,missing_mAP_Rank1_both_2pp_count=sum(value['mAP_Rank1_both_at_least_2pp'] for value in comparisons.values()),normal_feature_max_error=0,optimizer_updates=0,descriptor_dimension=5632,active_descriptor_dimension=5120,zero_padding_dimension=512)
    csv_path=project/'results/axis_collaboration_v9_relation_frequency_frozen_trial/MSVR310_full49_models_metrics_147.csv'
    assert not csv_path.exists()
    with csv_path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
report['limits']='Single-seed frozen interface B checkpoint. 00 is its own trained base, not DeMo. Four-state interventions are not retrained ablations, causal effects or information-theoretic synergy. All49 coverage does not imply improvement. Same missing-input protocol for baseline/V5; original base zero-input ghost behavior is unchanged. No official test, multi-seed or matched ordinary/control claim. Output5632 with512zero padding, active5120, no equal-compute claim.'
receipt=load('results/preflight/relation_frequency_frozen_launch.json')
report['source_sha256']=receipt['source_sha256']
report['fixed_original_inputs']=receipt['availability']['first_inputs']
path=project/'results/preflight'/args.output_name;assert not path.exists()
path.write_bytes(json.dumps(report,ensure_ascii=False,indent=2).encode('utf-8'))
print('INTERFACE_B_FROZEN49_AND4_AUDITED',json.dumps({dataset:dict(joint_vs_base=value['four_state_changes']['00_to_11'],joint_vs_M=value['four_state_changes']['10_to_11'],joint_vs_F=value['four_state_changes']['01_to_11'],missing_2pp_condition_count=value['missing_mAP_Rank1_both_2pp_count'],full_reference_correlations={key:row['Pearson_correlation_recomputed'] for key,row in value['contribution_calibration']['full11'].items()}) for dataset,value in report['datasets'].items()}),flush=True)
