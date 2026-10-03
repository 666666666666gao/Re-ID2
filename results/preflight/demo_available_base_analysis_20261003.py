import ast
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

project=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
namespace=dict(project=project,sources={})
helper=Path('C:/Users/gb/.codex_tmp/demo_axis_v4_first2_analysis_20261003.py')
definitions=[node for node in ast.parse(helper.read_text(encoding='utf-8')).body if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef))]
exec(compile(ast.Module(body=definitions,type_ignores=[]),str(helper),'exec'),namespace)
definition=next(node for node in definitions if isinstance(node,ast.FunctionDef) and node.name=='compare_queries')
source=ast.unparse(definition).replace('harmed / reference_correct','harmed / reference_correct if reference_correct else None').replace('rescued / reference_wrong','rescued / reference_wrong if reference_wrong else None')
exec(compile(source,str(helper)+'::undefined_ratio_null','exec'),namespace)
load,csv_rows,compare=[namespace[name] for name in ('load','csv_rows','compare_queries')]
keys=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
identity=('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')
helper=Path('C:/Users/gb/.codex_tmp/demo_retrieval_utility_incremental_frozen_analysis_20261003.py')
definition=next(node for node in ast.parse(helper.read_text(encoding='utf-8')).body if isinstance(node,ast.FunctionDef) and node.name=='metric_rows')
namespace.update(keys=keys,csv_rows=csv_rows)
exec(compile(ast.Module(body=[definition],type_ignores=[]),str(helper),'exec'),namespace)
metric_rows=namespace['metric_rows']
sets=('R','N','T','RN','RT','NT','RNT')
expected={'q_'+q+'_g_'+g for q in sets for g in sets}
report=dict(status='PASS_ACTUAL_FROZEN_AVAILABILITY_BASE_EVIDENCE_AUDIT',observed_at=datetime.now().isoformat(timespec='seconds'),
            runs={},optimizer_updates=0,official_test_uses=0,
            limits='Frozen mask intervention, not missing-input retraining or final+2 acceptance. DeMo5120/V55632 descriptors unchanged; own original checkpoint is the intervention reference. All49 reported, no subset selection.')
flat=[]
for host in ('2026','2027'):
    campaign='results/availability_base_frozen_'+host+'_20261003'
    launch=load('results/preflight/availability_base_'+host+'_launch.json')
    parent=load(campaign+'/controller_result.json')
    assert parent['status']=='COMPLETE' and len(parent['runs'])==len(launch['jobs'])
    for job in launch['jobs']:
        dataset,variant=job['dataset'],job['variant'];name=dataset+'_'+variant+'_s42'
        folder=campaign+'/'+name
        controller=load(folder+'/controller_result.json')
        assert controller['status']=='PASS' and len(controller['stages'])==2 and all(row['exit_code']==0 for row in controller['stages'])
        assert load(folder+'/frozen_inputs.json')==job['frozen_inputs']
        for stage in ('smoke','full'):
            intake=load(folder+'/'+stage+'/intake.json');assert intake['exit_code']==0
            for relative,proof in intake['files'].items():
                path=project/campaign/relative
                assert path.stat().st_size==proof['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==proof['sha256']
        smoke=load(folder+'/smoke/smoke.json')
        assert smoke['status']=='PASS' and smoke['normal_feature_max_error']==0
        assert smoke['base_mask_tensor_contract']['status']=='PASS' and smoke['state_tensor_versions_unchanged']
        assert smoke['parameter_objects_and_state_keys_unchanged'] and smoke['optimizer_updates']==0
        masked=load(folder+'/full/result.json')
        assert masked['status']=='COMPLETE' and masked['normal_feature_max_error']==0 and masked['optimizer_updates']==0
        assert masked['state_tensor_versions_unchanged'] and masked['parameter_objects_and_state_keys_unchanged']
        assert masked['original_input_files']==job['frozen_inputs'] and set(masked['measurements'])==expected
        for key,value in masked['runtime'].items():
            if key=='clean':continue
            assert value['descriptor_contract']['unavailable_global_coordinates_zero'] and value['descriptor_contract']['invalid_relation_coordinates_zero']
        if variant=='demo' and dataset=='RGBNT100':
            original_folder='results/anytoany49_demo_RGBNT100_samehost_recheck_20261003/'+name+'/full'
        else:
            original_folder='results/anytoany49_'+variant+'_20261003/'+name+'/full'
        original=load(original_folder+'/result.json')
        assert original['status']=='COMPLETE' and set(original['measurements'])==expected
        assert original['original_input_files']==job['frozen_inputs']
        comparisons={}
        for condition,item in masked['measurements'].items():
            current=item['metrics'];reference=original['measurements'][condition]['metrics']
            assert current==load(folder+'/full/'+condition+'.json') and reference==load(original_folder+'/'+condition+'.json')
            current_rows=metric_rows(folder+'/full/'+condition+'.csv',current)
            reference_rows=metric_rows(original_folder+'/'+condition+'.csv',reference)
            assert len(current_rows)==len(reference_rows) and all(all(a[key]==b[key] for key in identity) for a,b in zip(current_rows,reference_rows))
            paired=compare(current_rows,reference_rows)
            delta={key:current[key]-reference[key] for key in keys}
            assert abs(paired['mAP_delta_recomputed_from_per_query_pp']-delta['mAP'])<1e-8
            assert abs(paired['Rank1_delta_recomputed_from_per_query_pp']-delta['Rank-1'])<1e-8
            if condition=='q_RNT_g_RNT':assert all(value==0 for value in delta.values())
            comparisons[condition]=dict(original={key:reference[key] for key in keys},masked={key:current[key] for key in keys},delta_pp=delta,paired_queries=paired)
            q,g=condition[2:].split('_g_')
            for state,metrics in [('original',reference),('available_base',current)]:
                flat.append(dict(dataset=dataset,variant=variant,state=state,query_available=q,gallery_available=g,**{key:metrics[key] for key in keys}))
        report['runs'][name]=dict(host=host,conditions=comparisons,
                                 mAP_improved_conditions=sum(value['delta_pp']['mAP']>0 for value in comparisons.values()),
                                 Rank1_improved_conditions=sum(value['delta_pp']['Rank-1']>0 for value in comparisons.values()),
                                 original_inputs=job['frozen_inputs'],normal_feature_max_error=0)
assert len(report['runs'])==6 and len(flat)==588
report['total_comparisons']=294
output=project/'results/availability_base_comparison_20261003';output.mkdir(exist_ok=False)
with (output/'all3_demo_V5_original_and_available_base_full49_metrics_588.csv').open('w',encoding='utf-8',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
path=project/'results/preflight/availability_base_complete_analysis.json';assert not path.exists()
path.write_bytes(json.dumps(report,ensure_ascii=False,indent=2).encode('utf-8'))
print('AVAILABILITY_BASE_ALL3_DEMO_V5_FULL49_AUDITED',json.dumps({name:dict(mAP_improved=value['mAP_improved_conditions'],Rank1_improved=value['Rank1_improved_conditions'],RGB_only_query=value['conditions']['q_R_g_RNT']['delta_pp']) for name,value in report['runs'].items()}),flush=True)
