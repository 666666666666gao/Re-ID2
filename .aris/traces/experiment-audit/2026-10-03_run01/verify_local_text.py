"""Read the frozen audit snapshot; stdlib only, no experiment imports or network."""
from pathlib import Path
import ast
import collections
import csv
import datetime
import hashlib
import io
import json
import math
import re
import statistics
import struct

TRACE = Path(__file__).parent
snapshot = json.loads((TRACE / 'inputs.snapshot.json').read_text(encoding='utf-8'))
files = {r['path']: r for r in snapshot['files']}
checks = []

def check(name, ok, **evidence):
    checks.append({'name': name, 'status': 'PASS' if ok else 'FAIL', **evidence})

def text(path):
    return files[path]['text']

def obj(path):
    return json.loads(text(path))

def rows(path):
    return list(csv.DictReader(io.StringIO(text(path))))

def close(a, b, tol=1e-8):
    return abs(a-b) <= tol

def avg(values):
    return math.fsum(values) / len(values)

def labels(name, dataset):
    if dataset == 'RGBNT201':
        return int(name.split('_')[0][:6]), int(name.split('_')[1][3])-1, -1
    if dataset == 'RGBNT100':
        identity, camera = map(int, re.search(r'([-\d]+)_c([-\d]+)', name).groups())
        return identity, camera-1, -1
    return int(name[:4]), int(name[11]), int(name[6:9])

metrics = ['mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20']
splits = obj('splits.json')
table_results = {}
table_rows = {}

def verify_table(csv_path, summary, dataset):
    data = rows(csv_path)
    valid = [r for r in data if r['valid'] == 'True']
    check(csv_path + ':query_index', [int(r['query_index']) for r in data] == list(range(len(data))))
    check(csv_path + ':name_to_GT', all(labels(r['name'], dataset) == tuple(int(r[k]) for k in ['identity','camera','scene']) for r in data))
    check(csv_path + ':dev_membership', all(int(r['identity']) in splits[dataset]['dev_ids'] for r in data))
    check(csv_path + ':counts', len(data) == summary['query_count'] and len(valid) == summary['valid_queries'] and len(data)-len(valid) == summary['invalid_queries'])
    inp_error = max(abs(float(r['INP']) - int(r['relevant_gallery'])/int(r['last_match'])) for r in valid)
    check(csv_path + ':per_query_INP', inp_error <= 1e-15, max_abs_error=inp_error)
    check(csv_path + ':per_query_ranks', all(int(r[f'Rank-{k}']) == int(int(r['first_match']) <= k) for r in valid for k in [1,5,10,20]))
    check(csv_path + ':finite_bounds', all(0 <= float(r['AP']) <= 1 and 0 <= float(r['INP']) <= 1 and 1 <= int(r['first_match']) <= int(r['last_match']) <= int(r['kept_gallery']) <= summary['gallery_count'] for r in valid))
    values = {'mAP': 100*avg([float(r['AP']) for r in valid]), 'mINP':100*avg([int(r['relevant_gallery'])/int(r['last_match']) for r in valid])}
    values.update({f'Rank-{k}':100*avg([int(r['first_match'])<=k for r in valid]) for k in [1,5,10,20]})
    metric_error = max(abs(values[k]-summary[k]) for k in metrics)
    cmc = [100*avg([int(r['first_match'])<=k for r in valid]) for k in range(1,51)]
    cmc_error = max(abs(a-b) for a,b in zip(cmc, summary['CMC_1_to_50']))
    group_errors=[]; group_count=0
    for key in ['camera','scene','identity']:
        expected=collections.defaultdict(list)
        for r in valid: expected[int(r[key])].append(r)
        check(csv_path + ':group_values:'+key, sorted(expected) == [g['value'] for g in summary['groups'][key]])
        for group in summary['groups'][key]:
            selected=expected[group['value']]; group_count+=1
            check(csv_path+':group_count:'+key+':'+str(group['value']), len(selected) == group['queries'])
            computed={'mAP':100*avg([float(r['AP']) for r in selected]),'mINP':100*avg([int(r['relevant_gallery'])/int(r['last_match']) for r in selected])}
            computed.update({f'Rank-{k}':100*avg([int(r['first_match'])<=k for r in selected]) for k in [1,5,10,20]})
            group_errors += [abs(group[k]-computed[k]) for k in metrics]
    check(csv_path+':aggregate_metrics', metric_error < 1e-8, max_abs_error=metric_error)
    check(csv_path+':CMC1_50', len(summary['CMC_1_to_50'])==50 and cmc_error<1e-8, max_abs_error=cmc_error)
    check(csv_path+':all_groups', max(group_errors)<1e-8, groups=group_count,max_abs_error=max(group_errors))
    table_results[csv_path]={'dataset':dataset,'queries':len(data),'valid_queries':len(valid),'distinct_query_identities':len({r['identity'] for r in data}), 'metrics':values,'max_metric_error':metric_error,'max_cmc_error':cmc_error,'max_group_error':max(group_errors),'max_per_query_INP_error':inp_error,'groups':group_count,'CSV_data_lines':[2,len(data)+1],'AP_limit':'Aggregate AP mean independently recomputed; individual AP cannot be reconstructed from first/last rank alone.'}
    table_rows[csv_path]=data
    return data

names=['MSVR310_axis_scaled_fullref_s42','RGBNT201_axis_scaled_fullref_s42','RGBNT201_plain_scaled_fullref_s42','RGBNT100_axis_scaled_fullref_s42','RGBNT100_plain_scaled_fullref_s42']
development='results/axis_collaboration_v4_development/'
diag='results/axis_collaboration_v4_diagnostic/MSVR310_axis_scaled_fullref_s42_cross26/'
run_results={}
for name in names:
    prefix=development+name+'/'
    terminal=obj(prefix+'result.json'); info=obj(prefix+'run.json'); best=obj(prefix+'best.json')
    dataset=terminal['arguments']['dataset']; epoch_rows=rows(prefix+'epochs.csv')
    batches=[json.loads(line) for line in text(prefix+'batch_orders.jsonl').splitlines()]
    baseline=[json.loads(line) for line in text(f'results/full_suite/{dataset}_demo_s42/batch_orders.jsonl').splitlines()]
    check(name+':run_status_result', all(info[k]==terminal[k] for k in info) and obj(prefix+'status.json')==terminal and best==terminal['best'])
    check(name+':complete50_exit0',terminal['status']=='COMPLETE' and terminal['epochs']==50 and obj(development+name+'_exit.json')['exit_code']==0)
    check(name+':epoch_sequence', [int(r['epoch']) for r in epoch_rows]==list(range(1,51)))
    check(name+':finite_epochs', all(math.isfinite(float(v)) for r in epoch_rows for v in r.values()))
    max_map=max(float(r['mAP']) for r in epoch_rows); tied=[int(r['epoch']) for r in epoch_rows if float(r['mAP'])==max_map]
    check(name+':earliest_max', best['epoch']==min(tied) and best['mAP']==max_map, tied_epochs=tied)
    check(name+':best_row_metrics', all(float(epoch_rows[best['epoch']-1][k])==best[k] for k in ['mAP','Rank-1','Rank-5','Rank-10']))
    check(name+':strict_reload_receipt', all(terminal['strict_reload'][k]==best[k] for k in terminal['strict_reload']))
    check(name+':sampling', [(r['epoch'],r['step'],r['names']) for r in batches]==[(r['epoch'],r['step'],r['names']) for r in baseline])
    check(name+':step_sequence', [r['step'] for r in batches]==list(range(1,len(batches)+1)))
    check(name+':all_batch_size64', all(len(r['names'])==64 for r in batches))
    seen_names={n for r in batches for n in r['names']}; train_ids={labels(n,dataset)[0] for n in seen_names}
    check(name+':fit_dev_disjoint', train_ids.isdisjoint(splits[dataset]['dev_ids']),observed_fit_ids=len(train_ids),declared_fit_ids=terminal['classes'])
    check(name+':observed_fit_identity_count',len(train_ids)==terminal['classes'])
    updates=sum(r['optimizer_updated'] for r in batches)
    check(name+':attempts_updates_skips',len(batches)==terminal['steps'] and updates==terminal['optimizer_steps'] and len(batches)-updates==terminal['amp_skipped_steps'])
    per_epoch=[]
    for r in epoch_rows:
        epoch=int(r['epoch']); selected=[b for b in batches if b['epoch']<=epoch]
        count=len(selected); n_updates=sum(b['optimizer_updated'] for b in selected)
        per_epoch.append(count-int(r['steps'])==0 and n_updates-int(r['optimizer_steps'])==0 and count-n_updates==int(r['amp_skipped_steps']) and selected[-1]['amp_scale']==float(r['amp_scale']))
    check(name+':cumulative_epoch_counts',all(per_epoch))
    logs=text(development+name+'.log').splitlines()
    logged_rows=[json.loads(line.removeprefix('EPOCH ').split(' BEST ')[0]) for line in logs if line.startswith('EPOCH ')]
    check(name+':log_epochs_equal_CSV',len(logged_rows)==50 and all(all(close(float(a[k]),float(b[k]),0) for k in b) for a,b in zip(logged_rows,epoch_rows)))
    logged_terminal=[json.loads(line.removeprefix('TRAIN_COMPLETE ')) for line in logs if line.startswith('TRAIN_COMPLETE ')]
    check(name+':log_terminal', logged_terminal==[terminal])
    command=obj(development+name+'_launch.json')['command']; cli={command[i][2:].replace('-','_'):command[i+1] for i in range(len(command)-1) if command[i].startswith('--')}
    check(name+':launch_args', all((float(cli[k])==v if isinstance(v,(int,float)) else cli[k]==v) for k,v in terminal['arguments'].items()))
    config=terminal['config']
    check(name+':effective_config50_batch64', 'MAX_EPOCHS: 50' in config and '  IMS_PER_BATCH: 64' in config)
    computation=obj(prefix+'development_metrics/metric_computation.json')
    check(name+':metric_receipt',computation['selected_epoch']==best['epoch'] and computation['installed_ground_truth_ids_cameras_scenes_names_and_query_order_equal'] and computation['original_and_strict_reload_four_metrics_match'])
    summary=obj(prefix+'development_metrics/full_metrics.json')
    check(name+':full_metrics_best',all(summary[k]==best[k] for k in ['mAP','Rank-1','Rank-5','Rank-10','query_count','gallery_count']))
    verify_table(prefix+'development_metrics/full_metrics.csv',summary,dataset)
    run_results[name]={'dataset':dataset,'epochs':len(epoch_rows),'attempts':len(batches),'updates':updates,'AMP_skips':len(batches)-updates,'best_epoch':best['epoch'],'best_epoch_CSV_line':best['epoch']+1,'max_mAP_tied_epochs':tied,'parameters':terminal['parameters'],'trainable_parameters':terminal['trainable_parameters'],'descriptor_dim':terminal['descriptor_dim'],'fit_records_declared':terminal['fit_records'],'distinct_sampled_filenames':len(seen_names),'fit_identities':len(train_ids),'dev_records':terminal['dev_records'],'dev_queries':terminal['dev_queries'],'dev_identity_split_count':len(splits[dataset]['dev_ids']),'batch_orders_rows':len(batches),'wall_seconds':terminal['finished']-terminal['started'],'training_peak_memory_bytes':terminal['peak_memory']}

for dataset in ['RGBNT201','RGBNT100','MSVR310']:
    prefix=f'results/full_suite/{dataset}_demo_s42/full_evaluation/'
    verify_table(prefix+'dev_per_query.csv',obj(prefix+'metrics.json')['dev'],dataset)
for dataset in ['RGBNT201','MSVR310']:
    prefix=f'results/axis_collaboration_v3_development/{dataset}_axis_collaboration_s42/development_metrics/'
    verify_table(prefix+'full_metrics.csv',obj(prefix+'full_metrics.json'),dataset)
for state in ['00','10','01','11']:
    verify_table(diag+'full/state_'+state+'.csv',obj(diag+'full/state_'+state+'.json'),'MSVR310')

pair_results={}
def pair(source_path, target_path):
    source,target=table_rows[source_path],table_rows[target_path]
    gt_keys=['query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery']
    aligned=len(source)==len(target) and all(all(a[k]==b[k] for k in gt_keys) for a,b in zip(source,target))
    check(source_path+'->'+target_path+':GT_alignment',aligned)
    ap=[float(b['AP'])-float(a['AP']) for a,b in zip(source,target)]
    return {'queries':len(source),'GT_and_order_equal':aligned,'rank1_harmed_count':sum(a['Rank-1']=='1' and b['Rank-1']=='0' for a,b in zip(source,target)),'rank1_rescued_count':sum(a['Rank-1']=='0' and b['Rank-1']=='1' for a,b in zip(source,target)),'AP_improved_count':sum(x>0 for x in ap),'AP_worsened_count':sum(x<0 for x in ap),'AP_identical_count':sum(x==0 for x in ap),'mAP_delta_pp':100*avg(ap),'Rank1_delta_pp':100*avg([int(b['Rank-1'])-int(a['Rank-1']) for a,b in zip(source,target)])}

for name in names:
    dataset=run_results[name]['dataset']; target=development+name+'/development_metrics/full_metrics.csv'
    pair_results[name]={'demo':pair(f'results/full_suite/{dataset}_demo_s42/full_evaluation/dev_per_query.csv',target)}
    if dataset in ['RGBNT201','MSVR310'] and '_axis_' in name:
        pair_results[name]['V3_axis']=pair(f'results/axis_collaboration_v3_development/{dataset}_axis_collaboration_s42/development_metrics/full_metrics.csv',target)
    if '_plain_' in name:
        pair_results[name]['V4_axis_scaled_fullref']=pair(development+dataset+'_axis_scaled_fullref_s42/development_metrics/full_metrics.csv',target)
    summary=obj(development+name+'/development_metrics/full_metrics.json')
    baseline=obj(f'results/full_suite/{dataset}_demo_s42/full_evaluation/metrics.json')['dev']
    run_results[name]['delta_vs_demo']={k:summary[k]-baseline[k] for k in metrics}
    run_results[name]['mAP_and_Rank1_both_ge_2pp']=all(run_results[name]['delta_vs_demo'][k]>=2 for k in ['mAP','Rank-1'])
for dataset in ['RGBNT201','RGBNT100']:
    axis=run_results[dataset+'_axis_scaled_fullref_s42']; plain=run_results[dataset+'_plain_scaled_fullref_s42']
    check(dataset+':equal_capacity_dim',all(axis[k]==plain[k] for k in ['parameters','trainable_parameters','descriptor_dim']))
for state in ['00','10','01']:
    pair_results[state+'_to_11']=pair(diag+'full/state_'+state+'.csv',diag+'full/state_11.csv')

diagnostic=obj(diag+'full/diagnostic.json'); mechanism=obj('results/preflight/axis_collaboration_v4_msvr_mechanism_analysis.json')
for state in ['00','10','01','11']:
    check('diagnostic:state'+state,diagnostic['metrics'][state]==obj(diag+'full/state_'+state+'.json'))
check('diagnostic:normal11_csv_exact_training',text(diag+'full/state_11.csv')==text(development+'MSVR310_axis_scaled_fullref_s42/development_metrics/full_metrics.csv'))
for key in ['00_to_11','10_to_11','01_to_11']:
    computed=pair_results[key]; recorded=diagnostic['harm_rescue'][key]; claimed=mechanism['paired_query_comparisons'][key]
    check('diagnostic:harm_rescue:'+key,computed['rank1_harmed_count']==recorded['harmed_count'] and computed['rank1_rescued_count']==recorded['rescued_count'] and computed['queries']==recorded['valid_queries'])
    check('mechanism:AP_counts:'+key,all(computed[k]==claimed[k] for k in ['AP_improved_count','AP_worsened_count','AP_identical_count']))

def f32(x):
    return struct.unpack('f',struct.pack('f',x))[0]

calibration={}
for filename, key in [('contributions.csv','contribution_calibration'),('full_reference_contributions.csv','full_reference_contribution_calibration')]:
    data=rows(diag+'full/'+filename); per_column={}; delta_errors=[]
    check(filename+':query_names',[(r['query_index'],r['name']) for r in data]==[(r['query_index'],r['name']) for r in table_rows[diag+'full/state_11.csv']])
    check(filename+':reference_bounds',all(0<=int(r[k])<360 for r in data for k in ['positive_reference_index','negative_reference_index']))
    for r in data:
        s={k:float(r['R'+k]) for k in ['00','10','01','11']}
        expected=[f32(s['11']-s['01']),f32(s['11']-s['10']),f32(f32(f32(s['11']-s['10'])-s['01'])+s['00'])]
        delta_errors.extend(abs(a-float(r[k+'_target'])) for a,k in zip(expected,['delta_M_given_F','delta_F_given_M','empirical_interaction']))
    check(filename+':target_identities_float32',max(delta_errors)==0,max_abs_error=max(delta_errors))
    for column in ['delta_M_given_F','delta_F_given_M','empirical_interaction']:
        target=[float(r[column+'_target']) for r in data]; pred=[float(r[column+'_prediction']) for r in data]
        mt,mp=avg(target),avg(pred); st=statistics.pstdev(target); sp=statistics.pstdev(pred)
        covariance=avg([(a-mt)*(b-mp) for a,b in zip(target,pred)])
        computed={'target_mean':mt,'target_std':st,'prediction_mean':mp,'prediction_std':sp,'MAE':avg([abs(a-b) for a,b in zip(target,pred)]),'RMSE':math.sqrt(avg([(a-b)**2 for a,b in zip(target,pred)])),'positive_target_fraction':avg([a>0 for a in target]),'sign_agreement_fraction':avg([(a>0)-(a<0)==(b>0)-(b<0) for a,b in zip(target,pred)]),'prediction_target_covariance':covariance}
        error=max(abs(computed[k]-diagnostic[key][column][k]) for k in computed)
        check(filename+':calibration:'+column,error<1e-8,max_abs_error=error)
        computed.update(Pearson_correlation=covariance/(st*sp),MAE_over_target_std=computed['MAE']/st,negative_prediction_fraction=avg([p<0 for p in pred]))
        if filename=='full_reference_contributions.csv':
            claim=mechanism['full11_development_reference_calibration'][column]
            check('mechanism:Pearson:'+column,close(computed['Pearson_correlation'],claim['Pearson_correlation_recomputed'],1e-6))
            check('mechanism:MAE_std:'+column,close(computed['MAE_over_target_std'],claim['MAE_over_target_std'],1e-5))
        per_column[column]={**computed,'max_summary_abs_error':error}
    calibration[filename]=per_column

hash_checks=[]
for name in names:
    for path,base,key in [(development+name+'/intake.json',development,'files'),(development+name+'/development_metrics/intake.json',development+name+'/development_metrics/',None)]:
        intake=obj(path); mapping=intake if key is None else intake[key]
        for relative,expected in mapping.items():
            actual=files[base+relative]; ok=actual['sha256']==expected['sha256'] and actual['bytes']==expected['bytes']
            hash_checks.append({'receipt':path,'path':base+relative,'match':ok}); check('intake:'+base+relative,ok)
for relative,expected in obj(diag+'intake.json')['files'].items():
    actual=files[diag+relative]; ok=actual['sha256']==expected['sha256'] and actual['bytes']==expected['bytes']; hash_checks.append({'receipt':diag+'intake.json','path':diag+relative,'match':ok}); check('diagnostic_intake:'+relative,ok)
for path,record in files.items():
    if path.endswith('.json'):
        data=json.loads(record['text'])
        if isinstance(data,dict):
            for key in ['source_sha256','audited_reports_sha256']:
                for relative,expected in data.get(key,{}).items():
                    if relative in files:
                        check('source_hash:'+path+':'+relative,files[relative]['sha256']==expected)

analysis_paths=['results/preflight/axis_collaboration_v4_first2_development_analysis.json','results/preflight/axis_collaboration_v4_first_plain_control_analysis.json','results/preflight/axis_collaboration_v4_rgbnt100_development_analysis.json','results/preflight/axis_collaboration_v4_first5_development_analysis.json']
for path in analysis_paths:
    for name,r in obj(path)['runs'].items():
        computed=run_results[name]; summary=obj(development+name+'/development_metrics/full_metrics.json')
        check(path+':six_metrics:'+name,all(r['metrics'][k]==summary[k] for k in metrics))
        check(path+':budget_best:'+name,all(r[k]==computed[k] for k in ['best_epoch','parameters','trainable_parameters','descriptor_dim','attempts','updates']))
        for reference,pair_claim in r['paired_queries'].items():
            pair_computed=pair_results[name][reference]
            check(path+':paired:'+name+':'+reference,all(pair_claim[k]==pair_computed[k] for k in ['rank1_harmed_count','rank1_rescued_count','AP_improved_count','AP_worsened_count','AP_identical_count']))
            check(path+':paired_deltas:'+name+':'+reference,close(pair_computed['mAP_delta_pp'],pair_claim['mAP_delta_recomputed_from_per_query_pp']) and close(pair_computed['Rank1_delta_pp'],pair_claim['Rank1_delta_recomputed_from_per_query_pp']))
        dataset=computed['dataset']
        reference_metrics={'demo':obj(f'results/full_suite/{dataset}_demo_s42/full_evaluation/metrics.json')['dev']}
        if 'V3_axis' in r['delta']:
            reference_metrics['V3_axis']=obj(f'results/axis_collaboration_v3_development/{dataset}_axis_collaboration_s42/development_metrics/full_metrics.json')
        if 'V4_axis_scaled_fullref' in r['delta']:
            reference_metrics['V4_axis_scaled_fullref']=obj(development+dataset+'_axis_scaled_fullref_s42/development_metrics/full_metrics.json')
        for reference,claimed in r['delta'].items():
            check(path+':all_six_deltas:'+name+':'+reference,all(close(claimed[k],summary[k]-reference_metrics[reference][k]) for k in metrics))
        for reference,claimed in r['paired_queries'].items():
            computed_pair=pair_results[name][reference]; n=computed_pair['queries']; correct=round(reference_metrics[reference]['Rank-1']*n/100)
            fractions={'rank1_harmed_fraction_of_all_queries':computed_pair['rank1_harmed_count']/n,'rank1_rescued_fraction_of_all_queries':computed_pair['rank1_rescued_count']/n,'rank1_harmed_fraction_of_reference_correct':computed_pair['rank1_harmed_count']/correct,'rank1_rescued_fraction_of_reference_wrong':computed_pair['rank1_rescued_count']/(n-correct)}
            check(path+':fractions:'+name+':'+reference,all(close(claimed[k],v) for k,v in fractions.items()))

combined=obj(analysis_paths[-1])
for dataset,gate in combined['axis_gates'].items():
    computed=run_results[dataset+'_axis_scaled_fullref_s42']
    check('combined:axis_gate:'+dataset,gate['delta_demo_pp']==computed['delta_vs_demo'] and gate['mAP_Rank1_at_least_2pp']==computed['mAP_and_Rank1_both_ge_2pp'])
for dataset,value in combined['same_capacity_axis_vs_plain'].items():
    axis=obj(development+dataset+'_axis_scaled_fullref_s42/development_metrics/full_metrics.json'); plain=obj(development+dataset+'_plain_scaled_fullref_s42/development_metrics/full_metrics.json')
    check('combined:axis_minus_plain:'+dataset,all(close(value['axis_minus_plain_six_pp'][k],axis[k]-plain[k]) for k in metrics))

handoff=text('docs/实验交接.md').splitlines()
for number in [10,11,766,767]:
    cells=[s.strip() for s in handoff[number-1].strip('|').split('|')]; name=cells[0]; summary=obj(development+name+'/development_metrics/full_metrics.json')
    check('handoff:table_line'+str(number),int(cells[1])==run_results[name]['best_epoch'] and all(close(float(cells[i+2]),summary[k],0.0000005) for i,k in enumerate(metrics)))
for number,state in [(777,'00'),(778,'10'),(779,'01'),(780,'11')]:
    cells=[s.strip() for s in handoff[number-1].strip('|').split('|')]; summary=diagnostic['metrics'][state]
    check('handoff:four_state_table_line'+str(number),cells[0]==state and all(close(float(cells[i+1]),summary[k],0.0000005) for i,k in enumerate(metrics)))

snapshot_results={}
for filename in ['axis_collaboration_v4_first2_publication_observation.json','axis_collaboration_v4_latest_snapshot.json']:
    path='results/preflight/'+filename; data=obj(path); development_rows=[r for r in data['rows'] if r['phase']=='development']; complete=[r['name'] for r in development_rows if r.get('result',{}).get('status')=='COMPLETE' and r.get('exit',{}).get('exit_code')==0]
    smoke=[]
    for row in data['rows']:
        if 'smoke' in row:
            s=row['smoke']; gradients=s['gradients']; smoke.append({'name':row['name'],'steps':s['steps'],'gradient_tensors':len(gradients),'all_gradient_checks_true':all(gradients.values()),'strict_reload_equal':s['strict_reload_equal']})
            check(filename+':smoke:'+row['name'],s['status']=='SMOKE_PASS' and s['steps']==3 and all(gradients.values()) and row['exit']['exit_code']==0 and s['strict_reload_equal'])
    snapshot_results[filename]={'observed_at':data['observed_at'],'read_at':files[path]['read_at'],'sha256':files[path]['sha256'],'controller_live':data['controller_live'],'planned':len(development_rows),'completed':complete,'pending':[{k:r[k] for k in ['name','status'] if k in r} for r in development_rows if r['name'] not in complete],'smokes':smoke,'limit':'Frozen observation, not live or terminal campaign state.'}

for path in files:
    if path.endswith('.py'):
        ast.parse(text(path),filename=path); check('syntax:'+path,True)

old=obj('results/missing_modalities/verified_aggregate.json'); old_first=obj(analysis_paths[0])['V1_missing_terminal_summary']
check('V1:label_counts',old['runs']==old_first['runs'] and old['conditions']==old_first['conditions'] and old['seeds']==old_first['seeds'])
for dataset,claim in old_first['mean_mAP_Rank1_gate'].items():
    selected=[r for r in old['mean_comparisons'] if r['dataset']==dataset]
    check('V1:existing_gate_label:'+dataset,len(selected)==claim['total_conditions'] and sum(r['mAP_Rank1_at_least_2pp'] for r in selected)==claim['passed_conditions'])
check('handoff:old_sync_source_hash',obj('results/preflight/axis_collaboration_v4_first2_handoff_sync.json')['sha256']==files['docs/实验交接.md']['sha256'])
check('frozen_stats:checkpoint_and_array_hash_links',obj('results/preflight/axis_collaboration_v4_msvr_frozen_stats.json')['checkpoint_sha256']==diagnostic['checkpoint_sha256'] and obj('results/preflight/axis_collaboration_v4_msvr_frozen_stats.json')['best_dev_arrays_sha256']==obj(development+'MSVR310_axis_scaled_fullref_s42/development_metrics/metric_computation.json')['best_dev_arrays_sha256'])

output={'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'Independent standard-library CPU parsing and recomputation of frozen local text; no imports of torch/numpy/experiment modules, no network/GPU or binary array loading.','input_count':len(files),'checks':checks,'failures':[c for c in checks if c['status']=='FAIL'],'counts':dict(collections.Counter(c['status'] for c in checks)),'runs':run_results,'metric_tables':table_results,'query_pair_comparisons':pair_results,'calibration':calibration,'intake_hash_checks':hash_checks,'snapshots':snapshot_results,'V1_label_only':{'runs':old['runs'],'conditions':old['conditions'],'seeds':old['seeds'],'raw351_recomputed':False}}
(TRACE/'deterministic_verification.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS' if not output['failures'] else 'FAIL','counts':output['counts'],'failures':output['failures'],'runs':run_results,'metric_tables':{k:{x:v[x] for x in ['queries','max_metric_error','max_cmc_error','max_group_error','max_per_query_INP_error']} for k,v in table_results.items()}},ensure_ascii=False))
