from datetime import datetime
import json,sys
from statistics import fmean
sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,remote_python
from analyze_full_official_baseline_pairs import METRICS,SETS
from analyze_full_official_control_trial import condition_groups
pf=PROJECT/'results/preflight';output=pf/'full_official_availability_graph_axis_frozen_readout_20261005.json';assert not output.exists()
first=json.loads((pf/'full_official_availability_graph_first_closed_pair_20261005.json').read_text(encoding='utf-8'));source=first['source']
record=json.loads(remote_python('2026',f'''import json
from pathlib import Path
root=Path({source!r});name='MSVR310_graph_axis_shared_s42'
train=json.loads((root/'training'/name/'result.json').read_text())
audit=json.loads((root/'frozen49'/name/'independent_cpu_audit.json').read_text())
evaluation=json.loads((root/'frozen49'/name/'result.json').read_text())
assert train['status']==evaluation['status']=='COMPLETE' and train['epochs']==50 and train['training_heldout_identities']==0
assert (train['train_records'],train['query_records'],train['gallery_records'])==(1032,591,1055)
assert train['training_coverage']==dict(eligible=1032,visited=1032,unvisited=[])
assert audit['status']=='PASS' and audit['cases']==49 and audit['perquery_count']==28959
assert train['best']['epoch']==audit['selected_epoch']==evaluation['selected_epoch']
assert train['arguments']==evaluation['model_arguments']==audit['model_arguments']
for phase in ('training','frozen49','audit'):assert json.loads((root/phase/(name+'_exit.json')).read_text())['exit_code']==0
print(json.dumps(dict(training=train,audit=audit)))
'''))
audit=record['audit'];conditions={'q_'+q+'_g_'+g for q in SETS for g in SETS};assert set(audit['conditions'])==conditions
assert all(abs(record['training']['full_metrics'][m]-audit['normal'][m])<1e-8 for m in METRICS)
prior=json.loads((PROJECT/'results/full_official_modality_outlet_m4_20261005/analysis.json').read_text(encoding='utf-8'));assert prior['status']=='THREE_FULL_OFFICIAL_M4_RUNS_AND882_STATE_CASES_ANALYZED'
references={'original_demo':prior['references']['demo']['conditions'],'previous_shared_demo':prior['references']['demo_shared']['conditions'],
 'M5_demo_shared':first['models']['demo_shared']['audit']['conditions'],'M5_frequency_shared':first['models']['frequency_shared']['audit']['conditions'],
 'M4_axis_shared':{c:r['metrics']['11'] for c,r in prior['models']['axis_shared']['conditions'].items()}}
comparisons={}
for label,old in references.items():
 delta={c:{m:audit['conditions'][c][m]-old[c][m] for m in METRICS} for c in sorted(conditions)}
 groups={g:[r for c,r in delta.items() if g in condition_groups(c)] for g in ('same_availability','overlap_mismatch','source_disjoint','partial_query_full_gallery','both_partial')}
 comparisons[label]=dict(normal_delta_pp=delta['q_RNT_g_RNT'],condition_delta_pp=delta,all49_equal_condition_mean_delta_pp={m:fmean(r[m] for r in delta.values()) for m in METRICS},
  groups={g:dict(conditions=len(rows),equal_condition_mean_delta_pp={m:fmean(r[m] for r in rows) for m in METRICS}) for g,rows in groups.items()})
report=dict(status='ONE_REAL_CLOSED_AXIS_FULL50_AND49_GT_FROZEN_CASES_READ',read_at=datetime.now().isoformat(timespec='seconds'),source=source,axis=record,comparisons=comparisons,frozen_cases_read=49,normal_both_plus2_vs_original=all(comparisons['original_demo']['normal_delta_pp'][m]>=2 for m in ('mAP','Rank-1')),new_neural_execution=0,new_raw_distance_audit=0,goal_complete=False,
 limits='Full dual-axis49 scalar CPU-audited result with signed original/shared/ordinary-frequency/previous-axis comparisons, not final four-model analysis. Ordinary twins and actual candidate six-state inference utility remain pending.49 conditions repeat591 queries; no untouched-test, multiseed or three-dataset claim.')
output.write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
print('AXIS_FULL50_AND49_GT_FROZEN_RESULT',json.dumps(dict(epoch=audit['selected_epoch'],normal=audit['normal'],both_plus2=report['normal_both_plus2_vs_original'])),flush=True)
for label,r in comparisons.items():print(label,json.dumps(dict(normal=r['normal_delta_pp'],all49=r['all49_equal_condition_mean_delta_pp'],groups=r['groups'])),flush=True)