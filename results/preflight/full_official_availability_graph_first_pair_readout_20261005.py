"""Read only the two already completed graph endpoints and their signed scalar metrics."""
from datetime import datetime
import json
from pathlib import Path
from statistics import fmean
import sys
sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,remote_python
from analyze_full_official_baseline_pairs import METRICS,SETS
from analyze_full_official_control_trial import condition_groups
pf=PROJECT/'results/preflight'
output=pf/'full_official_availability_graph_first_closed_pair_20261005.json'
assert not output.exists()
source=json.loads((pf/'full_official_availability_graph_validated_launch_20261005.json').read_text(encoding='utf-8'))['launch']['output']
models=json.loads(remote_python('2026',f'''import json
from pathlib import Path
root=Path({source!r});result={{}}
for variant in ('demo_shared','frequency_shared'):
 name='MSVR310_graph_'+variant+'_s42'
 train=json.loads((root/'training'/name/'result.json').read_text())
 audit=json.loads((root/'frozen49'/name/'independent_cpu_audit.json').read_text())
 evaluated=json.loads((root/'frozen49'/name/'result.json').read_text())
 assert train['status']==evaluated['status']=='COMPLETE' and train['epochs']==50
 assert train['training_heldout_identities']==0 and train['training_coverage']==dict(eligible=1032,visited=1032,unvisited=[])
 assert (train['train_records'],train['query_records'],train['gallery_records'])==(1032,591,1055)
 assert audit['status']=='PASS' and audit['cases']==49 and audit['perquery_count']==28959
 assert evaluated['selected_epoch']==audit['selected_epoch']==train['best']['epoch']
 assert train['arguments']==evaluated['model_arguments']==audit['model_arguments']
 for phase in ('training','frozen49','audit'):
  assert json.loads((root/phase/(name+'_exit.json')).read_text())['exit_code']==0
 result[variant]=dict(training=train,audit=audit)
print(json.dumps(result))
'''))
prior=json.loads((PROJECT/'results/full_official_modality_outlet_m4_20261005/analysis.json').read_text(encoding='utf-8'))
assert prior['status']=='THREE_FULL_OFFICIAL_M4_RUNS_AND882_STATE_CASES_ANALYZED'
conditions={'q_'+q+'_g_'+g for q in SETS for g in SETS}
comparisons={}
for variant in ('demo_shared','frequency_shared'):
 trained,audit=models[variant]['training'],models[variant]['audit']
 assert set(audit['conditions'])==conditions
 assert all(abs(trained['full_metrics'][m]-audit['normal'][m])<1e-8 for m in METRICS)
 if variant=='demo_shared':
  old=prior['references']['demo_shared']['conditions']
 else:
  old={c:r['metrics']['11'] for c,r in prior['models'][variant]['conditions'].items()}
 delta={c:{m:audit['conditions'][c][m]-old[c][m] for m in METRICS} for c in sorted(conditions)}
 groups={g:[r for c,r in delta.items() if g in condition_groups(c)] for g in ('same_availability','overlap_mismatch','source_disjoint','partial_query_full_gallery','both_partial')}
 comparisons[variant]=dict(normal_delta_pp=delta['q_RNT_g_RNT'],condition_delta_pp=delta,
  all49_equal_condition_mean_delta_pp={m:fmean(r[m] for r in delta.values()) for m in METRICS},
  groups={g:dict(conditions=len(rows),equal_condition_mean_delta_pp={m:fmean(r[m] for r in rows) for m in METRICS}) for g,rows in groups.items()})
record=dict(status='TWO_REAL_CLOSED_FULL50_GRAPH_ENDPOINTS_AND98_GT_FROZEN_CASES_READ',read_at=datetime.now().isoformat(timespec='seconds'),source=source,models=models,comparison_with_same_model_previous_training=comparisons,actual_frozen_cases_read=98,new_neural_execution=0,new_raw_distance_audit=0,goal_complete=False,
 limits='Only two completed controls, not the final four-model method result. Signed six metrics from existing installed-GT CPU audits;49 conditions repeat591 queries and equal-condition group means are descriptive. Expert states and candidate/fair twins results remain pending; no three-dataset or positive-inference claim.')
output.write_bytes((json.dumps(record,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
for v in ('demo_shared','frequency_shared'):
 print(v,json.dumps(dict(epoch=models[v]['audit']['selected_epoch'],normal=models[v]['audit']['normal'],delta=comparisons[v]['normal_delta_pp'],all49=comparisons[v]['all49_equal_condition_mean_delta_pp'],groups=comparisons[v]['groups'])),flush=True)
print('TWO_CLOSED_GRAPH_ENDPOINTS_READ_ONCE',flush=True)