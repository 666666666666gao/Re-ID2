from datetime import datetime
import json,sys
from pathlib import Path
from statistics import fmean
sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,remote_python
from analyze_full_official_baseline_pairs import METRICS
from analyze_full_official_control_trial import STATES,comparison_groups
pf=PROJECT/'results/preflight';output=pf/'full_official_availability_graph_first_pair_frozen_utility_20261005.json';assert not output.exists()
closed=json.loads((pf/'full_official_availability_graph_first_closed_pair_20261005.json').read_text(encoding='utf-8'))
source=closed['source']+'/diagnosis/MSVR310_graph_frequency_shared_s42'
audit=json.loads(remote_python('2026','import json;from pathlib import Path;p=Path('+repr(source)+');a=json.loads((p/"independent_cpu_audit.json").read_text());assert a["status"]=="PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT" and a["cases"]==294;print(json.dumps(a))'))
assert audit['repeated_condition_query_rows']==173754 and audit['calibration_rows']==28959
assert audit['model_arguments']==closed['models']['frequency_shared']['training']['arguments']
frozen=closed['models']['frequency_shared']['audit'];assert set(audit['conditions'])==set(frozen['conditions'])
for c,row in audit['conditions'].items():
 assert set(row['metrics'])==set(STATES)
 assert all(abs(row['metrics']['11'][m]-frozen['conditions'][c][m])<1e-8 for m in METRICS)
effects={}
for key in ('full11_minus00','full11_minus10','full11_minus01'):
 rows={c:dict(r[key],both_metrics_plus2=all(r[key]['delta_pp'][m]>=2 for m in ('mAP','Rank-1'))) for c,r in audit['conditions'].items()}
 effects[key]=dict(normal=rows['q_RNT_g_RNT'],groups=comparison_groups(rows),all49_equal_condition_mean_delta_pp={m:fmean(r['delta_pp'][m] for r in rows.values()) for m in METRICS})
cal=[r['calibration'] for r in audit['conditions'].values()]
summary=dict(status='ONE_REAL_COMPLETED_ORDINARY_FREQUENCY294_STATE_GT_CASES_READ',read_at=datetime.now().isoformat(timespec='seconds'),source=source,state_cases=294,repeated_condition_query_rows=173754,audit=audit,inference_effects=effects,
 calibration_equal_condition_mean={k:[fmean(r[k][i] for r in cal) for i in range(3)] for k in ('target_mean','target_std','MAE','zero_prediction_MAE')},
 calibration_MAE_better_than_zero_conditions=[sum(r['MAE'][i]<r['zero_prediction_MAE'][i] for r in cal) for i in range(3)],
 new_neural_execution=0,new_raw_distance_audit=0,goal_complete=False,
 prior_readout_error=dict(error='KeyError: both_metrics_plus2',stage='Local scalar comparison_groups consumer before output write',correction='Existing state audit does not store the consumer-derived plus2 boolean; compute it from the original signed mAP/Rank-1 deltas. Original audit and metrics remain unchanged; no neural or CPU scoring repeated.'),
 limits='This is the completed ordinary frequency control only, not the unfinished dual-axis candidate or final four-model comparison.00 is this trained model base; conditions repeat591 queries. Existing installed-GT state CPU audit read, no new inference or scoring.')
output.write_bytes((json.dumps(summary,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
for k,v in effects.items():print(k,json.dumps(dict(normal=v['normal'],all49=v['all49_equal_condition_mean_delta_pp'],groups=v['groups'])),flush=True)
print('ORDINARY_GRAPH_FREQUENCY_CLOSED_CALIBRATION',json.dumps(dict(better_than_zero=summary['calibration_MAE_better_than_zero_conditions'],calibration=summary['calibration_equal_condition_mean'])),flush=True)