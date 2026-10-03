import csv
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT

root = PROJECT / 'results/common_coordinate_v12_trial_20261003/controlled_states'
report = json.loads((root / 'independent_cpu_audit.json').read_text())
assert report['status'] == 'PASS' and report['cases'] == 1176
metrics = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
states = ('00', '10', '01', '11', 'base_private', 'base_shared')
rows, variants = [], {}
for variant, run in report['runs'].items():
    conditions = run['conditions']
    assert len(conditions) == 49
    for condition, values in conditions.items():
        for state, values_ in values['metrics'].items():
            rows.append(dict(variant=variant, condition=condition, state=state,
                             selected_epoch=run['selected_epoch'], **values_))
    variants[variant] = dict(selected_epoch=run['selected_epoch'], residual_scale=run['residual_scale'],
        normal=conditions['q_RNT_g_RNT'],
        equal_condition_means={state: {key: statistics.mean(v['metrics'][state][key] for v in conditions.values()) for key in metrics} for state in states},
        expert_changes={state: dict(
            mean_delta_pp={key: statistics.mean(v[state]['delta_pp'][key] for v in conditions.values()) for key in metrics},
            Rank1_harm_queries=sum(v[state]['Rank1_harm_queries'] for v in conditions.values()),
            Rank1_rescue_queries=sum(v[state]['Rank1_rescue_queries'] for v in conditions.values()),
            AP_improved_queries=sum(v[state]['AP_improved_queries'] for v in conditions.values()),
            AP_worsened_queries=sum(v[state]['AP_worsened_queries'] for v in conditions.values()),
            double_plus2_conditions=sum(v[state]['delta_pp']['mAP'] >= 2 and v[state]['delta_pp']['Rank-1'] >= 2 for v in conditions.values()))
            for state in ('full_vs_base', 'M_vs_base', 'F_vs_base', 'full_vs_M', 'full_vs_F')},
        private_disjoint_conditions=[name for name in conditions if set(name.split('_')[1]).isdisjoint(set(name.split('_')[3]))])
    assert len(variants[variant]['private_disjoint_conditions']) == 12
output = PROJECT / 'results/preflight/common_coordinate_states_analysis.json'
assert not output.exists()
output.write_text(json.dumps(dict(status='ACTUAL_FROZEN_GT_CPU_RECOUNT_ANALYSIS',
    variants=variants, metric_cases=len(rows), independent_raw_distance_recount=report['status'],
    limits='equal-condition mean is diagnostic, not an official aggregate mAP; query counts across49 repeat queries and are condition-query pairs; frozen states do not replace separately trained ablations; no all3 or multi-seed acceptance'), indent=2) + '\n')
table = root / 'MSVR310_all4_sixstates_full49_metrics_1176.csv'
with table.open('x', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('ACTUAL_STATE_UTILITY', json.dumps({key: dict(normal_full_vs_base=value['normal']['full_vs_base'],
    mean_full_vs_base=value['expert_changes']['full_vs_base'],
    mean_common_metrics=value['equal_condition_means']['base_shared']) for key, value in variants.items()}), flush=True)
