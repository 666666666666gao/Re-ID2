"""Summarize a completed full-data M3a/M3b trial, its controls and frozen utility."""
import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

from analyze_full_official_baseline_pairs import METRICS, SETS, GROUPS, query_rows, summarize


MODES = ('measurement_only', 'independent_control')
VARIANTS = ('axis_shared', 'frequency_shared', 'twins_shared')
STATES = ('00', '10', '01', '11', 'base_private', 'base_shared')
GT_FIELDS = ('query_index', 'name', 'identity', 'camera', 'scene', 'valid',
             'relevant_gallery', 'kept_gallery')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def condition_groups(condition):
    _, qset, _, gset = condition.split('_')
    primary = 'same_availability' if qset == gset else (
        'overlap_mismatch' if set(qset) & set(gset) else 'source_disjoint')
    groups = [primary]
    if qset != 'RNT' and gset == 'RNT': groups.append('partial_query_full_gallery')
    if qset != 'RNT' and gset != 'RNT': groups.append('both_partial')
    return groups


def compare(before, after):
    assert len(before) == len(after) == 591
    assert all(all(a[k] == b[k] for k in GT_FIELDS) for a, b in zip(before, after))
    old, new = summarize(before), summarize(after)
    delta = {m: new[m] - old[m] for m in METRICS}
    ap = [float(b['AP']) - float(a['AP']) for a, b in zip(before, after)]
    return dict(delta_pp=delta,
        Rank1_harm_queries=sum(int(a['first_match']) == 1 and int(b['first_match']) != 1 for a, b in zip(before, after)),
        Rank1_rescue_queries=sum(int(a['first_match']) != 1 and int(b['first_match']) == 1 for a, b in zip(before, after)),
        AP_improved_queries=sum(d > 0 for d in ap), AP_worsened_queries=sum(d < 0 for d in ap),
        AP_unchanged_queries=sum(d == 0 for d in ap),
        both_metrics_plus2=all(delta[m] >= 2 for m in ('mAP', 'Rank-1')))


def comparison_groups(conditions):
    groups = {g: [] for g in GROUPS}
    for condition, row in conditions.items():
        for group in condition_groups(condition): groups[group].append(row)
    assert all(len(groups[g]) == n for g, n in GROUPS.items())
    return {g: dict(conditions=len(rows),
        equal_condition_mean_delta_pp={m: fmean(r['delta_pp'][m] for r in rows) for m in METRICS},
        Rank1_harm_repeated_query_rows=sum(r['Rank1_harm_queries'] for r in rows),
        Rank1_rescue_repeated_query_rows=sum(r['Rank1_rescue_queries'] for r in rows),
        both_metrics_plus2_conditions=sum(r['both_metrics_plus2'] for r in rows)) for g, rows in groups.items()}


def analyze(trial, baseline):
    terminal = read(trial / 'controller_result.json')
    assert terminal['status'] == 'COMPLETE' and len(terminal['runs']) == 6
    assert terminal['training_heldout_identities'] == 0 and terminal['paired_identity_and_partial_sampling_exact']
    assert terminal['all_frozen_state_cpu_audits_passed'] and terminal['frozen_state_metric_cases'] == 1764
    assert terminal['frozen_state_repeated_query_rows'] == 1042524 and terminal['contribution_rows'] == 173754
    expected_conditions = {'q_' + q + '_g_' + g for q in SETS for g in SETS}
    baseline_result = read(baseline / 'summary.json')
    assert baseline_result['status'] == 'SIX_FULL_OFFICIAL_BASELINES_COMPLETE_ALL294_CPU_AUDITED'
    assert baseline_result['paired_identity_sampling_exact'] and baseline_result['cases'] == 294
    refs, full_rows, training_orders, reference_orders = {}, {}, [], []
    for variant in ('demo', 'demo_shared'):
        name = 'MSVR310_' + variant + '_s42'
        trained = read(baseline / 'training' / name / 'result.json')
        audit = read(baseline / 'frozen49' / name / 'independent_cpu_audit.json')
        assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50 and trained['training_heldout_identities'] == 0
        assert trained['training_coverage'] == dict(eligible=1032, visited=1032, unvisited=[])
        assert audit['status'] == 'PASS' and audit['cases'] == 49 and set(audit['conditions']) == expected_conditions
        with (baseline / 'training' / name / 'batch_orders.jsonl').open(encoding='utf-8') as handle:
            reference_orders.append([(r['epoch'], r['step'], r['names']) for r in map(json.loads, handle)])
        refs[variant] = audit
        full_rows[variant] = {}
        for condition in expected_conditions:
            rows = query_rows(baseline / 'frozen49' / name / (condition + '.csv'), 591)
            assert all(abs(summarize(rows)[m] - audit['conditions'][condition][m]) < 1e-8 for m in METRICS)
            full_rows[variant][condition] = rows
    reports = {}
    for mode in MODES:
        for variant in VARIANTS:
            name = 'MSVR310_' + mode + '_' + variant + '_s42'
            run, frozen, diagnosis = (trial / phase / name for phase in ('training', 'frozen49', 'diagnosis'))
            trained, frozen_result, frozen_audit, states = map(read, (
                run / 'result.json', frozen / 'result.json', frozen / 'independent_cpu_audit.json',
                diagnosis / 'independent_cpu_audit.json'))
            assert trained['status'] == frozen_result['status'] == 'COMPLETE' and trained['epochs'] == 50
            arguments = trained['arguments']
            assert (arguments['dataset'], arguments['variant'], arguments['gate_gradient_mode'], arguments['seed']) == (
                'MSVR310', variant, mode, 42)
            assert (trained['train_records'], trained['query_records'], trained['gallery_records']) == (1032, 591, 1055)
            assert trained['training_heldout_identities'] == 0 and trained['descriptor_dim'] == 5632
            assert trained['training_coverage'] == dict(eligible=1032, visited=1032, unvisited=[])
            assert frozen_audit['status'] == 'PASS' and frozen_audit['cases'] == 49 and frozen_audit['perquery_count'] == 28959
            assert states['status'] == 'PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT' and states['cases'] == 294
            assert states['repeated_condition_query_rows'] == 173754 and states['calibration_rows'] == 28959
            assert states['model_arguments'] == frozen_audit['model_arguments'] == arguments
            assert (states['query_records'], states['gallery_records']) == (591, 1055)
            assert set(states['conditions']) == set(frozen_audit['conditions']) == expected_conditions
            assert frozen_audit['selected_epoch'] == trained['best']['epoch']
            for phase in ('training', 'frozen49', 'audit', 'diagnosis', 'diagnosis_audit'):
                assert read(trial / phase / (name + '_exit.json'))['exit_code'] == 0
            with (run / 'batch_orders.jsonl').open(encoding='utf-8') as handle:
                orders = list(map(json.loads, handle))
            assert len(orders) == trained['steps'] and sum(r['optimizer_updated'] for r in orders) == trained['optimizer_steps']
            training_orders.append([(r['epoch'], r['step'], r['names'], r['partial_set']) for r in orders])
            full_rows[mode + '/' + variant] = {}
            for condition in sorted(expected_conditions):
                assert set(states['conditions'][condition]['metrics']) == set(STATES)
                for state in STATES:
                    rows = query_rows(diagnosis / condition / ('state_' + state + '.csv'), 591)
                    measured = summarize(rows)
                    assert all(abs(measured[m] - states['conditions'][condition]['metrics'][state][m]) < 1e-8 for m in METRICS)
                    if state == '11':
                        assert all(abs(measured[m] - frozen_audit['conditions'][condition][m]) < 1e-8 for m in METRICS)
                        full_rows[mode + '/' + variant][condition] = rows
            assert all(abs(states['normal']['metrics']['11'][m] - trained['full_metrics'][m]) < 1e-8 for m in METRICS)
            effects = {c: dict(states['conditions'][c]['full11_minus00'], both_metrics_plus2=all(
                states['conditions'][c]['full11_minus00']['delta_pp'][m] >= 2 for m in ('mAP', 'Rank-1')))
                for c in sorted(expected_conditions)}
            calibrations = [states['conditions'][c]['calibration'] for c in sorted(expected_conditions)]
            reports[mode + '/' + variant] = dict(selected_epoch=trained['best']['epoch'], normal=states['normal'],
                conditions=states['conditions'], inference_effect_groups=comparison_groups(effects),
                all49_full11_equal_condition_mean={m: fmean(states['conditions'][c]['metrics']['11'][m]
                    for c in expected_conditions) for m in METRICS}, heads=states['heads'],
                calibration_equal_condition_mean={k: [fmean(r[k][i] for r in calibrations) for i in range(3)]
                    for k in ('target_mean', 'target_std', 'MAE', 'zero_prediction_MAE')},
                calibration_MAE_better_than_zero_conditions=[sum(r['MAE'][i] < r['zero_prediction_MAE'][i]
                    for r in calibrations) for i in range(3)],
                parameters=trained['parameters'], trainable_parameters=trained['trainable_parameters'],
                descriptor_dim=trained['descriptor_dim'], optimizer_steps=trained['optimizer_steps'],
                amp_skipped_steps=trained['amp_skipped_steps'], training_coverage=trained['training_coverage'],
                runtime=trained['runtime'], frozen_availability_runtime=frozen_result['runtime'],
                peak_training_memory_bytes=trained['peak_memory'])
    assert all(order == training_orders[0] for order in training_orders[1:])
    assert reference_orders[0] == reference_orders[1] == [row[:3] for row in training_orders[0]]
    for mode in MODES:
        assert len({(reports[mode + '/' + v]['parameters'], reports[mode + '/' + v]['trainable_parameters'],
            reports[mode + '/' + v]['descriptor_dim']) for v in VARIANTS}) == 1
    for variant in VARIANTS:
        assert reports['independent_control/' + variant]['parameters'] - reports['measurement_only/' + variant]['parameters'] == 27769
        assert reports['independent_control/' + variant]['trainable_parameters'] - reports['measurement_only/' + variant]['trainable_parameters'] == 27769
    pairs = []
    for mode in MODES:
        axis = mode + '/axis_shared'
        for before in ('demo', 'demo_shared', mode + '/frequency_shared', mode + '/twins_shared'):
            pairs.append((before, axis))
    pairs += [('measurement_only/' + variant, 'independent_control/' + variant) for variant in VARIANTS]
    comparisons = {}
    for before, after in pairs:
        conditions = {c: compare(full_rows[before][c], full_rows[after][c]) for c in sorted(expected_conditions)}
        comparisons[before + ' -> ' + after] = dict(before=before, after=after, normal=conditions['q_RNT_g_RNT'],
            conditions=conditions, groups=comparison_groups(conditions),
            both_metrics_plus2_conditions=sum(r['both_metrics_plus2'] for r in conditions.values()),
            all49_equal_condition_mean_delta_pp={m: fmean(r['delta_pp'][m] for r in conditions.values()) for m in METRICS})
    return dict(status='SIX_FULL_OFFICIAL_CONTROL_RUNS_AND1764_STATE_CASES_ANALYZED',
        dataset='MSVR310', seed=42, full_split_counts=dict(train=1032, query=591, gallery=1055),
        training_heldout_identities=0, paired_identity_and_partial_sampling_exact=True,
        metric_cases=1764, repeated_condition_query_rows=1042524, contribution_rows=173754,
        models=reports, comparisons=comparisons,
        original_DeMo_plus2_thresholds={m: refs['demo']['normal'][m] + 2 for m in ('mAP', 'Rank-1')},
        goal_complete=False,
        limits='Full-data single-seed MSVR mechanism trial, not the unified three-dataset/multiseed goal. '
        'All checkpoints chosen by official benchmark mAP. Equal-condition and overlapping group means '
        'reuse queries and are diagnostic, not independent observations or an official aggregate. '
        '11-versus00 closes experts at inference in the same trained model; it is not a retrained baseline. '
        'M3b adds27769 parameters relative to M3a; ordinary controls match capacity within each mode. '
        'Contribution targets are empirical fixed-reference query margins, not causal or information-theoretic synergy. '
        'This text analysis rechecks exported metrics and query changes; independent native-GT raw-distance '
        'and calibration arithmetic audits remain the saved CPU audit evidence. No NN rerun or new training.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial-root', required=True)
    parser.add_argument('--baseline-root', required=True)
    args = parser.parse_args()
    trial = Path(args.trial_root)
    output = trial / 'analysis.json'
    assert not output.exists()
    report = analyze(trial, Path(args.baseline_root))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fields = ['comparison', 'condition', *METRICS, 'Rank1_harm_queries', 'Rank1_rescue_queries',
        'AP_improved_queries', 'AP_worsened_queries', 'AP_unchanged_queries', 'both_metrics_plus2']
    with (trial / 'comparison_condition_deltas.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, comparison in report['comparisons'].items():
            for condition, row in comparison['conditions'].items():
                writer.writerow(dict(comparison=label, condition=condition, **row['delta_pp'],
                    **{k: row[k] for k in fields[8:]}))
    print('FULL_OFFICIAL_CONTROL_TRIAL_ALL1764_CASES_ANALYZED', flush=True)


if __name__ == '__main__': main()
