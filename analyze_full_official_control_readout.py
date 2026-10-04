"""Analyze1176 installed-GT audited frozen readouts without rerunning neural models."""
import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

from analyze_full_official_control_trial import MODES, GT_FIELDS, compare, condition_groups, comparison_groups
from analyze_full_official_baseline_pairs import METRICS, SETS, GROUPS, query_rows, summarize


STAGES = ('deployed', 'base_common', 'M_pre', 'M_post', 'F_pre', 'F_post', 'M_aux', 'F_aux')
READOUTS = {stage: (stage, stage) for stage in STAGES}
READOUTS.update(F_post_to_common=('F_post', 'base_common'), common_to_F_post=('base_common', 'F_post'),
    M_post_to_common=('M_post', 'base_common'), common_to_M_post=('base_common', 'M_post'))
COMPARISONS = dict(F_route=('F_aux', 'F_pre'), F_projection=('F_pre', 'F_post'),
    M_route=('M_aux', 'M_pre'), M_projection=('M_pre', 'M_post'),
    F_query_public_gallery=('base_common', 'F_post_to_common'),
    public_query_F_gallery=('base_common', 'common_to_F_post'),
    M_query_public_gallery=('base_common', 'M_post_to_common'),
    public_query_M_gallery=('base_common', 'common_to_M_post'))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def analyze(root, prior, mode):
    name = 'MSVR310_' + mode + '_axis_shared_s42'
    original = read(prior / 'training' / name / 'result.json')
    previous = read(prior / 'frozen49' / name / 'independent_cpu_audit.json')
    folder = root / 'frozen' / mode
    result = read(folder / 'result.json')
    audit = read(folder / 'independent_cpu_audit.json')
    assert original['status'] == 'COMPLETE' and original['epochs'] == 50
    assert original['training_coverage'] == dict(eligible=1032, visited=1032, unvisited=[])
    assert result['status'] == 'COMPLETE' and result['metric_cases'] == 588
    assert audit['status'] == 'PASS_FULL_OFFICIAL_READOUT_INSTALLED_GT' and audit['cases'] == 588
    assert audit['max_sixmetric_error_pp'] < 1e-8 and previous['status'] == 'PASS'
    assert result['model_arguments'] == audit['model_arguments'] == original['arguments'] == previous['model_arguments']
    assert result['selected_epoch'] == audit['selected_epoch'] == original['best']['epoch'] == previous['selected_epoch']
    assert result['training_heldout_identities'] == result['optimizer_updates'] == result['new_weights'] == 0
    assert (result['train_records'], result['query_records'], result['gallery_records']) == (1032, 591, 1055)
    assert result['repeated_condition_query_rows'] == audit['repeated_condition_query_rows'] == 347508
    assert result['previous_all49_deployed_metrics_perquery_exact'] and result['normal_feature_max_error'] == 0
    assert result['original_fuse_reconstructed_exact'] and result['state_tensor_versions_unchanged']
    assert result['readouts'] == audit['readouts'] == {key: list(value) for key, value in READOUTS.items()}
    assert tuple(result['stages']) == STAGES
    for phase in ('preflight', 'frozen', 'audit'):
        assert read(root / phase / (mode + '_exit.json'))['exit_code'] == 0
    smoke = read(root / 'preflight' / mode / 'smoke.json')
    assert smoke['status'] == 'PASS' and smoke['records'] == 64 and smoke['availability_sets'] == 7
    assert smoke['normal_feature_max_error'] == smoke['optimizer_updates'] == smoke['new_weights'] == 0
    expected = {'q_' + q + '_g_' + g for q in SETS for g in SETS}
    assert set(result['measurements']) == set(audit['conditions']) == set(previous['conditions']) == expected
    measurements, comparisons = {}, {key: {} for key in COMPARISONS}
    for condition in sorted(expected):
        rows, values = {}, {}
        assert set(result['measurements'][condition]) == set(audit['conditions'][condition]) == set(READOUTS)
        for stage in READOUTS:
            rows[stage] = query_rows(folder / condition / (stage + '.csv'), 591)
            values[stage] = summarize(rows[stage])
            assert all(abs(values[stage][metric] - result['measurements'][condition][stage][metric]) < 1e-8
                and abs(values[stage][metric] - audit['conditions'][condition][stage][metric]) < 1e-8 for metric in METRICS)
            assert all(all(a[key] == b[key] for key in GT_FIELDS) for a, b in zip(rows[stage], rows['deployed']))
        prior_rows = query_rows(prior / 'frozen49' / name / (condition + '.csv'), 591)
        assert rows['deployed'] == prior_rows
        assert all(abs(values['deployed'][metric] - previous['conditions'][condition][metric]) < 1e-8 for metric in METRICS)
        measurements[condition] = values
        for key, (before, after) in COMPARISONS.items():
            comparisons[key][condition] = compare(rows[before], rows[after])
    groups = {}
    for group, count in GROUPS.items():
        keys = [key for key in expected if group in condition_groups(key)]
        assert len(keys) == count
        groups[group] = dict(conditions=count, equal_condition_mean={stage: {
            metric: fmean(measurements[key][stage][metric] for key in keys) for metric in METRICS} for stage in READOUTS})
    equal_mean = {stage: {metric: fmean(values[stage][metric] for values in measurements.values()) for metric in METRICS}
        for stage in READOUTS}
    assert all(abs(equal_mean[stage][metric] - audit['equal_condition_mean'][stage][metric]) < 1e-8
        for stage in READOUTS for metric in METRICS)
    return dict(selected_epoch=result['selected_epoch'], model_arguments=result['model_arguments'],
        normal=measurements['q_RNT_g_RNT'], equal_condition_mean=equal_mean, groups=groups,
        measurements=measurements, comparisons={key: dict(before=COMPARISONS[key][0], after=COMPARISONS[key][1],
            conditions=values, groups=comparison_groups(values), normal=values['q_RNT_g_RNT'],
            all49_equal_condition_mean_delta_pp={metric: fmean(row['delta_pp'][metric] for row in values.values()) for metric in METRICS})
            for key, values in comparisons.items()}, deployed_all49_previous_perquery_exact=True,
        max_cpu_sixmetric_error_pp=audit['max_sixmetric_error_pp'], extraction_runtime=result['runtime'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--prior', required=True)
    args = parser.parse_args()
    root, prior = Path(args.root), Path(args.prior)
    intake = read(root / 'intake_receipt.json')
    terminal = read(root / 'controller_result.json')
    assert intake['status'] == 'BOTH_FROZEN_READOUTS_AND1176_CPU_CASES_TEXT_COLLECTED'
    assert terminal['status'] == 'COMPLETE' and terminal['metric_cases'] == 1176
    assert terminal['repeated_condition_query_rows'] == 695016
    assert terminal['training_heldout_identities'] == terminal['optimizer_updates'] == terminal['new_weights'] == 0
    assert {row['mode'] for row in terminal['runs']} == set(MODES)
    output = root / 'readout_analysis.json'
    assert not output.exists()
    report = dict(status='BOTH_FROZEN_FULL_OFFICIAL_READOUTS_AND1176_CASES_ANALYZED',
        modes={mode: analyze(root, prior, mode) for mode in MODES}, metric_cases=1176,
        repeated_condition_query_rows=695016, paired_diagnostic_condition_rows=784,
        training_heldout_identities=0, neural_reruns=0, training_updates=0, new_weights=0, goal_complete=False,
        readouts=READOUTS, comparison_definitions=COMPARISONS,
        limits='Single-seed MSVR fixed benchmark-selected weights. Eight same-stage and four directed public-coordinate '
            'readouts are empirical stage retrieval diagnostics, not fair new-model gains, information-loss proofs or causal effects. '
            'Seven stages are512D and deployed is5632D. Conditions reuse the591 queries; five groups overlap. '
            'Independent CPU audit recounts saved distances against installed official GT, without independent neural regeneration. '
            'Extraction runtime includes decoding/DataLoader first batch and all probed stages, not isolated GPU latency.')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fields = ['mode', 'comparison', 'before', 'after', 'condition', *METRICS, 'Rank1_harm_queries', 'Rank1_rescue_queries',
        'AP_improved_queries', 'AP_worsened_queries', 'AP_unchanged_queries']
    count = 0
    with (root / 'readout_condition_deltas.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for mode, result in report['modes'].items():
            for comparison, values in result['comparisons'].items():
                for condition, row in values['conditions'].items():
                    writer.writerow(dict(mode=mode, comparison=comparison, before=values['before'], after=values['after'],
                        condition=condition, **row['delta_pp'], **{key: row[key] for key in fields[11:]}))
                    count += 1
    assert count == 784
    print('FROZEN_FULL_OFFICIAL_ALL1176_READOUT_CASES_ANALYZED', flush=True)


if __name__ == '__main__':
    main()
