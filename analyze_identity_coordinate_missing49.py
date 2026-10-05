"""Reduce completed three-dataset fixed-checkpoint missing results; no NN."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path

from analyze_full_official_baseline_pairs import query_rows, summarize
from analyze_identity_coordinate_three_normal import COUNTS, METRICS, table

SETS = ('R', 'N', 'T', 'RN', 'RT', 'NT', 'RNT')
STATES = ('00', '10', '01', '11')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verified_rows(path, metrics, count, gallery):
    rows = query_rows(path, count)
    assert metrics['query_count'] == metrics['valid_queries'] == count
    assert metrics['gallery_count'] == gallery and metrics['invalid_queries'] == 0
    assert all(abs(summarize(rows)[key] - metrics[key]) < 1e-8 for key in METRICS)
    assert len(metrics['CMC_1_to_50']) == 50
    assert all(abs(100 * sum(int(row['first_match']) <= rank for row in rows) / count - metrics['CMC_1_to_50'][rank - 1]) < 1e-8 for rank in range(1, 51))
    for grouping in ('identity', 'camera', 'scene'):
        groups = metrics['groups'][grouping]
        assert {int(row[grouping]) for row in rows} == {group['value'] for group in groups}
        for group in groups:
            members = [row for row in rows if int(row[grouping]) == group['value']]
            assert group['queries'] == len(members)
            assert all(abs(summarize(members)[key] - group[key]) < 1e-8 for key in METRICS)
    return rows


def availability_groups(q, g):
    groups = ['all49', 'same_availability' if q == g else 'overlap_mismatch' if set(q) & set(g) else 'source_disjoint']
    if q != 'RNT' and g == 'RNT': groups.append('partial_query_full_gallery')
    if q == 'RNT' and g != 'RNT': groups.append('full_query_partial_gallery')
    if q != 'RNT' and g != 'RNT': groups.append('both_partial')
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', required=True)
    parser.add_argument('--rgbnt201', required=True)
    parser.add_argument('--baselines', required=True)
    parser.add_argument('--normal', required=True)
    args = parser.parse_args()
    stream, r201, baselines, normal = map(Path, (args.stream, args.rgbnt201, args.baselines, args.normal))
    controller = load(stream / 'controller_full_result.json')
    assert controller['status'] == 'COMPLETE' and controller['controls'] == 4
    assert controller['missing_GT_state_cases'] == 784 and controller['raw_conditions'] == 196
    assert controller['all_raw_local_verified'] and controller['optimizer_updates'] == controller['new_weights'] == 0
    assert load(r201 / 'controller_result.json')['closed_state_cases'] == 784
    assert load(normal / 'normal_analysis/result.json')['status'] == 'ACTUAL_THREE_NORMAL_DATASETS_CPU_ANALYSIS_COMPLETE'
    metrics_rows, deltas, group_rows, identities = [], [], [], []
    checked_queries = 0
    for dataset, (_, count, gallery, _) in COUNTS.items():
        original = baselines / 'frozen49' / (dataset + '_demo_s42')
        reference = load(original / 'result.json')['measurements']
        audit = load(original / 'independent_cpu_audit.json')
        assert audit['status'] == 'PASS' and audit['cases'] == 49
        models = {'original_DeMo': (reference, original)}
        for variant in ('frequency_shared', 'axis_shared'):
            name = dataset + '_identity_' + variant + '_narrow_s42'
            folder = r201 / 'frozen49' / name if dataset == 'RGBNT201' else stream / 'full' / name
            result = load(folder / 'result.json')
            audited = load(folder / ('independent_fourstate_cpu_audit.json' if dataset == 'RGBNT201' else 'independent_cpu_audit.json'))
            assert result['status'] == 'COMPLETE' and result['state_cases'] == audited['state_cases'] == 196
            assert audited['status'] == 'PASS'
            trained = load(normal / 'training' / name / 'result.json')
            assert result['selected_epoch'] == trained['best']['epoch']
            measurements = result['state_measurements'] if dataset == 'RGBNT201' else result['measurements']
            for state in STATES:
                key = variant + '/' + state
                models[key] = ({condition: values[state] for condition, values in measurements.items()}, folder)
            assert all(abs(measurements['q_RNT_g_RNT']['11'][metric] - trained['full_metrics'][metric]) < 1e-8 for metric in METRICS)
        pairs = [('axis_shared/11', ref) for ref in ('original_DeMo', 'axis_shared/00', 'axis_shared/10', 'axis_shared/01', 'frequency_shared/11')]
        pairs.extend(('frequency_shared/11', ref) for ref in ('original_DeMo', 'frequency_shared/00', 'frequency_shared/10', 'frequency_shared/01'))
        dataset_deltas = []
        for q in SETS:
            for g in SETS:
                condition = 'q_' + q + '_g_' + g
                read = {}
                for model, (measurements, folder) in models.items():
                    state = model.rsplit('/', 1)[-1]
                    if model == 'original_DeMo': path = folder / (condition + '.csv')
                    elif dataset == 'RGBNT201': path = folder / (condition + ('' if state == '11' else '_state' + state) + '.csv')
                    else: path = folder / condition / ('state_' + state + '.csv')
                    values = measurements[condition]
                    rows = verified_rows(path, values, count, gallery)
                    checked_queries += len(rows)
                    read[model] = rows
                    metrics_rows.append(dict(dataset=dataset, model=model, condition=condition, **{key: values[key] for key in METRICS}))
                    for group in values['groups']['identity']:
                        identities.append(dict(dataset=dataset, model=model, condition=condition, **group))
                assert read['axis_shared/00'] == read['frequency_shared/00']
                for improved, reference_name in pairs:
                    left, right = read[improved], read[reference_name]
                    assert all(tuple(a[key] for key in ('query_index', 'name', 'identity', 'camera', 'scene', 'valid', 'relevant_gallery', 'kept_gallery')) == tuple(b[key] for key in ('query_index', 'name', 'identity', 'camera', 'scene', 'valid', 'relevant_gallery', 'kept_gallery')) for a, b in zip(left, right))
                    a, b = models[improved][0][condition], models[reference_name][0][condition]
                    row = dict(dataset=dataset, condition=condition, improved=improved, reference=reference_name,
                        **{key: a[key] - b[key] for key in METRICS},
                        rescued=sum(int(x['Rank-1']) == 1 and int(y['Rank-1']) == 0 for x, y in zip(left, right)),
                        harmed=sum(int(x['Rank-1']) == 0 and int(y['Rank-1']) == 1 for x, y in zip(left, right)),
                        AP_improved=sum(float(x['AP']) > float(y['AP']) for x, y in zip(left, right)),
                        AP_worsened=sum(float(x['AP']) < float(y['AP']) for x, y in zip(left, right)),
                        both_plus2=a['mAP'] - b['mAP'] >= 2 and a['Rank-1'] - b['Rank-1'] >= 2)
                    dataset_deltas.append((availability_groups(q, g), row))
                    deltas.append(row)
        sizes = {'all49':49, 'same_availability':7, 'overlap_mismatch':30, 'source_disjoint':12,
                 'partial_query_full_gallery':6, 'full_query_partial_gallery':6, 'both_partial':36}
        for improved, ref in pairs:
            for group, size in sizes.items():
                rows = [row for groups, row in dataset_deltas if group in groups and row['improved'] == improved and row['reference'] == ref]
                assert len(rows) == size
                group_rows.append(dict(dataset=dataset, improved=improved, reference=ref, group=group, conditions=size,
                    **{key: math.fsum(row[key] for row in rows) / size for key in METRICS},
                    rescued=sum(row['rescued'] for row in rows), harmed=sum(row['harmed'] for row in rows),
                    both_plus2_conditions=sum(row['both_plus2'] for row in rows)))
    assert len(metrics_rows) == len(deltas) == 1323 and checked_queries == 49 * 9 * sum(x[1] for x in COUNTS.values())
    out = stream / 'three_dataset_missing_analysis'
    out.mkdir(exist_ok=False)
    for name, rows in [('six_metrics', metrics_rows), ('comparisons', deltas), ('availability_group_deltas', group_rows), ('identity_metrics', identities)]:
        table(out / (name + '.csv'), rows)
    result = dict(status='ACTUAL_THREE_DATASET_FIXED_BEST_ALL49_FOURSTATE_CPU_READOUT',
        completed_at=datetime.now().isoformat(timespec='seconds'), metric_rows=1323, comparisons=1323,
        checked_condition_query_rows=checked_queries, original_query_records=3142, seed=42,
        full_metrics_CMC50_identity_camera_scene_reduced=True, new_neural_calls=0, new_optimizer_updates=0,
        limits='Group means are equal-condition diagnostics, not official pooled mAP. Repeated condition-query rows are not independent samples. Original50 versus new100 budget is disclosed; frequency/axis new50 are matched. Original baseline missing processing differs from controlled availability-masked00; baseline gains alone cannot establish dual-axis utility. The12 disjoint-source conditions lack common private coordinates. Benchmark selected best and one seed do not establish independent final-test or multiseed success.')
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('ACTUAL_THREE_DATASET_MISSING_CPU_READOUT_COMPLETE', checked_queries, flush=True)


if __name__ == '__main__':
    main()
