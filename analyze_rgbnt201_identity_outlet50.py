"""Local CPU reduction of the completed four-run experiment; no inference."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path

METRICS = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
SETS = ('R', 'N', 'T', 'RN', 'RT', 'NT', 'RNT')
STATES = ('00', '10', '01', '11')
CONDITIONS = tuple('q_' + q + '_g_' + g for q in SETS for g in SETS)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_table(path, rows):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def query_rows(path, metrics):
    with path.open(encoding='utf-8', newline='') as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 836 and all(row['valid'] == 'True' for row in rows)
    computed = dict(mAP=100 * math.fsum(float(r['AP']) for r in rows) / 836,
        mINP=100 * math.fsum(float(r['INP']) for r in rows) / 836,
        **{m: 100 * sum(int(r[m]) for r in rows) / 836 for m in METRICS[2:]})
    assert all(abs(computed[m] - metrics[m]) < 1e-8 for m in METRICS)
    assert len(metrics['CMC_1_to_50']) == 50
    assert all(abs(100 * sum(int(r['first_match']) <= k for r in rows) / 836 -
        metrics['CMC_1_to_50'][k - 1]) < 1e-8 for k in range(1, 51))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--baseline', required=True)
    args = parser.parse_args()
    root, baseline = Path(args.root), Path(args.baseline)
    controller = load(root / 'controller_result.json')
    assert controller['status'] == 'COMPLETE' and controller['models'] == 4
    assert controller['additional_epochs'] == 200 and controller['successful_updates'] == 10588
    assert controller['closed_state_cases'] == 784 and controller['paired_sampling_exact']
    assert controller['all_raw_local_verified']
    models = {'original_DeMo': load(baseline / 'result.json')['measurements']}
    folders = {'original_DeMo': baseline}
    training = {}
    for run in controller['runs']:
        name = run['name']
        folder = root / 'frozen49' / name
        train = load(root / 'training' / name / 'result.json')
        audit = load(folder / 'independent_fourstate_cpu_audit.json')
        assert audit['status'] == 'PASS' and audit['state_cases'] == 196
        assert train['status'] == 'COMPLETE' and train['epochs'] == 50
        assert train['optimizer_steps'] == 2647 and train['amp_skipped_steps'] == 0
        assert train['training_coverage'] == dict(eligible=3951, visited=3951, unvisited=[])
        assert train['training_heldout_identities'] == 0 and train['descriptor_dim'] == 5120
        training[name] = train
        result = load(folder / 'result.json')
        assert result['status'] == 'COMPLETE' and result['state_cases'] == 196
        assert result['selected_epoch'] == train['best']['epoch']
        for state in STATES:
            key = name + '/' + state
            models[key] = {c: result['state_measurements'][c][state] for c in CONDITIONS}
            folders[key] = folder
    assert len({(t['parameters'], t['trainable_parameters']) for t in training.values()}) == 1
    assert all(set(values) == set(CONDITIONS) for values in models.values())
    assert all(all(abs(t['full_metrics'][m] - models[name + '/11']['q_RNT_g_RNT'][m]) < 1e-8
        for m in METRICS) for name, t in training.items())
    pairs = []
    for name in training:
        pairs.extend((name + '/11', reference) for reference in
            ('original_DeMo', name + '/00', name + '/10', name + '/01'))
    job = lambda variant, mode: 'RGBNT201_identity_' + variant + '_' + mode + '_s42/11'
    pairs.extend((job('axis_shared', mode), job('frequency_shared', mode)) for mode in ('narrow', 'bypass'))
    pairs.extend((job(variant, 'bypass'), job(variant, 'narrow')) for variant in ('frequency_shared', 'axis_shared'))
    metrics_table, comparisons, normal_deltas = [], [], []
    for condition in CONDITIONS:
        queries = {}
        for key, values in models.items():
            state = key.rsplit('/', 1)[-1] if key != 'original_DeMo' else '11'
            stem = condition if state == '11' else condition + '_state' + state
            queries[key] = query_rows(folders[key] / (stem + '.csv'), values[condition])
            metrics_table.append(dict(model=key, condition=condition,
                **{m: values[condition][m] for m in METRICS},
                **{'delta_original_' + m: values[condition][m] - models['original_DeMo'][condition][m] for m in METRICS}))
        anchors = [queries[name + '/00'] for name in training]
        assert all(rows == anchors[0] for rows in anchors[1:])
        if condition == 'q_RNT_g_RNT':
            assert anchors[0] == queries['original_DeMo']
        for current, reference in pairs:
            a, b = queries[current], queries[reference]
            assert all(all(x[k] == y[k] for k in ('query_index', 'name', 'identity', 'camera', 'scene',
                'valid', 'relevant_gallery', 'kept_gallery')) for x, y in zip(a, b))
            comparisons.append(dict(model=current, reference=reference, condition=condition,
                **{'delta_' + m: models[current][condition][m] - models[reference][condition][m] for m in METRICS},
                rank1_rescue=sum(int(x['Rank-1']) > int(y['Rank-1']) for x, y in zip(a, b)),
                rank1_harm=sum(int(x['Rank-1']) < int(y['Rank-1']) for x, y in zip(a, b)),
                AP_improved=sum(float(x['AP']) > float(y['AP']) for x, y in zip(a, b)),
                AP_worsened=sum(float(x['AP']) < float(y['AP']) for x, y in zip(a, b))))
            if condition == 'q_RNT_g_RNT':
                normal_deltas.extend(dict(model=current, reference=reference, name=x['name'], identity=x['identity'],
                    AP_delta=float(x['AP']) - float(y['AP']), INP_delta=float(x['INP']) - float(y['INP']),
                    **{m + '_delta': int(x[m]) - int(y[m]) for m in METRICS[2:]}) for x, y in zip(a, b))
    groups = dict(all49=CONDITIONS, normal=('q_RNT_g_RNT',),
        same_availability=tuple('q_' + s + '_g_' + s for s in SETS),
        partial_query_full_gallery=tuple('q_' + s + '_g_RNT' for s in SETS if s != 'RNT'),
        full_query_partial_gallery=tuple('q_RNT_g_' + s for s in SETS if s != 'RNT'),
        disjoint=tuple('q_' + q + '_g_' + g for q in SETS for g in SETS if not set(q) & set(g)),
        overlapping_different_partial_sets=tuple('q_' + q + '_g_' + g for q in SETS for g in SETS
            if q != g and q != 'RNT' and g != 'RNT' and set(q) & set(g)))
    assert len(groups['disjoint']) == 12
    normal = 'q_RNT_g_RNT'
    summary = dict(status='ACTUAL_R201C_FOUR50_ALL49_FOURSTATE_ANALYZED',
        analyzed_at=datetime.now().isoformat(timespec='seconds'), dataset='RGBNT201', seed=42,
        official_train=3951, official_query=836, official_gallery=836, artificial_holdout=0,
        metrics_rows=len(metrics_table), comparison_rows=len(comparisons),
        normal_both_plus2={name: all(models[name + '/11'][normal][m] >= models['original_DeMo'][normal][m] + 2
            for m in ('mAP', 'Rank-1')) for name in training},
        training={name: dict(selected_epoch=t['best']['epoch'], parameters=t['parameters'],
            trainable_parameters=t['trainable_parameters'], seconds=t['finished'] - t['started'],
            runtime=t['runtime']) for name, t in training.items()},
        groups={group: dict(count=len(cases), models={key: {m: math.fsum(values[c][m] for c in cases) / len(cases)
            for m in METRICS} for key, values in models.items()}) for group, cases in groups.items()},
        limits='Original50 plus additional50; one seed; benchmark mAP selects earliest best. '
            '49 conditions reuse836 queries and are diagnostic groups, not independent replications. '
            'The12 disjoint-source conditions lack shared identity coordinates and remain unsolved. '
            'Frozen state ablation is not retraining ablation; no3dataset/multiseed goal-completion claim.')
    write_table(root / 'metrics49_all_states_and_baseline.csv', metrics_table)
    write_table(root / 'paired49_expert_utility_and_fair_controls.csv', comparisons)
    write_table(root / 'normal_per_query_paired_deltas.csv', normal_deltas)
    (root / 'analysis.json').write_bytes((json.dumps(summary, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(dict(status=summary['status'], normal_both_plus2=summary['normal_both_plus2']), ensure_ascii=False))


if __name__ == '__main__':
    main()
