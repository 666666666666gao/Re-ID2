"""Full official baseline-pair analysis from installed-GT CPU-audited exports."""
import argparse
import csv
import json
from pathlib import Path
from statistics import fmean


METRICS = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
COUNTS = {'MSVR310': (1032, 591, 1055), 'RGBNT201': (3951, 836, 836),
          'RGBNT100': (8675, 1715, 8575)}
SETS = ('RNT', 'R', 'N', 'T', 'RN', 'RT', 'NT')
GROUPS = {'same_availability': 7, 'overlap_mismatch': 30, 'source_disjoint': 12,
          'partial_query_full_gallery': 6, 'both_partial': 36}


def query_rows(path, count):
    with path.open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == count and all(r['valid'] == 'True' for r in rows)
    assert [int(r['query_index']) for r in rows] == list(range(count))
    return rows


def summarize(rows):
    return dict(mAP=100 * fmean(float(r['AP']) for r in rows),
        mINP=100 * fmean(float(r['INP']) for r in rows),
        **{f'Rank-{k}': 100 * fmean(int(r['first_match']) <= k for r in rows) for k in (1, 5, 10, 20)})


def analyze_dataset(root, dataset):
    train_count, query_count, gallery_count = COUNTS[dataset]
    references, paired_orders = {}, []
    for variant in ('demo', 'demo_shared'):
        name = dataset + '_' + variant + '_s42'
        run = root / 'training' / name
        frozen = root / 'frozen49' / name
        trained = json.loads((run / 'result.json').read_text(encoding='utf-8'))
        audit = json.loads((frozen / 'independent_cpu_audit.json').read_text(encoding='utf-8'))
        assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50
        assert trained['arguments']['dataset'] == dataset and trained['arguments']['variant'] == variant
        assert trained['training_heldout_identities'] == 0 and trained['training_coverage'] == dict(
            eligible=train_count, visited=train_count, unvisited=[])
        assert audit['status'] == 'PASS' and audit['cases'] == 49 and audit['perquery_count'] == 49 * query_count
        assert (trained['train_records'], trained['query_records'], trained['gallery_records']) == COUNTS[dataset]
        assert all(json.loads((root / phase / (name + '_exit.json')).read_text())['exit_code'] == 0
                   for phase in ('training', 'frozen49', 'audit'))
        assert trained['best']['epoch'] == audit['selected_epoch']
        assert all(abs(trained['full_metrics'][m] - audit['normal'][m]) < 1e-8 for m in METRICS)
        with (run / 'batch_orders.jsonl').open(encoding='utf-8') as handle:
            paired_orders.append([(r['epoch'], r['step'], r['names']) for r in map(json.loads, handle)])
        references[variant] = dict(selected_epoch=trained['best']['epoch'], normal=audit['normal'],
            conditions=audit['conditions'], optimizer_steps=trained['optimizer_steps'], amp_skipped_steps=trained['amp_skipped_steps'],
            training_coverage=trained['training_coverage'], descriptor_dim=trained['descriptor_dim'])
    assert paired_orders[0] == paired_orders[1]
    groups = {name: [] for name in GROUPS}
    conditions = {}
    for qset in SETS:
        for gset in SETS:
            condition = 'q_' + qset + '_g_' + gset
            before = query_rows(root / 'frozen49' / (dataset + '_demo_s42') / (condition + '.csv'), query_count)
            after = query_rows(root / 'frozen49' / (dataset + '_demo_shared_s42') / (condition + '.csv'), query_count)
            assert all(all(a[k] == b[k] for k in ('query_index', 'name', 'identity', 'camera', 'scene',
                'valid', 'relevant_gallery', 'kept_gallery')) for a, b in zip(before, after))
            values = {}
            for variant, rows in (('demo', before), ('demo_shared', after)):
                values[variant] = summarize(rows)
                assert all(abs(values[variant][m] - references[variant]['conditions'][condition][m]) < 1e-8 for m in METRICS)
            delta = {m: values['demo_shared'][m] - values['demo'][m] for m in METRICS}
            ap_delta = [float(b['AP']) - float(a['AP']) for a, b in zip(before, after)]
            row = dict(normal=qset == gset == 'RNT', delta_pp=delta,
                Rank1_harm_queries=sum(int(a['first_match']) == 1 and int(b['first_match']) != 1 for a, b in zip(before, after)),
                Rank1_rescue_queries=sum(int(a['first_match']) != 1 and int(b['first_match']) == 1 for a, b in zip(before, after)),
                AP_improved_queries=sum(d > 0 for d in ap_delta), AP_worsened_queries=sum(d < 0 for d in ap_delta),
                AP_unchanged_queries=sum(d == 0 for d in ap_delta), query_records=query_count,
                both_metrics_plus2=all(delta[m] >= 2 for m in ('mAP', 'Rank-1')))
            conditions[condition] = row
            group = 'same_availability' if qset == gset else 'overlap_mismatch' if set(qset) & set(gset) else 'source_disjoint'
            groups[group].append(row)
            if qset != 'RNT' and gset == 'RNT': groups['partial_query_full_gallery'].append(row)
            if qset != 'RNT' and gset != 'RNT': groups['both_partial'].append(row)
    assert all(len(groups[name]) == count for name, count in GROUPS.items())
    grouped = {name: dict(conditions=len(rows), equal_condition_mean_delta_pp={m: fmean(r['delta_pp'][m] for r in rows) for m in METRICS},
        Rank1_harm_repeated_query_rows=sum(r['Rank1_harm_queries'] for r in rows),
        Rank1_rescue_repeated_query_rows=sum(r['Rank1_rescue_queries'] for r in rows),
        both_metrics_plus2_conditions=sum(r['both_metrics_plus2'] for r in rows)) for name, rows in groups.items()}
    return dict(references=references, conditions=conditions, groups=grouped, normal=conditions['q_RNT_g_RNT'],
        both_metrics_plus2_conditions=sum(r['both_metrics_plus2'] for r in conditions.values()),
        all49_equal_condition_mean_delta_pp={m: fmean(r['delta_pp'][m] for r in conditions.values()) for m in METRICS},
        full_split_counts=dict(train=train_count, query=query_count, gallery=gallery_count),
        paired_identity_sampling_exact=True, paired_batches=len(paired_orders[0]),
        original_DeMo_plus2_thresholds={m: references['demo']['normal'][m] + 2 for m in ('mAP', 'Rank-1')})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    completed = json.loads((root / 'controller_result.json').read_text(encoding='utf-8'))
    assert completed['status'] == 'COMPLETE' and len(completed['runs']) == 6 and completed['paired_identity_sampling_exact']
    output = root / 'baseline_pair_analysis.json'
    assert not output.exists()
    datasets = {dataset: analyze_dataset(root, dataset) for dataset in COUNTS}
    result = dict(status='THREE_COMPLETE_OFFICIAL_BASELINE_PAIRS_ALL294_CASES_ANALYZED', datasets=datasets,
        model_metric_cases=294, paired_conditions=147, repeated_condition_query_rows=49 * 2 * sum(c[1] for c in COUNTS.values()),
        same_seed=42, training_heldout_identities=0, goal_complete=False,
        limits='Public-CLIP original DeMo versus shared-identity/availability/missing-augmentation DeMo, without new dual-axis experts. Baseline diagnostic only, not proof of expert cooperation. One seed, official benchmark-selected checkpoints. Conditions and overlapping groups reuse the same queries; condition means are not an official aggregate or independent observations. No NN rerun or new training.')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fields = ['dataset', 'condition', *METRICS, 'Rank1_harm_queries', 'Rank1_rescue_queries',
        'AP_improved_queries', 'AP_worsened_queries', 'AP_unchanged_queries', 'both_metrics_plus2']
    with (root / 'baseline_pair_condition_deltas.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for dataset, report in datasets.items():
            for condition, row in report['conditions'].items():
                writer.writerow(dict(dataset=dataset, condition=condition, **row['delta_pp'], **{k: row[k] for k in fields[8:]}))
    print('THREE_FULL_OFFICIAL_BASELINE_PAIRS_ANALYZED', flush=True)


if __name__ == '__main__': main()
