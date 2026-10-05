"""CPU-only reduction of all three completed normal-input control pairs."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path

METRICS = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
COUNTS = {'RGBNT201': (3951, 836, 836, 2647), 'RGBNT100': (8675, 1715, 8575, 6357), 'MSVR310': (1032, 591, 1055, 705)}
VARIANTS = ('frequency_shared', 'axis_shared')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def six(rows):
    return dict(mAP=100 * math.fsum(float(row['AP']) for row in rows) / len(rows),
        mINP=100 * math.fsum(float(row['INP']) for row in rows) / len(rows),
        **{key: 100 * sum(int(row[key]) for row in rows) / len(rows) for key in METRICS[2:]})


def query_rows(folder, dataset):
    summary = load(folder / 'best_per_query.json')
    with (folder / 'best_per_query.csv').open(encoding='utf-8', newline='') as file:
        rows = list(csv.DictReader(file))
    _, queries, gallery, _ = COUNTS[dataset]
    assert len(rows) == queries and all(row['valid'] == 'True' for row in rows)
    assert [int(row['query_index']) for row in rows] == list(range(queries))
    assert summary['query_count'] == summary['valid_queries'] == queries and summary['invalid_queries'] == 0
    assert summary['gallery_count'] == gallery
    assert all(abs(six(rows)[key] - summary[key]) < 1e-8 for key in METRICS)
    assert len(summary['CMC_1_to_50']) == 50
    assert all(abs(100 * sum(int(row['first_match']) <= rank for row in rows) / queries - summary['CMC_1_to_50'][rank - 1]) < 1e-8 for rank in range(1, 51))
    for axis in ('identity', 'camera', 'scene'):
        assert {int(row[axis]) for row in rows} == {group['value'] for group in summary['groups'][axis]}
        for group in summary['groups'][axis]:
            members = [row for row in rows if int(row[axis]) == group['value']]
            assert len(members) == group['queries']
            assert all(abs(six(members)[key] - group[key]) < 1e-8 for key in METRICS)
    return rows, summary


def table(path, rows):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--baselines', required=True)
    args = parser.parse_args()
    root, baseline = Path(args.root), Path(args.baselines)
    controller = load(root / 'controller_result.json')
    assert controller['status'] == 'COMPLETE' and controller['new_models'] == 4
    assert controller['additional_epochs'] == 200 and controller['successful_updates'] == 14124
    assert controller['normal_datasets_completed'] == 3 and controller['normal_archives_local_verified'] == 4
    assert controller['paired_sampling_exact'] and controller['original_missing_controller_resumed']
    normal, groups, pairs, differences, training = [], [], [], [], {}
    for dataset, (train_count, queries, gallery, steps) in COUNTS.items():
        folders = {'original_DeMo': baseline / (dataset + '_demo_s42')}
        folders.update({variant: root / 'training' / (dataset + '_identity_' + variant + '_narrow_s42') for variant in VARIANTS})
        reads = {}
        for model, folder in folders.items():
            run = load(folder / 'result.json')
            assert run['status'] == 'COMPLETE' and run['epochs'] == 50
            assert run['optimizer_steps'] == run['steps'] == steps and run['amp_skipped_steps'] == 0
            assert run['training_heldout_identities'] == 0 and run['descriptor_dim'] == 5120
            assert (run['train_records'], run['query_records'], run['gallery_records']) == (train_count, queries, gallery)
            assert run['training_coverage'] == dict(eligible=train_count, visited=train_count, unvisited=[])
            rows, summary = query_rows(folder, dataset)
            assert all(abs(summary[key] - run['full_metrics'][key]) < 1e-8 for key in METRICS)
            with (folder / 'epochs.csv').open(encoding='utf-8', newline='') as file:
                epochs = list(csv.DictReader(file))
            assert [int(row['epoch']) for row in epochs] == list(range(1, 51))
            selected = max(epochs, key=lambda row: float(row['mAP']))
            assert int(selected['epoch']) == run['best']['epoch']
            assert all(abs(float(selected[key]) - summary[key]) < 1e-8 for key in METRICS)
            if model != 'original_DeMo':
                archive = load(folder / 'normal_local_archive.json')
                assert archive['status'] == 'NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
                if dataset == 'RGBNT201':
                    audit = load(root / 'rgbnt201_closed_GT' / folder.name / 'independent_fourstate_cpu_audit.json')
                    assert audit['status'] == 'PASS' and audit['state_cases'] == 196
                    frozen = load(root / 'rgbnt201_closed_GT' / folder.name / 'result.json')
                    assert all(abs(frozen['state_measurements']['q_RNT_g_RNT']['11'][key] - summary[key]) < 1e-8 for key in METRICS)
                else:
                    audit = load(folder / 'normal_cpu_audit.json')
                    assert audit['status'] == 'PASS' and audit['full_training_coverage'] and audit['max_metric_error'] < 1e-8
            training[dataset + '/' + model] = dict(selected_epoch=run['best']['epoch'], additional_epochs=0 if model == 'original_DeMo' else 50,
                stage_updates=steps, total_training_budget_epochs=50 if model == 'original_DeMo' else 100,
                parameters=run['parameters'], trainable_parameters=run['trainable_parameters'], descriptor_dim=5120,
                final_epoch_metrics={key: float(epochs[-1][key]) for key in METRICS}, peak_memory=run['peak_memory'], runtime=run['runtime'])
            normal.append(dict(dataset=dataset, model=model, selected_epoch=run['best']['epoch'], **{key: summary[key] for key in METRICS}))
            for axis, values in summary['groups'].items():
                groups.extend(dict(dataset=dataset, model=model, grouping=axis, **value) for value in values)
            reads[model] = rows, summary
        a, f = (training[dataset + '/' + variant] for variant in VARIANTS)
        assert (a['parameters'], a['trainable_parameters']) == (f['parameters'], f['trainable_parameters'])
        for improved, reference in (('axis_shared', 'original_DeMo'), ('frequency_shared', 'original_DeMo'), ('axis_shared', 'frequency_shared')):
            new_rows, new_metrics = reads[improved]
            old_rows, old_metrics = reads[reference]
            assert all(tuple(left[key] for key in ('name', 'identity', 'camera', 'scene')) == tuple(right[key] for key in ('name', 'identity', 'camera', 'scene')) for left, right in zip(new_rows, old_rows))
            delta = {key: new_metrics[key] - old_metrics[key] for key in METRICS}
            rescue = sum(int(a['Rank-1']) == 1 and int(b['Rank-1']) == 0 for a, b in zip(new_rows, old_rows))
            harm = sum(int(a['Rank-1']) == 0 and int(b['Rank-1']) == 1 for a, b in zip(new_rows, old_rows))
            differences.append(dict(dataset=dataset, improved=improved, reference=reference, **delta,
                both_plus2=delta['mAP'] >= 2 and delta['Rank-1'] >= 2, rank1_rescued=rescue, rank1_harmed=harm,
                AP_improved=sum(float(a['AP']) > float(b['AP']) for a, b in zip(new_rows, old_rows)),
                AP_worsened=sum(float(a['AP']) < float(b['AP']) for a, b in zip(new_rows, old_rows))))
            for a, b in zip(new_rows, old_rows):
                pairs.append(dict(dataset=dataset, improved=improved, reference=reference,
                    **{key: a[key] for key in ('query_index', 'name', 'identity', 'camera', 'scene')},
                    delta_AP_pp=100 * (float(a['AP']) - float(b['AP'])), delta_INP_pp=100 * (float(a['INP']) - float(b['INP'])),
                    delta_rank1=int(a['Rank-1']) - int(b['Rank-1'])))
    assert len(normal) == len(differences) == 9 and len(pairs) == 9426
    out = root / 'normal_analysis'
    out.mkdir(exist_ok=False)
    for name, values in (('six_metrics', normal), ('group_metrics', groups), ('paired_query_changes', pairs), ('comparisons', differences)):
        table(out / (name + '.csv'), values)
    summary = dict(status='ACTUAL_THREE_NORMAL_DATASETS_CPU_ANALYSIS_COMPLETE', completed_at=datetime.now().isoformat(timespec='seconds'),
        model_results=9, paired_query_rows=len(pairs), full_metrics_CMC50_and_groups_reduced=True,
        comparisons=differences, training=training, seed=42,
        all_three_original_both_plus2=all(row['both_plus2'] for row in differences if row['improved'] == 'axis_shared' and row['reference'] == 'original_DeMo'),
        missing_evaluation_not_proven_here=True, multi_seed_confirmation_not_proven_here=True, new_neural_calls=0, new_optimizer_updates=0,
        limits='Official benchmark-selected best, not an untouched final test. Original50 versus original50+new50; ordinary frequency has matched new50 budget. Six normal controlled runs do not establish missing-modal or multiseed success. CPU CSV reduction checks prior installed-GT audits; it is not a new neural inference or an independent GT rerun.')
    (out / 'result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('THREE_NORMAL_CPU_ANALYSIS_COMPLETE', json.dumps(dict(all_three_both_plus2=summary['all_three_original_both_plus2'], model_results=9, paired_query_rows=len(pairs))), flush=True)


if __name__ == '__main__':
    main()
