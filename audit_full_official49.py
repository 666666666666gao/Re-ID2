"""Independent CPU recount using filenames in the entire installed official splits."""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

from audit_shared_identity_states import METRICS, summarize


def installed_rows(root, dataset, split):
    directory = {
        'RGBNT201': {'train': 'RGBNT201/train_171/RGB', 'query': 'RGBNT201/test/RGB', 'gallery': 'RGBNT201/test/RGB'},
        'RGBNT100': {'train': 'RGBNT100/rgbir/bounding_box_train', 'query': 'RGBNT100/rgbir/query', 'gallery': 'RGBNT100/rgbir/bounding_box_test'},
        'MSVR310': {'train': 'MSVR310/bounding_box_train', 'query': 'MSVR310/query3', 'gallery': 'MSVR310/bounding_box_test'}}[dataset][split]
    folder = root / directory
    paths = sorted(folder.glob('*/vis/*') if dataset == 'MSVR310' else folder.glob('*.jpg'))
    rows = []
    for path in paths:
        name = path.name
        if dataset == 'MSVR310':
            identity, camera, scene = int(name[:4]), int(name[11]), int(name[6:9])
            assert (path.parent.parent / 'ni' / name).is_file() and (path.parent.parent / 'th' / name).is_file()
        elif dataset == 'RGBNT201':
            identity, camera, scene = int(name.split('_')[0][:6]), int(name.split('_')[1][3]) - 1, -1
            assert (path.parent.parent / 'NI' / name).is_file() and (path.parent.parent / 'TI' / name).is_file()
        else:
            identity, camera = map(int, re.search(r'([-\d]+)_c([-\d]+)', name).groups())
            camera, scene = camera - 1, -1
        rows.append(dict(name=name, identity=identity, camera=camera, scene=scene))
    assert rows
    return rows


def recount(distances, queries, gallery, dataset):
    ids = np.asarray([r['identity'] for r in gallery])
    selector = 'scene' if dataset == 'MSVR310' else 'camera'
    exclusion = np.asarray([r[selector] for r in gallery])
    assert distances.shape == (len(queries), len(gallery)) and np.isfinite(distances).all()
    rows = []
    gallery_indices = np.arange(len(gallery))
    for ordinal, query in enumerate(queries):
        order = np.lexsort((gallery_indices, distances[ordinal]))
        keep = order[~((ids[order] == query['identity']) & (exclusion[order] == query[selector]))]
        ranks = np.flatnonzero(ids[keep] == query['identity']) + 1
        assert len(ranks) > 0
        rows.append(dict(query_index=ordinal, **query, valid=True, relevant_gallery=len(ranks), kept_gallery=len(keep),
            AP=float(np.mean(np.arange(1, len(ranks) + 1) / ranks)), INP=float(len(ranks) / ranks[-1]),
            first_match=int(ranks[0]), last_match=int(ranks[-1]), **{f'Rank-{k}': int(ranks[0] <= k) for k in (1, 5, 10, 20)}))
    return rows


def verify(rows, reported, csv_path, gallery_count):
    values = summarize(rows)
    error = max(abs(values[m] - reported[m]) for m in METRICS)
    assert error < 1e-8
    assert reported['query_count'] == reported['valid_queries'] == len(rows)
    assert reported['invalid_queries'] == 0 and reported['gallery_count'] == gallery_count
    cmc = [100 * float(np.mean([r['first_match'] <= k for r in rows])) for k in range(1, 51)]
    assert np.max(np.abs(np.asarray(cmc) - reported['CMC_1_to_50'])) < 1e-8
    with csv_path.open() as handle:
        exported = list(csv.DictReader(handle))
    assert len(exported) == len(rows)
    for actual, expected in zip(exported, rows):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            if key in ('AP', 'INP'):
                assert abs(float(actual[key]) - value) < 1e-12
            else:
                assert actual[key] == str(value), (key, actual[key], value)
    for key in ('camera', 'scene', 'identity'):
        groups = reported['groups'][key]
        assert [g['value'] for g in groups] == sorted({r[key] for r in rows})
        for group in groups:
            members = [r for r in rows if r[key] == group['value']]
            assert len(members) == group['queries']
            assert all(abs(summarize(members)[m] - group[m]) < 1e-8 for m in METRICS)
    return values, error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--evaluation', required=True)
    args = parser.parse_args()
    run, evaluation = Path(args.run_dir), Path(args.evaluation)
    trained = json.loads((run / 'result.json').read_text())
    assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50 and trained['training_heldout_identities'] == 0
    arguments = trained['arguments']
    dataset, root = arguments['dataset'], Path(arguments['data_root'])
    splits = {s: installed_rows(root, dataset, s) for s in ('train', 'query', 'gallery')}
    manifest = json.loads((run / 'official_split_manifest.json').read_text())
    assert all(splits[s] == manifest[s] for s in splits)
    train_ids = {r['identity'] for r in splits['train']}
    assert train_ids.isdisjoint({r['identity'] for r in splits['query'] + splits['gallery']})
    assert manifest['training_labels'] == {str(pid): i for i, pid in enumerate(sorted(train_ids))}
    assert len(splits['train']) == trained['train_records']
    assert len(splits['query']) == trained['query_records'] and len(splits['gallery']) == trained['gallery_records']
    eligible = {r['name'] for r in splits['train']}
    with (run / 'batch_orders.jsonl').open() as handle:
        orders = [json.loads(line) for line in handle]
    visited = {n for order in orders for n in order['names']}
    assert visited <= eligible
    assert len(orders) == trained['steps'] and sum(o['optimizer_updated'] for o in orders) == trained['optimizer_steps']
    assert trained['training_coverage'] == dict(eligible=len(eligible), visited=len(visited), unvisited=sorted(eligible - visited))
    with (run / 'epochs.csv').open() as handle:
        epochs = list(csv.DictReader(handle))
    assert [int(e['epoch']) for e in epochs] == list(range(1, 51))
    selected = max(epochs, key=lambda e: float(e['mAP']))
    assert int(selected['epoch']) == trained['best']['epoch']
    assert all(abs(float(selected[m]) - trained['full_metrics'][m]) < 1e-8 for m in METRICS)
    reported = json.loads((evaluation / 'result.json').read_text())
    assert reported['status'] == 'COMPLETE' and reported['training_heldout_identities'] == reported['optimizer_updates'] == 0
    assert len(reported['measurements']) == 49 and reported['normal_feature_max_error'] == 0 and reported['state_tensor_versions_unchanged']
    values, errors = {}, []
    for condition, summary in reported['measurements'].items():
        with np.load(evaluation / (condition + '.npz')) as arrays:
            rows = recount(arrays['distances'], splits['query'], splits['gallery'], dataset)
        actual, error = verify(rows, summary, evaluation / (condition + '.csv'), len(splits['gallery']))
        values[condition] = actual
        errors.append(error)
    normal = values['q_RNT_g_RNT']
    assert all(abs(normal[m] - trained['full_metrics'][m]) < 1e-8 for m in METRICS)
    result = dict(status='PASS', dataset=dataset, variant=arguments['variant'], cases=49,
        perquery_count=49 * len(splits['query']), train_records=len(splits['train']), query_records=len(splits['query']),
        gallery_records=len(splits['gallery']), training_heldout_identities=0, training_coverage=trained['training_coverage'],
        selected_epoch=trained['best']['epoch'], max_sixmetric_error_pp=max(errors), conditions=values, normal=normal,
        equal_condition_mean={m: float(np.mean([v[m] for v in values.values()])) for m in METRICS},
        limits='Equal-condition means are diagnostics, not an official aggregate. Benchmark checkpoint selection, one seed. Original demo missing-input protocol is reported separately from corrected-source shared demo.')
    (evaluation / 'independent_cpu_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print('FULL_OFFICIAL_ALL49_GT_CPU_PASS', dataset, arguments['variant'], flush=True)


if __name__ == '__main__':
    main()
