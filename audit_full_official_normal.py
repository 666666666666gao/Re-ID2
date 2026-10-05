"""Installed-filename CPU GT recount of a selected full-modality checkpoint."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_full_official49 import installed_rows, recount, verify, METRICS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    trained = json.loads((run / 'result.json').read_text())
    assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50 and trained['training_heldout_identities'] == 0
    assert json.loads((run.parent / (run.name + '_exit.json')).read_text())['exit_code'] == 0
    dataset = trained['arguments']['dataset']
    splits = {s: installed_rows(Path(trained['arguments']['data_root']), dataset, s) for s in ('train', 'query', 'gallery')}
    manifest = json.loads((run / 'official_split_manifest.json').read_text())
    assert all(splits[s] == manifest[s] for s in splits)
    train_ids = {row['identity'] for row in splits['train']}
    assert train_ids.isdisjoint({row['identity'] for row in splits['query'] + splits['gallery']})
    assert manifest['training_labels'] == {str(pid): i for i, pid in enumerate(sorted(train_ids))}
    assert all(len(splits[s]) == trained[s + '_records'] for s in splits)
    with (run / 'batch_orders.jsonl').open() as file:
        orders = [json.loads(line) for line in file]
    eligible = {r['name'] for r in splits['train']}
    visited = {name for order in orders for name in order['names']}
    assert visited == eligible and len(orders) == trained['steps']
    assert sum(row['optimizer_updated'] for row in orders) == trained['optimizer_steps'] == trained['steps']
    assert trained['amp_skipped_steps'] == 0
    with (run / 'epochs.csv').open() as file:
        epochs = list(csv.DictReader(file))
    assert [int(row['epoch']) for row in epochs] == list(range(1, 51))
    selected = max(epochs, key=lambda row: float(row['mAP']))
    assert int(selected['epoch']) == trained['best']['epoch']
    assert all(abs(float(selected[m]) - trained['full_metrics'][m]) < 1e-8 for m in METRICS)
    with np.load(run / 'best_official_arrays.npz') as arrays:
        for prefix in ('query', 'gallery'):
            for suffix, key in (('names', 'name'), ('ids', 'identity'), ('cameras', 'camera'), ('scenes', 'scene')):
                assert arrays[prefix + '_' + suffix].tolist() == [r[key] for r in splits[prefix]]
            features = arrays[prefix + '_features']
            assert features.shape == (len(splits[prefix]), 5120) and np.isfinite(features).all()
            assert np.max(np.abs(np.linalg.norm(features, axis=1) - 1)) < 1e-5
        rows = recount(arrays['distances'], splits['query'], splits['gallery'], dataset)
    reported = json.loads((run / 'best_per_query.json').read_text())
    metrics, error = verify(rows, reported, run / 'best_per_query.csv', len(splits['gallery']))
    assert all(abs(metrics[m] - trained['full_metrics'][m]) < 1e-8 for m in METRICS)
    (run / 'normal_cpu_audit.json').write_text(json.dumps(dict(status='PASS', dataset=dataset,
        selected_epoch=trained['best']['epoch'], installed_split_counts={k: len(v) for k,v in splits.items()},
        six_metrics=metrics, max_metric_error=error, CMC50_and_per_query_and_groups=True,
        full_training_coverage=True, new_optimizer_updates=0, missing_evaluation=False,
        limits='One selected full-modality condition, not49 missing conditions. Benchmark-selected checkpoint.'), indent=2) + '\n')
    print('FULL_NORMAL_INSTALLED_GT_AUDIT_PASS', dataset, flush=True)


if __name__ == '__main__':
    main()
