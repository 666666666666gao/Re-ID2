"""Full RGBNT100 B128/P8K16 GT audit with actual update and AMP-skip accounting."""
import argparse
import csv
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np

from audit_full_official49 import installed_rows, recount, verify, METRICS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    run = Path(parser.parse_args().run_dir)
    trained = json.loads((run/'result.json').read_text())
    assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50
    assert trained['training_heldout_identities'] == 0
    assert json.loads((run.parent/(run.name+'_exit.json')).read_text())['exit_code'] == 0
    dataset = trained['arguments']['dataset']
    assert dataset == 'RGBNT100' and trained['arguments']['training_batch'] == 128
    heads, dimension, expected_counts = trained['arguments']['atmoe_heads'], 5120, (8675,1715,8575)
    assert heads in (8,32) and trained['descriptor_dim'] == dimension
    splits = {s: installed_rows(Path(trained['arguments']['data_root']), dataset, s) for s in ('train','query','gallery')}
    manifest = json.loads((run/'official_split_manifest.json').read_text())
    assert all(splits[s] == manifest[s] for s in splits)
    assert tuple(len(splits[s]) for s in ('train','query','gallery')) == expected_counts
    assert all(len(splits[s]) == trained[s+'_records'] for s in splits)
    ids = {r['identity'] for r in splits['train']}
    assert ids.isdisjoint({r['identity'] for r in splits['query']+splits['gallery']})
    assert manifest['training_labels'] == {str(pid): i for i,pid in enumerate(sorted(ids))}
    orders = [json.loads(line) for line in (run/'batch_orders.jsonl').read_text().splitlines()]
    attempts = len(orders)
    updates = sum(row['optimizer_updated'] for row in orders)
    skips = attempts-updates
    assert attempts == trained['steps'] > 0
    identity_by_name = {row['name']:row['identity'] for row in splits['train']}
    assert all(len(row['names']) == 128 and sorted(Counter(identity_by_name[name] for name in row['names']).values()) == [16]*8 for row in orders)
    assert updates == trained['optimizer_steps'] and 0 < updates <= attempts
    assert skips == trained['amp_skipped_steps']
    assert [row['step'] for row in orders] == list(range(1,attempts+1))
    assert all(isinstance(row['optimizer_updated'],bool) for row in orders)
    assert all(math.isfinite(row['loss']) and math.isfinite(row['amp_scale']) and row['amp_scale']>0 for row in orders)
    eligible = {r['name'] for r in splits['train']}
    assert {n for row in orders for n in row['names']} == eligible
    assert trained['training_coverage'] == dict(eligible=len(eligible),visited=len(eligible),unvisited=[])
    updated_records = {n for row in orders if row['optimizer_updated'] for n in row['names']}
    epochs = list(csv.DictReader((run/'epochs.csv').open(encoding='utf-8',newline='')))
    assert [int(row['epoch']) for row in epochs] == list(range(1,51))
    visited = set()
    cumulative_attempts = cumulative_updates = 0
    for epoch,row in enumerate(epochs,1):
        batches = [batch for batch in orders if batch['epoch']==epoch]
        assert batches
        cumulative_attempts += len(batches)
        cumulative_updates += sum(batch['optimizer_updated'] for batch in batches)
        current = {n for batch in batches for n in batch['names']}
        visited.update(current)
        assert int(row['steps']) == cumulative_attempts
        assert int(row['optimizer_steps']) == cumulative_updates
        assert int(row['amp_skipped_steps']) == cumulative_attempts-cumulative_updates
        assert int(row['epoch_unique_train_records']) == len(current)
        assert int(row['cumulative_unique_train_records']) == len(visited)
        assert math.isfinite(float(row['loss'])) and math.isfinite(float(row['amp_scale'])) and float(row['amp_scale'])>0
        assert all(math.isfinite(float(row[key])) for key in METRICS)
    assert cumulative_attempts == attempts and cumulative_updates == updates
    selected = max(epochs,key=lambda row: float(row['mAP']))
    assert int(selected['epoch']) == trained['best']['epoch']
    assert all(abs(float(selected[k])-trained['full_metrics'][k])<1e-8 for k in METRICS)
    with np.load(run/'best_official_arrays.npz') as arrays:
        for side in ('query','gallery'):
            for suffix,key in (('names','name'),('ids','identity'),('cameras','camera'),('scenes','scene')):
                assert arrays[side+'_'+suffix].tolist() == [r[key] for r in splits[side]]
            features = arrays[side+'_features']
            assert features.shape == (len(splits[side]),dimension) and np.isfinite(features).all()
            assert np.max(np.abs(np.linalg.norm(features,axis=1)-1))<1e-5
            if dataset == 'MSVR310':
                assert np.max(np.abs((features[:,:5120]**2).sum(1)-.75))<1e-5
                assert np.max(np.abs((features[:,5120:]**2).sum(1)-.25))<1e-5
        rows = recount(arrays['distances'],splits['query'],splits['gallery'],dataset)
    reported = json.loads((run/'best_per_query.json').read_text())
    metrics,error = verify(rows,reported,run/'best_per_query.csv',len(splits['gallery']))
    assert all(abs(metrics[k]-trained['full_metrics'][k])<1e-8 for k in METRICS)
    output = dict(status='PASS',dataset=dataset,descriptor_dim=dimension,atmoe_heads=heads,batch_size=128,P=8,K=16,
        selected_epoch=trained['best']['epoch'],installed_split_counts={s:len(v) for s,v in splits.items()},
        six_metrics=metrics,max_metric_error=error,CMC50_and_per_query_and_groups=True,
        attempts=attempts,successful_optimizer_updates=updates,amp_skips=skips,
        all_epoch_cumulative_accounting=True,full_training_coverage=True,new_optimizer_updates=0,
        forward_record_coverage=len(eligible),successful_update_record_coverage=len(updated_records),
        records_without_successful_update=sorted(eligible-updated_records),
        missing_evaluation=False,
        limits='Same original dynamic GradScaler512 and50epochs; skipped batches are not optimizer updates. Nonzero skips are disclosed, not a proof of global numeric health or equal update budget. Official benchmark selected the checkpoint.')
    (run/'normal_cpu_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print('VEHICLE_HEAD_FULL_NORMAL_GT_AND_ACTUAL_UPDATE_AUDIT_PASS',dataset,flush=True)


if __name__ == '__main__':
    main()
