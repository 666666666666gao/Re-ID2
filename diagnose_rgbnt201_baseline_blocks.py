"""CPU-only outlet diagnosis of the two completed full-official RGBNT201 baselines."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from full_evaluation import distance, full_metrics
from official_training_data import full_records, metadata
from run_experiment import write_json


METRICS = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root, out = Path(args.baseline_root), Path(args.output)
    records, arrays = {}, {}
    for variant, dim in (('demo', 5120), ('demo_shared', 5632)):
        run = root / 'training' / ('RGBNT201_' + variant + '_s42')
        record = json.loads((run / 'result.json').read_text())
        assert record['status'] == 'COMPLETE' and record['epochs'] == 50
        assert record['arguments']['dataset'] == 'RGBNT201' and record['arguments']['variant'] == variant
        assert record['arguments']['seed'] == 42 and record['training_heldout_identities'] == 0
        assert record['optimizer_steps'] == 2647 and record['amp_skipped_steps'] == 0
        assert record['descriptor_dim'] == dim
        assert json.loads((run.parent / (run.name + '_exit.json')).read_text())['exit_code'] == 0
        train, query, gallery, _, _, manifest = full_records(record['arguments']['data_root'], 'RGBNT201')
        assert (len(train), len(query), len(gallery)) == (3951, 836, 836)
        assert json.loads((run / 'official_split_manifest.json').read_text()) == manifest
        installed = metadata(query, 'query') | metadata(gallery, 'gallery')
        with np.load(run / 'best_official_arrays.npz') as saved:
            values = {key: saved[key].copy() for key in saved.files}
        assert all(np.array_equal(values[key], value) for key, value in installed.items())
        for side in ('query', 'gallery'):
            value = values[side + '_features']
            assert value.shape == (836, dim) and value.dtype == np.float32 and np.isfinite(value).all()
        records[variant], arrays[variant] = record, values
    assert all(np.array_equal(arrays['demo'][key], arrays['demo_shared'][key]) for key in installed)
    shared = arrays['demo_shared']
    systems = {
        'original_DeMo': (arrays['demo']['query_features'], arrays['demo']['gallery_features']),
        'shared_fused': (shared['query_features'], shared['gallery_features']),
        'shared_private_only': (shared['query_features'][:, :5120], shared['gallery_features'][:, :5120]),
        'shared_public_only': (shared['query_features'][:, 5120:], shared['gallery_features'][:, 5120:]),
    }
    out.mkdir(parents=True, exist_ok=False)
    measurements, per_query = {}, {}
    for label, (query_features, gallery_features) in systems.items():
        q, g = torch.from_numpy(query_features.copy()), torch.from_numpy(gallery_features.copy())
        if label.endswith('_only'):
            q, g = F.normalize(q, dim=1), F.normalize(g, dim=1)
        distances = distance(q, g)
        if label in ('original_DeMo', 'shared_fused'):
            variant = 'demo' if label == 'original_DeMo' else 'demo_shared'
            assert np.array_equal(distances, arrays[variant]['distances'])
        result = full_metrics(distances, shared['query_ids'], shared['gallery_ids'],
            shared['query_cameras'], shared['gallery_cameras'], shared['query_names'],
            shared['query_cameras'], shared['query_scenes'], out / label)
        if label in ('original_DeMo', 'shared_fused'):
            assert all(abs(result[key] - records[variant]['best'][key]) < 1e-8 for key in METRICS)
        measurements[label] = result
        with (out / (label + '.csv')).open(encoding='utf-8', newline='') as stream:
            per_query[label] = list(csv.DictReader(stream))
    comparisons = {}
    for left, right in (('shared_fused', 'original_DeMo'), ('shared_private_only', 'original_DeMo'),
                        ('shared_public_only', 'original_DeMo'), ('shared_fused', 'shared_private_only')):
        a, b = per_query[left], per_query[right]
        assert len(a) == len(b) == 836
        assert all(x['name'] == y['name'] and x['identity'] == y['identity'] and x['valid'] == y['valid'] for x, y in zip(a, b))
        valid = [(x, y) for x, y in zip(a, b) if x['valid'] == 'True']
        comparisons[left + '_minus_' + right] = {
            'delta_pp': {key: measurements[left][key] - measurements[right][key] for key in METRICS},
            'Rank1_harm_queries': sum(int(x['Rank-1']) < int(y['Rank-1']) for x, y in valid),
            'Rank1_rescue_queries': sum(int(x['Rank-1']) > int(y['Rank-1']) for x, y in valid),
            'AP_improved_queries': sum(float(x['AP']) > float(y['AP']) for x, y in valid),
            'AP_worsened_queries': sum(float(x['AP']) < float(y['AP']) for x, y in valid),
            'valid_queries': len(valid),
        }
    write_json(out / 'result.json', dict(status='COMPLETE_CPU_RGBNT201_SELECTED_BASELINE_BLOCK_DIAGNOSIS',
        dataset='RGBNT201', full_split_counts=dict(train=3951, query=836, gallery=836),
        training_heldout_identities=0, optimizer_updates=0, neural_inference_calls=0, device='CPU',
        checkpoints={key: value['best']['epoch'] for key, value in records.items()},
        measurements=measurements, comparisons=comparisons,
        scope='Four fixed descriptors, entire official normal query/gallery and installed same-ID/same-camera exclusion. Existing full49 results remain separate; no new missing feature banks or weight tuning.',
        limits='Single-seed benchmark-selected baselines. Private/public block ablation diagnoses the selected shared checkpoint, not retraining, causal attribution, or a new best method. No new weights or raw distance files.'))
    print(json.dumps({key: {metric: value[metric] for metric in METRICS} for key, value in measurements.items()}), flush=True)


if __name__ == '__main__':
    main()
