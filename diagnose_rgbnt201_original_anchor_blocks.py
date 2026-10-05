"""CPU readout of one completed R201A control against its original DeMo anchor."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from diagnose_rgbnt201_baseline_blocks import METRICS
from full_evaluation import distance, full_metrics
from official_training_data import full_records, metadata
from run_experiment import write_json


def main():
    parser = argparse.ArgumentParser()
    for key in ('run-dir', 'reference-run', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    records, arrays = {}, {}
    for label, folder, variant, dim in (
        ('original', args.reference_run, 'demo', 5120),
        ('control', args.run_dir, 'demo_shared', 5632),
    ):
        run = Path(folder)
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
            value = {key: saved[key].copy() for key in saved.files}
        assert all(np.array_equal(value[key], expected) for key, expected in installed.items())
        for side in ('query', 'gallery'):
            features = value[side + '_features']
            assert features.shape == (836, dim) and features.dtype == np.float32 and np.isfinite(features).all()
        records[label], arrays[label] = record, value
    anchor = records['control']['anchor']
    assert Path(anchor['run_dir']) == Path(args.reference_run)
    assert anchor['anchor_model'] == 'demo' and anchor['selected_epoch'] == records['original']['best']['epoch']
    assert anchor['selected_anchor_full_metrics'] == records['original']['full_metrics']
    assert all(np.array_equal(arrays['original'][key], arrays['control'][key]) for key in installed)
    ref, control = arrays['original'], arrays['control']
    systems = {
        'original_DeMo': (ref['query_features'], ref['gallery_features']),
        'control_fused': (control['query_features'], control['gallery_features']),
        'control_private_only': (control['query_features'][:, :5120], control['gallery_features'][:, :5120]),
        'control_public_only': (control['query_features'][:, 5120:], control['gallery_features'][:, 5120:]),
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    measurements, per_query, private_errors = {}, {}, {}
    for label, (query_features, gallery_features) in systems.items():
        q, g = torch.from_numpy(query_features.copy()), torch.from_numpy(gallery_features.copy())
        if label.endswith('_only'):
            q, g = F.normalize(q, dim=1), F.normalize(g, dim=1)
        if label == 'control_private_only':
            private_errors = {side: float((feature - torch.from_numpy(ref[side + '_features'])).abs().max())
                              for side, feature in (('query', q), ('gallery', g))}
        distances = distance(q, g)
        if label in ('original_DeMo', 'control_fused'):
            source = 'original' if label == 'original_DeMo' else 'control'
            assert np.array_equal(distances, arrays[source]['distances'])
        result = full_metrics(distances, control['query_ids'], control['gallery_ids'],
            control['query_cameras'], control['gallery_cameras'], control['query_names'],
            control['query_cameras'], control['query_scenes'], out / label)
        if label in ('original_DeMo', 'control_fused'):
            assert all(abs(result[key] - records[source]['best'][key]) < 1e-8 for key in METRICS)
        measurements[label] = result
        with (out / (label + '.csv')).open(encoding='utf-8', newline='') as stream:
            per_query[label] = list(csv.DictReader(stream))
    comparisons = {}
    for left, right in (('control_fused', 'original_DeMo'), ('control_private_only', 'original_DeMo'),
                        ('control_public_only', 'original_DeMo'), ('control_fused', 'control_private_only')):
        a, b = per_query[left], per_query[right]
        assert len(a) == len(b) == 836
        assert all(x['name'] == y['name'] and x['identity'] == y['identity'] and x['valid'] == y['valid'] for x, y in zip(a, b))
        valid = [(x, y) for x, y in zip(a, b) if x['valid'] == 'True']
        comparisons[left + '_minus_' + right] = dict(
            delta_pp={key: measurements[left][key] - measurements[right][key] for key in METRICS},
            Rank1_harm_queries=sum(int(x['Rank-1']) < int(y['Rank-1']) for x, y in valid),
            Rank1_rescue_queries=sum(int(x['Rank-1']) > int(y['Rank-1']) for x, y in valid),
            AP_improved_queries=sum(float(x['AP']) > float(y['AP']) for x, y in valid),
            AP_worsened_queries=sum(float(x['AP']) < float(y['AP']) for x, y in valid), valid_queries=len(valid))
    write_json(out / 'result.json', dict(status='COMPLETE_CPU_RGBNT201_ORIGINAL_ANCHOR_BLOCK_READOUT',
        dataset='RGBNT201', full_split_counts=dict(train=3951, query=836, gallery=836),
        training_heldout_identities=0, optimizer_updates=0, neural_inference_calls=0, device='CPU',
        checkpoints={key: value['best']['epoch'] for key, value in records.items()},
        freeze_identity_encoder=anchor['freeze_identity_encoder'], original_anchor=anchor,
        measurements=measurements, comparisons=comparisons, private_feature_max_abs_error=private_errors,
        scope='Four fixed selected descriptors, all official normal query/gallery and installed same-ID/same-camera exclusion. Existing full49 remains separate; no missing feature banks or metric tuning.',
        limits='Original50 plus additional50, one seed, benchmark-selected. Coordinate error includes strip-and-renormalize floating-point rounding; frozen-weight protection is verified by native/training checks. Block ablation is not retraining, causal proof, or unique dual-axis gain. No new weights/raw distances.'))
    print(json.dumps({key: {metric: value[metric] for metric in METRICS} for key, value in measurements.items()}), flush=True)


if __name__ == '__main__':
    main()
