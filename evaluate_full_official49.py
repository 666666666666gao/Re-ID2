"""Frozen best checkpoint: complete official query/gallery in all 49 conditions."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from full_evaluation import distance, full_metrics
from missing_evaluation import extract_missing, verify_original_mask, METRICS
from official_training_data import full_records, metadata
from run_experiment import configuration, write_json
from run_full_official_experiment import build


SETS = {'RNT': (), 'R': ('NI', 'TI'), 'N': ('RGB', 'TI'), 'T': ('RGB', 'NI'),
        'RN': ('TI',), 'RT': ('NI',), 'NT': ('RGB',)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run, out = Path(args.run_dir), Path(args.output)
    result = json.loads((run / 'result.json').read_text())
    assert result['status'] == 'COMPLETE' and result['epochs'] == 50 and result['training_heldout_identities'] == 0
    assert json.loads((run.parent / (run.name + '_exit.json')).read_text())['exit_code'] == 0
    arguments = argparse.Namespace(**result['arguments'])
    cfg = configuration(arguments)
    train, query, gallery, classes, cameras, manifest = full_records(arguments.data_root, arguments.dataset)
    assert json.loads((run / 'official_split_manifest.json').read_text()) == manifest
    torch.set_num_threads(4)
    model = build(arguments, cfg, classes, cameras)
    model.load_state_dict(torch.load(run / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    model.eval()
    versions = {n: v._version for n, v in model.state_dict().items()}
    out.mkdir(exist_ok=False)
    started = time.time()
    original_mask = verify_original_mask(model, query, cfg, arguments.seed) if arguments.variant == 'demo' else None
    arrays = metadata(query, 'query') | metadata(gallery, 'gallery')
    saved = np.load(run / 'best_official_arrays.npz')
    assert all(np.array_equal(saved[k], v) for k, v in arrays.items())
    banks, runtime = {}, {}
    for available, missing in SETS.items():
        feature, performance = extract_missing(model, query + gallery, cfg, arguments.seed, missing)
        assert feature.shape == (len(query) + len(gallery), result['descriptor_dim']) and torch.isfinite(feature).all()
        banks[available] = (feature[:len(query)], feature[len(query):])
        runtime[available] = performance
    assert np.array_equal(banks['RNT'][0].numpy(), saved['query_features'])
    assert np.array_equal(banks['RNT'][1].numpy(), saved['gallery_features'])
    selector = 'scenes' if arguments.dataset == 'MSVR310' else 'cameras'
    measurements = {}
    for qs, (q, _) in banks.items():
        for gs, (_, g) in banks.items():
            condition = 'q_' + qs + '_g_' + gs
            distances = distance(q, g)
            if qs == gs == 'RNT':
                assert np.array_equal(distances, saved['distances'])
            values = full_metrics(distances, arrays['query_ids'], arrays['gallery_ids'],
                arrays['query_' + selector], arrays['gallery_' + selector], arrays['query_names'],
                arrays['query_cameras'], arrays['query_scenes'], out / condition)
            np.savez_compressed(out / (condition + '.npz'), distances=distances)
            measurements[condition] = values
            print('OFFICIAL_49', arguments.dataset, arguments.variant, condition,
                  json.dumps({m: values[m] for m in METRICS}), flush=True)
    assert len(measurements) == 49
    assert versions == {n: v._version for n, v in model.state_dict().items()}
    write_json(out / 'result.json', dict(status='COMPLETE', dataset=arguments.dataset, variant=arguments.variant,
        model_arguments=result['arguments'],
        seed=arguments.seed, selected_epoch=result['best']['epoch'], measurements=measurements, runtime=runtime,
        query_records=len(query), gallery_records=len(gallery), train_records=len(train),
        training_heldout_identities=0, optimizer_updates=0, normal_feature_max_error=0,
        state_tensor_versions_unchanged=True, original_mask=original_mask, seconds=time.time() - started,
        selection=result['checkpoint_rule'], protocol='All official queries and galleries, all 7x7 nonempty retained modality combinations; seven independent descriptor banks, frozen full-benchmark mAP-best, original GT junk exclusion',
        limits='Official benchmark used for checkpoint selection; single seed. Architecture, source handling and training revision are specified by the saved training record and model arguments; no training occurs in this frozen evaluation.'))


if __name__ == '__main__':
    main()
