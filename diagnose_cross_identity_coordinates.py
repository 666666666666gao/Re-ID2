"""Frozen M2 cross-coordinate retrieval; no training or descriptor changes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import diagnose_trained_outlet_utility as probe
from diagnose_shared_identity_states import SETS
from experiment_data import split_records
from full_evaluation import distance, full_metrics
from missing_evaluation import METRICS
from run_experiment import configuration, write_json


PAIRS = {
    'common_common': ('base_common', 'base_common'),
    'F_pre_F_pre': ('F_pre', 'F_pre'),
    'F_post_F_post': ('F_post', 'F_post'),
    'F_pre_common': ('F_pre', 'base_common'),
    'common_F_pre': ('base_common', 'F_pre'),
    'F_post_common': ('F_post', 'base_common'),
    'common_F_post': ('base_common', 'F_post'),
}


def main():
    parser = argparse.ArgumentParser()
    for name in ('run-dir', 'previous-utility', 'output', 'data-root', 'pretrained'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run, previous, output = map(Path, (args.run_dir, args.previous_utility, args.output))
    terminal = json.loads((run / 'result.json').read_text())
    assert terminal['status'] == 'COMPLETE' and terminal['epochs'] == 50
    arguments = argparse.Namespace(**terminal['arguments'])
    assert arguments.variant in ('axis_shared', 'frequency_shared', 'twins_shared')
    assert arguments.dataset == 'MSVR310' and arguments.pooling == 'original_mean'
    arguments.data_root, arguments.pretrained = args.data_root, args.pretrained
    exit_file = run.parent / (run.name + '_exit.json')
    assert json.loads(exit_file.read_text())['exit_code'] == 0
    prior = json.loads((previous / 'result.json').read_text())
    assert prior['status'] == 'COMPLETE' and prior['metric_count'] == 392
    assert prior['variant'] == arguments.variant and prior['selected_epoch'] == terminal['best']['epoch']
    protected = (run / 'best.pth', run / 'best_dev_arrays.npz', run / 'result.json', exit_file, previous / 'result.json')
    inputs = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    cfg = configuration(arguments)
    _, dev, query, classes, cameras = split_records(args.data_root, 'MSVR310')
    with np.load(run / 'best_dev_arrays.npz') as saved:
        assert np.array_equal(saved['query_indices'], query)
        for key, values in (('ids', [r[1] for r in dev]), ('cameras', [r[2] for r in dev]),
                            ('scenes', [r[3] for r in dev]),
                            ('names', [Path(r[0] if isinstance(r[0], str) else r[0][0]).name for r in dev])):
            assert np.array_equal(saved[key], values)
        saved = {key: saved[key] for key in saved.files}
    model = probe.build_trained(arguments, cfg, classes, cameras)
    model.load_state_dict(torch.load(run / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    model.eval()
    assert all(not m.training for m in model.modules())
    versions = {name: value._version for name, value in model.state_dict().items()}
    handles = probe.attach_probe(model)
    output.mkdir(exist_ok=False)
    started = time.time()
    records = dev[:64] if args.smoke else dev
    banks, scalar_banks = {}, {}
    for availability, retained in SETS.items():
        banks[availability], scalar_banks[availability] = probe.framework.extract(model, records, cfg, arguments.seed, retained)
    for handle in handles:
        handle.remove()
    assert versions == {name: value._version for name, value in model.state_dict().items()}
    parity = float(np.abs(banks['RNT']['deployed'].numpy() - saved['features'][:len(records)]).max())
    assert parity == 0
    for availability in SETS:
        for stage in ('base_common', 'F_pre', 'F_post'):
            feature = banks[availability][stage]
            assert feature.shape == (len(records), 512) and torch.isfinite(feature).all()
            assert torch.allclose(feature.norm(dim=1), torch.ones(len(records)), atol=1e-6, rtol=0)
    if args.smoke:
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == inputs[str(p)] for p in protected)
        write_json(output / 'smoke.json', dict(status='PASS', variant=arguments.variant,
            normal_feature_max_error=parity, source_unchanged=True, state_tensor_versions_unchanged=True,
            original_fuse_reconstructed_exact=True, seven_banks=True, records=len(records),
            optimizer_updates=0, new_weights=0, pairs=PAIRS))
        return
    query = np.asarray(query)
    ids, cams, scenes, names = (saved[key] for key in ('ids', 'cameras', 'scenes', 'names'))
    measurements, raw = {}, {}
    for qset in SETS:
        for gset in SETS:
            condition = 'q_' + qset + '_g_' + gset
            folder = output / condition
            folder.mkdir()
            values = {}
            for pair, (qstage, gstage) in PAIRS.items():
                distances = distance(banks[qset][qstage][query], banks[gset][gstage])
                raw[condition + '_' + pair] = distances
                values[pair] = full_metrics(distances, ids[query], ids, scenes[query], scenes,
                    names[query], cams[query], scenes[query], folder / pair)
                if qstage == gstage:
                    assert all(values[pair][m] == prior['measurements'][condition][qstage][m] for m in METRICS)
                    with (previous / condition / (qstage + '.csv')).open() as f:
                        prior_rows = list(csv.DictReader(f))
                    with (folder / (pair + '.csv')).open() as f:
                        assert list(csv.DictReader(f)) == prior_rows
            measurements[condition] = values
            print('CROSS_COORDINATE_CONDITION', condition, flush=True)
    metadata = {key: saved[key] for key in ('query_indices', 'ids', 'cameras', 'scenes', 'names')}
    np.savez_compressed(output / 'raw_distances.npz', **raw, **metadata)
    features = {availability + '_' + stage: banks[availability][stage].numpy()
                for availability in SETS for stage in ('base_common', 'F_pre', 'F_post')}
    np.savez_compressed(output / 'normalized_features.npz', **features, **metadata)
    for availability, values in scalar_banks.items():
        with (output / ('scale_' + availability + '.csv')).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['row', 'availability', *values])
            writer.writeheader()
            writer.writerows(dict(row=i, availability=availability, **{key: value[i] for key, value in values.items()}) for i in range(len(dev)))
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == inputs[str(p)] for p in protected)
    write_json(output / 'result.json', dict(status='COMPLETE', variant=arguments.variant, dataset='MSVR310',
        selected_epoch=terminal['best']['epoch'], pairs=PAIRS, measurements=measurements,
        metric_count=343, records=len(dev), normal_feature_max_error=parity,
        original_fuse_reconstructed_exact=True, previous_same_coordinate_metrics_perquery_exact=True,
        state_tensor_versions_unchanged=True, original_inputs=inputs, optimizer_updates=0, new_weights=0,
        official_test_uses=0, seconds=time.time() - started,
        protocol='Fixed original M2 best. Independently normalized512D actual base_common/F_pre/F_post. Four cross-coordinate directions plus three exact within-coordinate controls across all7x7 availability pairs.',
        limits='Cross-coordinate retrieval is a compatibility diagnostic, not a new deployed descriptor, a causal effect, or evidence that larger gates solve fusion. No test selection or weight tuning.'))


if __name__ == '__main__':
    main()
