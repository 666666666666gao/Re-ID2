"""Frozen held-out development retrieval under the six DeMo modality masks."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiment_data import split_records
from full_evaluation import distance, full_metrics
from missing_evaluation import MISSING, METRICS, extract_missing
from run_routed_metric_experiment import build, configuration, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run, output = Path(args.run_dir), Path(args.output)
    terminal = json.loads((run / 'result.json').read_text())
    assert terminal['status'] == 'COMPLETE' and terminal['epochs'] == 50
    arguments = argparse.Namespace(**terminal['arguments'])
    assert arguments.variant == 'axis_metric_routed_fullref'
    exit_file = run.parent / (run.name + '_exit.json')
    assert json.loads(exit_file.read_text())['exit_code'] == 0
    original_inputs = {path.name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
                       for path in (run / 'best.pth', run / 'best_dev_arrays.npz', run / 'result.json', exit_file)}
    arguments.data_root, arguments.pretrained = args.data_root, args.pretrained
    torch.set_num_threads(4)
    cfg = configuration(arguments)
    assert cfg.TEST.MISS == 'nothing'
    _, dev, query, classes, cameras = split_records(arguments.data_root, arguments.dataset)
    saved = np.load(run / 'best_dev_arrays.npz')
    assert np.array_equal(saved['query_indices'], query)
    assert np.array_equal(saved['ids'], [row[1] for row in dev])
    assert np.array_equal(saved['cameras'], [row[2] for row in dev])
    assert np.array_equal(saved['scenes'], [row[3] for row in dev])
    assert np.array_equal(saved['names'], [Path(row[0] if isinstance(row[0], str) else row[0][0]).name for row in dev])
    model = build(arguments, cfg, classes, cameras)
    model.load_state_dict(torch.load(run / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    model.eval()
    assert all(not module.training for module in model.modules())
    versions = {name: value._version for name, value in model.state_dict().items()}
    records = dev[:64] if args.smoke else dev
    output.mkdir(exist_ok=False)
    started = time.time()
    clean, clean_runtime = extract_missing(model, records, cfg, arguments.seed, ())
    parity = float(np.abs(clean.numpy() - saved['features'][:len(records)]).max())
    assert parity == 0
    original_masks = None
    diagnostics = {'clean': clean_runtime}
    if args.smoke:
        for code, missing in MISSING.items():
            feature, runtime = extract_missing(model, records, cfg, arguments.seed, missing)
            assert feature.shape == clean.shape and torch.isfinite(feature).all()
            diagnostics[code] = runtime
        assert versions == {name: value._version for name, value in model.state_dict().items()}
        write_json(output / 'smoke.json', {'status': 'PASS', 'triplets': len(records),
                   'all_six_masks_finite': True, 'normal_feature_max_error': parity,
                   'optimizer_updates': 0, 'original_mask_check': original_masks,
                   'state_tensor_versions_unchanged': True, 'runtime': diagnostics})
        return
    query = np.asarray(query)
    ids, cams, scenes, names = [saved[key] for key in ('ids', 'cameras', 'scenes', 'names')]
    exclusion = scenes if arguments.dataset == 'MSVR310' else cams
    measurements = {}

    def measure(name, distances, missing_query, missing_gallery):
        metrics = full_metrics(distances, ids[query], ids, exclusion[query], exclusion,
                               names[query], cams[query], scenes[query], output / name)
        measurements[name] = {'missing_query': missing_query, 'missing_gallery': missing_gallery, 'metrics': metrics}
        print('DEVELOPMENT_MISSING_CONDITION', name, json.dumps({key: metrics[key] for key in METRICS}), flush=True)

    measure('clean', saved['distances'], [], [])
    assert all(abs(measurements['clean']['metrics'][key] - terminal['strict_reload'][key]) < 1e-8
               for key in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10'))
    for code, missing in MISSING.items():
        feature, runtime = extract_missing(model, dev, cfg, arguments.seed, missing)
        assert feature.shape == clean.shape and torch.isfinite(feature).all()
        measure('both_missing_' + code, distance(feature[query], feature), list(missing), list(missing))
        measure('query_missing_' + code, distance(feature[query], clean), list(missing), [])
        diagnostics[code] = runtime
    assert len(measurements) == 13
    assert versions == {name: value._version for name, value in model.state_dict().items()}
    for path in (run / 'best.pth', run / 'best_dev_arrays.npz', run / 'result.json', exit_file):
        proof = original_inputs[path.name]
        assert path.stat().st_size == proof['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == proof['sha256']
    write_json(output / 'result.json', {'status': 'COMPLETE', 'dataset': arguments.dataset,
               'variant': arguments.variant, 'seed': arguments.seed, 'selected_epoch': terminal['best']['epoch'],
               'measurements': measurements, 'runtime': diagnostics, 'seconds': time.time() - started,
               'normal_feature_max_error': parity, 'optimizer_updates': 0,
               'normal_distance_source': 'saved best_dev_arrays.npz/distances after exact clean feature parity; missing conditions use full_evaluation.distance',
               'state_tensor_versions_unchanged': True, 'original_mask_check': original_masks,
               'original_input_files': original_inputs, 'peak_memory_bytes': torch.cuda.max_memory_allocated(),
               'protocol': 'Fixed identity-heldout development queries and same development gallery. Six missing sets r/n/t/rn/rt/nt use normalized zero input tensors; both-missing uses identical masks on query/gallery, query-only uses clean gallery. Frozen dev-selected checkpoint, installed GT and original junk exclusion. No official test or optimizer updates.',
               'limits': 'Single-seed frozen robustness diagnostic, not full official-train reproduction, independent training with missing inputs, or multi-seed acceptance.'})


if __name__ == '__main__':
    main()
