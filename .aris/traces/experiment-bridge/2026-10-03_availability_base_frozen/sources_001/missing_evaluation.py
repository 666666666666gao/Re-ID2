"""Frozen dev-selected checkpoints under all six DeMo missing-modality masks."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiment_data import make_loader, split_records
from full_evaluation import official_records, full_metrics, distance
from run_experiment import build, configuration, write_json

MISSING = {'r': ('RGB',), 'n': ('NI',), 't': ('TI',),
           'rn': ('RGB', 'NI'), 'rt': ('RGB', 'TI'), 'nt': ('NI', 'TI')}
METRICS = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')


def mask_images(images, missing):
    return {key: torch.zeros_like(value) if key in missing else value
            for key, value in images.items()}


@torch.no_grad()
def extract_missing(model, records, cfg, seed, missing):
    features, routes = [], []
    started = time.time()
    for images, _, cam, scene, _ in make_loader(records, cfg, False, seed):
        images = {key: value.cuda(non_blocking=True) for key, value in images.items()}
        images = mask_images(images, missing)
        feature = model(images, cam_label=cam.cuda(), view_label=scene.cuda())
        features.append(F.normalize(feature.float(), dim=1).cpu())
        if hasattr(model, 'last_route'):
            routes.append(model.last_route.cpu())
    torch.cuda.synchronize()
    diagnostics = {'triplets': len(records), 'seconds': time.time() - started}
    if routes:
        route = torch.cat(routes)
        diagnostics['mean_route'] = route.mean(0).tolist()
        if route.ndim == 3:
            product = route.sum(2, keepdim=True) * route.sum(1, keepdim=True)
            diagnostics['joint_minus_marginal_product_L1'] = float((route - product).abs().sum((1, 2)).mean())
    return torch.cat(features), diagnostics


@torch.no_grad()
def verify_original_mask(model, records, cfg, seed):
    assert model.miss_type == 'nothing'
    images, _, cam, scene, _ = next(iter(make_loader(records[:8], cfg, False, seed)))
    images = {key: value.cuda() for key, value in images.items()}
    checks = []
    for code, missing in MISSING.items():
        external = model(mask_images(images, missing), cam_label=cam.cuda(), view_label=scene.cuda())
        model.miss_type = code
        original = model(images, cam_label=cam.cuda(), view_label=scene.cuda())
        assert torch.equal(external, original), code
        model.miss_type = 'nothing'
        checks.append(code)
    return {'original_forward_bitwise_equal_masks': checks, 'batch': len(cam), 'optimizer_updates': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    result = json.loads((run / 'result.json').read_text())
    assert result['status'] == 'COMPLETE' and json.loads((run / 'exit.json').read_text())['exit_code'] == 0
    arguments = argparse.Namespace(**result['arguments'])
    torch.set_num_threads(4)
    cfg = configuration(arguments)
    assert cfg.TEST.MISS == 'nothing'
    _, _, _, classes, cameras = split_records(arguments.data_root, arguments.dataset)
    model = build(arguments, cfg, classes, cameras)
    model.load_state_dict(torch.load(run / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    model.eval()
    queries, gallery = official_records(arguments.data_root, arguments.dataset)
    original_check = verify_original_mask(model, queries, cfg, arguments.seed) if arguments.variant == 'demo' else None
    qids, qcams, qscenes = [np.asarray([row[k] for row in queries]) for k in (1, 2, 3)]
    gids, gcams, gscenes = [np.asarray([row[k] for row in gallery]) for k in (1, 2, 3)]
    qexclude, gexclude = (qscenes, gscenes) if arguments.dataset == 'MSVR310' else (qcams, gcams)
    names = np.asarray([Path(row[0] if isinstance(row[0], str) else row[0][0]).name for row in queries])
    out = run / 'missing_evaluation'
    out.mkdir(exist_ok=False)
    clean_query, query_runtime = extract_missing(model, queries, cfg, arguments.seed, ())
    clean_gallery, gallery_runtime = extract_missing(model, gallery, cfg, arguments.seed, ())
    measurements = {}

    def measure(name, q, g, missing_query, missing_gallery):
        metrics = full_metrics(distance(q, g), qids, gids, qexclude, gexclude,
                               names, qcams, qscenes, out / name)
        measurements[name] = {'missing_query': missing_query, 'missing_gallery': missing_gallery, 'metrics': metrics}
        print('MISSING_CONDITION', name, json.dumps({m: metrics[m] for m in METRICS}), flush=True)

    measure('clean', clean_query, clean_gallery, [], [])
    normal = json.loads((run / 'full_evaluation/metrics.json').read_text())['official_test']
    assert all(abs(measurements['clean']['metrics'][m] - normal[m]) < 1e-8 for m in METRICS)
    runtimes = {'clean_query': query_runtime, 'clean_gallery': gallery_runtime}
    for code, missing in MISSING.items():
        q, qr = extract_missing(model, queries, cfg, arguments.seed, missing)
        g, gr = extract_missing(model, gallery, cfg, arguments.seed, missing)
        measure('both_missing_' + code, q, g, list(missing), list(missing))
        measure('query_missing_' + code, q, clean_gallery, list(missing), [])
        runtimes[code] = {'query': qr, 'gallery': gr}
    assert len(measurements) == 13
    write_json(out / 'result.json', {'status': 'COMPLETE', 'dataset': arguments.dataset,
               'variant': arguments.variant, 'seed': arguments.seed, 'selected_epoch': result['best']['epoch'],
               'measurements': measurements, 'runtime': runtimes, 'original_mask_check': original_check,
               'peak_memory_bytes': torch.cuda.max_memory_allocated(), 'optimizer_updates': 0,
               'protocol': 'All six missing sets r/n/t/rn/rt/nt. Missing normalized input tensors are zeros, matching original DeMo TEST.MISS semantics. Both-missing uses the SAME set for all queries and gallery; query-only uses clean gallery. All weights fixed; same official GT/junk exclusion and dev-selected checkpoint. No generated modalities, reranking or test-time optimization.',
               'limits': 'Fixed fit subset, not full official-train paper reproduction. Zero-input robustness diagnostic; does not establish learned availability masking or training with missing modalities.'})


if __name__ == '__main__':
    main()
