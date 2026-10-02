"""Fixed query-only blur and frozen-weight joint-router diagnostics."""
import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from experiment_data import make_loader, split_records
from full_evaluation import official_records, full_metrics, extract, distance
from run_experiment import build, configuration, write_json


@torch.no_grad()
def query_features(model, records, cfg, seed, modality, sigma):
    features, routes = [], []
    for images, _, cam, scene, _ in make_loader(records, cfg, False, seed):
        images = {key: value.cuda(non_blocking=True) for key, value in images.items()}
        if modality is not None:
            images[modality] = gaussian_blur(images[modality], kernel_size=[9, 9], sigma=[sigma, sigma])
        feature = model(images, cam_label=cam.cuda(), view_label=scene.cuda())
        features.append(F.normalize(feature.float(), dim=1).cpu())
        if hasattr(model, 'last_route'):
            routes.append(model.last_route.cpu())
    diagnostics = {}
    if routes:
        route = torch.cat(routes)
        diagnostics['mean_route'] = route.mean(0).tolist()
        if route.ndim == 3:
            product = route.sum(2, keepdim=True) * route.sum(1, keepdim=True)
            diagnostics['joint_minus_marginal_product_L1'] = float((route - product).abs().sum((1, 2)).mean())
    return torch.cat(features), diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    result = json.loads((run / 'result.json').read_text())
    assert result['status'] == 'COMPLETE' and json.loads((run / 'exit.json').read_text())['exit_code'] == 0
    arguments = argparse.Namespace(**result['arguments'])
    cfg = configuration(arguments)
    torch.set_num_threads(4)
    _, _, _, classes, cameras = split_records(arguments.data_root, arguments.dataset)
    model = build(arguments, cfg, classes, cameras)
    model.load_state_dict(torch.load(run / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    model.eval()
    queries, gallery = official_records(arguments.data_root, arguments.dataset)
    gallery_features, gallery_runtime = extract(model, gallery, cfg, arguments.seed)
    qids, qcams, qscenes = [np.asarray([row[k] for row in queries]) for k in (1, 2, 3)]
    gids, gcams, gscenes = [np.asarray([row[k] for row in gallery]) for k in (1, 2, 3)]
    qexclude, gexclude = (qscenes, gscenes) if arguments.dataset == 'MSVR310' else (qcams, gcams)
    names = np.asarray([Path(row[0] if isinstance(row[0], str) else row[0][0]).name for row in queries])
    out = run / 'stress_evaluation'
    out.mkdir(exist_ok=False)
    conditions = [('clean', None, None)]
    conditions += [(f'{m}_blur_sigma{tag}', m, sigma) for m in ('RGB', 'NI', 'TI') for tag, sigma in (('1p5', 1.5), ('3', 3.0))]
    if arguments.variant == 'dual':
        conditions.append(('psi_off_frozen', None, None))
    measurements = {}
    for name, modality, sigma in conditions:
        condition_gallery = gallery_features
        if name == 'psi_off_frozen':
            model.collaboration.joint = partial(model.collaboration.joint, interaction=False)
            condition_gallery, _ = extract(model, gallery, cfg, arguments.seed)
        features, routing = query_features(model, queries, cfg, arguments.seed, modality, sigma)
        metrics = full_metrics(distance(features, condition_gallery), qids, gids, qexclude, gexclude,
                               names, qcams, qscenes, out / name)
        if name == 'psi_off_frozen':
            assert routing['joint_minus_marginal_product_L1'] < 1e-6
        measurements[name] = {'metrics': metrics, 'routing': routing}
        print('STRESS_CONDITION', name, json.dumps({k: metrics[k] for k in ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')}), flush=True)
    if (run / 'full_evaluation/metrics.json').exists():
        clean = json.loads((run / 'full_evaluation/metrics.json').read_text())['official_test']
        assert all(abs(measurements['clean']['metrics'][k] - clean[k]) < 1e-8 for k in ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20'))
    write_json(out / 'result.json', {'status': 'COMPLETE', 'dataset': arguments.dataset, 'variant': arguments.variant,
               'seed': arguments.seed, 'selected_epoch': result['best']['epoch'], 'measurements': measurements,
               'gallery_runtime': gallery_runtime, 'peak_memory_bytes': torch.cuda.max_memory_allocated(),
               'protocol': 'Official test; clean gallery images fixed. Blur conditions reuse the identical original gallery descriptors. Psi-off recomputes gallery descriptors using the same altered encoder as queries. One query modality blurred after resize/normalization, kernel9 sigma1.5/3 pixels. All model weights fixed.',
               'psi_warning': 'Frozen-weight inference diagnostic with psi removed for query AND gallery; not a trained independent-router ablation.',
               'training_warning': 'Fixed fit subset, not full official training; no contribution supervision.'})


if __name__ == '__main__':
    main()
