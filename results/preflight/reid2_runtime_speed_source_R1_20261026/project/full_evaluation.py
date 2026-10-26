"""Ground-truth full metrics for dev-selected checkpoints; test never selects them."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiment_data import make_loader, source_records, split_records
from data.datasets.RGBNT201 import RGBNT201
from data.datasets.RGBNT100 import RGBNT100
from data.datasets.msvr310 import MSVR310
from run_experiment import build, configuration, write_json
from utils.reid_evaluation import evaluate_reid


def full_metrics(distances, qids, gids, qexclude, gexclude, names, cameras, scenes, output):
    """INP = relevant count / last relevant rank, after the SAME junk exclusion."""
    ranked = np.argsort(distances, axis=1, kind='stable')
    rows = []
    for i, order in enumerate(ranked):
        junk = (gids[order] == qids[i]) & (gexclude[order] == qexclude[i])
        matches = gids[order][~junk] == qids[i]
        positives = np.flatnonzero(matches) + 1
        valid = len(positives) > 0
        row = {'query_index': i, 'name': str(names[i]), 'identity': int(qids[i]),
               'camera': int(cameras[i]), 'scene': int(scenes[i]), 'valid': valid,
               'relevant_gallery': len(positives), 'kept_gallery': len(matches)}
        if valid:
            row.update(AP=float(np.mean(np.arange(1, len(positives) + 1) / positives)),
                       INP=float(len(positives) / positives[-1]), first_match=int(positives[0]),
                       last_match=int(positives[-1]))
            row.update({f'Rank-{r}': int(positives[0] <= r) for r in (1, 5, 10, 20)})
        else:
            row.update(AP=None, INP=None, first_match=None, last_match=None)
            row.update({f'Rank-{r}': None for r in (1, 5, 10, 20)})
        rows.append(row)
    valid = [r for r in rows if r['valid']]
    assert valid
    summary = {'mAP': 100 * float(np.mean([r['AP'] for r in valid])),
               'mINP': 100 * float(np.mean([r['INP'] for r in valid])),
               **{f'Rank-{k}': 100 * float(np.mean([r[f'Rank-{k}'] for r in valid])) for k in (1, 5, 10, 20)},
               'query_count': len(rows), 'valid_queries': len(valid), 'invalid_queries': len(rows) - len(valid),
               'gallery_count': len(gids)}
    cmc, map_ = evaluate_reid(distances, qids, gids, qexclude, gexclude)
    assert abs(summary['mAP'] - 100 * map_) < 1e-8
    assert all(abs(summary[f'Rank-{r}'] - 100 * cmc[r - 1]) < 1e-8 for r in (1, 5, 10, 20))
    summary['CMC_1_to_50'] = (100 * cmc).tolist()
    summary['groups'] = {}
    for key in ('camera', 'scene', 'identity'):
        summary['groups'][key] = []
        for value in sorted({r[key] for r in valid}):
            group = [r for r in valid if r[key] == value]
            summary['groups'][key].append({'value': value, 'queries': len(group),
                                         'mAP': 100 * float(np.mean([r['AP'] for r in group])),
                                         'mINP': 100 * float(np.mean([r['INP'] for r in group])),
                                         **{f'Rank-{k}': 100 * float(np.mean([r[f'Rank-{k}'] for r in group])) for k in (1, 5, 10, 20)}})
    with output.with_suffix('.csv').open('w', encoding='utf-8', newline='') as table:
        writer = csv.DictWriter(table, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(output.with_suffix('.json'), summary)
    return summary


def official_records(root, dataset):
    cls, query_dir, gallery_dir = {
        'RGBNT201': (RGBNT201, 'RGBNT201/test', 'RGBNT201/test'),
        'RGBNT100': (RGBNT100, 'RGBNT100/rgbir/query', 'RGBNT100/rgbir/bounding_box_test'),
        'MSVR310': (MSVR310, 'MSVR310/query3', 'MSVR310/bounding_box_test'),
    }[dataset]
    query, gallery = [sorted(cls._process_dir(None, str(Path(root) / path), relabel=False), key=lambda r: str(r[0]))
                      for path in (query_dir, gallery_dir)]
    train_ids = {r[1] for r in source_records(root, dataset)}
    assert train_ids.isdisjoint({r[1] for r in query + gallery})
    assert query and gallery
    return query, gallery


@torch.no_grad()
def extract(model, records, cfg, seed):
    features, routes, gates = [], [], []
    start = time.time()
    for images, _, cam, scene, _ in make_loader(records, cfg, False, seed):
        images = {k: v.cuda(non_blocking=True) for k, v in images.items()}
        features.append(F.normalize(model(images, cam_label=cam.cuda(), view_label=scene.cuda()).float(), dim=1).cpu())
        if hasattr(model, 'last_route'):
            routes.append(model.last_route.cpu())
            gates.append(model.last_gates.cpu())
    torch.cuda.synchronize()
    result = {'end_to_end_seconds': time.time() - start, 'triplets': len(records),
              'includes_decode_and_first_batch': True}
    if routes:
        route, gate = torch.cat(routes), torch.cat(gates)
        result['mean_route'] = route.mean(0).tolist()
        result['mean_outer_gates'] = gate.mean(0).tolist()
        result['mean_route_entropy'] = float(torch.special.entr(route).flatten(1).sum(1).mean())
        if route.ndim == 3:
            independent = route.sum(2, keepdim=True) * route.sum(1, keepdim=True)
            result['mean_joint_minus_marginal_product_L1'] = float((route - independent).abs().sum((1, 2)).mean())
    return torch.cat(features), result


def distance(q, g):
    return (q.square().sum(1, keepdim=True) + g.square().sum(1)[None] - 2 * q @ g.T).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    terminal = json.loads((run / 'result.json').read_text())
    assert terminal['status'] == 'COMPLETE'
    assert json.loads((run / 'exit.json').read_text())['exit_code'] == 0
    arguments = argparse.Namespace(**terminal['arguments'])
    torch.set_num_threads(4)
    cfg = configuration(arguments)
    _, dev, indices, classes, cameras = split_records(arguments.data_root, arguments.dataset)
    model = build(arguments, cfg, classes, cameras)
    model.load_state_dict(torch.load(run / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    model.eval()
    out = run / 'full_evaluation'
    out.mkdir(exist_ok=False)
    arrays = np.load(run / 'best_dev_arrays.npz')
    assert np.array_equal(arrays['query_indices'], indices)
    assert np.array_equal(arrays['ids'], [r[1] for r in dev])
    exclusion = arrays['scenes'] if arguments.dataset == 'MSVR310' else arrays['cameras']
    dev_metrics = full_metrics(arrays['distances'], arrays['ids'][indices], arrays['ids'],
                              exclusion[indices], exclusion, arrays['names'][indices],
                              arrays['cameras'][indices], arrays['scenes'][indices], out / 'dev_per_query')
    assert all(abs(dev_metrics[k] - terminal['best'][k]) < 1e-8 for k in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10'))
    queries, gallery = official_records(arguments.data_root, arguments.dataset)
    q, q_runtime = extract(model, queries, cfg, arguments.seed)
    g, g_runtime = extract(model, gallery, cfg, arguments.seed)
    qids, qcam, qscene = [np.asarray([r[i] for r in queries]) for i in (1, 2, 3)]
    gids, gcam, gscene = [np.asarray([r[i] for r in gallery]) for i in (1, 2, 3)]
    qexclude, gexclude = (qscene, gscene) if arguments.dataset == 'MSVR310' else (qcam, gcam)
    qnames = np.asarray([Path(r[0] if isinstance(r[0], str) else r[0][0]).name for r in queries])
    distances = distance(q, g)
    test_metrics = full_metrics(distances, qids, gids, qexclude, gexclude, qnames, qcam, qscene, out / 'test_per_query')
    np.savez_compressed(out / 'test_arrays.npz', query_features=q.numpy(), gallery_features=g.numpy(),
                        distances=distances, query_ids=qids, gallery_ids=gids, query_cameras=qcam, gallery_cameras=gcam,
                        query_scenes=qscene, gallery_scenes=gscene, query_names=qnames)
    result = {'dataset': arguments.dataset, 'variant': arguments.variant, 'seed': arguments.seed,
              'checkpoint': 'best.pth', 'selected_epoch': terminal['best']['epoch'],
              'selection': 'highest identity-heldout development mAP; official test never selects checkpoint',
              'training_scope': terminal['evaluation_scope'],
              'test_warning': 'Trained on fixed fit subset, not the entire official train split; not a full-training paper reproduction.',
              'exclusion': 'same identity and scene' if arguments.dataset == 'MSVR310' else 'same identity and camera',
              'mINP_reference': 'https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/evaluation/rank.py',
              'dev': dev_metrics, 'official_test': test_metrics, 'query_runtime': q_runtime, 'gallery_runtime': g_runtime,
              'parameters': terminal['parameters'], 'descriptor_dim': q.shape[1],
              'train_peak_memory_bytes': terminal['peak_memory'], 'eval_peak_memory_bytes': torch.cuda.max_memory_allocated(),
              'optimizer_updates': terminal['optimizer_steps'], 'amp_skips': terminal['amp_skipped_steps']}
    result['batch_attempts'] = terminal['steps']
    result['training_wall_seconds'] = terminal['finished'] - terminal['started']
    write_json(out / 'metrics.json', result)
    print('FULL_EVALUATION_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
