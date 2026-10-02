"""CPU sanity against real saved retrieval arrays and installed official datasets."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiment_data import split_records, Triplets
from full_evaluation import official_records, full_metrics
from run_experiment import configuration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    terminal = json.loads((run / 'result.json').read_text())
    original = terminal['arguments']
    report = {'status': 'CPU_RUNTIME_PASS', 'official_ground_truth': {}}
    for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
        fit, dev, indices, classes, cameras = split_records(original['data_root'], dataset)
        queries, gallery = official_records(original['data_root'], dataset)
        assert max(r[2] for r in queries + gallery) < cameras
        cfg = configuration(argparse.Namespace(**{**original, 'dataset': dataset}))
        for records in (queries, gallery):
            sample = Triplets(records, cfg, False)[0]
            assert set(sample[0]) == {'RGB', 'NI', 'TI'}
            assert all(tuple(t.shape) == (3, *cfg.INPUT.SIZE_TRAIN) for t in sample[0].values())
        selector = 3 if dataset == 'MSVR310' else 2
        valid = sum(any(t[1] == r[1] and t[selector] != r[selector] for t in gallery) for r in queries)
        report['official_ground_truth'][dataset] = {'queries': len(queries), 'gallery': len(gallery),
                                                   'valid_queries': valid, 'fit_classes': classes, 'camera_embeddings': cameras}
    out = run / 'full_metric_cpu_check'
    out.mkdir(parents=True, exist_ok=False)
    arrays = np.load(run / 'best_dev_arrays.npz')
    indices = arrays['query_indices']
    exclusion = arrays['scenes'] if original['dataset'] == 'MSVR310' else arrays['cameras']
    metrics = full_metrics(arrays['distances'], arrays['ids'][indices], arrays['ids'], exclusion[indices], exclusion,
                           arrays['names'][indices], arrays['cameras'][indices], arrays['scenes'][indices], out / 'real_dev')
    assert all(abs(metrics[k] - terminal['best'][k]) < 1e-8 for k in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10'))
    assert 0 <= metrics['mINP'] <= 100
    report['real_saved_array_metrics'] = {k: metrics[k] for k in ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')}
    (out / 'runtime.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
