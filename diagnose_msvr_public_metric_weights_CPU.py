"""Finite cached-descriptor metric-weight comparison; no neural inference or fitting."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from full_evaluation import distance, full_metrics
from official_training_data import full_records, metadata
from run_experiment import write_json

METRICS = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
WEIGHTS = (.125, .25, .5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-proof', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    torch.set_num_threads(4)
    proof = json.loads(Path(args.input_proof).read_text('utf-8'))
    assert proof['status'] == 'FINITE_MSVR_EXISTING_PUBLIC_METRIC_WEIGHT_PLAN_NOT_RUN'
    assert proof['public_metric_weights'] == list(WEIGHTS)
    train, query, gallery, _, _, manifest = full_records(proof['data_root'], 'MSVR310')
    assert (len(train), len(query), len(gallery)) == (1032, 591, 1055)
    installed = metadata(query, 'query') | metadata(gallery, 'gallery')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    all_results, cache_records = {}, {}
    started = time.time()
    for label, record in proof['records'].items():
        path = Path(record['remote'])
        assert path.stat().st_size == record['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        with np.load(path, allow_pickle=False) as saved:
            values = {key: saved[key].copy() for key in saved.files}
        assert all(np.array_equal(values[key], value) for key, value in installed.items())
        tensors = {}
        for split, count in (('query', 591), ('gallery', 1055)):
            array = values[split+'_features']
            assert array.shape == (count, 5632) and array.dtype == np.float32 and np.isfinite(array).all()
            features = torch.from_numpy(array)
            assert features.device.type == 'cpu'
            assert (features[:, :5120].square().sum(1)-.75).abs().max() < 1e-5
            assert (features[:, 5120:].square().sum(1)-.25).abs().max() < 1e-5
            tensors[split] = features
        measurements, raw = {}, dict(installed)
        for weight in WEIGHTS:
            tag = 'public_'+str(weight).replace('.', '_')
            pair = []
            for split in ('query', 'gallery'):
                source = tensors[split]
                if weight == .25:
                    feature = source
                else:
                    feature = torch.cat((math.sqrt((1-weight)/.75)*source[:, :5120],
                                         math.sqrt(weight/.25)*source[:, 5120:]), 1)
                assert feature.shape[1] == 5632 and (feature.square().sum(1)-1).abs().max() < 2e-5
                pair.append(feature)
            distances = distance(*pair)
            assert distances.shape == (591, 1055) and distances.dtype == np.float32 and np.isfinite(distances).all()
            if weight == .25:
                assert np.array_equal(distances, values['distances'])
            measured = full_metrics(distances, installed['query_ids'], installed['gallery_ids'],
                installed['query_scenes'], installed['gallery_scenes'], installed['query_names'],
                installed['query_cameras'], installed['query_scenes'], output/(label+'_'+tag))
            if weight == .25:
                assert all(abs(measured[key]-record['six'][key]) < 1e-8 for key in METRICS)
            measurements[tag] = dict(public_metric_weight=weight, selected_epoch=record['selected_epoch'],
                **{key: measured[key] for key in METRICS},
                paper_both_plus1=measured['mAP'] >= 50.2 and measured['Rank-1'] >= 60.8)
            raw['distances_'+tag] = distances
        for tag, measured in measurements.items():
            measured['delta_from_same_checkpoint_weight025'] = {
                key: measured[key]-measurements['public_0_25'][key] for key in METRICS}
        raw_path = output/(label+'_cached_weight_distances.npz')
        np.savez_compressed(raw_path, **raw)
        cache_records[label] = dict(path=str(raw_path), bytes=raw_path.stat().st_size,
            sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
        all_results[label] = measurements
        assert path.stat().st_size == record['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
    assert set(all_results) == {'strong_DeMo', 'AH_frequency_shared', 'AH_axis_shared'}
    comparisons = []
    for tag in all_results['strong_DeMo']:
        for label in ('AH_frequency_shared', 'AH_axis_shared'):
            for reference in ('strong_DeMo', 'AH_frequency_shared'):
                if label == reference:
                    continue
                comparisons.append(dict(model=label, reference=reference, public_metric_weight=all_results[label][tag]['public_metric_weight'],
                    delta_pp={key: all_results[label][tag][key]-all_results[reference][tag][key] for key in METRICS}))
    assert not torch.cuda.is_initialized()
    result = dict(status='ACTUAL_MSVR_FINITE_CACHED_PUBLIC_WEIGHT_FULL_QG_CPU_COMPLETE',
        dataset='MSVR310', public_metric_weights=list(WEIGHTS), model_checkpoints=3, cached_full_QG_cases=9,
        full_split_counts=dict(train=1032, query=591, gallery=1055), descriptor_dim=5632,
        input_proof_sha256=hashlib.sha256(Path(args.input_proof).read_bytes()).hexdigest(),
        measurements=all_results, comparisons=comparisons, raw=cache_records,
        new_neural_calls=0, optimizer_updates=0, checkpoint_writes=0, device='CPU', CPU_threads=4,
        CUDA_initialized=False, missing_evaluated=False, new49=False, wall_seconds=time.time()-started,
        limits='Nine full-QG cached-descriptor evaluations, same finite three weights and same installed GT for strong DeMo/ordinary/axis. Checkpoints were selected and trained at0.25; no epoch/seed retraining or new neural forward. Benchmark weight search is disclosed and is not an untouched test, fresh deployment equivalence or causal expert claim. Values at0.25 must exactly reproduce stored distance matrices and metrics. Any chosen weight needs actual descriptor-path implementation/parity before replacing the preserved deployed model.')
    write_json(output/'result.json', result)
    print(json.dumps(dict(status=result['status'], measurements=all_results, wall_seconds=result['wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
