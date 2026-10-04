"""Frozen eight-stage readout utility on all official records and49 modality pairs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from diagnose_trained_outlet_utility import attach_probe, STAGES, framework
from evaluate_full_official49 import SETS
from full_evaluation import distance, full_metrics
from independent_control_axis import build
from missing_evaluation import METRICS
from official_training_data import full_records, metadata
from run_experiment import configuration, write_json


READOUTS = {stage: (stage, stage) for stage in STAGES}
READOUTS.update(F_post_to_common=('F_post', 'base_common'), common_to_F_post=('base_common', 'F_post'),
    M_post_to_common=('M_post', 'base_common'), common_to_M_post=('base_common', 'M_post'))


def main():
    parser = argparse.ArgumentParser()
    for name in ('run-dir', 'previous-frozen', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run, previous, output = map(Path, (args.run_dir, args.previous_frozen, args.output))
    trained = json.loads((run / 'result.json').read_text())
    old = json.loads((previous / 'result.json').read_text())
    assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50 and trained['training_heldout_identities'] == 0
    assert json.loads((run.parent / (run.name + '_exit.json')).read_text())['exit_code'] == 0
    arguments = argparse.Namespace(**trained['arguments'])
    assert arguments.dataset == 'MSVR310' and arguments.variant == 'axis_shared' and arguments.pooling == 'original_mean'
    assert old['status'] == 'COMPLETE' and len(old['measurements']) == 49
    assert old['selected_epoch'] == trained['best']['epoch'] and old['model_arguments'] == trained['arguments']
    cfg = configuration(arguments)
    train, query, gallery, classes, cameras, manifest = full_records(arguments.data_root, arguments.dataset)
    assert manifest == json.loads((run / 'official_split_manifest.json').read_text())
    assert (len(train), len(query), len(gallery)) == (1032, 591, 1055)
    arrays = metadata(query, 'query') | metadata(gallery, 'gallery')
    protected = (run / 'best.pth', run / 'best_official_arrays.npz', run / 'result.json', previous / 'result.json')
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    with np.load(run / 'best_official_arrays.npz') as saved:
        assert all(np.array_equal(saved[key], value) for key, value in arrays.items())
        reference = {key: saved[key].copy() for key in ('query_features', 'gallery_features')}
    torch.set_num_threads(4)
    model = build(arguments, cfg, classes, cameras)
    model.load_state_dict(torch.load(run / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    model.eval()
    versions = {name: value._version for name, value in model.state_dict().items()}
    handles = attach_probe(model)
    output.mkdir(exist_ok=False)
    records = query[:64] if args.smoke else query + gallery
    banks, scalars, runtime = {}, {}, {}
    started = time.time()
    for name, missing in SETS.items():
        retained = tuple(index for index, key in enumerate(('RGB', 'NI', 'TI')) if key not in missing)
        before = time.time()
        banks[name], scalars[name] = framework.extract(model, records, cfg, arguments.seed, retained)
        torch.cuda.synchronize()
        runtime[name] = dict(records=len(records), seconds=time.time() - before,
            includes_decode_and_first_batch=True)
        assert set(banks[name]) == set(STAGES)
    for handle in handles: handle.remove()
    assert versions == {name: value._version for name, value in model.state_dict().items()}
    assert np.array_equal(banks['RNT']['deployed'][:len(query) if not args.smoke else len(records)].numpy(),
        reference['query_features'][:len(query) if not args.smoke else len(records)])
    if not args.smoke:
        assert np.array_equal(banks['RNT']['deployed'][len(query):].numpy(), reference['gallery_features'])
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == inputs[str(path)] for path in protected)
    if args.smoke:
        write_json(output / 'smoke.json', dict(status='PASS', records=64, availability_sets=7, stages=list(STAGES),
            original_fuse_reconstructed_exact=True, normal_feature_max_error=0,
            state_tensor_versions_unchanged=True, optimizer_updates=0, new_weights=0))
        return
    for name, values in scalars.items():
        assert len(values['names']) == len(query) + len(gallery)
        with (output / ('scale_' + name + '.csv')).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['row', 'availability'] + list(values))
            writer.writeheader()
            writer.writerows(dict(row=i, availability=name, **{key: value[i] for key, value in values.items()})
                for i in range(len(records)))
    measurements = {}
    for qs in SETS:
        for gs in SETS:
            condition = 'q_' + qs + '_g_' + gs
            folder = output / condition; folder.mkdir()
            raw, values = {}, {}
            for readout, (qstage, gstage) in READOUTS.items():
                raw[readout] = distance(banks[qs][qstage][:len(query)], banks[gs][gstage][len(query):])
                values[readout] = full_metrics(raw[readout], arrays['query_ids'], arrays['gallery_ids'],
                    arrays['query_scenes'], arrays['gallery_scenes'], arrays['query_names'],
                    arrays['query_cameras'], arrays['query_scenes'], folder / readout)
            assert all(values['deployed'][m] == old['measurements'][condition][m] for m in METRICS)
            with (folder / 'deployed.csv').open() as handle, (previous / (condition + '.csv')).open() as prior:
                assert list(csv.DictReader(handle)) == list(csv.DictReader(prior))
            np.savez_compressed(folder / 'raw.npz', **raw, **arrays)
            measurements[condition] = values
            print('FULL_OFFICIAL_READOUT_STAGES', condition, flush=True)
    assert versions == {name: value._version for name, value in model.state_dict().items()}
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == inputs[str(path)] for path in protected)
    write_json(output / 'result.json', dict(status='COMPLETE', model_arguments=trained['arguments'],
        selected_epoch=trained['best']['epoch'], train_records=len(train), query_records=len(query),
        gallery_records=len(gallery), training_heldout_identities=0, stages=list(STAGES), readouts=READOUTS, metric_cases=588,
        repeated_condition_query_rows=588 * len(query), measurements=measurements, runtime=runtime,
        previous_all49_deployed_metrics_perquery_exact=True, normal_feature_max_error=0,
        original_fuse_reconstructed_exact=True, state_tensor_versions_unchanged=True,
        original_inputs=inputs, optimizer_updates=0, new_weights=0, seconds=time.time() - started,
        protocol='Unchanged fixed full-official best. Existing read-only probe: independent M/F auxiliary outputs, '
            'actual weighted routed M/F evidence, PM/PF outputs, public base and deployed descriptor. '
            'All7 availabilities,49 pairs,8 same-stage and4 directed PM/PF-public readouts; original GT scene exclusion. '
            'No training or deployment change.',
        limits='Single-seed MSVR benchmark-selected diagnostic. Stage retrieval changes are empirical utility; '
            'they do not establish information destruction, strict semantic disentanglement or causality. '
            'Seven stages are512D and deployed is5632D. Repeated pairs are not independent queries.'))


if __name__ == '__main__':
    main()
