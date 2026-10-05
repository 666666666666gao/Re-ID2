"""Complete NN states, immediate installed-GT audit and acknowledged per-condition raw archive."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import sys

from audit_full_official_condition import installed_context, audit_condition
from audit_full_official_control_states import verify_heads

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_full_official49 import SETS
from experiment_data import make_loader
from full_evaluation import distance, full_metrics
from independent_control_axis import build
from missing_evaluation import mask_images, METRICS
from official_training_data import full_records, metadata
from run_experiment import configuration, write_json


from diagnose_full_official_control_states import extract, calibration, STATES, TARGETS

STREAM_PREFIX = "OFFICIAL_STREAM "


def emit(value):
    print(STREAM_PREFIX + json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for key in ('run-dir', 'previous-frozen', 'output'): parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    run, previous, out = map(Path, (args.run_dir, args.previous_frozen, args.output))
    trained = json.loads((run / 'result.json').read_text())
    prior = json.loads((previous / 'result.json').read_text())
    assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50 and trained['training_heldout_identities'] == 0
    assert json.loads((run.parent / (run.name + '_exit.json')).read_text())['exit_code'] == 0
    assert prior['status'] == 'COMPLETE' and len(prior['measurements']) == 49
    assert prior['selected_epoch'] == trained['best']['epoch'] and prior['model_arguments'] == trained['arguments']
    context, dataset, installed = installed_context(run)
    assert context == trained
    assert prior['training_heldout_identities'] == 0
    frozen_audit = json.loads((previous / 'independent_cpu_audit.json').read_text())
    assert frozen_audit['cases'] == 49
    arguments = argparse.Namespace(**trained['arguments'])
    cfg = configuration(arguments)
    train, query, gallery, classes, cameras, manifest = full_records(arguments.data_root, arguments.dataset)
    assert manifest == json.loads((run / 'official_split_manifest.json').read_text())
    protected = (run / 'best.pth', run / 'result.json', previous / 'result.json')
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    torch.set_num_threads(4)
    model = build(arguments, cfg, classes, cameras)
    model.load_state_dict(torch.load(run / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    model.eval()
    versions = {key: value._version for key, value in model.state_dict().items()}
    out.mkdir(exist_ok=False)
    arrays = metadata(query, 'query') | metadata(gallery, 'gallery')
    selector = 'scenes' if arguments.dataset == 'MSVR310' else 'cameras'
    saved = np.load(run / 'best_official_arrays.npz')
    assert all(np.array_equal(saved[key], value) for key, value in arrays.items())
    banks, predictions, runtime = {}, {}, {}
    started = time.time()
    for available, missing in SETS.items():
        feature, prediction, gates, controls, performance = extract(model, query + gallery, cfg,
            arguments.seed, missing, arguments.gate_gradient_mode)
        banks[available] = {state: (value[:len(query)], value[len(query):]) for state, value in feature.items()}
        predictions[available] = prediction[:len(query)]
        runtime[available] = performance
        np.savez_compressed(out / ('heads_' + available + '.npz'), predictions=prediction, gates=gates,
            control_logits=controls, **arrays)
        with (out / ('heads_' + available + '.csv')).open('w', newline='', encoding='utf-8') as handle:
            fields = ['role', 'row', 'name', 'identity', 'camera', 'scene', 'prediction_M', 'prediction_F',
                'prediction_I', 'gate_M', 'gate_F', 'gate_I', 'control_M', 'control_F', 'control_I']
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for i, record in enumerate(query + gallery):
                role, index = ('query', i) if i < len(query) else ('gallery', i - len(query))
                writer.writerow(dict(role=role, row=index, name=str(arrays[role + '_names'][index]),
                    identity=int(arrays[role + '_ids'][index]), camera=int(arrays[role + '_cameras'][index]),
                    scene=int(arrays[role + '_scenes'][index]),
                    **{key: float(value[i, j]) for value, keys in ((prediction, ('prediction_M', 'prediction_F', 'prediction_I')),
                        (gates, ('gate_M', 'gate_F', 'gate_I')), (controls, ('control_M', 'control_F', 'control_I')))
                       for j, key in enumerate(keys)}))
    assert np.array_equal(banks['RNT']['11'][0].numpy(), saved['query_features'])
    assert np.array_equal(banks['RNT']['11'][1].numpy(), saved['gallery_features'])
    heads, head_predictions = verify_heads(out, installed['query'], installed['gallery'], arguments.gate_gradient_mode)
    measurements, diagnostics, conditions, receipts, errors = {}, {}, {}, {}, []
    peak_raw_bytes = 0
    for qs in SETS:
        for gs in SETS:
            condition = 'q_' + qs + '_g_' + gs
            folder = out / condition; folder.mkdir()
            raw, values = {}, {}
            for state in STATES:
                q, g = banks[qs][state][0], banks[gs][state][1]
                distances = (2 - 2 * q @ g.T).numpy() if state.startswith('base_') else distance(q, g)
                if state == '11':
                    with np.load(previous / (condition + '.npz')) as old:
                        assert np.array_equal(distances, old['distances'])
                values[state] = full_metrics(distances, arrays['query_ids'], arrays['gallery_ids'],
                    arrays['query_' + selector], arrays['gallery_' + selector], arrays['query_names'],
                    arrays['query_cameras'], arrays['query_scenes'], folder / ('state_' + state))
                raw['distance_' + state] = distances
            assert all(values['11'][m] == prior['measurements'][condition][m] for m in METRICS)
            with (folder / 'state_11.csv').open() as handle, (previous / (condition + '.csv')).open() as before:
                assert list(csv.DictReader(handle)) == list(csv.DictReader(before))
            qbank = {state: banks[qs][state][0] for state in ('00', '10', '01', '11')}
            gbank = {state: banks[gs][state][1] for state in ('00', '10', '01', '11')}
            reference, detail = calibration(qbank, gbank, predictions[qs], arrays['query_ids'], arrays['gallery_ids'],
                arrays['query_' + selector], arrays['gallery_' + selector], folder / 'contributions.csv')
            np.savez_compressed(folder / 'raw.npz', **raw, **reference, **arrays)
            measurements[condition], diagnostics[condition] = values, detail
            checked, error = audit_condition(folder, previous, values, detail, dataset, installed, head_predictions[qs])
            conditions[condition] = checked
            errors.append(error)
            raw_path = folder / 'raw.npz'
            raw_info = dict(bytes=raw_path.stat().st_size, sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
            peak_raw_bytes = max(peak_raw_bytes, raw_info['bytes'])
            assert len(list(out.glob('q_*_g_*/raw.npz'))) == 1
            emit(dict(event='RAW_READY', condition=condition, path=str(raw_path), file=raw_info,
                      installed_gt_cases=6, full11_exact=True))
            acknowledgment = json.loads(sys.stdin.readline())
            assert acknowledgment == dict(event='ARCHIVED', condition=condition, file=raw_info)
            assert raw_path.resolve().is_relative_to(out.resolve())
            assert raw_path.stat().st_size == raw_info['bytes']
            assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == raw_info['sha256']
            raw_path.unlink()
            assert not list(out.glob('q_*_g_*/raw.npz'))
            receipts[condition] = dict(file=raw_info, local_verified_acknowledged=True,
                                       server_copy_cleared=True, max_sixmetric_error_pp=error)
            write_json(folder / 'condition_cpu_audit.json', dict(measured=checked, archive=receipts[condition]))
            emit(dict(event='RAW_CLEARED', condition=condition))
    assert versions == {key: value._version for key, value in model.state_dict().items()}
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == inputs[str(path)] for path in protected)
    assert len(conditions) == len(receipts) == len(measurements) == 49
    normal = conditions['q_RNT_g_RNT']
    assert all(normal['metrics']['11'][m] == trained['full_metrics'][m] for m in METRICS)
    audit = dict(status='PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT', model_arguments=trained['arguments'],
        cases=294, repeated_condition_query_rows=294 * len(query), query_records=len(query), gallery_records=len(gallery),
        max_sixmetric_error_pp=max(errors), conditions=conditions, normal=normal, heads=heads,
        full11_minus00_equal_condition_mean={m: float(np.mean([v['full11_minus00']['delta_pp'][m] for v in conditions.values()])) for m in METRICS},
        all49_harm=sum(v['full11_minus00']['Rank1_harm_queries'] for v in conditions.values()),
        all49_rescue=sum(v['full11_minus00']['Rank1_rescue_queries'] for v in conditions.values()),
        calibration_rows=49 * len(query), head_records=7 * (len(query) + len(gallery)),
        limits='Installed-GT CPU recount of all6x49 cases immediately after NN inference. Same original reference legality/head/group audits; raw files locally archived before bounded server unlink. Repeated conditions are not independent queries; not causal synergy.')
    write_json(out / 'independent_cpu_audit.json', audit)
    write_json(out / 'stream_archive.json', dict(status='ALL49_CONDITIONS_AUDITED_ARCHIVED_AND_SERVER_RAW_CLEARED',
        conditions=receipts, max_live_enhanced_raw_files=1, peak_live_enhanced_raw_bytes=peak_raw_bytes,
        frozen49_bank_retained_for_matrix_equality=True, optimizer_updates=0, cases=294))
    write_json(out / 'result.json', dict(status='COMPLETE', model_arguments=trained['arguments'], dataset=arguments.dataset,
        variant=arguments.variant, selected_epoch=trained['best']['epoch'], states=list(STATES), metric_count=294,
        measurements=measurements, contributions=diagnostics, runtime=runtime, query_records=len(query),
        gallery_records=len(gallery), train_records=len(train), training_heldout_identities=0,
        normal_feature_max_error=0, previous_all49_full11_distances_metrics_perquery_exact=True,
        state_tensor_versions_unchanged=True, original_inputs=inputs, optimizer_updates=0, new_weights=0,
        residual_scale=model.residual_scale.detach().cpu().tolist(), seconds=time.time() - started, timing_includes_gt_audit_and_archive_wait=True,
        protocol='Four-state retrieval closes each expert, its conditional messages and joint interaction on both query and gallery sides. Calibration intervenes on query only against the same full11 gallery with fixed legal references. Private/shared block cosines are diagnostics, not new deployment descriptors.',
        limits='Full official benchmark-selected checkpoint, one seed; frozen empirical mechanism evidence only. No retraining, missing-view synthesis, test-time optimization or causal/information-theoretic claims.'))

    emit(dict(event='DIAGNOSIS_COMPLETE', cases=294, conditions=49, neural_inference=True, optimizer_updates=0))


if __name__ == '__main__': main()
