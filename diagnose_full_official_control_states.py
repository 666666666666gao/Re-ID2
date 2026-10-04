"""Frozen full-official four states, base blocks and contribution calibration."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

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


STATES = ('00', '10', '01', '11', 'base_private', 'base_shared')
TARGETS = ('delta_M_given_F', 'delta_F_given_M', 'empirical_interaction')


@torch.no_grad()
def extract(model, records, cfg, seed, missing, mode):
    banks = {state: [] for state in STATES}
    predictions, gates, controls = [], [], []
    captured, control_captured = [], []
    handle = model.calibrator.register_forward_hook(lambda module, inputs, result: captured.append(result))
    if mode == 'independent_control':
        control_handle = model.calibrator.control.register_forward_hook(
            lambda module, inputs, result: control_captured.append(result))
    start = time.time()
    for images, _, cam, scene, _ in make_loader(records, cfg, False, seed):
        captured.clear(); control_captured.clear()
        images = mask_images({key: value.cuda(non_blocking=True) for key, value in images.items()}, missing)
        states = model(images, cam_label=cam.cuda(), view_label=scene.cuda(), return_states=True)
        assert set(states) == {'00', '10', '01', '11'} and len(captured) == 3
        prediction, gate = captured[0]
        predictions.append(prediction.float().cpu())
        gates.append(gate.float().cpu())
        if mode == 'independent_control':
            assert len(control_captured) == 3
            controls.append(control_captured[0].float().cpu())
        else:
            controls.append(torch.zeros_like(predictions[-1]))
        for state, value in states.items():
            assert value.shape == (len(cam), 5632) and torch.isfinite(value).all()
            banks[state].append(F.normalize(value.float(), dim=1).cpu())
        banks['base_private'].append(F.normalize(states['00'][:, :5120].float(), dim=1).cpu())
        banks['base_shared'].append(F.normalize(states['00'][:, 5120:].float(), dim=1).cpu())
    handle.remove()
    if mode == 'independent_control': control_handle.remove()
    torch.cuda.synchronize()
    return ({state: torch.cat(values) for state, values in banks.items()},
        torch.cat(predictions).numpy(), torch.cat(gates).numpy(), torch.cat(controls).numpy(),
        dict(records=len(records), seconds=time.time() - start))


def calibration(qbank, gbank, predictions, qids, gids, qexclude, gexclude, output):
    # Query interventions use the same full11 gallery and the same legal references.
    similarity = (qbank['11'] @ gbank['11'].T).numpy()
    positive = (qids[:, None] == gids[None]) & (qexclude[:, None] != gexclude[None])
    negative = qids[:, None] != gids[None]
    assert positive.any(1).all() and negative.any(1).all()
    pos = np.where(positive, similarity, np.inf).argmin(1)
    neg = np.where(negative, similarity, -np.inf).argmax(1)
    direction = gbank['11'][pos] - gbank['11'][neg]
    scores = {state: (qbank[state] * direction).sum(1).numpy() for state in ('00', '10', '01', '11')}
    targets = np.stack((scores['11'] - scores['01'], scores['11'] - scores['10'],
        scores['11'] - scores['10'] - scores['01'] + scores['00']), 1)
    fields = ['query_index', 'gallery_positive_index', 'gallery_negative_index',
        'R00', 'R10', 'R01', 'R11', *TARGETS, 'prediction_M', 'prediction_F', 'prediction_I']
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for i in range(len(qids)):
            writer.writerow(dict(query_index=i, gallery_positive_index=int(pos[i]), gallery_negative_index=int(neg[i]),
                **{'R' + state: float(values[i]) for state, values in scores.items()},
                **{key: float(targets[i, j]) for j, key in enumerate(TARGETS)},
                **{key: float(predictions[i, j]) for j, key in enumerate(('prediction_M', 'prediction_F', 'prediction_I'))}))
    return dict(reference_similarity=similarity, positive_indices=pos, negative_indices=neg,
        raw_scores=np.stack([scores[s] for s in ('00', '10', '01', '11')], 1),
        targets=targets, predictions=predictions), dict(target_mean=targets.mean(0).tolist(),
        target_std=targets.std(0).tolist(), MAE=np.abs(targets - predictions).mean(0).tolist(),
        zero_prediction_MAE=np.abs(targets).mean(0).tolist(), same_reference_all_query_states=True,
        scope='Query-state margins against fixed full11 gallery of this availability. GT-valid hardest positive/negative references; empirical diagnostic, not causal or information-theoretic synergy.')


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
    measurements, diagnostics = {}, {}
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
            print('FULL_OFFICIAL_CONTROL_STATES', condition, flush=True)
    assert versions == {key: value._version for key, value in model.state_dict().items()}
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == inputs[str(path)] for path in protected)
    write_json(out / 'result.json', dict(status='COMPLETE', model_arguments=trained['arguments'], dataset=arguments.dataset,
        variant=arguments.variant, selected_epoch=trained['best']['epoch'], states=list(STATES), metric_count=294,
        measurements=measurements, contributions=diagnostics, runtime=runtime, query_records=len(query),
        gallery_records=len(gallery), train_records=len(train), training_heldout_identities=0,
        normal_feature_max_error=0, previous_all49_full11_distances_metrics_perquery_exact=True,
        state_tensor_versions_unchanged=True, original_inputs=inputs, optimizer_updates=0, new_weights=0,
        residual_scale=model.residual_scale.detach().cpu().tolist(), seconds=time.time() - started,
        protocol='Four-state retrieval closes each expert, its conditional messages and joint interaction on both query and gallery sides. Calibration intervenes on query only against the same full11 gallery with fixed legal references. Private/shared block cosines are diagnostics, not new deployment descriptors.',
        limits='Full official benchmark-selected checkpoint, one seed; frozen empirical mechanism evidence only. No retraining, missing-view synthesis, test-time optimization or causal/information-theoretic claims.'))


if __name__ == '__main__': main()
