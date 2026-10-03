"""Frozen dev-best: four controlled states and base-block utility over full availability49."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiment_data import make_loader, split_records
from full_evaluation import distance, full_metrics
from missing_evaluation import METRICS
from run_shared_identity_experiment import build, configuration, write_json
from shared_identity_axis import KEYS, VARIANTS


SETS = {'RNT': (0, 1, 2), 'R': (0,), 'N': (1,), 'T': (2,), 'RN': (0, 1), 'RT': (0, 2), 'NT': (1, 2)}
STATES = ('00', '10', '01', '11', 'base_private', 'base_shared')


@torch.no_grad()
def extract(model, records, cfg, seed, retained):
    banks = {key: [] for key in STATES}
    predictions = []
    captures = []
    hook = None
    if hasattr(model, 'calibrator'):
        hook = model.calibrator.register_forward_hook(lambda module, inputs, result: captures.append(result[0].detach()))
    for images, _, cam, scene, _ in make_loader(records, cfg, False, seed):
        captures.clear()
        images = {key: value.cuda(non_blocking=True) if index in retained else torch.zeros_like(value).cuda()
                  for index, (key, value) in enumerate(images.items())}
        states = model(images, cam_label=cam.cuda(), view_label=scene.cuda(), return_states=True)
        assert set(states) == {'00', '10', '01', '11'}
        if hook is not None:
            assert len(captures) == 3
            predictions.append(captures[0].float().cpu())
        for key, value in states.items():
            assert value.shape == (len(cam), 5632) and torch.isfinite(value).all()
            banks[key].append(F.normalize(value.float(), dim=1).cpu())
        banks['base_private'].append(F.normalize(states['00'][:, :5120].float(), dim=1).cpu())
        banks['base_shared'].append(F.normalize(states['00'][:, 5120:].float(), dim=1).cpu())
    if hook is not None:
        hook.remove()
    return {key: torch.cat(value) for key, value in banks.items()}, torch.cat(predictions) if predictions else None


def contributions(qbank, gbank, query, ids, exclusion, predictions, output):
    # All query states use the SAME full11 gallery, positive and negative rows.
    similarity = qbank['11'][query] @ gbank['11'].T
    positive = (ids[query, None] == ids[None]) & (exclusion[query, None] != exclusion[None])
    negative = ids[query, None] != ids[None]
    assert positive.any(1).all() and negative.any(1).all()
    pos = similarity.masked_fill(~torch.from_numpy(positive), torch.inf).argmin(1)
    neg = similarity.masked_fill(~torch.from_numpy(negative), -torch.inf).argmax(1)
    direction = gbank['11'][pos] - gbank['11'][neg]
    scores = {key: (qbank[key][query] * direction).sum(1).numpy() for key in ('00', '10', '01', '11')}
    targets = np.stack((scores['11'] - scores['01'], scores['11'] - scores['10'],
                        scores['11'] - scores['10'] - scores['01'] + scores['00']), 1)
    fields = ['query_index', 'gallery_positive_index', 'gallery_negative_index'] + ['R' + key for key in scores]
    fields += ['delta_M_given_F', 'delta_F_given_M', 'empirical_interaction']
    estimate = None if predictions is None else predictions[query].numpy()
    if estimate is not None:
        fields += ['prediction_M', 'prediction_F', 'prediction_I']
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(query):
            item = dict(query_index=int(row), gallery_positive_index=int(pos[index]), gallery_negative_index=int(neg[index]),
                        **{'R' + key: float(value[index]) for key, value in scores.items()},
                        **{key: float(targets[index, i]) for i, key in enumerate(fields[7:10])})
            if estimate is not None:
                item.update({key: float(estimate[index, i]) for i, key in enumerate(('prediction_M', 'prediction_F', 'prediction_I'))})
            writer.writerow(item)
    return dict(target_mean=targets.mean(0).tolist(), target_std=targets.std(0).tolist(),
                MAE=None if estimate is None else np.abs(targets - estimate).mean(0).tolist(),
                same_reference_all_states=True, GT_positive_negative_valid=True,
                scope='query-state margins against frozen full11 gallery of this availability; empirical diagnostic, not causal/information-theoretic synergy')


def main(model_builder=build):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--previous-frozen', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run, output, previous = Path(args.run_dir), Path(args.output), Path(args.previous_frozen)
    terminal = json.loads((run / 'result.json').read_text())
    assert terminal['status'] == 'COMPLETE' and terminal['epochs'] == 50
    arguments = argparse.Namespace(**terminal['arguments'])
    assert arguments.variant in VARIANTS and arguments.dataset == 'MSVR310'
    arguments.data_root, arguments.pretrained = args.data_root, args.pretrained
    exit_file = run.parent / (run.name + '_exit.json')
    assert json.loads(exit_file.read_text())['exit_code'] == 0
    previous_result = json.loads((previous / 'result.json').read_text())
    assert previous_result['status'] == 'COMPLETE' and len(previous_result['measurements']) == 49
    protected = (run / 'best.pth', run / 'best_dev_arrays.npz', run / 'result.json', exit_file, previous / 'result.json')
    inputs = {str(path): dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in protected}
    cfg = configuration(arguments)
    _, dev, query, classes, cameras = split_records(args.data_root, arguments.dataset)
    saved = np.load(run / 'best_dev_arrays.npz')
    assert np.array_equal(saved['query_indices'], query)
    for key, values in (('ids', [r[1] for r in dev]), ('cameras', [r[2] for r in dev]), ('scenes', [r[3] for r in dev]),
                        ('names', [Path(r[0] if isinstance(r[0], str) else r[0][0]).name for r in dev])):
        assert np.array_equal(saved[key], values)
    model = model_builder(arguments, cfg, classes, cameras)
    model.load_state_dict(torch.load(run / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    model.eval()
    assert all(not module.training for module in model.modules())
    versions = {name: value._version for name, value in model.state_dict().items()}
    output.mkdir(exist_ok=False)
    started = time.time()
    records = dev[:64] if args.smoke else dev
    banks, predictions = {}, {}
    for name, retained in SETS.items():
        banks[name], predictions[name] = extract(model, records, cfg, arguments.seed, retained)
    parity = float(np.abs(banks['RNT']['11'].numpy() - saved['features'][:len(records)]).max())
    assert parity == 0
    assert versions == {name: value._version for name, value in model.state_dict().items()}
    if args.smoke:
        write_json(output / 'smoke.json', dict(status='PASS', normal_feature_max_error=parity,
                   seven_availability_banks=True, six_states_finite=True, state_tensor_versions_unchanged=True,
                   optimizer_updates=0, triplets=len(records)))
        return
    query = np.asarray(query)
    ids, cams, scenes, names = (saved[key] for key in ('ids', 'cameras', 'scenes', 'names'))
    exclusion = scenes
    measurements, diagnostics, raw = {}, {}, {}
    for qset in SETS:
        for gset in SETS:
            condition = 'q_' + qset + '_g_' + gset
            folder = output / condition
            folder.mkdir()
            values = {}
            for state in STATES:
                q, g = banks[qset][state][query], banks[gset][state]
                if state.startswith('base_'):
                    # Private source-disjoint banks have exactly zero dot products.
                    # Unit-vector cosine preserves those ties instead of norm-roundoff ordering.
                    distances = (2 - 2 * q @ g.T).numpy()
                elif state == '11' and qset == gset == 'RNT':
                    distances = saved['distances']
                else:
                    distances = distance(q, g)
                raw[condition + '_' + state] = distances
                values[state] = full_metrics(distances, ids[query], ids, exclusion[query], exclusion,
                                             names[query], cams[query], scenes[query], folder / ('state_' + state))
            old = previous_result['measurements'][condition]['metrics']
            assert all(values['11'][key] == old[key] for key in METRICS)
            with (previous / (condition + '.csv')).open(encoding='utf-8') as handle:
                old_rows = list(csv.DictReader(handle))
            with (folder / 'state_11.csv').open(encoding='utf-8') as handle:
                assert list(csv.DictReader(handle)) == old_rows
            measurements[condition] = values
            diagnostics[condition] = contributions(banks[qset], banks[gset], query, ids, exclusion, predictions[qset], folder / 'contributions.csv')
            print('FROZEN_STATES', condition, json.dumps({state: {key: value[key] for key in METRICS} for state, value in values.items()}), flush=True)
    np.savez_compressed(output / 'raw_distances.npz', **raw, query_indices=query, ids=ids, cameras=cams, scenes=scenes, names=names)
    assert versions == {name: value._version for name, value in model.state_dict().items()}
    for path in protected:
        assert path.stat().st_size == inputs[str(path)]['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == inputs[str(path)]['sha256']
    write_json(output / 'result.json', dict(status='COMPLETE', dataset=arguments.dataset, variant=arguments.variant,
               selected_epoch=terminal['best']['epoch'], measurements=measurements, contributions=diagnostics,
               normal_feature_max_error=parity, previous_all49_full11_sixmetric_and_perquery_exact=True,
               state_tensor_versions_unchanged=True, original_inputs=inputs, optimizer_updates=0, official_test_uses=0,
               residual_scale=None if arguments.variant == 'demo_shared' else model.residual_scale.detach().cpu().tolist(),
               seconds=time.time() - started, metric_count=49 * len(STATES), raw_distance_archive='raw_distances.npz',
               protocol='same frozen dev-selected checkpoint of the specified model; states close expert/conditional messages/psi together; all49 query/gallery retained sets; four-state retrieval closes experts on both sides; block-only cosine diagnostics are base00 private/shared, not deployment descriptors',
               limits='single-dataset seed42 frozen mechanism diagnostic; not retraining, new method, causal interaction or final all3 acceptance'))


if __name__ == '__main__':
    main()
