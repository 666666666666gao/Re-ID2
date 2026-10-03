"""Frozen development four-state retrieval and contribution diagnostic, zero updates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from dual_axis import RELATIONS
from experiment_data import make_loader, split_records
from full_evaluation import distance, full_metrics
from run_mass_experiment import build, configuration, write_json


@torch.no_grad()
def extract_states(model, records, cfg, seed):
    features = {key: [] for key in ('00', '10', '01', '11')}
    routes, gates, predictions, norms = [], [], [], []
    independent_m, independent_f, calibrator_inputs = [], [], []
    read_m, read_f = model.modality_expert.read, model.frequency_expert.read

    def modality_read(*args, **kwargs):
        result = read_m(*args, **kwargs)
        independent_m.append(result)
        return result

    def frequency_read(*args, **kwargs):
        result = read_f(*args, **kwargs)
        independent_f.append(result)
        return result

    model.modality_expert.read = modality_read
    model.frequency_expert.read = frequency_read
    hook = model.calibrator.register_forward_pre_hook(lambda module, inputs: calibrator_inputs.append(inputs))
    for images, _, cam, scene, _ in make_loader(records, cfg, False, seed):
        independent_m.clear(); independent_f.clear(); calibrator_inputs.clear()
        images = {key: value.cuda(non_blocking=True) for key, value in images.items()}
        full = model(images, cam_label=cam.cuda(), view_label=scene.cuda())
        assert len(independent_m) == 2 and len(independent_f) == 2 and len(calibrator_inputs) == 1
        base = calibrator_inputs[0][0]
        available = torch.stack([images[key].flatten(1).ne(0).any(1) for key in ('RGB', 'NI', 'TI')], 1)
        assert available.all(), 'this diagnostic uses clean development triplets'
        eligible = torch.stack([available[:, subset].all(1) for subset in RELATIONS], 1)
        routes.append(model.last_route.cpu())
        gates.append(model.last_gates.cpu())
        predictions.append(model.last_contribution_prediction.float().cpu())
        states = model.controlled_states(base, independent_m[0], independent_f[0], eligible, available, full)
        assert len(calibrator_inputs) == 3
        norms.append(torch.stack((base.norm(dim=1), (states['10'] - base).norm(dim=1),
                                  (states['01'] - base).norm(dim=1), (full - base).norm(dim=1)), 1).cpu())
        for key, value in states.items():
            features[key].append(F.normalize(value.float(), dim=1).cpu())
    hook.remove()
    model.modality_expert.read, model.frequency_expert.read = read_m, read_f
    return {key: torch.cat(value) for key, value in features.items()}, torch.cat(routes), torch.cat(gates), torch.cat(predictions), torch.cat(norms)


def reference_contributions(features, query, ids, exclusion, reference_key):
    reference = features[reference_key]
    similarity = reference[query] @ reference.T
    positive = (ids[query, None] == ids[None]) & (exclusion[query, None] != exclusion[None])
    negative = ids[query, None] != ids[None]
    assert positive.any(1).all() and negative.any(1).all()
    pos = similarity.masked_fill(~torch.from_numpy(positive), torch.inf).argmin(1)
    neg = similarity.masked_fill(~torch.from_numpy(negative), -torch.inf).argmax(1)
    direction = reference[pos] - reference[neg]
    scores = {key: (value[query] * direction).sum(1) for key, value in features.items()}
    target = torch.stack((scores['11'] - scores['01'], scores['11'] - scores['10'],
                          scores['11'] - scores['10'] - scores['01'] + scores['00']), 1)
    return target.numpy(), pos.numpy(), neg.numpy(), {key: value.numpy() for key, value in scores.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--data-root')
    parser.add_argument('--pretrained')
    args = parser.parse_args()
    run, output = Path(args.run_dir), Path(args.output)
    terminal = json.loads((run / 'result.json').read_text())
    assert terminal['status'] == 'COMPLETE' and terminal['epochs'] == 50
    arguments = argparse.Namespace(**terminal['arguments'])
    assert arguments.variant == 'axis_mass_fullref'
    exit_file = run.parent / (run.name + '_exit.json')
    assert json.loads(exit_file.read_text())['exit_code'] == 0
    assert (args.data_root is None) == (args.pretrained is None), 'cross-host data and pretrained paths must be supplied together'
    original_paths = {'data_root': arguments.data_root, 'pretrained': arguments.pretrained}
    if args.data_root is not None:
        arguments.data_root, arguments.pretrained = args.data_root, args.pretrained
    output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    cfg = configuration(arguments)
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
    started = time.time()
    # Match the saved inference batch shape for exact feature parity.
    records = dev[:64] if args.smoke else dev
    features, routes, gates, predictions, norms = extract_states(model, records, cfg, arguments.seed)
    assert versions == {name: value._version for name, value in model.state_dict().items()}
    assert all(torch.isfinite(value).all() for value in (*features.values(), routes, gates, predictions, norms))
    if args.smoke:
        parity = float(np.abs(features['11'].numpy() - saved['features'][:64]).max())
        assert parity == 0
        write_json(output / 'smoke.json', {'status': 'PASS', 'triplets': len(records), 'normal_feature_max_error': parity,
                                         'all_states_finite': True, 'state_tensor_versions_unchanged': True,
                                         'optimizer_updates': 0, 'installed_ground_truth_order_equal': True,
                                         'original_paths': original_paths,
                                         'execution_paths': {'data_root': arguments.data_root, 'pretrained': arguments.pretrained},
                                         'scope': 'Actual first64 clean development triplets, same inference batch shape as frozen arrays, no metric inference'})
        print('FOUR_STATE_SMOKE_PASS', arguments.dataset, flush=True)
        return
    query = np.asarray(query)
    assert np.array_equal(query, saved['query_indices'])
    assert np.array_equal(saved['ids'], [row[1] for row in dev])
    parity = float(np.abs(features['11'].numpy() - saved['features']).max())
    assert parity == 0, 'normal FP32 inference must reproduce frozen development features exactly'
    ids, cameras_, scenes, names = (saved[key] for key in ('ids', 'cameras', 'scenes', 'names'))
    exclusion = scenes if arguments.dataset == 'MSVR310' else cameras_
    metrics, per_query = {}, {}
    for key, value in features.items():
        distances = saved['distances'] if key == '11' else distance(value[query], value)
        metrics[key] = full_metrics(distances, ids[query], ids, exclusion[query], exclusion,
                                    names[query], cameras_[query], scenes[query], output / ('state_' + key))
        with (output / ('state_' + key + '.csv')).open(encoding='utf-8', newline='') as table:
            per_query[key] = list(csv.DictReader(table))
    assert all(abs(metrics['11'][key] - terminal['best'][key]) < 1e-8 for key in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10'))
    targets, pos, neg, scores = reference_contributions(features, query, ids, exclusion, '00')
    full_targets, full_pos, full_neg, full_scores = reference_contributions(features, query, ids, exclusion, '11')
    prediction = predictions.numpy()[query]
    columns = ('delta_M_given_F', 'delta_F_given_M', 'empirical_interaction')
    calibration = {}
    for index, key in enumerate(columns):
        target, estimate = targets[:, index], prediction[:, index]
        calibration[key] = {'target_mean': float(target.mean()), 'target_std': float(target.std()),
                            'prediction_mean': float(estimate.mean()), 'prediction_std': float(estimate.std()),
                            'MAE': float(np.abs(estimate - target).mean()), 'RMSE': float(np.sqrt(np.square(estimate - target).mean())),
                            'positive_target_fraction': float((target > 0).mean()),
                            'sign_agreement_fraction': float((np.sign(estimate) == np.sign(target)).mean()),
                            'prediction_target_covariance': float(np.mean((target - target.mean()) * (estimate - estimate.mean())))}
    full_calibration = {}
    for index, key in enumerate(columns):
        target, estimate = full_targets[:, index], prediction[:, index]
        full_calibration[key] = {'target_mean': float(target.mean()), 'target_std': float(target.std()),
                                 'prediction_mean': float(estimate.mean()), 'prediction_std': float(estimate.std()),
                                 'MAE': float(np.abs(estimate - target).mean()), 'RMSE': float(np.sqrt(np.square(estimate - target).mean())),
                                 'positive_target_fraction': float((target > 0).mean()),
                                 'sign_agreement_fraction': float((np.sign(estimate) == np.sign(target)).mean()),
                                 'prediction_target_covariance': float(np.mean((target - target.mean()) * (estimate - estimate.mean())))}
    with (output / 'contributions.csv').open('w', encoding='utf-8', newline='') as table:
        fieldnames = ['query_index', 'name', 'positive_reference_index', 'negative_reference_index']
        fieldnames += ['R' + key for key in scores]
        fieldnames += [key + suffix for key in columns for suffix in ('_target', '_prediction')]
        writer = csv.DictWriter(table, fieldnames=fieldnames)
        writer.writeheader()
        for i, name in enumerate(names[query]):
            row = {'query_index': i, 'name': str(name), 'positive_reference_index': int(pos[i]), 'negative_reference_index': int(neg[i])}
            row.update({'R' + key: float(value[i]) for key, value in scores.items()})
            row.update({key + suffix: float(value[i, j]) for j, key in enumerate(columns) for suffix, value in (('_target', targets), ('_prediction', prediction))})
            writer.writerow(row)
    with (output / 'full_reference_contributions.csv').open('w', encoding='utf-8', newline='') as table:
        writer = csv.DictWriter(table, fieldnames=fieldnames)
        writer.writeheader()
        for i, name in enumerate(names[query]):
            row = {'query_index': i, 'name': str(name), 'positive_reference_index': int(full_pos[i]), 'negative_reference_index': int(full_neg[i])}
            row.update({'R' + key: float(value[i]) for key, value in full_scores.items()})
            row.update({key + suffix: float(value[i, j]) for j, key in enumerate(columns) for suffix, value in (('_target', full_targets), ('_prediction', prediction))})
            writer.writerow(row)
    harm_rescue = {}
    for key in ('00', '10', '01'):
        source = np.asarray([int(row['Rank-1']) for row in per_query[key]])
        joint = np.asarray([int(row['Rank-1']) for row in per_query['11']])
        harm_rescue[key + '_to_11'] = {'valid_queries': len(query), 'harmed_count': int(((source == 1) & (joint == 0)).sum()),
                                     'rescued_count': int(((source == 0) & (joint == 1)).sum()),
                                     'both_correct_count': int(((source == 1) & (joint == 1)).sum()),
                                     'both_wrong_count': int(((source == 0) & (joint == 0)).sum())}
    independent = routes.sum(2, keepdim=True) * routes.sum(1, keepdim=True)
    np.savez_compressed(output / 'states.npz', **{key: value.numpy() for key, value in features.items()}, query_indices=query,
                        route=routes.numpy(), gates=gates.numpy(), predictions=predictions.numpy(), norms=norms.numpy(),
                        ids=ids, cameras=cameras_, scenes=scenes, names=names, contribution_targets=targets,
                        positive_indices=pos, negative_indices=neg, full_reference_targets=full_targets,
                        full_reference_positive_indices=full_pos, full_reference_negative_indices=full_neg)
    report = {'dataset': arguments.dataset, 'variant': arguments.variant, 'seed': arguments.seed, 'best_epoch': terminal['best']['epoch'],
              'trained_contribution_reference': terminal.get('contribution_reference', 'base00'),
              'installed_ground_truth_order_equal': True, 'original_paths': original_paths,
              'execution_paths': {'data_root': arguments.data_root, 'pretrained': arguments.pretrained},
              'checkpoint_sha256': hashlib.sha256((run / 'best.pth').read_bytes()).hexdigest(),
              'scope': 'Frozen identity-heldout development diagnostic, zero updates, no official test, one shared backbone inference per batch',
              'states': '00 base;10 independent M only;01 independent F only;11 both, query conditions, joint psi and interaction projection. Disabled experts have no conditional messages/psi in10/01.',
              'contribution_reference': 'Detached development base gallery; same hardest positive and negative for every state, ground-truth identity and camera/scene exclusions. Diagnostic references differ from the current-batch training targets.',
              'additional_full_reference': 'Detached full11 development gallery; independent fixed positive/negative selection in that gallery, same indices for all4states. No training target is changed.',
              'interpretation': 'Empirical retrieval interaction, not information-theoretic synergy or a causal effect; fixed trained model interventions are not trained ablations.',
              'normal_inference_feature_max_error': parity, 'optimizer_updates': 0, 'state_tensor_versions_unchanged': True,
              'normal_distance_source': 'saved best_dev_arrays.npz/distances after exact full11 feature parity; 00/10/01 use full_evaluation.distance',
              'metrics': metrics, 'contribution_calibration': calibration, 'full_reference_contribution_calibration': full_calibration,
              'base_reference_tail_512_max_abs': float(features['00'][:, -512:].abs().max()), 'harm_rescue': harm_rescue,
              'mean_route': routes.mean(0).tolist(), 'mean_gates': gates.mean(0).tolist(),
              'mean_joint_minus_marginal_product_L1': float((routes - independent).abs().sum((1, 2)).mean()),
              'mean_norms_base_M_residual_F_residual_full_residual': norms.mean(0).tolist(),
              'wall_seconds': time.time() - started}
    write_json(output / 'diagnostic.json', report)
    print('FOUR_STATE_DEVELOPMENT_DIAGNOSTIC_COMPLETE', json.dumps({key: {metric: value[metric] for metric in ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')} for key, value in metrics.items()}), flush=True)


if __name__ == '__main__':
    main()
