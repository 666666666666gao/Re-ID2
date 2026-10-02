"""Real-data checks of V4 relative residuals and the common full reference."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from diagnose_axis_collaboration import extract_states
from dual_axis import RELATIONS
from experiment_data import make_loader, seed_all, split_records
from run_experiment import build, configuration, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('RGBNT201', 'RGBNT100', 'MSVR310'), required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    args.seed, args.variant, args.contribution_weight = 42, 'axis_scaled_fullref', .05
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    cfg = configuration(args)
    fit, dev, _, classes, cameras = split_records(args.data_root, args.dataset)
    models = []
    variants = ('axis_scaled_fullref', 'plain_scaled_fullref', 'frequency_scaled_fullref')
    for variant in variants:
        args.variant = variant
        models.append(build(args, cfg, classes, cameras).eval())
    counts = [{ 'parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)} for model in models]
    assert counts[0] == counts[1] == counts[2]
    initial = models[0].state_dict()
    assert all(all(torch.equal(value, model.state_dict()[name]) for name, value in initial.items()) for model in models[1:])
    args.variant = 'demo'
    baseline = build(args, cfg, classes, cameras).eval()
    assert all(torch.equal(value, initial[name]) for name, value in baseline.state_dict().items())
    images, _, cam, scene, _ = next(iter(make_loader(dev[:8], cfg, False, args.seed)))
    images = {key: value.cuda() for key, value in images.items()}
    with torch.no_grad():
        original = baseline(images, cam_label=cam.cuda(), view_label=scene.cuda())
        base = F.normalize(torch.cat((original, torch.zeros_like(original[:, :512])), 1), dim=1).cpu()
    shapes = []
    for model in models:
        features, _, _, _, _ = extract_states(model, dev[:8], cfg, args.seed)
        assert features['11'].shape == (8, 5632) and all(torch.isfinite(value).all() for value in features.values())
        assert torch.equal(features['00'], base)
        shapes.append(list(features['11'].shape))
    model = models[0]
    observed_gates = []
    hook = model.calibrator.register_forward_hook(lambda module, inputs, result: observed_gates.append(result[1].detach()))
    features, _, _, _, _ = extract_states(model, dev[:8], cfg, args.seed)
    hook.remove()
    assert len(observed_gates) == 3
    actual_ratio = features['01'][:, -512:].norm(dim=1) / features['01'][:, :-512].norm(dim=1)
    expected_ratio = (model.residual_scale[1].abs() * observed_gates[2][:, 1]).cpu()
    ratio_error = float((actual_ratio - expected_ratio).abs().max())
    assert ratio_error < 1e-6
    with torch.no_grad():
        scales = model.residual_scale.clone()
        model.residual_scale.zero_()
        disabled = model(images, cam_label=cam.cuda(), view_label=scene.cuda())
        assert torch.equal(F.normalize(disabled, dim=1).cpu(), base)
        model.residual_scale.copy_(scales)
    missing = {}
    for mask in ('r', 'n', 't', 'rn', 'rt', 'nt'):
        damaged = {key: value.clone() for key, value in images.items()}
        for letter in mask:
            damaged[{'r':'RGB', 'n':'NI', 't':'TI'}[letter]].zero_()
        available = torch.stack([damaged[key].flatten(1).ne(0).any(1) for key in ('RGB','NI','TI')], 1)
        eligible = torch.stack([available[:, subset].all(1) for subset in RELATIONS], 1)
        with torch.no_grad():
            result = model(damaged, cam_label=cam.cuda(), view_label=scene.cuda())
        assert torch.isfinite(result).all() and model.last_route[~eligible].count_nonzero() == 0
        missing[mask] = {'finite':True, 'invalid_relation_route_mass':float(model.last_route[~eligible].sum())}
    captured = {}
    states_method = model.controlled_states

    def states(*values, **keywords):
        result = states_method(*values, **keywords)
        captured.update(result)
        return result

    model.controlled_states = states
    seed_all(args.seed)
    batch = next(iter(make_loader(fit, cfg, True, args.seed)))
    images, labels, cam, scene, _ = batch
    images = {key: value.cuda() for key, value in images.items()}
    model.train()
    with torch.autocast('cuda'):
        model(images, label=labels.cuda(), cam_label=cam.cuda(), view_label=scene.cuda())
    model.controlled_states = states_method
    target, pos, neg, scores = model.calibrator.targets(captured, labels.cuda())
    assert not target.requires_grad and torch.isfinite(target).all() and target.count_nonzero() > 0
    reference = captured['11'].detach().float().cpu().numpy().astype(np.float64)
    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
    pos_, neg_ = pos.cpu().numpy(), neg.cpu().numpy()
    label_ = labels.numpy()
    assert np.all(label_[pos_] == label_) and np.all(pos_ != np.arange(len(label_))) and np.all(label_[neg_] != label_)
    direction = reference[pos_] - reference[neg_]
    cpu_scores = {}
    for key, value in captured.items():
        array = value.detach().float().cpu().numpy().astype(np.float64)
        array /= np.linalg.norm(array, axis=1, keepdims=True)
        cpu_scores[key] = np.sum(array * direction, axis=1)
    cpu_target = np.stack((cpu_scores['11'] - cpu_scores['01'], cpu_scores['11'] - cpu_scores['10'],
                           cpu_scores['11'] - cpu_scores['10'] - cpu_scores['01'] + cpu_scores['00']), 1)
    target_error = float(np.abs(cpu_target - target.cpu().numpy()).max())
    assert target_error < 2e-6
    write_json(output/'tensor_contract.json', {'status':'PASS', 'dataset':args.dataset, 'variants':variants,
        'equal_parameters':counts, 'all_initial_state_tensors_equal':True, 'original_DeMo_initial_tensors_equal':True,
        'descriptor_shapes':shapes, 'base00_and_zero_scales_equal_original_DeMo':True,
        'F_only_relative_norm_max_error':ratio_error, 'missing':missing,
        'full_reference_target_CPU_max_error':target_error, 'target_requires_grad':False,
        'full_reference_tail_mean_norm':float(torch.linalg.vector_norm(F.normalize(captured['11'].detach().float(),dim=1)[:,-512:],dim=1).mean()),
        'target_mean':target.mean(0).tolist(), 'target_std':target.std(0).tolist(),
        'target_nonzero_count':int(target.count_nonzero()), 'reference':model.contribution_audit['reference'],
        'semantic_training_forward_batches':1, 'optimizer_updates':0,
        'scope':'Initial actual8dev +64fit tensor checks; semantic fit forward updates training BN buffers, no optimizer; no efficacy or official test'})
    print('SCALED_AXIS_TENSOR_CONTRACT_PASS', args.dataset, flush=True)


if __name__ == '__main__':
    main()
