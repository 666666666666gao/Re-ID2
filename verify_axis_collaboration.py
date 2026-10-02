"""Actual-data witnesses for equal capacity, information access, and calibration."""
import argparse
import torch
from torch.nn import functional as F

from axis_collaboration import ContributionCalibrator
from dual_axis import RELATIONS
from experiment_data import split_records, make_loader
from run_experiment import build, configuration, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['RGBNT201', 'RGBNT100', 'MSVR310'], required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    args.seed, args.contribution_weight = 42, .05
    torch.set_num_threads(4)
    cfg = configuration(args)
    fit, dev, _, classes, cameras = split_records(args.data_root, args.dataset)
    models = {}
    for variant in ('demo', 'axis_collaboration', 'plain_twins', 'ordinary_frequency'):
        args.variant = variant
        models[variant] = build(args, cfg, classes, cameras).eval()
    axis = models['axis_collaboration']
    initial = axis.state_dict()
    for variant in ('plain_twins', 'ordinary_frequency'):
        state = models[variant].state_dict()
        assert state.keys() == initial.keys() and all(torch.equal(state[key], initial[key]) for key in state)
    assert all(torch.equal(value, initial[key]) for key, value in models['demo'].state_dict().items())
    counts = {key: sum(p.numel() for p in model.parameters()) for key, model in models.items()}
    active = {key: sum(p.numel() for p in model.parameters() if p.requires_grad) for key, model in models.items()}
    assert len({counts[key] for key in models if key != 'demo'}) == 1
    assert len({active[key] for key in models if key != 'demo'}) == 1
    images, _, cam, scene, _ = next(iter(make_loader(dev[:8], cfg, False, args.seed)))
    images = {key: value.cuda() for key, value in images.items()}
    kw = {'cam_label': cam.cuda(), 'view_label': scene.cuda()}
    reference = models['demo'](images, **kw)
    normal = {key: model(images, **kw) for key, model in models.items() if key != 'demo'}
    assert all(feature.shape == (len(cam), 5632) and torch.isfinite(feature).all() for feature in normal.values())
    assert torch.equal(normal['ordinary_frequency'][:, :5120], reference)
    route = axis.last_route
    joint_error = float((route - route.sum(2, keepdim=True) * route.sum(1, keepdim=True)).abs().sum((1, 2)).mean())
    assert joint_error > 1e-8
    axis.joint_interaction = False
    axis(images, **kw)
    independent = axis.last_route
    factor_error = float((independent - independent.sum(2, keepdim=True) * independent.sum(1, keepdim=True)).abs().sum((1, 2)).mean())
    assert factor_error < 1e-6
    axis.joint_interaction = True
    scales = axis.residual_scale.clone()
    axis.residual_scale.zero_()
    anchor = axis(images, **kw)
    assert torch.equal(anchor[:, :5120], reference) and torch.count_nonzero(anchor[:, 5120:]) == 0
    axis.residual_scale.copy_(scales)
    available = torch.ones((len(cam), 3), device='cuda', dtype=torch.bool)
    spatial = torch.stack([axis.BACKBONE(images[key], **kw)[0] for key in ('RGB', 'NI', 'TI')], 1)
    tokens = axis.bands(spatial)
    masks = axis.bands.masks
    height, width = masks.shape[-2:]
    reflected = masks[:, (-torch.arange(height, device='cuda')) % height][:, :, (-torch.arange(width, device='cuda')) % width]
    assert torch.equal(masks, reflected) and (masks.sum(0) == 1).all()
    reconstruction = float((tokens.sum(2) - spatial).abs().max())
    assert torch.allclose(tokens.sum(2), spatial, atol=5e-5, rtol=1e-5)
    eligible = torch.ones((len(cam), 7), device='cuda', dtype=torch.bool)
    u = axis.modality_expert.read(axis.modality_expert.prepare(tokens, available), eligible, True)
    v = axis.frequency_expert.read(axis.frequency_expert.prepare(tokens, available, True), available, True)
    changed = tokens.clone()
    changed[:, 2, 2] += torch.randn_like(changed[:, 2, 2])
    altered_u = axis.modality_expert.read(axis.modality_expert.prepare(changed, available), eligible, True)
    altered_v = axis.frequency_expert.read(axis.frequency_expert.prepare(changed, available, True), available, True)
    protected = [s for s, subset in enumerate(RELATIONS) if 2 not in subset]
    assert torch.equal(u[:, protected], altered_u[:, protected])
    assert torch.equal(u[:, :, :2], altered_u[:, :, :2])
    assert torch.equal(v[:, :2], altered_v[:, :2])
    affected_u = float((u[:, 2, 2] - altered_u[:, 2, 2]).abs().max())
    affected_v = float((v[:, 2] - altered_v[:, 2]).abs().max())
    assert affected_u > 1e-7 and affected_v > 1e-7
    generic = models['plain_twins']
    full = spatial[:, :, None].expand(-1, -1, 3, -1, -1)
    generic_u = generic.modality_expert.read(generic.modality_expert.prepare(full, available), eligible, False)
    generic_changed = full.clone()
    generic_changed[:, 2] += torch.randn_like(generic_changed[:, 2])
    generic_altered = generic.modality_expert.read(generic.modality_expert.prepare(generic_changed, available), eligible, False)
    generic_sensitivity = float((generic_u[:, 0] - generic_altered[:, 0]).abs().max())
    assert generic_sensitivity > 1e-7
    missing = {}
    for code in ('r', 'n', 't', 'rn', 'rt', 'nt'):
        masked = dict(images)
        for letter, key in zip('rnt', ('RGB', 'NI', 'TI')):
            if letter in code:
                masked[key] = torch.zeros_like(images[key])
        feature = axis(masked, **kw)
        flags = torch.tensor([letter not in code for letter in 'rnt'], device='cuda')
        valid = torch.tensor([bool(flags[list(subset)].all()) for subset in RELATIONS], device='cuda')
        assert torch.isfinite(feature).all() and torch.count_nonzero(axis.last_route[:, ~valid]) == 0
        assert torch.allclose(axis.last_route.sum((1, 2)), torch.ones(len(cam), device='cuda'))
        missing[code] = {'finite': True, 'invalid_relation_route_mass': 0., 'eligible_relations': valid.tolist()}
    # Capture actual production states. The backbone runs three times once;
    # the four contribution states do not rerun it or update its BN statistics.
    batch = next(iter(make_loader(fit, cfg, True, args.seed)))
    train_images, labels, train_cam, train_scene, _ = batch
    train_images = {key: value.cuda() for key, value in train_images.items()}
    captured, backbone_calls = {}, []
    original_states = axis.controlled_states

    def capture(*values):
        states = original_states(*values)
        captured['values'] = values
        captured['states'] = {key: value.detach().float() for key, value in states.items()}
        return states

    axis.controlled_states = capture
    hook = axis.BACKBONE.register_forward_hook(lambda *values: backbone_calls.append(1))
    axis.train()
    with torch.autocast('cuda'):
        output = axis(train_images, label=labels.cuda(), cam_label=train_cam.cuda(), view_label=train_scene.cuda())
    hook.remove()
    axis.controlled_states = original_states
    assert len(backbone_calls) == 3 and torch.isfinite(output[-1])
    states = captured['states']
    target, pos, neg, scores = ContributionCalibrator.targets(states, labels.cuda())
    ref = F.normalize(states['00'].cpu(), dim=1)
    direction = ref[pos.cpu()] - ref[neg.cpu()]
    independently_scored = {key: (F.normalize(feature.cpu(), dim=1) * direction).sum(1) for key, feature in states.items()}
    expected = torch.stack((independently_scored['11'] - independently_scored['01'],
                            independently_scored['11'] - independently_scored['10'],
                            independently_scored['11'] - independently_scored['10'] - independently_scored['01'] + independently_scored['00']), 1)
    error = float((target.cpu() - expected).abs().max())
    assert error < 1e-6
    assert labels[pos.cpu()].eq(labels).all() and labels[neg.cpu()].ne(labels).all()
    with torch.enable_grad():
        differentiable = {key: value.clone().requires_grad_() for key, value in states.items()}
        stopped, _, _, _ = ContributionCalibrator.targets(differentiable, labels.cuda())
        assert not stopped.requires_grad
    conditions, routing = axis.router.conditions, axis.router.route

    def forbidden(*values, **kwargs):
        raise AssertionError('closed state or inference evaluated a forbidden operation')

    axis.router.conditions, axis.router.route = forbidden, forbidden
    with torch.autocast('cuda'):
        closed = original_states(*captured['values'])
    assert all(torch.equal(closed[key].float(), states[key]) for key in states)
    axis.router.conditions, axis.router.route = conditions, routing
    axis.eval()
    axis.controlled_states = forbidden
    inference = axis(images, **kw)
    assert torch.isfinite(inference).all()
    axis.controlled_states = original_states
    receipt = {'status': 'PASS', 'dataset': args.dataset, 'actual_dev_samples': len(cam),
               'parameters': counts, 'trainable_parameters': active, 'matched_full_initial_state': True,
               'original_DeMo_initial_state_equal': True, 'descriptor_dim': 5632,
               'ordinary_frequency_base_bitwise_equal': True, 'axis_zero_residual_base_bitwise_equal': True,
               'FFT_partition_and_symmetry': True, 'FFT_reconstruction_max_error': reconstruction,
               'outside_source_and_band_unchanged': True, 'inside_source_band_max_change': affected_u,
               'frequency_other_source_unchanged': True, 'frequency_own_source_max_change': affected_v,
               'plain_twin_outside_source_sensitivity': generic_sensitivity,
               'joint_nonfactor_L1': joint_error, 'psi_off_factor_error_L1': factor_error,
               'six_actual_missing_checks': missing, 'training_batch': len(labels), 'backbone_calls_for_four_states': len(backbone_calls),
               'same_actual_references': True, 'contribution_independent_CPU_max_error': error,
               'target_stops_gradient': True, 'closed_states_without_router_calls': True,
               'inference_without_counterfactuals': True, 'optimizer_updates': 0,
               'limits': 'Initialization/access/controlled-state witnesses on 8 actual dev and one actual training batch; no retrieval improvement or complete-dataset parity claim.'}
    from pathlib import Path
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    write_json(output/'verification.json', receipt)
    print('AXIS_COLLABORATION_VERIFY_PASS', receipt, flush=True)


if __name__ == '__main__':
    main()
