"""Real-data V5 routing checks before fresh development training."""
import argparse

import torch
from torch.nn import functional as F

from dual_axis import RELATIONS
from experiment_data import make_loader, seed_all, split_records
from mass_axis_collaboration import MassAxisCollaborationDeMo
from run_experiment import build, configuration, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('MSVR310', 'RGBNT201', 'RGBNT100'), required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    args.seed, args.contribution_weight = 42, .05
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    cfg = configuration(args)
    fit, dev, _, classes, cameras = split_records(args.data_root, args.dataset)
    seed_all(args.seed)
    model = MassAxisCollaborationDeMo(classes, cfg, cameras).float().cuda().eval()
    initial = model.state_dict()
    count = (sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad))
    controls = {}
    for variant in ('axis_scaled_fullref', 'plain_scaled_fullref', 'frequency_scaled_fullref'):
        args.variant = variant
        old = build(args, cfg, classes, cameras).eval()
        assert count == (sum(p.numel() for p in old.parameters()), sum(p.numel() for p in old.parameters() if p.requires_grad))
        assert all(torch.equal(value, old.state_dict()[name]) for name, value in initial.items())
        controls[variant] = {'initial_state_equal': True, 'parameter_count': count}
        if variant == 'axis_scaled_fullref':
            reference = old
    images, _, cam, scene, _ = next(iter(make_loader(dev[:8], cfg, False, args.seed)))
    images = {key: value.cuda() for key, value in images.items()}
    captured = []
    hook = model.calibrator.register_forward_pre_hook(lambda module, inputs: captured.append(inputs))
    with torch.no_grad():
        full = model(images, cam_label=cam.cuda(), view_label=scene.cuda())
    hook.remove()
    assert full.shape == (8, 5632) and torch.isfinite(full).all() and len(captured) == 1
    base, modality, frequency = captured[0]
    eligible = torch.ones((8, 7), dtype=torch.bool, device=base.device)
    gates = torch.ones((8, 3), device=base.device)
    with torch.no_grad():
        original = reference.fuse(base, modality, frequency, gates, True, True, eligible)
        uniform = model.fuse(base, modality, frequency, gates, True, True, eligible, torch.full((8, 7), 1 / 7, device=base.device))
    assert torch.equal(original, uniform)

    # Fix both evidence tensors, c_b and psi. Perturb only the actual a_S output;
    # inspect M-channel retrieval residuals with F/I outer gates closed.
    seed_all(args.seed)
    u = torch.randn(8, 7, 3, 512, device=base.device)
    v = torch.randn(8, 3, 3, 512, device=base.device)
    only_m_gate = torch.zeros_like(gates)
    only_m_gate[:, 0] = 1
    with torch.no_grad():
        m1, f1, pi1 = model.router.route(u, v, eligible)
        shift = torch.linspace(-2, 2, 7, device=base.device)[None, :, None]
        hook = model.router.relation_score.register_forward_hook(lambda module, inputs, output: output + shift)
        m2, f2, pi2 = model.router.route(u, v, eligible)
        hook.remove()
        residual1 = model.fuse(base, m1, f1, only_m_gate, True, True, eligible, pi1.sum(2)) - base
        residual2 = model.fuse(base, m2, f2, only_m_gate, True, True, eligible, pi2.sum(2)) - base
    conditional_error = float((m1 - m2).abs().max())
    routed_change = float((residual1[:, 1536:5120] - residual2[:, 1536:5120]).abs().max())
    assert conditional_error < 2e-6 and routed_change > 1e-4
    m, f, pi = model.router.route(u, v, eligible)
    residual = model.fuse(base, m, f, only_m_gate, True, True, eligible, pi.sum(2)) - base
    probe = torch.randn_like(residual[:, 1536:5120])
    gradient = torch.autograd.grad((residual[:, 1536:5120] * probe).sum(), model.router.relation_score.weight)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0

    # Intervene on the closed expert's independent evidence. These two states
    # must retain exact outputs without accessing any cached full-state route.
    available = torch.ones((8, 3), dtype=torch.bool, device=base.device)
    with torch.no_grad():
        states = model.controlled_states(base, u, v, eligible, available, full)
        changed_f = model.controlled_states(base, u, -v, eligible, available, full)
        changed_m = model.controlled_states(base, -u, v, eligible, available, full)
    assert torch.equal(states['10'], changed_f['10']) and torch.equal(states['01'], changed_m['01'])
    assert torch.equal(states['00'], base)
    missing = {}
    for mask in ('r', 'n', 't', 'rn', 'rt', 'nt'):
        damaged = {key: value.clone() for key, value in images.items()}
        for letter in mask:
            damaged[{'r': 'RGB', 'n': 'NI', 't': 'TI'}[letter]].zero_()
        present = torch.stack([damaged[key].flatten(1).ne(0).any(1) for key in ('RGB', 'NI', 'TI')], 1)
        allowed = torch.stack([present[:, subset].all(1) for subset in RELATIONS], 1)
        with torch.no_grad():
            feature = model(damaged, cam_label=cam.cuda(), view_label=scene.cuda())
        assert torch.isfinite(feature).all() and model.last_route[~allowed].count_nonzero() == 0
        weight = model.last_route.sum(2) * allowed.sum(1, keepdim=True)
        mean = weight.sum(1) / allowed.sum(1)
        assert torch.allclose(mean, torch.ones_like(mean), atol=1e-6, rtol=0)
        missing[mask] = {'finite': True, 'invalid_relation_mass': 0, 'mean_eligible_weight_max_error': float((mean - 1).abs().max())}
    seed_all(args.seed)
    train_images, labels, train_cam, train_scene, _ = next(iter(make_loader(fit, cfg, True, args.seed)))
    train_images = {key: value.cuda() for key, value in train_images.items()}
    model.train()
    with torch.autocast('cuda'):
        output = model(train_images, label=labels.cuda(), cam_label=train_cam.cuda(), view_label=train_scene.cuda())
    assert torch.isfinite(output[-1]) and not model.contribution_audit['target_requires_grad']
    assert any(abs(value) > 0 for row in model.contribution_audit['targets'] for value in row)
    write_json(args.output, {'status': 'PASS', 'dataset': args.dataset, 'controls': controls,
        'descriptor_dim': 5632, 'uniform_mass_exact_V4_fuse_equivalence': True,
        'fixed_evidence_aS_intervention_conditional_m_max_error': conditional_error,
        'fixed_evidence_aS_intervention_M_residual_max_change': routed_change,
        'aS_M_residual_gradient_abs_sum': float(gradient.abs().sum()),
        'closed_F_evidence_cannot_change_state10': True, 'closed_M_evidence_cannot_change_state01': True,
        'missing': missing, 'semantic_training_forward_batches': 1, 'optimizer_updates': 0,
        'target_requires_grad': False, 'reference': model.contribution_audit['reference'],
        'scope': '8 actual development triplets, six masks, controlled evidence intervention and one B64 semantic fit forward. No efficacy/test or optimizer updates.'})
    print('MASS_AXIS_TENSOR_CONTRACT_PASS', args.dataset, flush=True)


if __name__ == '__main__':
    main()
