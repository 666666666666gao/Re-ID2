"""CUDA parity with M2b and direct task/regression gradient isolation checks."""
import argparse
import gc
from pathlib import Path

import torch

from experiment_data import make_loader, seed_all, split_records
from identity_alignment_axis import build as parent_build
from layers.make_loss import make_loss
from layers.softmax_loss import CrossEntropyLabelSmooth
from measurement_gate_axis import build
from run_mass_experiment import configuration, write_json
from run_shared_identity_experiment import PARTIAL_SETS
from shared_identity_axis import partial_gallery_triplet


def main():
    parser = argparse.ArgumentParser()
    for name in ('data-root', 'pretrained', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    args.dataset, args.seed, args.pooling = 'MSVR310', 42, 'original_mean'
    args.relation_weight, args.alignment_weight, args.alignment_temperature = .1, .1, .07
    torch.set_num_threads(4)
    out = Path(args.output); out.mkdir(exist_ok=False)
    cfg = configuration(args)
    fit, dev, _, classes, cameras = split_records(args.data_root, args.dataset)
    image, _, cam, scene, _ = next(iter(make_loader(dev[:8], cfg, False, args.seed)))
    image = {k: v.cuda() for k, v in image.items()}; cam, scene = cam.cuda(), scene.cuda()
    loss_fn, _ = make_loss(cfg, classes)
    xent = CrossEntropyLabelSmooth(classes)
    counts, checks = {}, []
    for variant in ('axis_shared', 'frequency_shared', 'twins_shared'):
        args.variant = variant
        parent = parent_build(args, cfg, classes, cameras).eval()
        model = build(args, cfg, classes, cameras).eval()
        assert set(model.state_dict()) == set(parent.state_dict())
        assert all(torch.equal(v, parent.state_dict()[n]) for n, v in model.state_dict().items())
        counts[variant] = dict(parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        for retained in (*PARTIAL_SETS, (0, 1, 2)):
            partial = {k: v if i in retained else torch.zeros_like(v) for i, (k, v) in enumerate(image.items())}
            with torch.no_grad():
                expected = parent(partial, cam_label=cam, view_label=scene, return_states=True)
                actual = model(partial, cam_label=cam, view_label=scene, return_states=True)
            assert all(torch.equal(actual[s], expected[s]) for s in actual)
            checks.append(dict(variant=variant, retained=retained, all_four_states_exact=True))
        images, labels, cam_fit, scene_fit, names = next(iter(make_loader(fit, cfg, True, args.seed)))
        images = {k: v.cuda() for k, v in images.items()}
        labels, cam_fit, scene_fit = labels.cuda(), cam_fit.cuda(), scene_fit.cuda()
        parent.train(); model.train()
        parent.alignment_names = model.alignment_names = tuple(names)
        with torch.no_grad(), torch.autocast('cuda'):
            seed_all(1234)
            expected = parent(images, label=labels, cam_label=cam_fit, view_label=scene_fit)
            seed_all(1234)
            actual = model(images, label=labels, cam_label=cam_fit, view_label=scene_fit)
        assert len(expected) == len(actual) and all(torch.equal(a, b) for a, b in zip(actual, expected))
        checks.append(dict(variant=variant, full_training_forward_and_all_loss_values_exact=True))
        del parent, expected, actual
        gc.collect(); torch.cuda.empty_cache()
        with torch.autocast('cuda'):
            values = model(images, label=labels, cam_label=cam_fit, view_label=scene_fit)
            assert len(values) % 2 == 1
            task = sum(model.loss_weights[i // 2] * loss_fn(values[i], values[i + 1], labels, cam_fit)
                       for i in range(0, len(values) - 1, 2))
        predictor = tuple(model.calibrator.predictor.parameters())
        gradient = torch.autograd.grad(task * 512, predictor, allow_unused=True, retain_graph=True)
        assert all(g is None for g in gradient)
        regression = torch.autograd.grad(values[-1] * 512, predictor)
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in regression)
        checks.append(dict(variant=variant, full_task_predictor_gradients_all_None=True,
            weighted_contribution_predictor_gradients_finite_nonzero=True,
            regression_gradient_L2=float(torch.cat([g.float().flatten() / 512 for g in regression]).norm()),
            relation_weight=model.relation_weight, alignment_weight=model.alignment_weight,
            target_requires_grad=model.contribution_audit['target_requires_grad']))
        gallery = values[1].detach()
        del values, task, gradient, regression
        model.last_frequency_relation_loss = None
        model.last_identity_alignment_loss = None
        gc.collect(); torch.cuda.empty_cache()
        for retained in PARTIAL_SETS:
            partial = {k: v if i in retained else torch.zeros_like(v) for i, (k, v) in enumerate(images.items())}
            with torch.autocast('cuda'):
                score, query = model(partial, label=labels, cam_label=cam_fit, view_label=scene_fit, partial=True)
                triplet, _, _ = partial_gallery_triplet(query, gallery, labels)
                partial_task = .25 * xent(score, labels) + .5 * triplet
            gradient = torch.autograd.grad(partial_task * 512, predictor, allow_unused=True)
            assert all(g is None for g in gradient)
            checks.append(dict(variant=variant, retained=retained, partial_task_predictor_gradients_all_None=True))
            del score, query, triplet, partial_task, gradient
        del model, gallery, predictor
        gc.collect(); torch.cuda.empty_cache()
    assert len({tuple(v.values()) for v in counts.values()}) == 1
    write_json(out / 'result.json', dict(status='PASS_MEASUREMENT_GATE_GRADIENT_CONTRACT', checks=checks,
        parameter_counts=counts, inference_and_full_loss_values_equal_M2b=True,
        full_and_all_partial_task_gradient_blocked=True, regression_gradient_real_nonzero=True,
        optimizer_updates=0, official_test_uses=0,
        limits='Initial same-state CUDA/gradient checks only; no retrieval improvement claim. Partial training still has no contribution-regression target.'))
    print('MEASUREMENT_GATE_GRADIENT_CONTRACT_PASS', flush=True)


if __name__ == '__main__':
    main()
