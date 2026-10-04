"""Actual CUDA contract: zero-control parity and measurement/control gradients."""
import argparse
import gc
from pathlib import Path

import torch

from experiment_data import make_loader, seed_all
from independent_control_axis import build
from layers.make_loss import make_loss
from layers.softmax_loss import CrossEntropyLabelSmooth
from official_training_data import full_records
from run_experiment import configuration, write_json
from run_shared_identity_experiment import PARTIAL_SETS
from shared_identity_axis import partial_gallery_triplet


def finite_nonzero(values):
    return all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in values)


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    args.dataset, args.seed, args.pooling = 'MSVR310', 42, 'original_mean'
    args.relation_weight, args.alignment_weight, args.alignment_temperature = .1, .1, .07
    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    cfg = configuration(args)
    train, query, gallery, classes, cameras, _ = full_records(args.data_root, args.dataset)
    image, _, cam, scene, _ = next(iter(make_loader(query[:8], cfg, False, args.seed)))
    image = {key: value.cuda() for key, value in image.items()}
    cam, scene = cam.cuda(), scene.cuda()
    loss_fn, _ = make_loss(cfg, classes)
    xent = CrossEntropyLabelSmooth(classes)
    counts, checks = {}, []
    for variant in ('axis_shared', 'frequency_shared', 'twins_shared'):
        args.variant, args.gate_gradient_mode = variant, 'measurement_only'
        parent = build(args, cfg, classes, cameras).eval()
        args.gate_gradient_mode = 'independent_control'
        model = build(args, cfg, classes, cameras).eval()
        control_keys = {n for n in model.state_dict() if n.startswith('calibrator.control.')}
        assert set(model.state_dict()) - control_keys == set(parent.state_dict())
        assert all(torch.equal(v, parent.state_dict()[n]) for n, v in model.state_dict().items() if n not in control_keys)
        parent_count = sum(p.numel() for p in parent.parameters())
        parent_trainable = sum(p.numel() for p in parent.parameters() if p.requires_grad)
        counts[variant] = dict(parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            descriptor_dim=5632, added_control_parameters=27769)
        assert counts[variant]['parameters'] - parent_count == 27769
        assert counts[variant]['trainable_parameters'] - parent_trainable == 27769
        for retained in (*PARTIAL_SETS, (0, 1, 2)):
            partial = {k: v if i in retained else torch.zeros_like(v) for i, (k, v) in enumerate(image.items())}
            for amp in (False, True):
                with torch.no_grad(), torch.autocast('cuda', enabled=amp):
                    expected = parent(partial, cam_label=cam, view_label=scene, return_states=True)
                    actual = model(partial, cam_label=cam, view_label=scene, return_states=True)
                assert set(actual) == set(expected) == {'00', '10', '01', '11'}
                assert all(torch.equal(actual[key], expected[key]) for key in actual)
                checks.append(dict(variant=variant, retained=retained, AMP=amp, all_four_states_exact=True))
        images, labels, train_cam, train_scene, names = next(iter(make_loader(train, cfg, True, args.seed)))
        images = {k: v.cuda() for k, v in images.items()}
        labels, train_cam, train_scene = labels.cuda(), train_cam.cuda(), train_scene.cuda()
        parent.train(); model.train()
        parent.alignment_names = model.alignment_names = tuple(names)
        with torch.no_grad(), torch.autocast('cuda'):
            seed_all(1234)
            expected = parent(images, label=labels, cam_label=train_cam, view_label=train_scene)
            seed_all(1234)
            actual = model(images, label=labels, cam_label=train_cam, view_label=train_scene)
        assert len(expected) == len(actual) and all(torch.equal(a, b) for a, b in zip(actual, expected))
        with torch.no_grad(), torch.autocast('cuda'):
            parent_full_loss = sum(parent.loss_weights[i // 2] * loss_fn(expected[i], expected[i + 1], labels, train_cam)
                for i in range(0, len(expected) - 1, 2)) + expected[-1]
            model_full_loss = sum(model.loss_weights[i // 2] * loss_fn(actual[i], actual[i + 1], labels, train_cam)
                for i in range(0, len(actual) - 1, 2)) + actual[-1]
        assert torch.equal(parent_full_loss, model_full_loss)
        checks.append(dict(variant=variant, full_training_outputs_and_full_objective_exact=True))
        for ordinal, retained in enumerate(PARTIAL_SETS):
            partial = {k: v if i in retained else torch.zeros_like(v) for i, (k, v) in enumerate(images.items())}
            with torch.no_grad(), torch.autocast('cuda'):
                seed_all(1235 + ordinal)
                expected_score, expected_feature = parent(partial, label=labels, cam_label=train_cam, view_label=train_scene, partial=True)
                seed_all(1235 + ordinal)
                actual_score, actual_feature = model(partial, label=labels, cam_label=train_cam, view_label=train_scene, partial=True)
                expected_ce, actual_ce = xent(expected_score, labels), xent(actual_score, labels)
                expected_triplet, ep, en = partial_gallery_triplet(expected_feature, expected[1], labels)
                actual_triplet, ap, an = partial_gallery_triplet(actual_feature, actual[1], labels)
                expected_loss = .25 * expected_ce + .5 * expected_triplet
                actual_loss = .25 * actual_ce + .5 * actual_triplet
            assert torch.equal(expected_score, actual_score) and torch.equal(expected_feature, actual_feature)
            assert torch.equal(expected_ce, actual_ce) and torch.equal(expected_triplet, actual_triplet)
            assert torch.equal(expected_loss, actual_loss) and torch.equal(ep, ap) and torch.equal(en, an)
            checks.append(dict(variant=variant, retained=retained, AMP_partial_outputs_CE_triplet_objective_indices_exact=True))
        del parent, expected, actual
        del parent_full_loss, model_full_loss, expected_score, expected_feature, actual_score, actual_feature
        del expected_ce, actual_ce, expected_triplet, actual_triplet, expected_loss, actual_loss
        gc.collect(); torch.cuda.empty_cache()
        with torch.autocast('cuda'):
            values = model(images, label=labels, cam_label=train_cam, view_label=train_scene)
            task = sum(model.loss_weights[i // 2] * loss_fn(values[i], values[i + 1], labels, train_cam)
                for i in range(0, len(values) - 1, 2))
        predictor = tuple(model.calibrator.predictor.parameters())
        control = tuple(model.calibrator.control.parameters())
        predictor_task = torch.autograd.grad(task * 512, predictor, allow_unused=True, retain_graph=True)
        assert all(g is None for g in predictor_task)
        control_task = torch.autograd.grad(task * 512, control, retain_graph=True)
        assert all(torch.isfinite(g).all() for g in control_task)
        assert all(g.abs().sum() == 0 for g in control_task[:-2])
        assert finite_nonzero(control_task[-2:])
        predictor_regression = torch.autograd.grad(values[-1] * 512, predictor, retain_graph=True)
        assert finite_nonzero(predictor_regression)
        control_regression = torch.autograd.grad(values[-1] * 512, control, allow_unused=True)
        assert all(g is None for g in control_regression)
        checks.append(dict(variant=variant, full_task_predictor_gradients_all_None=True,
            regression_predictor_gradients_finite_nonzero=True, regression_control_gradients_all_None=True,
            full_task_control_output_gradients_finite_nonzero=True, initial_hidden_control_gradients_zero=True))
        full_gallery = values[1].detach()
        del values, task, predictor_task, control_task, predictor_regression, control_regression
        model.last_frequency_relation_loss = model.last_identity_alignment_loss = None
        gc.collect(); torch.cuda.empty_cache()
        for retained in PARTIAL_SETS:
            partial = {k: v if i in retained else torch.zeros_like(v) for i, (k, v) in enumerate(images.items())}
            with torch.autocast('cuda'):
                score, feature = model(partial, label=labels, cam_label=train_cam, view_label=train_scene, partial=True)
                triplet, _, _ = partial_gallery_triplet(feature, full_gallery, labels)
                partial_task = .25 * xent(score, labels) + .5 * triplet
            predictor_task = torch.autograd.grad(partial_task * 512, predictor, allow_unused=True, retain_graph=True)
            assert all(g is None for g in predictor_task)
            control_task = torch.autograd.grad(partial_task * 512, control)
            assert all(torch.isfinite(g).all() for g in control_task)
            assert finite_nonzero(control_task[-2:])
            checks.append(dict(variant=variant, retained=retained, partial_task_predictor_gradients_all_None=True,
                partial_task_control_output_gradients_finite_nonzero=True))
            del score, feature, triplet, partial_task, predictor_task, control_task
        del model, full_gallery, predictor, control
        gc.collect(); torch.cuda.empty_cache()
    assert len({tuple(count.values()) for count in counts.values()}) == 1
    write_json(out / 'result.json', dict(status='PASS_FULL_OFFICIAL_CONTROL_GRADIENT_CONTRACT', checks=checks,
        parameter_counts=counts, train_records=len(train), query_records=len(query), gallery_records=len(gallery),
        training_heldout_identities=0, classes=classes, optimizer_updates=0,
        limits='Initialization/source-gradient contract only. No performance claim. Separate three-actual-update smoke must activate every trainable parameter and strictly reload in memory before fresh50.'))
    print('FULL_OFFICIAL_CONTROL_GRADIENT_CONTRACT_PASS', flush=True)


if __name__ == '__main__':
    main()
