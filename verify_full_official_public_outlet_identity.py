"""Four native updates: parent, repeat, disabled, and observed public CE inputs."""
import argparse
import gc
import math
from pathlib import Path

import torch

from public_outlet_identity_training import step
from experiment_data import make_loader, seed_all
from layers.make_loss import make_loss
from layers.softmax_loss import CrossEntropyLabelSmooth
from official_training_data import full_records
from run_experiment import configuration, write_json
from run_full_official_public_outlet_identity_experiment import builder, VARIANTS
from run_full_official_modality_outlet_experiment import step as m4_step
from run_shared_identity_experiment import step as shared_step
from solver.make_optimizer import make_optimizer


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    args = parser.parse_args()
    args.dataset, args.seed = 'MSVR310', 42
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    out = Path(args.output); out.mkdir(exist_ok=False)
    cfg = configuration(args)
    train, query, gallery, classes, cameras, _ = full_records(args.data_root, args.dataset)
    assert (len(train), len(query), len(gallery)) == (1032, 591, 1055)
    model = builder(args, cfg, classes, cameras)
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    batch = next(iter(make_loader(train, cfg, True, args.seed)))
    loss_fn, center = make_loss(cfg, classes)
    xent = CrossEntropyLabelSmooth(classes)
    expected, expected_grad, expected_detail, expected_rng = {}, {}, {}, None
    comparisons, positive, optimizer_checks = [], {}, []
    for phase in ('parent', 'parent_repeat', 'disabled', 'positive'):
        model.load_state_dict(initial, strict=True)
        model.public_identity_weight = .1 if phase == 'positive' else 0
        model.train()
        optimizer, _ = make_optimizer(cfg, model, center)
        scaler = torch.amp.GradScaler('cuda', init_scale=512)
        raw, projection, outlets, neck_inputs, shared_logits, ce_calls = [], [], [], [], [], []

        def capture_raw(module, inputs, output):
            patch, cls = output
            raw.append(torch.cat((cls.detach(), model.pool(patch.detach().permute(0, 2, 1)).squeeze(-1)), -1))

        def capture_projection(module, inputs, output):
            projection.append(inputs[0].detach().clone())

        def capture_outlet(module, inputs, output):
            outlets.append((output[1][:, 5120:].detach().clone(), output[0].detach().clone()))

        def capture_neck(module, inputs, output):
            neck_inputs.append(inputs[0].detach().clone())

        def capture_classifier(module, inputs, output):
            shared_logits.append(output.detach().clone())

        def capture_ce(module, inputs, output):
            ce_calls.append((inputs[0].detach().clone(), inputs[1].detach().clone(), float(output.detach())))

        hooks = [model.BACKBONE.register_forward_hook(capture_raw),
                 model.shared_projection.register_forward_hook(capture_projection),
                 model.register_forward_hook(capture_outlet),
                 model.shared_neck.register_forward_hook(capture_neck),
                 model.shared_classifier.register_forward_hook(capture_classifier),
                 xent.register_forward_hook(capture_ce)]
        seed_all(1234)
        if phase in ('parent', 'parent_repeat'):
            if args.variant == 'demo_shared':
                detail = shared_step(model, batch, optimizer, scaler, loss_fn, xent, (0,))
            else:
                detail = m4_step(model, batch, optimizer, scaler, loss_fn, xent, (0,), args.variant)
        else:
            detail = step(model, batch, optimizer, scaler, loss_fn, xent, (0,), args.variant)
        for hook in hooks:
            hook.remove()
        assert detail['optimizer_updated'] and len(raw) == 4 and len(projection) == len(outlets) == 2
        assert torch.equal(projection[0], torch.stack(raw[:3], 1).mean(1))
        assert torch.equal(projection[1], raw[3])
        current = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        gradients = {key: value.grad.detach().cpu().clone() for key, value in model.named_parameters() if value.grad is not None}
        names_by_id = {id(value): key for key, value in model.named_parameters()}
        assert isinstance(optimizer, torch.optim.Adam)
        maximum_update_error, checked_parameters = 0., 0
        for group in optimizer.param_groups:
            assert len(group['params']) == 1 and not group['amsgrad'] and not group['maximize']
            parameter = group['params'][0]
            key = names_by_id[id(parameter)]
            if parameter.grad is None:
                assert torch.equal(current[key], initial[key])
                continue
            state = optimizer.state[parameter]
            assert float(state['step']) == 1
            beta1, beta2 = group['betas']
            moment = state['exp_avg'].detach().cpu().double()
            second = state['exp_avg_sq'].detach().cpu().double()
            expected_update = initial[key].double() - (group['lr'] / (1 - beta1)) * moment / (
                second.sqrt() / math.sqrt(1 - beta2) + group['eps'])
            maximum_update_error = max(maximum_update_error, float((current[key].double() - expected_update).abs().max()))
            torch.testing.assert_close(current[key].double(), expected_update, rtol=1e-5, atol=1e-6)
            checked_parameters += 1
        optimizer_checks.append(dict(phase=phase, actual_optimizer_updates=1,
            checked_parameter_tensors=checked_parameters, maximum_own_adam_reconstruction_error=maximum_update_error))
        rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
        if phase == 'parent':
            expected, expected_grad, expected_detail, expected_rng = current, gradients, detail, rng
            assert len(neck_inputs) == len(shared_logits) == len(ce_calls) == 1
        elif phase in ('parent_repeat', 'disabled'):
            assert set(current) == set(expected) and set(gradients) == set(expected_grad)
            state_errors = sorted([(float((current[key] - expected[key]).abs().max()), key)
                for key in current if current[key].is_floating_point()], reverse=True)
            gradient_errors = sorted([(float((gradients[key] - expected_grad[key]).abs().max()), key)
                for key in gradients], reverse=True)
            comparisons.append(dict(phase=phase, maximum_state_absolute_error=state_errors[0][0],
                maximum_gradient_absolute_error=gradient_errors[0][0], largest_state_errors=state_errors[:5],
                largest_gradient_errors=gradient_errors[:5]))
            write_json(out / 'comparison_progress.json', dict(status='COMPARISON_RECORDED_BEFORE_ASSERT',
                variant=args.variant, comparisons=comparisons, detail=detail))
            for key in current:
                if key not in names_by_id.values():
                    assert torch.equal(current[key], expected[key]), phase + ' buffer ' + key
            for key in gradients:
                torch.testing.assert_close(gradients[key], expected_grad[key], rtol=1e-5, atol=1e-6,
                    msg=lambda message, key=key: phase + ' gradient ' + key + ': ' + message)
            assert all(detail[key] == value for key, value in expected_detail.items())
            assert len(neck_inputs) == len(shared_logits) == len(ce_calls) == 1
            if phase == 'disabled':
                assert detail['public_identity_disabled'] and detail['public_identity_weight'] == 0
        else:
            assert len(neck_inputs) == len(shared_logits) == len(ce_calls) == 3
            assert torch.equal(neck_inputs[1], outlets[0][0])
            assert torch.equal(neck_inputs[2], outlets[1][0])
            assert neck_inputs[1].shape == neck_inputs[2].shape == (64, 512)
            assert torch.equal(ce_calls[0][0], shared_logits[1])
            assert torch.equal(ce_calls[1][0], outlets[1][1])
            assert torch.equal(ce_calls[2][0], shared_logits[2])
            assert all(torch.equal(call[1].cpu(), batch[1]) for call in ce_calls)
            assert detail['public_identity_weight'] == .1 and detail['public_identity_extra_backbone_passes'] == 0
            assert detail['public_identity_extra_parameters'] == 0 and detail['public_identity_extra_shared_neck_calls'] == 2
            assert not detail['reference_requires_grad']
            assert all(math.isfinite(detail[key]) and detail[key] > 0 for key in ('public_full_ce', 'public_partial_ce'))
            assert detail['public_full_ce'] == ce_calls[0][2] and detail['public_partial_ce'] == ce_calls[2][2]
            assert math.isclose(detail['original_full_loss'], expected_detail['full_loss'], rel_tol=1e-5, abs_tol=1e-6)
            assert math.isclose(detail['full_loss'], expected_detail['full_loss'] + .05 * detail['public_full_ce'], rel_tol=1e-5, abs_tol=1e-6)
            for key in ('partial_ce', 'cross_triplet'):
                assert math.isclose(detail[key], expected_detail[key], rel_tol=1e-5, abs_tol=1e-6)
            for key in ('positive_indices', 'negative_indices', 'names', 'partial_set'):
                assert detail[key] == expected_detail[key]
            total = detail['full_loss'] + .25 * detail['partial_ce'] + .5 * detail['cross_triplet'] + .05 * detail['public_partial_ce']
            assert math.isclose(detail['loss'], total, rel_tol=1e-5, abs_tol=1e-6)
            assert set(current) == set(expected)
            intended_bn = {'shared_neck.running_mean', 'shared_neck.running_var', 'shared_neck.num_batches_tracked'}
            for key in current:
                if key not in names_by_id.values() and key not in intended_bn:
                    assert torch.equal(current[key], expected[key]), 'positive buffer ' + key
            assert int(current['shared_neck.num_batches_tracked'] - expected['shared_neck.num_batches_tracked']) == 2
            assert all(torch.isfinite(current[key]).all() for key in intended_bn)
            assert all(torch.isfinite(value).all() for value in gradients.values())
            for prefix in ('shared_projection.', 'shared_classifier.'):
                assert any(key.startswith(prefix) and value.abs().sum() > 0 for key, value in gradients.items())
            positive = dict(actual_optimizer_updates=1, actual_public_outlet_inputs_exact=True,
                classifier_inputs_and_official_training_labels_exact=True, existing_head_calls=3,
                intended_extra_shared_BN_updates=2, other_buffers_exact=True,
                original_full_and_partial_losses_retained=True, scalar_objective_arithmetic_verified=True,
                backbone_calls=4, projection_calls=2, added_backbone_calls=0, RNG_exact=True,
                public_full_ce=detail['public_full_ce'], public_partial_ce=detail['public_partial_ce'])
        assert all(torch.equal(a, b) for a, b in zip(rng, expected_rng))
        del optimizer, scaler, raw, projection, outlets, neck_inputs, shared_logits, ce_calls
        gc.collect(); torch.cuda.empty_cache()
    write_json(out / 'result.json', dict(status='PASS_FULL_OFFICIAL_PUBLIC_IDENTITY_PARENT_AMP_STEP',
        variant=args.variant, parent_optimizer_updates=2, disabled_delegation_optimizer_updates=1,
        positive_public_identity_optimizer_updates=1, parent_step_loss_gradient_equivalence=True,
        optimizer_update_consistency=True, own_adam_update_checks=optimizer_checks,
        comparison_tolerance=dict(rtol=1e-5, atol=1e-6), repeatability_and_disabled_comparisons=comparisons,
        cross_execution_updated_parameter_equality='Diagnostic only; retain the M5v3 own-Adam criterion backed by three original-parent repeated updates.',
        criterion_evidence='results/preflight/full_official_availability_graph_adam_diagnostic_intake_20261005.json',
        positive_public_identity_branch=positive, parent_buffers_exact=True, positive_other_buffers_exact=True,
        CPU_and_selected_CUDA_RNG_exact=True, parent_detail_fields_exact=True, added_parameters=0,
        training_heldout_identities=0, descriptor_dim=5632, new_weights=0,
        scope='Four actual native64 optimizer updates; no benchmark performance measured.'))
    print('FULL_OFFICIAL_PUBLIC_IDENTITY_NATIVE_CONTRACT_PASS', args.variant, flush=True)


if __name__ == '__main__':
    main()
