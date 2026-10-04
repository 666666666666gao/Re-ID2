"""One native64 diagnostic of the observed disabled-parent Adam difference."""
import argparse
import gc
import math
from pathlib import Path

import torch

from availability_graph_training import step as graph_step
from experiment_data import make_loader, seed_all
from layers.make_loss import make_loss
from layers.softmax_loss import CrossEntropyLabelSmooth
from official_training_data import full_records
from run_experiment import configuration, write_json
from run_full_official_availability_graph_experiment import builder
from run_full_official_modality_outlet_experiment import step as parent_step
from solver.make_optimizer import make_optimizer

KEY = 'router.to_modality_query.1.weight'
INDEX = (50, 121)


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    args.dataset, args.seed, args.variant = 'MSVR310', 42, 'frequency_shared'
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
    parameter = dict(model.named_parameters())[KEY]
    rows, first_grad, first_state, first_detail, first_rng = [], {}, {}, {}, None
    for phase in ('parent1', 'parent2', 'parent3', 'disabled'):
        model.load_state_dict(initial, strict=True)
        model.graph_weight, model.graph_step = 0, 1
        model.train()
        optimizer, _ = make_optimizer(cfg, model, center)
        assert isinstance(optimizer, torch.optim.Adam)
        scaler = torch.amp.GradScaler('cuda', init_scale=512)
        raw, projection = [], []

        def capture_raw(module, inputs, output):
            patch, cls = output
            raw.append(torch.cat((cls.detach(), model.pool(patch.detach().permute(0, 2, 1)).squeeze(-1)), -1))

        def capture_projection(module, inputs, output):
            projection.append((inputs[0].detach().clone(), output.requires_grad))

        raw_hook = model.BACKBONE.register_forward_hook(capture_raw)
        projection_hook = model.shared_projection.register_forward_hook(capture_projection)
        seed_all(1234)
        detail = (graph_step if phase == 'disabled' else parent_step)(
            model, batch, optimizer, scaler, loss_fn, xent, (0,), args.variant)
        raw_hook.remove(); projection_hook.remove()
        assert detail['optimizer_updated'] and len(raw) == 4 and len(projection) == 2
        gradients = {key: value.grad.detach().cpu().clone() for key, value in model.named_parameters() if value.grad is not None}
        current = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
        group = next(group for group in optimizer.param_groups if any(value is parameter for value in group['params']))
        state = optimizer.state[parameter]
        assert len(group['params']) == 1 and not group['amsgrad'] and not group['maximize']
        assert float(state['step']) == 1
        beta1, beta2 = group['betas']
        theta0, theta1 = float(initial[KEY][INDEX]), float(current[KEY][INDEX])
        gradient = float(gradients[KEY][INDEX])
        moment = float(state['exp_avg'][INDEX])
        second = float(state['exp_avg_sq'][INDEX])
        denominator = math.sqrt(second) / math.sqrt(1 - beta2) + group['eps']
        reconstructed = theta0 - (group['lr'] / (1 - beta1)) * moment / denominator
        row = dict(phase=phase, theta0=theta0, theta1=theta1, unscaled_gradient=gradient,
            lr=group['lr'], weight_decay=group['weight_decay'], eps=group['eps'], betas=group['betas'],
            exp_avg=moment, exp_avg_sq=second, adam_effective_gradient=moment / (1 - beta1),
            adam_reconstructed_theta1=reconstructed, reconstruction_error=abs(theta1 - reconstructed),
            full_loss=detail['full_loss'], loss=detail['loss'], optimizer_updated=True)
        write_json(out / 'progress.json', dict(status='ACTUAL_ADAM_UPDATE_RECORDED_BEFORE_ASSERT',
            key=KEY, index=INDEX, updates=rows + [row]))
        assert math.isclose(theta1, reconstructed, rel_tol=1e-5, abs_tol=1e-6)
        if phase == 'parent1':
            first_grad, first_state, first_detail, first_rng = gradients, current, detail, rng
            row.update(maximum_gradient_error=0., maximum_state_error=0., culprit_state_error=0.)
        else:
            assert set(gradients) == set(first_grad) and set(current) == set(first_state)
            for key in gradients:
                torch.testing.assert_close(gradients[key], first_grad[key], rtol=1e-5, atol=1e-6)
            assert all(detail[key] == value for key, value in first_detail.items())
            assert all(torch.equal(a, b) for a, b in zip(rng, first_rng))
            assert all(torch.equal(current[key], first_state[key]) for key in current
                if key.endswith(('num_batches_tracked', 'running_mean', 'running_var')))
            row.update(maximum_gradient_error=max(float((gradients[key] - first_grad[key]).abs().max()) for key in gradients),
                maximum_state_error=max(float((current[key] - first_state[key]).abs().max()) for key in current if current[key].is_floating_point()),
                culprit_state_error=abs(theta1 - rows[0]['theta1']),
                culprit_gradient_error=abs(gradient - rows[0]['unscaled_gradient']),
                reconstructed_difference=reconstructed - rows[0]['adam_reconstructed_theta1'],
                RNG_exact=True, BN_exact=True, original_detail_exact=True, all_gradients_original_tolerance_pass=True)
        rows.append(row)
        write_json(out / 'progress.json', dict(status='ACTUAL_ADAM_DIAGNOSTIC_PROGRESS', key=KEY, index=INDEX, updates=rows))
        del optimizer, scaler, gradients, current, raw, projection
        gc.collect(); torch.cuda.empty_cache()
    write_json(out / 'result.json', dict(status='FOUR_ACTUAL_ADAM_DIAGNOSTIC_UPDATES_COMPLETE',
        key=KEY, index=INDEX, updates=rows, native_batch=64, graph_weight=0,
        tolerance=dict(rtol=1e-5, atol=1e-6), benchmark_results=0, new_weights=0,
        scope='Three direct original parent updates plus one exact disabled delegation; culprit own Adam moments reconstruct each actual update. No positive graph claim or full training.'))
    print('FOUR_ACTUAL_ADAM_DIAGNOSTIC_UPDATES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
