"""Six-update controlled diagnostic of the observed query-condition AMP zeros."""
import argparse
from pathlib import Path

import torch

from experiment_data import make_loader, seed_all, split_records
from layers.make_loss import make_loss
from run_experiment import build, configuration, step, write_json
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    args.dataset, args.variant, args.seed, args.contribution_weight = 'RGBNT201', 'axis_collaboration', 42, .05
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    cfg = configuration(args)
    fit, _, _, classes, cameras = split_records(args.data_root, args.dataset)
    reports = {}
    for mode in ('native_conditions', 'fp32_conditions'):
        model = build(args, cfg, classes, cameras)
        original_conditions = model.router.conditions
        if mode == 'fp32_conditions':
            def conditions(modality, frequency):
                with torch.autocast('cuda', enabled=False):
                    return original_conditions(modality.float(), frequency.float())
            model.router.conditions = conditions
        hooks, observed = [], {}

        def watch(name):
            def forward_hook(module, values, output):
                observed[name] = {'output_dtype': str(output.dtype), 'output_max': float(output.detach().abs().max())}
                def backward_hook(gradient):
                    observed[name]['scaled_output_gradient'] = {
                        'dtype': str(gradient.dtype), 'finite': bool(torch.isfinite(gradient).all()),
                        'nonzero_elements': int(gradient.count_nonzero()), 'elements': gradient.numel(),
                        'max_abs': float(gradient.abs().max()), 'sum_abs': float(gradient.abs().sum())}
                output.register_hook(backward_hook)
            return forward_hook

        for name in ('to_frequency_query', 'to_modality_query'):
            for index, module in enumerate(getattr(model.router, name)):
                hooks.append(module.register_forward_hook(watch(f'{name}.{index}')))
        loss_fn, center = make_loss(cfg, classes)
        optimizer, _ = make_optimizer(cfg, model, center)
        scheduler = create_scheduler(cfg, optimizer)
        scaler = torch.amp.GradScaler('cuda', init_scale=512)
        seed_all(args.seed)
        model.train()
        losses = []
        for batch in make_loader(fit, cfg, True, args.seed):
            loss, _, updated, scale = step(model, batch, optimizer, scaler, loss_fn)
            assert updated
            losses.append(loss)
            if len(losses) == 3:
                break
        gradients = {name: {'finite': bool(torch.isfinite(parameter.grad).all()),
                           'nonzero_elements': int(parameter.grad.count_nonzero()),
                           'elements': parameter.numel(), 'max_abs': float(parameter.grad.abs().max()),
                           'sum_abs': float(parameter.grad.abs().sum())}
                     for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is not None}
        missing = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
        reports[mode] = {'actual_updates': len(losses), 'losses': losses, 'scale': scale,
                         'optimizer_learning_rates': sorted({group['lr'] for group in optimizer.param_groups}),
                         'missing_gradients': missing, 'zero_gradients': [n for n, g in gradients.items() if g['nonzero_elements'] == 0],
                         'nonfinite_gradients': [n for n, g in gradients.items() if not g['finite']],
                         'condition_parameter_gradients': {n:g for n,g in gradients.items() if n.startswith('router.to_')},
                         'forward_backward_witness': observed}
        write_json(root/(mode+'.json'), reports[mode])
        print(mode, reports[mode], flush=True)
        for hook in hooks:
            hook.remove()
        del model, optimizer, scheduler, scaler, loss_fn, center
        torch.cuda.empty_cache()
    write_json(root/'diagnostic.json', {'reports': reports, 'scope': 'Controlled six-update numerical diagnostic, not efficacy or formal checkpoint'})


if __name__ == '__main__':
    main()
