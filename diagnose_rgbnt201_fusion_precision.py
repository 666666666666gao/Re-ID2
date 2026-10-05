"""Same warm model/batch: AMP fusion versus local FP32 fusion; no optimizer update."""
import argparse
from pathlib import Path

import torch

from experiment_data import make_loader, seed_all
from layers.make_loss import make_loss
from layers.softmax_loss import CrossEntropyLabelSmooth
from official_training_data import full_records
from original_identity_anchor import build
from run_experiment import configuration, write_json
import shared_identity_axis as shared


def gradient_state(model):
    result = {}
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        grad = value.grad
        result[name] = dict(grad_is_none=grad is None, parameter_dtype=str(value.dtype))
        if grad is not None:
            result[name].update(all_finite=bool(torch.isfinite(grad).all()),
                nonzero_count=int(torch.count_nonzero(grad)), max_abs=float(grad.abs().max()) / 512,
                grad_dtype=str(grad.dtype))
    return result


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'anchor-run-dir', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    args.dataset, args.variant, args.seed, args.freeze_identity_encoder = 'RGBNT201', 'demo_shared', 42, 0
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = configuration(args)
    train, query, gallery, classes, cameras, _ = full_records(args.data_root, args.dataset)
    assert (len(train), len(query), len(gallery), cfg.DATALOADER.NUM_INSTANCE) == (3951, 836, 836, 8)
    model = build(args, cfg, classes, cameras)
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    seed_all(args.seed)
    images, labels, cam, scene, names = next(iter(make_loader(train, cfg, True, args.seed)))
    images = {key: value.cuda() for key, value in images.items()}
    labels, cam, scene = labels.cuda(), cam.cuda(), scene.cuda()
    assert len(labels) == 64 and torch.unique(labels).numel() == 8
    loss_fn, _ = make_loss(cfg, classes)
    xent = CrossEntropyLabelSmooth(classes)
    original_fusion = shared.available_base_fusion

    def fp32_fusion(module, patches, globals_, available):
        with torch.autocast('cuda', enabled=False):
            return original_fusion(module, [value.float() for value in patches],
                [value.float() for value in globals_], available)

    measurements = {}
    for condition in ('original_AMP_fusion', 'local_FP32_fusion'):
        shared.available_base_fusion = original_fusion if condition == 'original_AMP_fusion' else fp32_fusion
        model.load_state_dict(initial, strict=True)
        model.train()
        model.zero_grad(set_to_none=True)
        seed_all(args.seed)
        with torch.autocast('cuda'):
            output = model(images, label=labels, cam_label=cam, view_label=scene)
            assert len(output) == 2 * len(model.loss_weights)
            full_loss = sum(model.loss_weights[i // 2] * loss_fn(output[i], output[i + 1], labels, cam)
                            for i in range(0, len(output), 2))
            reference = output[1].detach()
        assert torch.isfinite(full_loss)
        (512 * full_loss).backward()
        del output
        partial = {key: value if key == 'RGB' else torch.zeros_like(value) for key, value in images.items()}
        with torch.autocast('cuda'):
            score, feature = model(partial, label=labels, cam_label=cam, view_label=scene, partial=True)
            ce = xent(score, labels)
            triplet, _, _ = shared.partial_gallery_triplet(feature, reference, labels)
            partial_loss = .25 * ce + .5 * triplet
        assert torch.isfinite(partial_loss) and not reference.requires_grad
        (512 * partial_loss).backward()
        stats = gradient_state(model)
        parameters_unchanged = all(torch.equal(value.detach().cpu(), initial[name])
                                   for name, value in model.named_parameters())
        assert parameters_unchanged
        record = dict(loss=float(full_loss.detach() + partial_loss.detach()), gradient_state=stats,
            missing_gradients=[name for name, value in stats.items() if value['grad_is_none']],
            nonfinite_gradients=[name for name, value in stats.items() if not value['grad_is_none'] and not value['all_finite']],
            finite_zero_gradients=[name for name, value in stats.items() if not value['grad_is_none'] and value['all_finite'] and value['nonzero_count'] == 0],
            parameters_unchanged=parameters_unchanged, optimizer_updates=0)
        measurements[condition] = record
        write_json(out / (condition + '.json'), record)
        print('PRECISION_PROBE', condition, 'zeros', record['finite_zero_gradients'], flush=True)
        del full_loss, partial_loss, score, feature, reference, ce, triplet
    shared.available_base_fusion = original_fusion
    write_json(out / 'result.json', dict(status='COMPLETE_RGBNT201_FUSION_PRECISION_GRADIENT_PROBE',
        measurements=measurements, batch_names=list(names), batch_size=64, P=8, K=8,
        full_split_counts=dict(train=3951, query=836, gallery=836), training_heldout_identities=0,
        identical_initial_state_and_batch=True, random_seed_reset=42, optimizer_updates=0, new_weight_files=0,
        precision='CLIP/reducers/heads remain AMP in both; only available base fusion runs FP32 in the second diagnostic',
        limits='Single training batch gradient diagnosis with restored parameters and buffers; stochastic kernels may differ across dtypes. No training/validation score, no revised acceptance gate, no completed50/all49 or method-gain claim.'))


if __name__ == '__main__':
    main()
