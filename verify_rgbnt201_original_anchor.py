"""One real update per control, matched initial tensors and original private identity proof."""
import argparse
import io
from pathlib import Path

import torch

from evaluate_full_official49 import SETS
from experiment_data import make_loader, seed_all
from layers.make_loss import make_loss
from layers.softmax_loss import CrossEntropyLabelSmooth
from missing_evaluation import mask_images
from modeling.make_model import make_model
from official_training_data import full_records
from original_identity_anchor import build, assert_anchor_unchanged
from run_experiment import configuration, write_json
from run_rgbnt201_original_anchor import step
from shared_identity_axis import encode_available
from solver.make_optimizer import make_optimizer


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'output', 'anchor-run-dir'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    args.dataset, args.variant, args.seed = 'RGBNT201', 'demo_shared', 42
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = configuration(args)
    train, query, gallery, classes, cameras, _ = full_records(args.data_root, args.dataset)
    assert (len(train), len(query), len(gallery)) == (3951, 836, 836)
    models = {}
    for freeze in (1, 0):
        args.freeze_identity_encoder = freeze
        models[freeze] = build(args, cfg, classes, cameras)
    fixed, adaptive = models[1], models[0]
    assert fixed.state_dict().keys() == adaptive.state_dict().keys()
    assert all(torch.equal(value, adaptive.state_dict()[name]) for name, value in fixed.state_dict().items())
    reference = make_model(cfg, classes, cameras).float().cuda()
    reference.load_state_dict(torch.load(Path(args.anchor_run_dir) / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    reference.eval()
    images, _, cam, scene, _ = next(iter(make_loader(query[:8], cfg, False, args.seed)))
    images = {key: value.cuda() for key, value in images.items()}
    cam, scene = cam.cuda(), scene.cuda()
    initial_private = {}
    with torch.no_grad():
        expected = reference(images, cam_label=cam, view_label=scene)
        for freeze, model in models.items():
            model.eval()
            raw = encode_available(model, images, cam, scene)[2][:, :5120]
            assert torch.equal(raw, expected)
            initial_private[freeze] = True
        fixed.train()
        private_by_set = {}
        for available, missing in SETS.items():
            masked = mask_images(images, missing)
            value = encode_available(fixed, masked, cam, scene)[2][:, :5120]
            fixed.eval()
            actual = encode_available(fixed, masked, cam, scene)[2][:, :5120]
            assert torch.equal(value, actual)
            private_by_set[available] = actual.clone()
            fixed.train()
    del reference
    results = {}
    for freeze, model in models.items():
        model.train()
        loss_fn, center = make_loss(cfg, classes)
        optimizer, _ = make_optimizer(cfg, model, center)
        named = {id(parameter): name for name, parameter in model.named_parameters()}
        groups, pairs = [], []
        for group in optimizer.param_groups:
            cloned = []
            for parameter in group['params']:
                copy = torch.nn.Parameter(parameter.detach().clone())
                cloned.append(copy)
                pairs.append((named[id(parameter)], parameter, copy))
            groups.append({**{key: value for key, value in group.items() if key != 'params'}, 'params': cloned})
        assert isinstance(optimizer, torch.optim.Adam)
        replay = torch.optim.Adam(groups, **optimizer.defaults)
        scaler = torch.amp.GradScaler('cuda', init_scale=512)
        seed_all(args.seed)
        batch = next(iter(make_loader(train, cfg, True, args.seed)))
        assert cfg.DATALOADER.NUM_INSTANCE == 8
        assert len(batch[1]) == 64 and torch.unique(batch[1]).numel() == 8
        assert all(int((batch[1] == value).sum()) == 8 for value in torch.unique(batch[1]))
        detail = step(model, batch, optimizer, scaler, loss_fn, CrossEntropyLabelSmooth(classes), (0,), args.variant)
        assert detail['optimizer_updated'] and scaler.get_scale() == 512 and not detail['reference_requires_grad']
        active, gradient_state = {}, {}
        for name, parameter, copy in pairs:
            grad = parameter.grad
            gradient_state[name] = dict(grad_is_none=grad is None, parameter_dtype=str(parameter.dtype))
            if grad is not None:
                gradient_state[name].update(all_finite=bool(torch.isfinite(grad).all()),
                    nonzero_count=int(torch.count_nonzero(grad)), max_abs=float(grad.abs().max()),
                    grad_dtype=str(grad.dtype))
            active[name] = bool(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0)
        failed = [name for name, positive in active.items() if not positive]
        write_json(out / ('gradient_state_' + str(freeze) + '.json'), dict(freeze=freeze,
            batch_names=list(batch[-1]), batch_size=64, P=8, K=8, detail=detail,
            amp_scale=scaler.get_scale(), gradient_state=gradient_state, failed_names=failed,
            acceptance='Original all-parameter finite-nonzero native condition is unchanged; this file diagnoses the failed predicate, not a waived gate.'))
        assert not failed, failed
        for name, parameter, copy in pairs:
            copy.grad = parameter.grad.detach().clone()
        replay.step()
        max_error = 0.
        for name, parameter, copy in pairs:
            max_error = max(max_error, float((parameter.detach() - copy.detach()).abs().max()))
            assert torch.allclose(parameter, copy, rtol=1e-6, atol=1e-7), name
        if freeze:
            assert_anchor_unchanged(model)
        model.eval()
        with torch.no_grad():
            before = model(images, cam_label=cam, view_label=scene)
            if freeze:
                for available, missing in SETS.items():
                    value = encode_available(model, mask_images(images, missing), cam, scene)[2][:, :5120]
                    assert torch.equal(value, private_by_set[available])
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model.load_state_dict(torch.load(buffer, map_location='cuda', weights_only=True), strict=True)
        with torch.no_grad():
            after = model(images, cam_label=cam, view_label=scene)
        assert torch.equal(before, after) and list(before.shape) == [8, 5632]
        if freeze:
            assert_anchor_unchanged(model)
        results[str(freeze)] = dict(actual_optimizer_updates=1, amp_skipped_steps=0,
            active_gradients=active, own_adam_replay_max_error=max_error, strict_reload_equal=True,
            initial_private_exact_original=initial_private[freeze], fixed_private_all7_unchanged=bool(freeze),
            parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            anchor=model.anchor_record, detail=detail)
        del replay, optimizer, pairs, groups, buffer
    write_json(out / 'result.json', dict(status='PASS_RGBNT201_ORIGINAL_ANCHOR_TWO_NATIVE_UPDATES',
        full_split_counts=dict(train=3951, query=836, gallery=836), training_heldout_identities=0,
        matched_initial_state_tensors_exact=True, actual_optimizer_updates=2, amp_skipped_steps=0,
        controls=results, descriptor_dim=5632, batch_size=64, P=8, K=8, new_weight_files=0,
        limits='Native real updates/8-query identity and reload gates only; not full50 or final49 results. Entire official record inventory; no heldout training identities.'))
    print('RGBNT201_ORIGINAL_ANCHOR_NATIVE_PASS', flush=True)


if __name__ == '__main__':
    main()
