"""Real native update, exact warm-start identity coordinates and fixed-encoder proof."""
import argparse
import io
import json
from pathlib import Path

import torch

from evaluate_full_official49 import SETS
from experiment_data import make_loader, seed_all
from frozen_identity_anchor_axis import build, assert_anchor_unchanged, VARIANTS
from layers.make_loss import make_loss
from layers.softmax_loss import CrossEntropyLabelSmooth
from missing_evaluation import mask_images
from official_training_data import full_records
from run_experiment import configuration, write_json
from run_full_official_frozen_anchor_experiment import step
from shared_identity_axis import SharedIdentityDeMo, encode_available, metric_descriptor
from solver.make_optimizer import make_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('MSVR310',), required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    for key in ('data-root', 'pretrained', 'output', 'anchor-run-dir'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--freeze-identity-encoder', type=int, choices=(0, 1), required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    cfg = configuration(args)
    train, query, gallery, classes, cameras, manifest = full_records(args.data_root, args.dataset)
    model = build(args, cfg, classes, cameras)
    seed_all(args.seed)
    reference = SharedIdentityDeMo(classes, cfg, cameras, args.seed).float().cuda()
    reference.load_state_dict(torch.load(Path(args.anchor_run_dir) / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    reference.eval()
    images, _, cam, scene, _ = next(iter(make_loader(query[:8], cfg, False, args.seed)))
    images = {k: v.cuda() for k,v in images.items()}
    cam, scene = cam.cuda(), scene.cuda()
    initial_equal = {}
    model.eval()
    with torch.no_grad():
        for available, missing in SETS.items():
            masked = mask_images(images, missing)
            expected = reference(masked, cam_label=cam, view_label=scene)
            actual = model(masked, cam_label=cam, view_label=scene, return_states=True)['00']
            assert torch.equal(actual, expected)
            initial_equal[available] = True
    del reference
    if model.freeze_identity_encoder:
        model.train()
        with torch.no_grad():
            for available, missing in SETS.items():
                masked = mask_images(images, missing)
                _, _, value, _, _, _ = encode_available(model, masked, cam, scene)
                # Values below compare the frozen encoder in train-mode surroundings to its eval surroundings.
                train_value = metric_descriptor(value)
                model.eval()
                actual = model(masked, cam_label=cam, view_label=scene, return_states=True)['00']
                assert torch.equal(train_value, actual)
                model.train()
        assert_anchor_unchanged(model)
    model.train()
    loss_fn, center = make_loss(cfg, classes)
    xent = CrossEntropyLabelSmooth(classes)
    optimizer, _ = make_optimizer(cfg, model, center)
    named = {id(p): name for name,p in model.named_parameters()}
    replay_groups, pairs = [], []
    for group in optimizer.param_groups:
        cloned = []
        for parameter in group['params']:
            assert parameter.requires_grad
            copy = torch.nn.Parameter(parameter.detach().clone())
            cloned.append(copy)
            pairs.append((named[id(parameter)], parameter, copy))
        replay_groups.append({**{k:v for k,v in group.items() if k != 'params'}, 'params': cloned})
    assert isinstance(optimizer, torch.optim.Adam)
    replay = torch.optim.Adam(replay_groups, **optimizer.defaults)
    scaler = torch.amp.GradScaler('cuda', init_scale=512)
    seed_all(args.seed)
    batch = next(iter(make_loader(train, cfg, True, args.seed)))
    labels = batch[1]
    assert len(labels) == 64 and torch.unique(labels).numel() == 16
    assert all(int((labels == label).sum()) == 4 for label in torch.unique(labels))
    detail = step(model, batch, optimizer, scaler, loss_fn, xent, (0,), args.variant)
    assert detail['optimizer_updated'] and not detail['reference_requires_grad']
    active = {}
    for name, parameter, copy in pairs:
        active[name] = bool(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0)
        assert active[name], name
        copy.grad = parameter.grad.detach().clone()
    replay.step()
    max_error = 0.
    for name, parameter, copy in pairs:
        max_error = max(max_error, float((parameter.detach() - copy.detach()).abs().max()))
        assert torch.allclose(parameter, copy, rtol=1e-6, atol=1e-7), name
    if model.freeze_identity_encoder:
        assert_anchor_unchanged(model)
    model.eval()
    with torch.no_grad():
        before = model(images, cam_label=cam, view_label=scene)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    model.load_state_dict(torch.load(buffer, map_location='cuda', weights_only=True), strict=True)
    with torch.no_grad():
        after = model(images, cam_label=cam, view_label=scene)
    assert torch.equal(before, after)
    if model.freeze_identity_encoder:
        assert_anchor_unchanged(model)
    result = dict(status='PASS_FULL_OFFICIAL_ANCHOR_NATIVE_UPDATE_AND_IDENTITY_REFERENCE',
                  arguments=vars(args), actual_optimizer_updates=1, amp_skipped_steps=0,
                  anchor=model.anchor_record, initial_all7_base_descriptors_exact=initial_equal,
                  fixed_encoder_unchanged_after_update_and_reload=bool(model.freeze_identity_encoder),
                  own_adam_replay_max_error=max_error, own_adam_replay_tolerance=dict(rtol=1e-6,atol=1e-7),
                  active_gradients=active, parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  strict_reload_equal=True, descriptor_dim=5632, batch_size=64, P=16, K=4,
                  full_split_counts=dict(train=len(train),query=len(query),gallery=len(gallery)),
                  training_heldout_identities=0, detail=detail,
                  limits='Real one-update native contract and8-query reload/initial-coordinate gate, not completed50 or all49 final evaluation.')
    write_json(out / 'result.json', result)
    print('FULL_OFFICIAL_ANCHOR_NATIVE_CONTRACT_PASS', args.variant, args.freeze_identity_encoder, flush=True)


if __name__ == '__main__':
    main()
