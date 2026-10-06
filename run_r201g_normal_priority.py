"""R201G: retain the unit-Triplet fixture and disable only partial loss gradients."""
import torch

import run_full_official_frozen_anchor_experiment as runner
from identity_coordinate_three_dataset import build as coordinate_build
from original_identity_anchor import assert_anchor_unchanged
from rgbnt201_projection_probe import VARIANTS
from run_experiment import write_json
from shared_identity_axis import partial_gallery_triplet

REVISION = ('R201G normal-priority objective ablation relative to the closed R201C narrow/unit-Triplet '
    'controls. Identical frozen original DeMo identity anchor, fresh expert/heads, 5120D unit metric, '
    'full CE/soft Triplet/auxiliary/contribution losses, seed42/B64/P8K8/newAdam/additional50. '
    'Only the existing partial-view CE.25 and cross-gallery Triplet.5 weights become zero. '
    'Partial sets/images/forward/BN calls/mining/two backwards are retained, so this is not '
    'a claim that removing all missing-source computation helps. Raw partial losses are diagnostic '
    'only. No new architecture, parameters or novelty claim. Official full train/query/gallery, '
    'normal mAP-best earliest ties, benchmark-selected rather than untouched test. '
    'Other two normal datasets and missing49 remain required after candidate confirmation.')


def build(args, cfg, classes, cameras):
    assert cfg.MODEL.ID_LOSS_WEIGHT == .25 and cfg.MODEL.TRIPLET_LOSS_WEIGHT == 1
    model = coordinate_build(args, cfg, classes, cameras)
    model.normal_priority_smoke = args.mode == 'smoke'
    model.anchor_record['method'] = REVISION
    return model


def record(path, value):
    if 'descriptor_dim' in value:
        value['descriptor_dim'] = 5120
    if 'pretraining' in value:
        dataset = value['arguments']['dataset']
        epoch = value['anchor']['original_best_epoch']
        value['pretraining'] = f'Full-official original {dataset} DeMo50 bestE{epoch} plus additional50/new Adam; original identity encoder fixed'
    write_json(path, value)


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    assert variant in VARIANTS
    images, target, cam, scene, names = batch
    images = {key: value.cuda(non_blocking=True) for key, value in images.items()}
    target, cam, scene = target.cuda(), cam.cuda(), scene.cuda()
    optimizer.zero_grad(set_to_none=True)
    bn_before = int(model.fused_neck.num_batches_tracked) if model.normal_priority_smoke else None
    with torch.autocast('cuda'):
        output = model(images, label=target, cam_label=cam, view_label=scene)
        end = len(output) - len(output) % 2
        assert len(model.loss_weights) == end // 2
        assert output[1].shape == (64, 5120)
        full_loss = sum(model.loss_weights[i // 2] * loss_fn(output[i], output[i + 1], target, cam)
                        for i in range(0, end, 2))
        if len(output) % 2:
            full_loss = full_loss + output[-1]
        gallery = output[1].detach()
    assert torch.isfinite(full_loss)
    scaler.scale(full_loss).backward()
    del output
    partial = {key: value if index in retained else torch.zeros_like(value)
               for index, (key, value) in enumerate(images.items())}
    with torch.autocast('cuda'):
        score, query = model(partial, label=target, cam_label=cam, view_label=scene, partial=True)
        partial_ce = xent(score, target)
        cross_triplet, pos, neg = partial_gallery_triplet(query, gallery, target)
        partial_loss = 0. * partial_ce + 0. * cross_triplet
    assert torch.isfinite(partial_ce) and torch.isfinite(cross_triplet) and partial_loss == 0
    scaler.scale(partial_loss).backward()
    scaler.unscale_(optimizer)
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    detail = dict(loss=float(full_loss.detach() + partial_loss.detach()), full_loss=float(full_loss.detach()),
        partial_ce=float(partial_ce.detach()), cross_triplet=float(cross_triplet.detach()),
        partial_loss_effective=float(partial_loss.detach()), partial_CE_weight=0., partial_triplet_weight=0.,
        partial_set=''.join('RNT'[i] for i in retained), positive_indices=pos.tolist(), negative_indices=neg.tolist(),
        reference_requires_grad=gallery.requires_grad, optimizer_updated=scaler.get_scale() >= previous_scale,
        amp_scale=scaler.get_scale(), names=list(names), primary_full_metric='unit5120_soft_triplet',
        freeze_identity_encoder=True, added_losses=0, descriptor_dim=5120)
    assert not detail['reference_requires_grad']
    if model.normal_priority_smoke:
        detail['fused_BN_calls'] = int(model.fused_neck.num_batches_tracked) - bn_before
        assert detail['fused_BN_calls'] == 2
        detail['full_unit_max_error'] = float((gallery.norm(dim=1) - 1).abs().max())
        assert detail['full_unit_max_error'] < 1e-6
    return detail


if __name__ == '__main__':
    runner.assert_anchor_unchanged = assert_anchor_unchanged
    runner.write_json = record
    runner.main(builder=build, update=step, variants=VARIANTS, method_revision=REVISION,
        partial_training='Same six proper sets, augmented images, forwards, BN calls and mining as R201C; '
                         'partial CE/triplet both measured, zero weight/backward, only full objectives update parameters.')
