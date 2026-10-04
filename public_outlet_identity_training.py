"""One training factor: directly classify the actual public retrieval outlet."""
import torch

from run_shared_identity_experiment import step as shared_parent_step
from run_full_official_modality_outlet_experiment import step as m4_parent_step
from shared_identity_axis import partial_gallery_triplet


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    assert variant == model.variant
    if model.public_identity_weight == 0:
        if variant == 'demo_shared':
            detail = shared_parent_step(model, batch, optimizer, scaler, loss_fn, xent, retained)
        else:
            detail = m4_parent_step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant)
        detail.update(public_identity_disabled=True, public_identity_weight=0)
        return detail
    images, target, cam, scene, names = batch
    images = {key: value.cuda(non_blocking=True) for key, value in images.items()}
    target, cam, scene = target.cuda(), cam.cuda(), scene.cuda()
    model.alignment_names = tuple(names)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda'):
        output = model(images, label=target, cam_label=cam, view_label=scene)
        end = len(output) - len(output) % 2
        assert len(model.loss_weights) == end // 2
        original_full_loss = sum(model.loss_weights[i // 2] * loss_fn(output[i], output[i + 1], target, cam)
                                 for i in range(0, end, 2))
        if len(output) % 2:
            original_full_loss = original_full_loss + output[-1]
        full_public = output[1][:, 5120:]
        full_public_ce = xent(model.shared_classifier(model.shared_neck(full_public)), target)
        full_loss = original_full_loss + .5 * model.public_identity_weight * full_public_ce
        gallery = output[1].detach()
    assert torch.isfinite(full_loss)
    scaler.scale(full_loss).backward()
    del output, full_public
    partial = {key: value if index in retained else torch.zeros_like(value)
               for index, (key, value) in enumerate(images.items())}
    with torch.autocast('cuda'):
        score, query = model(partial, label=target, cam_label=cam, view_label=scene, partial=True)
        partial_ce = xent(score, target)
        cross_triplet, pos, neg = partial_gallery_triplet(query, gallery, target)
        partial_public_ce = xent(model.shared_classifier(model.shared_neck(query[:, 5120:])), target)
        original_partial_loss = .25 * partial_ce + .5 * cross_triplet
        partial_loss = original_partial_loss + .5 * model.public_identity_weight * partial_public_ce
    assert torch.isfinite(partial_loss)
    scaler.scale(partial_loss).backward()
    scaler.unscale_(optimizer)
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    detail = dict(loss=float(full_loss.detach() + partial_loss.detach()),
        full_loss=float(full_loss.detach()), partial_ce=float(partial_ce.detach()),
        cross_triplet=float(cross_triplet.detach()), partial_set=''.join('RNT'[i] for i in retained),
        positive_indices=pos.tolist(), negative_indices=neg.tolist(),
        reference_requires_grad=gallery.requires_grad,
        optimizer_updated=scaler.get_scale() >= previous_scale, amp_scale=scaler.get_scale(), names=list(names),
        original_full_loss=float(original_full_loss.detach()), original_partial_loss=float(original_partial_loss.detach()),
        public_identity_weight=model.public_identity_weight, public_full_ce=float(full_public_ce.detach()),
        public_partial_ce=float(partial_public_ce.detach()),
        public_identity_student='Actual deployed fused descriptor slice5120:5632 for full and partial views',
        public_identity_classifier='Existing shared_neck and shared_classifier, same head for both availability views',
        public_identity_labels='Official training identity labels only', public_identity_extra_backbone_passes=0,
        public_identity_extra_parameters=0, public_identity_extra_shared_neck_calls=2)
    assert not gallery.requires_grad
    if variant != 'demo_shared':
        detail.update(model.relation_audit)
        detail.update(model.alignment_audit)
        detail.update(model.modality_alignment_audit)
    return detail
