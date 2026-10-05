"""One objective factor: GT-aware smooth AP on the deployed R201A descriptor."""
import torch

from shared_identity_axis import partial_gallery_triplet


TEMPERATURE = 0.05
RANK_WEIGHT = 1.0


def smooth_ap(query, gallery, labels, cameras):
    """Same records in both views; RGBNT201 excludes same-ID/same-camera junk."""
    with torch.autocast(query.device.type, enabled=False):
        similarity = query.float() @ gallery.float().T
        same = labels[:, None].eq(labels[None])
        junk = same & cameras[:, None].eq(cameras[None])
        positive = same & ~junk
        eligible = positive.any(1)
        count = int(eligible.sum())
        if count == 0:
            # PK sampling has no camera constraint; skip the rank term, retain ReID.
            return similarity.sum() * 0, count
        size = len(labels)
        different_candidate = ~torch.eye(size, dtype=torch.bool, device=query.device)
        ahead = torch.sigmoid((similarity[:, None, :] - similarity[:, :, None]) / TEMPERATURE)
        ahead = ahead * different_candidate[None]
        rank = 1 + (ahead * (~junk)[:, None, :]).sum(-1)
        positive_rank = 1 + (ahead * positive[:, None, :]).sum(-1)
        precision = positive_rank / rank
        # Only positive candidates and queries with a legal positive participate.
        ap = (precision[eligible] * positive[eligible]).sum(-1) / positive[eligible].sum(-1)
        return 1 - ap.mean(), count


def step(model, batch, optimizer, scaler, loss_fn, xent, retained):
    images, target, cam, scene, names = batch
    images = {key: value.cuda(non_blocking=True) for key, value in images.items()}
    target, cam, scene = target.cuda(), cam.cuda(), scene.cuda()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda'):
        output = model(images, label=target, cam_label=cam, view_label=scene)
        assert len(output) == 2 * len(model.loss_weights), 'This factor targets demo_shared only'
        original_full_loss = sum(model.loss_weights[i // 2] * loss_fn(output[i], output[i + 1], target, cam)
                                 for i in range(0, len(output), 2))
        full_rank, full_valid = smooth_ap(output[1], output[1], target, cam)
        full_loss = original_full_loss + RANK_WEIGHT * full_rank
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
        partial_rank, partial_valid = smooth_ap(query, gallery, target, cam)
        partial_loss = .25 * partial_ce + .5 * cross_triplet + RANK_WEIGHT * partial_rank
    assert torch.isfinite(partial_loss)
    scaler.scale(partial_loss).backward()
    scaler.unscale_(optimizer)
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    detail = dict(loss=float(full_loss.detach() + partial_loss.detach()), full_loss=float(full_loss.detach()),
        partial_ce=float(partial_ce.detach()), cross_triplet=float(cross_triplet.detach()),
        partial_set=''.join('RNT'[i] for i in retained), positive_indices=pos.tolist(), negative_indices=neg.tolist(),
        reference_requires_grad=gallery.requires_grad, optimizer_updated=scaler.get_scale() >= previous_scale,
        amp_scale=scaler.get_scale(), names=list(names), full_rank_loss=float(full_rank.detach()),
        partial_rank_loss=float(partial_rank.detach()), rank_full_valid_queries=full_valid,
        rank_partial_valid_queries=partial_valid, rank_temperature=TEMPERATURE, rank_loss_weight=RANK_WEIGHT)
    assert not detail['reference_requires_grad']
    return detail
