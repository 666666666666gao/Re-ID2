"""One training factor: public identity supervision across available-source sets."""
import torch

from identity_alignment_axis import identity_alignment_loss
from run_shared_identity_experiment import PARTIAL_SETS, step as shared_parent_step
from run_full_official_modality_outlet_experiment import step as m4_parent_step
from shared_identity_axis import partial_gallery_triplet

REFERENCE_SETS = (*PARTIAL_SETS, (0, 1, 2))


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    assert variant == model.variant
    if model.graph_weight == 0:
        if variant == 'demo_shared':
            detail = shared_parent_step(model, batch, optimizer, scaler, loss_fn, xent, retained)
        else:
            detail = m4_parent_step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant)
        detail.update(graph_disabled=True, graph_weight=0, graph_step=model.graph_step)
        model.graph_step += 1
        return detail
    images, target, cam, scene, names = batch
    images = {key: value.cuda(non_blocking=True) for key, value in images.items()}
    target, cam, scene = target.cuda(), cam.cuda(), scene.cuda()
    model.alignment_names = tuple(names)
    optimizer.zero_grad(set_to_none=True)
    reference_set = REFERENCE_SETS[model.graph_step % len(REFERENCE_SETS)]
    raw_sources = []

    def capture(module, inputs, output):
        patch, cls = output
        raw_sources.append(torch.cat((cls.detach(),
            model.pool(patch.detach().permute(0, 2, 1)).squeeze(-1)), -1))

    handle = model.BACKBONE.register_forward_hook(capture)
    with torch.autocast('cuda'):
        output = model(images, label=target, cam_label=cam, view_label=scene)
        handle.remove()
        assert len(raw_sources) == 3
        with torch.no_grad():
            reference = model.shared_projection(torch.stack(
                [raw_sources[index] for index in reference_set], 1).mean(1)).detach()
        end = len(output) - len(output) % 2
        assert len(model.loss_weights) == end // 2
        full_loss = sum(model.loss_weights[i // 2] * loss_fn(output[i], output[i + 1], target, cam)
                        for i in range(0, end, 2))
        if len(output) % 2:
            full_loss = full_loss + output[-1]
        full_graph, full_detail = identity_alignment_loss(output[1][:, 5120:], reference,
            target, names, model.graph_temperature)
        if model.graph_weight > 0:
            full_loss = full_loss + .5 * model.graph_weight * full_graph
        gallery = output[1].detach()
    assert torch.isfinite(full_loss)
    scaler.scale(full_loss).backward()
    del output, raw_sources
    partial = {key: value if index in retained else torch.zeros_like(value)
               for index, (key, value) in enumerate(images.items())}
    with torch.autocast('cuda'):
        score, query = model(partial, label=target, cam_label=cam, view_label=scene, partial=True)
        partial_ce = xent(score, target)
        cross_triplet, pos, neg = partial_gallery_triplet(query, gallery, target)
        partial_graph, partial_detail = identity_alignment_loss(query[:, 5120:], reference,
            target, names, model.graph_temperature)
        partial_loss = .25 * partial_ce + .5 * cross_triplet
        if model.graph_weight > 0:
            partial_loss = partial_loss + .5 * model.graph_weight * partial_graph
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
        optimizer_updated=scaler.get_scale() >= previous_scale, amp_scale=scaler.get_scale(),
        names=list(names), graph_step=model.graph_step,
        graph_weight=model.graph_weight, graph_temperature=model.graph_temperature,
        graph_reference_set=''.join('RNT'[i] for i in reference_set),
        graph_query_sets=['RNT', ''.join('RNT'[i] for i in retained)],
        graph_full_raw=float(full_graph.detach()), graph_partial_raw=float(partial_graph.detach()),
        graph_full_audit=full_detail, graph_partial_audit=partial_detail,
        graph_reference_requires_grad=reference.requires_grad,
        graph_reference='Stopped shared projection of only selected real sources, from the same full-training backbone pass; positive observations are distinct.',
        graph_extra_backbone_passes=0)
    assert not gallery.requires_grad and not reference.requires_grad
    if variant != 'demo_shared':
        detail.update(model.relation_audit)
        detail.update(model.alignment_audit)
        detail.update(model.modality_alignment_audit)
    model.graph_step += 1
    return detail
