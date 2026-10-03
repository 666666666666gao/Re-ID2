"""Available DeMo coordinates plus a shared identity reference and routed increments."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from axis_collaboration import AxisCollaborationDeMo
from dual_axis import RELATIONS
from modeling.make_model import DeMo, __factory_T_type as factory
from modeling.meta_arch import weights_init_classifier, weights_init_kaiming
from scaled_axis_collaboration import full_reference_targets


VARIANTS = ('axis_shared', 'frequency_shared', 'twins_shared', 'demo_shared')
KEYS = ('RGB', 'NI', 'TI')
COMMON_WEIGHT = .25


def metric_descriptor(value):
    """The common coordinates have .25 of the cosine metric, not amplitude .25."""
    with torch.autocast('cuda', enabled=False):
        value = value.float()
        return torch.cat((math.sqrt(1 - COMMON_WEIGHT) * F.normalize(value[:, :5120], dim=1),
                          math.sqrt(COMMON_WEIGHT) * F.normalize(value[:, 5120:], dim=1)), 1)


def initialize_shared(model, classes, seed):
    # Same shared head initialization for DeMo and all augmented controls.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 10000)
        model.shared_projection = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 512),
                                                nn.GELU(), nn.LayerNorm(512))
        model.shared_neck = nn.BatchNorm1d(512)
        model.shared_neck.bias.requires_grad_(False)
        model.shared_neck.apply(weights_init_kaiming)
        model.shared_classifier = nn.Linear(512, classes, bias=False)
        model.shared_classifier.apply(weights_init_classifier)


def available_base_fusion(module, patches, globals_, available):
    """Uniform availability per batch; skip BN experts for absent relations."""
    assert torch.equal(available, available[:1].expand_as(available))
    assert module.HDM and module.ATM
    if bool(available.all()):
        result = module(*patches, *globals_)
        return result[0] if module.training else result
    batch = len(available)
    eligible = torch.stack([available[:, subset].all(1) for subset in RELATIONS], 1)
    relations = torch.stack([value.reshape(batch, module.feat_dim)
                             for value in module.forward_HDM(*patches, *globals_)], 1)
    relations = relations * eligible[..., None]
    moe = module.moe
    gate = moe.gating_network.gate
    query = gate.linear_re(relations.flatten(1))
    q = gate.q_(query).reshape(batch, 1, gate.num_heads, module.feat_dim // gate.num_heads).permute(0, 2, 1, 3)
    k = gate.k_(relations).reshape(batch, 7, gate.num_heads, module.feat_dim // gate.num_heads).permute(0, 2, 1, 3)
    logits = (q @ k.transpose(-2, -1)) * gate.scale
    weights = logits.masked_fill(~eligible[:, None, None, :], -torch.inf).softmax(-1)
    chunks = relations.chunk(moe.head, -1)
    heads = []
    for index, head in enumerate(moe.experts):
        outputs = [expert(chunks[index][:, relation]) if bool(eligible[0, relation])
                   else torch.zeros_like(chunks[index][:, relation])
                   for relation, expert in enumerate(head.expertHead)]
        heads.append(torch.stack(outputs, 1) * weights[:, index].squeeze(1)[..., None])
    return torch.cat(heads, -1).flatten(1)


def encode_available(model, x, cam_label, view_label):
    available = torch.stack([x[key].flatten(1).ne(0).any(1) for key in KEYS], 1)
    assert available.any(1).all()
    assert torch.equal(available, available[:1].expand_as(available)), 'feature banks and partial training use one availability set per batch'
    patches, globals_, common = [None] * 3, [None] * 3, []
    for index, (key, reduce) in enumerate(zip(KEYS, (model.rgb_reduce, model.nir_reduce, model.tir_reduce))):
        if bool(available[0, index]):
            patch, cls = model.BACKBONE(x[key], cam_label=cam_label, view_label=view_label)
            local = model.pool(patch.permute(0, 2, 1)).squeeze(-1)
            raw = torch.cat((cls, local), -1)
            patches[index], globals_[index] = patch, reduce(raw)
            common.append(raw)
    first = next(index for index in range(3) if patches[index] is not None)
    for index in range(3):
        if patches[index] is None:
            patches[index] = torch.zeros_like(patches[first])
            globals_[index] = torch.zeros_like(globals_[first])
    shared = model.shared_projection(torch.stack(common, 1).mean(1))
    original = torch.cat(globals_, -1)
    moe = available_base_fusion(model.generalFusion, patches, globals_, available)
    base = torch.cat((original, moe, shared), -1)
    eligible = torch.stack([available[:, subset].all(1) for subset in RELATIONS], 1)
    return patches, globals_, base, eligible, available, shared


def base_supervision(model, globals_, base, shared):
    original, moe = base[:, :1536], base[:, 1536:5120]
    output = [model.classifier_moe(model.bottleneck_moe(moe)), moe,
              model.shared_classifier(model.shared_neck(shared)), shared]
    assert not model.direct
    for feature, classifier, neck in zip(globals_, (model.classifier_r, model.classifier_n, model.classifier_t),
                                        (model.bottleneck_r, model.bottleneck_n, model.bottleneck_t)):
        output.extend((classifier(neck(feature)), feature))
    return output


class SharedIdentityDeMo(DeMo):
    def __init__(self, classes, cfg, cameras, seed):
        super().__init__(classes, cfg, cameras, 0, factory)
        initialize_shared(self, classes, seed)
        self.fused_neck = nn.BatchNorm1d(5632)
        self.fused_neck.bias.requires_grad_(False)
        self.fused_neck.apply(weights_init_kaiming)
        self.fused_classifier = nn.Linear(5632, classes, bias=False)
        self.fused_classifier.apply(weights_init_classifier)
        self.variant = 'demo_shared'
        self.loss_weights = [1., 1., .25, 1., 1., 1.]

    def forward(self, x, label=None, cam_label=None, view_label=None, partial=False, return_states=False):
        _, globals_, base, _, _, shared = encode_available(self, x, cam_label, view_label)
        fused = metric_descriptor(base)
        if not self.training:
            return {'00': fused, '10': fused, '01': fused, '11': fused} if return_states else fused
        output = [self.fused_classifier(self.fused_neck(fused)), fused]
        if not partial:
            output.extend(base_supervision(self, globals_, base, shared))
        return tuple(output)


class SharedIdentityAxis(AxisCollaborationDeMo):
    def __init__(self, classes, cfg, cameras, variant, seed):
        base_variant = {'axis_shared': 'axis_collaboration', 'frequency_shared': 'ordinary_frequency',
                        'twins_shared': 'plain_twins'}[variant]
        super().__init__(classes, cfg, cameras, base_variant)
        self.variant = variant
        initialize_shared(self, classes, seed)
        self.calibrator.targets = full_reference_targets
        self.loss_weights = [1., .1, .1, 1., .25, 1., 1., 1.]

    def fuse(self, base, modality, frequency, gates, use_m, use_f, eligible, relation_mass=None):
        with torch.autocast('cuda', enabled=False):
            base, modality, frequency, gates = (value.float() for value in (base, modality, frequency, gates))
            batch, _, dim = modality.shape
            anchor_m = base[:, 1536:5120].reshape(batch, 7, dim).norm(dim=-1, keepdim=True).detach()
            anchor_f = base[:, 5120:].norm(dim=1, keepdim=True).detach()
            delta_m, delta_f = torch.zeros_like(base[:, 1536:5120]), torch.zeros_like(frequency)
            if use_m:
                if use_f:
                    assert relation_mass is not None
                else:
                    logits = self.router.relation_score(modality).squeeze(-1)
                    relation_mass = logits.masked_fill(~eligible, -torch.inf).softmax(1)
                weights = relation_mass * eligible.sum(1, keepdim=True)
                projected = F.normalize(self.modality_projection(modality), dim=-1) * anchor_m * eligible[..., None]
                delta_m = self.residual_scale[0] * gates[:, :1] * (projected * weights[..., None]).flatten(1)
            if use_f:
                delta_f = self.residual_scale[1] * gates[:, 1:2] * F.normalize(self.frequency_projection(frequency), dim=1) * anchor_f
            if use_m and use_f:
                interaction = F.normalize(self.interaction_projection(torch.cat((modality.mean(1), frequency), -1)), dim=1)
                projected = interaction[:, None] * anchor_m * eligible[..., None] * weights[..., None]
                delta_m = delta_m + self.residual_scale[2] * gates[:, 2:] * projected.flatten(1)
            if self.frequency_only:
                delta_f = delta_f + delta_m.reshape(batch, 7, dim).mean(1)
                delta_m = torch.zeros_like(delta_m)
            return metric_descriptor(base + torch.cat((torch.zeros_like(base[:, :1536]), delta_m, delta_f), 1))

    def controlled_states(self, base, m0, f0, eligible, available, full):
        modality = self.modality_expert.pool(m0, eligible)
        frequency = self.frequency_expert.pool(f0, available, self.structured)
        _, gm = self.calibrator(base, modality, torch.zeros_like(frequency))
        _, gf = self.calibrator(base, torch.zeros_like(modality), frequency)
        return {'00': metric_descriptor(base),
                '10': self.fuse(base, modality, torch.zeros_like(frequency), gm, True, False, eligible),
                '01': self.fuse(base, torch.zeros_like(modality), frequency, gf, False, True, eligible), '11': full}

    def forward(self, x, label=None, cam_label=None, view_label=None, partial=False, return_states=False):
        patches, globals_, base, eligible, available, shared = encode_available(self, x, cam_label, view_label)
        spatial = torch.stack(patches, 1)
        tokens = self.bands(spatial) if self.structured or self.frequency_only else spatial[:, :, None].expand(-1, -1, 3, -1, -1)
        mvalues = self.modality_expert.prepare(tokens, available)
        fvalues = self.frequency_expert.prepare(tokens, available, self.structured)
        m0 = self.modality_expert.read(mvalues, eligible, self.structured)
        f0 = self.frequency_expert.read(fvalues, available, self.structured)
        cm, cf = self.router.conditions(m0, f0)
        m1 = self.modality_expert.read(mvalues, eligible, self.structured, cm)
        f1 = self.frequency_expert.read(fvalues, available, self.structured, cf)
        modality, frequency, route = self.router.route(m1, f1, eligible, self.joint_interaction)
        prediction, gates = self.calibrator(base, modality, frequency)
        fused = self.fuse(base, modality, frequency, gates, True, True, eligible, route.sum(2))
        self.last_route, self.last_gates = route.detach(), gates.detach()
        if not self.training and not return_states:
            return fused
        if self.training and partial:
            return self.fused_classifier(self.fused_neck(fused)), fused
        states = self.controlled_states(base, m0, f0, eligible, available, fused)
        if not self.training:
            return states
        target, pos, neg, scores = self.calibrator.targets(states, label)
        contribution = F.smooth_l1_loss(prediction.float(), target)
        self.contribution_audit = {'target_requires_grad': target.requires_grad,
                                  'positive_indices': pos.tolist(), 'negative_indices': neg.tolist(),
                                  'targets': target.tolist(), 'predictions': prediction.detach().float().tolist(),
                                  'scores': {key: value.tolist() for key, value in scores.items()}, 'loss': float(contribution.detach())}
        standalone_m = self.modality_expert.pool(m0, eligible).mean(1)
        standalone_f = self.frequency_expert.pool(f0, available, self.structured)
        output = [self.fused_classifier(self.fused_neck(fused)), fused,
                  self.modality_classifier(self.modality_neck(standalone_m)), standalone_m,
                  self.frequency_classifier(self.frequency_neck(standalone_f)), standalone_f]
        output.extend(base_supervision(self, globals_, base, shared))
        output.append(self.contribution_loss_weight * contribution)
        return tuple(output)


def partial_gallery_triplet(query, gallery, labels):
    """Different availability, same training identities; gallery is stopped."""
    with torch.autocast('cuda', enabled=False):
        query = F.normalize(query.float(), dim=1)
        reference = F.normalize(gallery.detach().float(), dim=1)
        similarity = query @ reference.T
        same = labels[:, None].eq(labels[None])
        positive = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        negative = ~same
        assert positive.any(1).all() and negative.any(1).all()
        pos = similarity.detach().masked_fill(~positive, torch.inf).argmin(1)
        neg = similarity.detach().masked_fill(~negative, -torch.inf).argmax(1)
        rows = torch.arange(len(labels), device=labels.device)
        positive_distance = (2 - 2 * similarity[rows, pos]).clamp(min=1e-12).sqrt()
        negative_distance = (2 - 2 * similarity[rows, neg]).clamp(min=1e-12).sqrt()
        loss = F.softplus(positive_distance - negative_distance).mean()
    return loss, pos, neg
