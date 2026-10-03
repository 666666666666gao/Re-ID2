"""Separate residual magnitude and reference-gallery interventions after V3 collapse."""
import torch
from torch.nn import functional as F

from axis_collaboration import AxisCollaborationDeMo


VARIANTS = {
    'axis_scaled_base': ('axis_collaboration', True, False),
    'axis_raw_fullref': ('axis_collaboration', False, True),
    'axis_scaled_fullref': ('axis_collaboration', True, True),
    'plain_scaled_fullref': ('plain_twins', True, True),
    'frequency_scaled_fullref': ('ordinary_frequency', True, True),
}


@torch.no_grad()
def full_reference_targets(states, labels):
    reference = F.normalize(states['11'].float(), dim=1)
    with torch.autocast('cuda', enabled=False):
        similarity = reference @ reference.T
    same = labels[:, None].eq(labels[None])
    positive = same & ~torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    negative = ~same
    assert positive.any(1).all() and negative.any(1).all()
    pos = similarity.masked_fill(~positive, torch.inf).argmin(1)
    neg = similarity.masked_fill(~negative, -torch.inf).argmax(1)
    direction = reference[pos] - reference[neg]
    scores = {key: (F.normalize(value.float(), dim=1) * direction).sum(1) for key, value in states.items()}
    target = torch.stack((scores['11'] - scores['01'], scores['11'] - scores['10'],
                          scores['11'] - scores['10'] - scores['01'] + scores['00']), 1)
    return target, pos, neg, scores


class ScaledAxisCollaborationDeMo(AxisCollaborationDeMo):
    def __init__(self, classes, cfg, cameras, variant):
        base_variant, normalize_residual, full_reference = VARIANTS[variant]
        super().__init__(classes, cfg, cameras, base_variant)
        self.normalize_residual, self.full_reference = normalize_residual, full_reference
        if self.full_reference:
            self.calibrator.targets = full_reference_targets

    def fuse(self, base, modality, frequency, gates, use_m, use_f, eligible):
        if not self.normalize_residual:
            return super().fuse(base, modality, frequency, gates, use_m, use_f, eligible)
        # V3 residual/base norm ratios reached 1e-14..1e-5. Use relative
        # directions and stopped base norms instead of the raw projection size.
        with torch.autocast('cuda', enabled=False):
            base, modality, frequency, gates = (value.float() for value in (base, modality, frequency, gates))
            batch, _, dim = modality.shape
            anchor_m = base[:, 3 * dim:10 * dim].reshape(batch, 7, dim).norm(dim=-1, keepdim=True).detach()
            anchor_f = base.norm(dim=1, keepdim=True).detach()
            delta_m = torch.zeros_like(base[:, 3 * dim:10 * dim])
            delta_f = torch.zeros_like(frequency)
            if use_m:
                projected = F.normalize(self.modality_projection(modality), dim=-1) * anchor_m * eligible[:, :, None]
                delta_m = self.residual_scale[0] * gates[:, :1] * projected.flatten(1)
            if use_f:
                projected = F.normalize(self.frequency_projection(frequency), dim=1) * anchor_f
                delta_f = self.residual_scale[1] * gates[:, 1:2] * projected
            if use_m and use_f:
                interaction = F.normalize(self.interaction_projection(torch.cat((modality.mean(1), frequency), -1)), dim=1)
                projected = interaction[:, None].expand(batch, 7, dim) * anchor_m * eligible[:, :, None]
                delta_m = delta_m + self.residual_scale[2] * gates[:, 2:] * projected.flatten(1)
            if self.frequency_only:
                delta_f = delta_f + delta_m.reshape(batch, 7, dim).mean(1)
                delta_m = torch.zeros_like(delta_m)
            return base + torch.cat((torch.zeros_like(base[:, :3 * dim]), delta_m, delta_f), -1)

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        if self.training and self.full_reference:
            self.contribution_audit['reference'] = 'one stopped-gradient current-batch full11 gallery; same indices for all four states, no EMA'
        return result
