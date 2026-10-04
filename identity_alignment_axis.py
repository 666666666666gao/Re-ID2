"""M2b: align actual routed PF with a stopped public identity gallery."""
import os

import torch
from torch.nn import functional as F

from experiment_data import seed_all
from frequency_relation_axis import FrequencyRelationAxis
from gpu_thermal_execute import check_limits


def identity_alignment_loss(student, reference, labels, names, temperature):
    assert len(student) == len(reference) == len(labels) == len(names)
    with torch.autocast('cuda', enabled=False):
        query = F.normalize(student.float(), dim=1)
        gallery = F.normalize(reference.detach().float(), dim=1)
        different_observation = torch.tensor([[q != g for g in names] for q in names],
            device=labels.device, dtype=torch.bool)
        same_identity = labels[:, None].eq(labels[None])
        positive = same_identity & different_observation
        valid = positive.any(1)
        assert valid.any() and (~same_identity[valid]).any(1).all()
        logits = (query[valid] @ gallery.T) / temperature
        log_probability = logits - torch.logsumexp(
            logits.masked_fill(~different_observation[valid], -torch.inf), dim=1, keepdim=True)
        loss = -(log_probability.masked_fill(~positive[valid], 0).sum(1) / positive[valid].sum(1)).mean()
        assert torch.isfinite(loss) and not gallery.requires_grad
    return loss, dict(identity_alignment_valid_anchors=int(valid.sum()),
        identity_alignment_excluded_anchors=int((~valid).sum()),
        identity_alignment_positive_pairs=int(positive.sum()),
        identity_alignment_reference_requires_grad=gallery.requires_grad,
        identity_alignment_positive_observations_distinct=True)


class IdentityAlignmentAxis(FrequencyRelationAxis):
    def __init__(self, classes, cfg, cameras, variant, seed, relation_weight, alignment_weight, temperature):
        super().__init__(classes, cfg, cameras, variant, seed, relation_weight)
        self.alignment_weight = alignment_weight
        self.alignment_temperature = temperature
        self.alignment_names = None
        self.alignment_labels = None
        self.last_identity_alignment_loss = None
        self.alignment_audit = {}

    def fuse(self, base, modality, frequency, gates, use_m, use_f, eligible, relation_mass=None):
        measure = self.training and not self.partial_relation and use_m and use_f and self.alignment_weight > 0
        captured = []
        if measure:
            handle = self.frequency_projection.register_forward_hook(
                lambda module, inputs, output: captured.append(output))
        result = super().fuse(base, modality, frequency, gates, use_m, use_f, eligible, relation_mass)
        if measure:
            handle.remove()
            assert len(captured) == 1
            self.last_identity_alignment_loss, detail = identity_alignment_loss(
                captured[0], base[:, 5120:], self.alignment_labels, self.alignment_names, self.alignment_temperature)
            self.alignment_audit = dict(**detail, identity_alignment_raw=float(self.last_identity_alignment_loss.detach()),
                identity_alignment_weight=self.alignment_weight, identity_alignment_temperature=self.alignment_temperature,
                identity_alignment_reference='same_full_view_stopped_base_common_other_observations')
        return result

    def forward(self, x, label=None, cam_label=None, view_label=None, partial=False, return_states=False):
        if self.training and not partial:
            self.alignment_labels = label
            self.last_identity_alignment_loss = None
        result = super().forward(x, label=label, cam_label=cam_label, view_label=view_label,
            partial=partial, return_states=return_states)
        if self.training and not partial and self.alignment_weight > 0:
            assert self.last_identity_alignment_loss is not None and len(result) % 2 == 1
            result = (*result[:-1], result[-1] + self.alignment_weight * self.last_identity_alignment_loss)
        return result


def build(args, cfg, classes, cameras):
    assert args.pooling == 'original_mean' and args.variant in ('axis_shared', 'frequency_shared', 'twins_shared')
    check_limits(int(os.environ['CUDA_VISIBLE_DEVICES']))
    seed_all(args.seed)
    return IdentityAlignmentAxis(classes, cfg, cameras, args.variant, args.seed,
        args.relation_weight, args.alignment_weight, args.alignment_temperature).float().cuda()
