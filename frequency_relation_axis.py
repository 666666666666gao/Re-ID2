"""M2: preserve actual routed F neighborhoods across the existing PF outlet."""
import os

import torch
from torch.nn import functional as F

from common_outlet_axis import CommonOutletAxis
from experiment_data import seed_all
from gpu_thermal_execute import check_limits


def normalized_distances(value):
    value = F.normalize(value.float(), dim=1)
    mask = ~torch.eye(len(value), device=value.device, dtype=torch.bool)
    distances = (2 - 2 * (value @ value.T)).clamp(min=1e-12).sqrt()[mask]
    return distances / distances.mean()


def relation_loss(student, reference):
    with torch.autocast('cuda', enabled=False):
        target = normalized_distances(reference.detach())
        prediction = normalized_distances(student)
        assert not target.requires_grad
        return F.smooth_l1_loss(prediction, target), target


class FrequencyRelationAxis(CommonOutletAxis):
    def __init__(self, classes, cfg, cameras, variant, seed, weight):
        super().__init__(classes, cfg, cameras, variant, seed, 'original_mean')
        self.relation_weight = weight
        self.partial_relation = False
        self.last_frequency_relation_loss = None
        self.relation_audit = {}

    def fuse(self, base, modality, frequency, gates, use_m, use_f, eligible, relation_mass=None):
        measure = self.training and not self.partial_relation and use_m and use_f and self.relation_weight > 0
        captured = []
        if measure:
            handle = self.frequency_projection.register_forward_hook(
                lambda module, inputs, output: captured.append((inputs[0], output)))
        result = super().fuse(base, modality, frequency, gates, use_m, use_f, eligible, relation_mass)
        if measure:
            handle.remove()
            assert len(captured) == 1
            reference, student = captured[0]
            self.last_frequency_relation_loss, target = relation_loss(student, reference)
            assert torch.isfinite(self.last_frequency_relation_loss)
            self.relation_audit = dict(frequency_relation_raw=float(self.last_frequency_relation_loss.detach()),
                frequency_relation_weight=self.relation_weight, frequency_relation_target_requires_grad=target.requires_grad,
                frequency_relation_pairs=target.numel(), frequency_relation_reference='actual_routed_same_full_view_F_pre')
        return result

    def forward(self, x, label=None, cam_label=None, view_label=None, partial=False, return_states=False):
        self.partial_relation = partial
        if self.training and not partial:
            self.last_frequency_relation_loss = None
        result = super().forward(x, label=label, cam_label=cam_label, view_label=view_label,
                                 partial=partial, return_states=return_states)
        if self.training and not partial and self.relation_weight > 0:
            assert self.last_frequency_relation_loss is not None and len(result) % 2 == 1
            result = (*result[:-1], result[-1] + self.relation_weight * self.last_frequency_relation_loss)
        return result


def build(args, cfg, classes, cameras):
    assert args.pooling == 'original_mean' and args.variant in ('axis_shared', 'frequency_shared', 'twins_shared')
    check_limits(int(os.environ['CUDA_VISIBLE_DEVICES']))
    seed_all(args.seed)
    return FrequencyRelationAxis(classes, cfg, cameras, args.variant, args.seed, args.relation_weight).float().cuda()
