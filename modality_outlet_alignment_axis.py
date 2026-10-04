"""M4: supervise the actual routed PM common-coordinate outlet, without new parameters."""
import os

import torch
from torch.nn import functional as F

from experiment_data import seed_all
from gpu_thermal_execute import check_limits
from identity_alignment_axis import identity_alignment_loss
from measurement_gate_axis import MeasurementGateAxis


def modality_outlet(projected, base, eligible, relation_mass):
    with torch.autocast('cuda', enabled=False):
        batch, _, dim = projected.shape
        anchor = base.float()[:, 1536:5120].reshape(batch, 7, dim).norm(dim=-1, keepdim=True).detach()
        weights = relation_mass.float() * eligible.sum(1, keepdim=True)
        rows = F.normalize(projected.float(), dim=-1) * anchor * eligible[..., None]
        return (rows * weights[..., None]).mean(1)


class ModalityOutletAlignmentAxis(MeasurementGateAxis):
    def __init__(self, *args, modality_alignment_weight):
        super().__init__(*args)
        self.modality_alignment_weight = modality_alignment_weight
        self.last_modality_alignment_loss = None
        self.modality_alignment_audit = {}

    def fuse(self, base, modality, frequency, gates, use_m, use_f, eligible, relation_mass=None):
        measure = self.training and not self.partial_relation and use_m and use_f and self.modality_alignment_weight > 0
        captured = []
        if measure:
            handle = self.modality_projection.register_forward_hook(lambda module, inputs, output: captured.append(output))
        result = super().fuse(base, modality, frequency, gates, use_m, use_f, eligible, relation_mass)
        if measure:
            handle.remove()
            assert len(captured) == 1 and relation_mass is not None
            self.last_modality_outlet = modality_outlet(captured[0], base, eligible, relation_mass)
            self.last_modality_alignment_loss, detail = identity_alignment_loss(self.last_modality_outlet,
                base[:, 5120:], self.alignment_labels, self.alignment_names, self.alignment_temperature)
            self.modality_alignment_audit = dict(**detail,
                modality_alignment_raw=float(self.last_modality_alignment_loss.detach()),
                modality_alignment_weight=self.modality_alignment_weight,
                modality_alignment_temperature=self.alignment_temperature,
                modality_alignment_reference='same_full_view_stopped_base_common_other_observations',
                modality_alignment_student='actual_PM_normalized_anchor_relation_mass_mean7_before_gate',
                modality_alignment_eligible_relations=eligible.sum(1).detach().cpu().tolist())
        return result

    def forward(self, x, label=None, cam_label=None, view_label=None, partial=False, return_states=False):
        if self.training and not partial:
            self.last_modality_alignment_loss = None
        result = super().forward(x, label=label, cam_label=cam_label, view_label=view_label,
            partial=partial, return_states=return_states)
        if self.training and not partial and self.modality_alignment_weight > 0:
            assert self.last_modality_alignment_loss is not None and len(result) % 2 == 1
            result = (*result[:-1], result[-1] + self.modality_alignment_weight * self.last_modality_alignment_loss)
        return result


def build(args, cfg, classes, cameras):
    assert args.variant in ('axis_shared', 'frequency_shared', 'twins_shared')
    assert args.gate_gradient_mode == 'measurement_only' and args.pooling == 'original_mean'
    assert (args.relation_weight, args.alignment_weight, args.alignment_temperature) == (.1, .1, .07)
    assert args.modality_alignment_weight in (0, .1)
    check_limits(int(os.environ['CUDA_VISIBLE_DEVICES']))
    seed_all(args.seed)
    return ModalityOutletAlignmentAxis(classes, cfg, cameras, args.variant, args.seed,
        args.relation_weight, args.alignment_weight, args.alignment_temperature,
        modality_alignment_weight=args.modality_alignment_weight).float().cuda()
