"""M1: compare only the seven-relation common outlet, with active matched queries."""
import math
import os

import torch
from torch import nn
from torch.nn import functional as F

from common_coordinate_axis import CommonCoordinateAxis
from experiment_data import seed_all
from gpu_thermal_execute import check_limits
from shared_identity_axis import SharedIdentityDeMo, metric_descriptor


POOLING = ('original_mean', 'eligible_mean', 'seed_query')


class SharedReadout(nn.Module):
    """The same query residual is trained in every pooling control, including DeMo."""
    def __init__(self, projection, seed):
        super().__init__()
        self.projection = projection
        generator = torch.Generator().manual_seed(seed + 20000)
        self.query = nn.Parameter(torch.randn(512, generator=generator) * .02)

    def forward(self, value):
        result = self.projection(value)
        return result + self.query.to(dtype=result.dtype)


def outlet_weights(pooling, query, modality, eligible):
    if pooling == 'original_mean':
        return eligible.float() / 7
    if pooling == 'eligible_mean':
        return eligible.float() / eligible.sum(1, keepdim=True)
    assert pooling == 'seed_query'
    # Keys are actual routed relation evidence, before PM; invalid sources are masked.
    logits = (F.normalize(modality.float(), dim=-1) * F.normalize(query.float(), dim=0)).sum(-1)
    return (logits * math.sqrt(modality.shape[-1])).masked_fill(~eligible, -torch.inf).softmax(1)


class CommonOutletDeMo(SharedIdentityDeMo):
    def __init__(self, classes, cfg, cameras, seed):
        super().__init__(classes, cfg, cameras, seed)
        self.shared_projection = SharedReadout(self.shared_projection, seed)


class CommonOutletAxis(CommonCoordinateAxis):
    def __init__(self, classes, cfg, cameras, variant, seed, pooling):
        super().__init__(classes, cfg, cameras, variant, seed)
        assert pooling in POOLING
        self.pooling = pooling
        self.shared_projection = SharedReadout(self.shared_projection, seed)

    def fuse(self, base, modality, frequency, gates, use_m, use_f, eligible, relation_mass=None):
        if self.pooling == 'original_mean':
            if use_m and use_f:
                self.last_outlet_weights = eligible.float() / 7
            return super().fuse(base, modality, frequency, gates, use_m, use_f, eligible, relation_mass)
        with torch.autocast('cuda', enabled=False):
            base, modality, frequency, gates = (value.float() for value in (base, modality, frequency, gates))
            batch, _, dim = modality.shape
            anchor_m = base[:, 1536:5120].reshape(batch, 7, dim).norm(dim=-1, keepdim=True).detach()
            anchor_f = base[:, 5120:].norm(dim=1, keepdim=True).detach()
            delta_m = torch.zeros_like(modality)
            delta_f = torch.zeros_like(frequency)
            if use_m:
                if use_f:
                    assert relation_mass is not None
                else:
                    logits = self.router.relation_score(modality).squeeze(-1)
                    relation_mass = logits.masked_fill(~eligible, -torch.inf).softmax(1)
                weights = relation_mass * eligible.sum(1, keepdim=True)
                projected = F.normalize(self.modality_projection(modality), dim=-1) * anchor_m * eligible[..., None]
                delta_m = self.residual_scale[0] * gates[:, :1, None] * (projected * weights[..., None])
            if use_f:
                delta_f = self.residual_scale[1] * gates[:, 1:2] * F.normalize(self.frequency_projection(frequency), dim=1) * anchor_f
            if use_m and use_f:
                interaction = F.normalize(self.interaction_projection(torch.cat((modality.mean(1), frequency), -1)), dim=1)
                projected = interaction[:, None] * anchor_m * eligible[..., None] * weights[..., None]
                delta_m = delta_m + self.residual_scale[2] * gates[:, 2:, None] * projected
            if use_m:
                coefficients = outlet_weights(self.pooling, self.shared_projection.query, modality, eligible)
                delta_f = delta_f + (delta_m * coefficients[..., None]).sum(1)
                if use_f:
                    self.last_outlet_weights = coefficients.detach()
            return metric_descriptor(torch.cat((base[:, :5120], base[:, 5120:] + delta_f), 1))


def build(args, cfg, classes, cameras):
    assert args.pooling in POOLING
    check_limits(int(os.environ['CUDA_VISIBLE_DEVICES']))
    seed_all(args.seed)
    if args.variant == 'demo_shared':
        assert args.pooling == 'original_mean'
        model = CommonOutletDeMo(classes, cfg, cameras, args.seed)
    else:
        model = CommonOutletAxis(classes, cfg, cameras, args.variant, args.seed, args.pooling)
    return model.float().cuda()
