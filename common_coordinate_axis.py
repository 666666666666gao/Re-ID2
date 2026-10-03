"""V12 single change: all augmented controls place M/F/I increments in common coordinates."""
import torch
from torch.nn import functional as F

from experiment_data import seed_all
from shared_identity_axis import SharedIdentityAxis, SharedIdentityDeMo, metric_descriptor


class CommonCoordinateAxis(SharedIdentityAxis):
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
            # Exactly the V11 ordinary-frequency interface, now shared by all controls.
            delta_f = delta_f + delta_m.reshape(batch, 7, dim).mean(1)
            return metric_descriptor(torch.cat((base[:, :5120], base[:, 5120:] + delta_f), 1))


def build(args, cfg, classes, cameras):
    seed_all(args.seed)
    if args.variant == 'demo_shared':
        model = SharedIdentityDeMo(classes, cfg, cameras, args.seed)
    else:
        model = CommonCoordinateAxis(classes, cfg, cameras, args.variant, args.seed)
    return model.float().cuda()
