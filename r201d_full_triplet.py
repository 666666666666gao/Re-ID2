"""R201D: retain unit retrieval/CE, restore raw fusion only for full Triplet."""
import torch

from rgbnt201_identity_outlet import IdentityCoordinateAxis
import identity_coordinate_three_dataset as coordinates


class RawFullTripletAxis(IdentityCoordinateAxis):
    def fuse(self, base, modality, frequency, gates, use_m, use_f, eligible,
             relation_mass=None, relation_frequency=None):
        unit = super().fuse(base, modality, frequency, gates, use_m, use_f,
                            eligible, relation_mass, relation_frequency)
        if self.training and use_m and use_f:
            # The first1536 globals are unchanged by fusion. Their norm ratio
            # recovers the pre-normalization norm without changing old sources.
            with torch.autocast('cuda', enabled=False):
                scale = base[:, :1536].float().norm(dim=1, keepdim=True) / unit[:, :1536].norm(dim=1, keepdim=True)
                self.full_metric_feature = unit * scale
        return unit

    def forward(self, x, label=None, cam_label=None, view_label=None, partial=False, return_states=False):
        output = super().forward(x, label, cam_label, view_label, partial, return_states)
        if self.training and not partial:
            raw = self.full_metric_feature
            self.full_metric_audit = dict(raw_norm_mean=float(raw.detach().norm(dim=1).mean()),
                unit_norm_max_error=float((output[1].detach().norm(dim=1) - 1).abs().max()),
                normalized_raw_max_error=float((torch.nn.functional.normalize(raw.detach(), dim=1) - output[1].detach()).abs().max()))
            return (output[0], raw, *output[2:])
        return output


def build(args, cfg, classes, cameras):
    args.bypass = 0
    args.full_triplet_feature = 'pre_normalization_fused_private5120'
    coordinates.IdentityCoordinateAxis = RawFullTripletAxis
    return coordinates.build(args, cfg, classes, cameras)
