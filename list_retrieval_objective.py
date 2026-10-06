"""Label-defined Smooth-AP for the actual normalized deployment descriptor.

Reference: Brown et al., Smooth-AP, ECCV2020, arXiv:2007.12163.
Training positives are other observations with the same training identity;
camera-agnostic, matching the existing supervised training protocol. Official
evaluation retains its camera exclusions. This is a known list objective, not
a claimed new method. Actual RGBNT201 B64 batches have unique filenames.
"""
import torch
from torch.nn import functional as F


def smooth_ap(features, labels, temperature=.01):
    with torch.autocast(features.device.type, enabled=False):
        features = F.normalize(features.float(), dim=1)
        scores = features @ features.T
        count = len(labels)
        diagonal = torch.eye(count, device=features.device, dtype=torch.bool)
        eligible = ~diagonal
        positives = labels[:, None].eq(labels[None, :]) & eligible
        assert positives.any(1).all()
        assert (eligible & ~positives).any(1).all()
        # Axes: query i, positive candidate k, competing gallery candidate j.
        difference = scores[:, None, :] - scores[:, :, None]
        outranking = torch.sigmoid(difference / temperature)
        competitors = eligible[:, None, :] & ~diagonal[None, :, :]
        all_rank = 1 + (outranking * competitors).sum(2)
        positive_rank = 1 + (outranking * competitors * positives[:, None, :]).sum(2)
        ap = ((positive_rank / all_rank) * positives).sum(1) / positives.sum(1)
        return 1 - ap.mean()
