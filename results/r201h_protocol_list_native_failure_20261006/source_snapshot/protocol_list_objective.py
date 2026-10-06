"""Known Smooth-AP with label-only or official same-ID/same-camera junk rules."""
import torch
from torch.nn import functional as F


def ap_from_scores(scores, labels, cameras, reference, temperature=.01):
    assert reference in ('label', 'camera') and temperature==.01
    assert scores.shape==(len(labels),len(labels)) and cameras.shape==labels.shape
    diagonal=torch.eye(len(labels),device=scores.device,dtype=torch.bool)
    same=labels[:,None].eq(labels[None,:])
    same_camera=cameras[:,None].eq(cameras[None,:])
    eligible=~diagonal if reference=='label' else ~(same & same_camera)
    positives=same & eligible
    valid=positives.any(1)
    detail=dict(reference_mask=reference,queries=len(labels),valid_AP_queries=int(valid.sum()),
        label_positive_pairs=int((same & ~diagonal).sum()),used_positive_pairs=int(positives.sum()),
        same_camera_label_positive_pairs=int((same & same_camera & ~diagonal).sum()),
        zero_valid_batch=not bool(valid.any()),temperature=temperature)
    if not valid.any():
        # Actual G201 PK logs contain 85/2647 batches with no cross-camera positives.
        # Other objectives still use the complete batch; no record/identity is dropped.
        return scores.sum()*0,detail
    assert (eligible & ~positives).any(1)[valid].all()
    difference=scores[:,None,:]-scores[:,:,None]
    outranking=torch.sigmoid(difference/temperature)
    competitors=eligible[:,None,:] & ~diagonal[None,:,:]
    all_rank=1+(outranking*competitors).sum(2)
    positive_rank=1+(outranking*competitors*positives[:,None,:]).sum(2)
    ap=((positive_rank/all_rank)*positives).sum(1)/positives.sum(1).clamp_min(1)
    return 1-ap[valid].mean(),detail


def smooth_ap(features, labels, cameras, reference):
    with torch.autocast(features.device.type,enabled=False):
        normalized=F.normalize(features.float(),dim=1)
        return ap_from_scores(normalized@normalized.T,labels,cameras,reference)
