"""Limit the complete expert update in the deployed unit identity coordinates."""
import math
from types import MethodType

import torch
from torch.nn import functional as F


MAX_UNIT_SHIFT = .10


def bound_descriptor(reference, candidate, epsilon=MAX_UNIT_SHIFT):
    """Preserve inactive candidates exactly and bound normalized distance to b.

    For unit b and any perturbation of norm <=r<1, the normalized result is
    at distance <=sqrt(2-2*sqrt(1-r*r)) from b. Set
    r=epsilon*sqrt(1-epsilon*epsilon/4), so this upper bound is epsilon.
    This protects a geometric budget, not correct retrieval on unseen IDs.
    """
    assert 0 < epsilon < 1
    with torch.autocast('cuda', enabled=False):
        base = F.normalize(reference.float(), dim=1)
        candidate = candidate.float()
        delta = candidate - base
        before = delta.norm(dim=1, keepdim=True)
        radius = epsilon * math.sqrt(1-epsilon*epsilon/4)
        ratio = (radius / before.clamp_min(1e-12)).clamp_max(1)
        clipped = before > radius
        limited = F.normalize(base + delta*ratio, dim=1)
        output = torch.where(clipped, limited, candidate)
        after = (output-base).norm(dim=1)
        return output, dict(epsilon=epsilon,perturbation_radius=radius,
            clipped_fraction=float(clipped.float().mean().detach()),
            candidate_shift_mean=float(before.mean().detach()),
            candidate_shift_max=float(before.max().detach()),
            bounded_shift_mean=float(after.mean().detach()),
            bounded_shift_max=float(after.max().detach()))


def geometry_fixture():
    """CPU vector arithmetic only: active/inactive budget and finite derivatives."""
    base = torch.eye(4)
    candidate = F.normalize(torch.tensor([[1.,0.,0.,0.],[.02,1.,0.,0.],
                                         [0.,.5,.5,0.],[0.,0.,0.,-1.]]),dim=1)
    candidate.requires_grad_(True)
    output,report = bound_descriptor(base,candidate)
    assert report['clipped_fraction']==.5
    assert torch.equal(output[:2],candidate[:2])
    assert report['bounded_shift_max']<=MAX_UNIT_SHIFT+1e-6
    assert torch.allclose(output.norm(dim=1),torch.ones(4),atol=1e-6,rtol=0)
    derivative=torch.autograd.grad(output.sum(),candidate)[0]
    assert torch.isfinite(derivative).all()
    return dict(status='PASS_CPU_GEOMETRY_FIXTURE',vectors=4,neural_forwards=0,
        active_and_inactive_cases=True,finite_derivatives=True,**report)


def install_bound(model):
    parent_fuse = model.fuse
    model.descriptor_bound_reports = {}

    def bounded_fuse(self,base,modality,frequency,gates,use_m,use_f,eligible,
                     relation_mass=None,relation_frequency=None):
        candidate=parent_fuse(base,modality,frequency,gates,use_m,use_f,eligible,
                              relation_mass,relation_frequency)
        output,report=bound_descriptor(base[:,:5120],candidate)
        full=bool(eligible.all())
        state=('1' if use_m else '0')+('1' if use_f else '0')
        if full and state=='11':self.descriptor_bound_reports={}
        self.descriptor_bound_reports[(state,full)]=report
        return output

    model.fuse=MethodType(bounded_fuse,model)
    model.descriptor_bound_fixture=geometry_fixture()
    model.anchor_record['final_unit_descriptor_bound']=dict(epsilon=MAX_UNIT_SHIFT,
        placement='After complete parent M/F/I aggregation, gates and relation weights; before final deployed return',
        inactive_output='Exact original candidate bytes; 00 remains the existing frozen identity path',
        parameter_change=0,geometry_fixture=model.descriptor_bound_fixture,
        limits='Known norm bound; not retrieval correctness or guaranteed +1. K PI10/01 native equality is checked before this new output bound.')
    return model
