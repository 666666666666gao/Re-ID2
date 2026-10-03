"""Tensor checks for the frozen source-masking intervention, never training."""
import torch

from availability_base_intervention import masked_base_fusion
from dual_axis import RELATIONS


@torch.no_grad()
def verify_masked_base_fusion(module):
    available = torch.tensor([[index in subset for index in range(3)] for subset in RELATIONS],
                             device=next(module.parameters()).device)
    dim = module.feat_dim
    patches = [torch.randn(7, 9, dim, device=available.device) for _ in range(3)]
    globals_ = [torch.randn(7, dim, device=available.device) for _ in range(3)]
    original = masked_base_fusion(module, patches, globals_, available)
    changed_patches = [value + 100 * (~available[:, index, None, None])
                       for index, value in enumerate(patches)]
    changed_globals = [value - 100 * (~available[:, index, None])
                       for index, value in enumerate(globals_)]
    changed = masked_base_fusion(module, changed_patches, changed_globals, available)
    assert torch.equal(original, changed)
    audit = module.availability_audit
    eligible, gates, outputs = (audit[key] for key in ('eligible', 'gates', 'relation_outputs'))
    assert torch.isfinite(original).all() and torch.isfinite(gates).all()
    assert gates.masked_select(~eligible[:, None, None, :].expand_as(gates)).eq(0).all()
    assert outputs.masked_select(~eligible[..., None].expand_as(outputs)).eq(0).all()
    assert torch.allclose(gates.sum(-1), torch.ones_like(gates.sum(-1)), atol=1e-6, rtol=0)
    return dict(status='PASS',all_seven_availability_sets=True,unavailable_source_perturbation_exact_invariance=True,
                invalid_relation_gates_zero=True,invalid_relation_outputs_zero=True,legal_gates_sum_one=True,
                optimizer_updates=0)


def verify_masked_descriptor(feature, missing):
    available = torch.tensor([key not in missing for key in ('RGB','NI','TI')], device=feature.device)
    eligible = torch.tensor([available[list(subset)].all() for subset in RELATIONS], device=feature.device)
    blocks = feature[:, :5120].reshape(len(feature), 10, 512)
    valid = torch.cat((available, eligible))
    assert blocks[:, ~valid].eq(0).all()
    return dict(unavailable_global_coordinates_zero=True,invalid_relation_coordinates_zero=True)
