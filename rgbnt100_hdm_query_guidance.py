"""AO: use already-computed, source-constrained HDM evidence as a query condition."""
from types import MethodType

import torch


def install(model):
    """Preserve band-specific values and reuse the existing 512-to-64 query projection."""
    before = set(model.state_dict())
    original_hdm = model.generalFusion.forward_HDM
    original_read = model.modality_expert.read
    model.hdm_query_reference = None
    model.hdm_query_native_reads = []

    def hdm(self, rgb, nir, tir, rgb_global, nir_global, tir_global):
        result = original_hdm(rgb, nir, tir, rgb_global, nir_global, tir_global)
        assert len(result) == 7
        model.hdm_query_reference = torch.stack([r.reshape(len(rgb), 512) for r in result], 1).detach()
        assert model.hdm_query_reference.shape == (len(rgb), 7, 512)
        return result

    def read(self, values, eligible, structured, condition=None):
        reference = model.hdm_query_reference
        assert reference is not None and reference.shape == (len(values), 7, 512)
        assert not reference.requires_grad
        with torch.autocast('cuda', enabled=False):
            guide = .1 * model.router.to_modality_query(reference.float())
            guide = guide[:, :, None].expand(-1, -1, 3, -1)
            assert guide.shape == (len(values), 7, 3, self.query.shape[-1])
            effective = guide if condition is None else condition.float() + guide
        if model.normal_priority_smoke:
            model.hdm_query_native_reads.append(dict(
                reference_shape=list(reference.shape), reference_requires_grad=False,
                guide_shape=list(guide.shape), guide_dtype=str(guide.dtype),
                guide_l2_mean=float(guide.detach().norm(dim=-1).mean()),
                previous_cross_condition_present=condition is not None,
                existing_band_values_unchanged=True))
        return original_read(values, eligible, structured, effective)

    model.generalFusion.forward_HDM = MethodType(hdm, model.generalFusion)
    model.modality_expert.read = MethodType(read, model.modality_expert)
    assert set(model.state_dict()) == before
    model.anchor_record['HDM_query_guidance'] = dict(
        source='Already-computed frozen pre-ATMoE HDM relation outputs, each with its original allowed sources',
        target='M expert initial and cross-conditioned attention queries; shared across its three bands',
        projection='Existing router.to_modality_query, FP32, coefficient .1',
        parameters_added=0, new_HDM_or_backbone_calls=0,
        unchanged='FFT masks, band/source-specific attention values, frequency expert, all heads/losses, T2/cap.10/5120',
        limits='Teacher-semantic query guidance, not copied unfiltered values or a proved retrieval improvement. Ordinary comparator receives the same guide.')
    return model
