"""Keep aligned patch regions through existing within-modality cross-band attention.

Only F.prepare/F.read change. M keeps its original all-position read because
paired sensor images need not share a vehicle view. No learned parameter or
descriptor coordinate is introduced.
"""
from types import MethodType

import torch

from axis_collaboration import FrequencyAxisExpert, ModalityAxisExpert
from local_patch_regions import patch_regions


def frequency_region_prepare(self, tokens, available, structured):
    batch, _, _, patches, _ = tokens.shape
    indices = self.region_patch_indices
    regions = indices.shape[0]
    pooled = []
    for band in range(3):
        value = self.values[band](tokens[:, :, band]) + self.source[None, :, None]
        value = value * available[:, :, None, None]
        selected_indices = indices
        if not structured:
            value = value.flatten(1, 2)[:, None].expand(-1, 3, -1, -1)
            selected_indices = torch.cat([indices + modality * patches for modality in range(3)], 1)
        rank = value.shape[-1]
        value = value[:, :, selected_indices].reshape(batch * 3 * regions, -1, rank)
        query = (self.query[band] + self.source)[None, :, None]
        query = query.expand(batch, -1, regions, -1).reshape(batch * 3 * regions, 1, rank)
        feature = self.attention[band](query, value, value, need_weights=False)[0]
        pooled.append(feature.reshape(batch, 3, regions, rank))
    return torch.stack(pooled, 2)


def frequency_region_read(self, pooled, available, structured, condition=None):
    batch, groups, bands, regions, rank = pooled.shape
    # L/M/H share a region: one source if structured, mixed sources otherwise.
    values = pooled.permute(0, 1, 3, 2, 4).reshape(batch * groups * regions, bands, rank)
    query = values
    if condition is not None:
        context = condition[:, :, None].expand(-1, -1, regions, -1, -1)
        query = values + context.reshape_as(values)
    evidence = values + self.cross_band(query, values, values, need_weights=False)[0]
    evidence = self.output(evidence).reshape(batch, groups, regions, bands, -1)
    evidence = evidence.mean(2)
    if structured:
        evidence = evidence * available[:, :, None, None]
    return evidence


def install_region_reading(model, height, width, grid):
    """Install before optimizer creation; use the same grid for both controls."""
    assert (height, width) == (8, 16)
    modality, frequency = model.modality_expert, model.frequency_expert
    assert modality.read.__func__ is ModalityAxisExpert.read
    assert frequency.prepare.__func__ is FrequencyAxisExpert.prepare
    assert frequency.read.__func__ is FrequencyAxisExpert.read
    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
    state_keys = set(model.state_dict())
    device = frequency.query.device
    indices = torch.tensor(patch_regions(height, width, grid), dtype=torch.long, device=device)
    frequency.register_buffer('region_patch_indices', indices, persistent=False)
    frequency.prepare = MethodType(frequency_region_prepare, frequency)
    frequency.read = MethodType(frequency_region_read, frequency)
    assert modality.read.__func__ is ModalityAxisExpert.read
    assert {name: id(parameter) for name, parameter in model.named_parameters()} == parameter_ids
    assert set(model.state_dict()) == state_keys
    model.anchor_record.update(local_reading_grid=grid, local_reading_regions=grid * grid,
        local_reading_parameters_added=0, local_reading_patch_grid=[height, width],
        local_reading_modality_policy='Original full-spatial M reader unchanged',
        local_reading_method='F-only regional support across bands: structured per source, ordinary mixed-source by region; cross-band attention and nonlinear output before region mean',
        local_reading_limit='Fixed image regions, no semantic part alignment claim within or across sensors. Existing M/global conditions/router/projections remain. Runtime parity, gradients, latency, normal retrieval gain and missing performance are unverified.')
    return model
