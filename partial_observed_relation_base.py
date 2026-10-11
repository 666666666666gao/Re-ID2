"""V inference control: keep observed-partial relations in the frozen DeMo base.

Missing sensors remain zero feature placeholders from encode_available. Only
the base HDM/ATMoE relation eligibility changes from all to any observed source.
The caller's returned eligibility and both new experts retain their old masks.
"""
import torch

from dual_axis import RELATIONS
import shared_identity_axis as shared


def observed_partial_base_fusion(module, patches, globals_, available):
    assert not module.training
    assert torch.equal(available, available[:1].expand_as(available))
    assert module.HDM and module.ATM
    with torch.autocast('cuda', enabled=False):
        patches = [value.float() for value in patches]
        globals_ = [value.float() for value in globals_]
        eligible = torch.stack([available[:, subset].any(1) for subset in RELATIONS], 1)
        module.partial_relation_base_audit = dict(
            available=available[0].tolist(), base_eligible=eligible[0].tolist(),
            base_relations=int(eligible[0].sum()),
            new_expert_eligibility='Unchanged encode_available all-required source rule',
            absent_backbones='Unchanged skipped encoder and zero feature placeholders',
            weights_or_parameters_added=0)
        if bool(available.all()):
            return module(*patches, *globals_)
        batch = len(available)
        relations = torch.stack([value.reshape(batch, module.feat_dim)
                                 for value in module.forward_HDM(*patches, *globals_)], 1)
        relations = relations * eligible[..., None]
        moe = module.moe
        gate = moe.gating_network.gate
        query = gate.linear_re(relations.flatten(1))
        q = gate.q_(query).reshape(batch, 1, gate.num_heads, module.feat_dim // gate.num_heads).permute(0, 2, 1, 3)
        k = gate.k_(relations).reshape(batch, 7, gate.num_heads, module.feat_dim // gate.num_heads).permute(0, 2, 1, 3)
        logits = (q @ k.transpose(-2, -1)) * gate.scale
        weights = logits.masked_fill(~eligible[:, None, None, :], -torch.inf).softmax(-1)
        chunks = relations.chunk(moe.head, -1)
        heads = []
        for index, head in enumerate(moe.experts):
            outputs = [expert(chunks[index][:, relation]) if bool(eligible[0, relation])
                       else torch.zeros_like(chunks[index][:, relation])
                       for relation, expert in enumerate(head.expertHead)]
            heads.append(torch.stack(outputs, 1) * weights[:, index].squeeze(1)[..., None])
        return torch.cat(heads, -1).flatten(1)


def install_partial_relation_base(model):
    """One frozen evaluation model per process; existing builders run first."""
    assert not model.training
    shared.available_base_fusion = observed_partial_base_fusion
    model.partial_relation_base_policy = 'base_any_observed_new_experts_all_required'
    return model
