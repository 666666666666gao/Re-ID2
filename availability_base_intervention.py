"""Frozen-only intervention on DeMo's unavailable base sources and relations."""
import torch

from dual_axis import RELATIONS


def masked_base_fusion(module, patches, globals_, available):
    assert not module.training and module.HDM and module.ATM
    assert available.any(1).all()
    patches = [value * available[:, index, None, None] for index, value in enumerate(patches)]
    globals_ = [value * available[:, index, None] for index, value in enumerate(globals_)]
    eligible = torch.stack([available[:, subset].all(1) for subset in RELATIONS], 1)
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
    heads = [expert([chunks[index][:, relation] for relation in range(7)], weights[:, index])
             for index, expert in enumerate(moe.experts)]
    output = torch.cat(heads, -1) * eligible[..., None]
    module.availability_audit = dict(eligible=eligible.detach(), gates=weights.detach(),
                                    relation_outputs=output.detach())
    return output.flatten(1)


def install_available_base(model):
    """Keep all tensors/keys; replace only frozen base computation when missing."""
    assert not model.training and all(not module.training for module in model.modules())
    original_forward = model.forward
    original_fusion = model.generalFusion.forward

    def forward(x, *args, **kwargs):
        assert not model.training
        model.base_available = torch.stack([x[key].flatten(1).ne(0).any(1)
                                            for key in ('RGB', 'NI', 'TI')], 1)
        assert model.base_available.any(1).all()
        return original_forward(x, *args, **kwargs)

    def fusion(*inputs):
        if model.base_available.all():
            return original_fusion(*inputs)
        return masked_base_fusion(model.generalFusion, inputs[:3], inputs[3:], model.base_available)

    def reduction_hook(index):
        def mask(module, inputs, output):
            if model.base_available.all():
                return output
            return output * model.base_available[:, index, None]
        return mask

    model.forward = forward
    model.generalFusion.forward = fusion
    for index, module in enumerate((model.rgb_reduce, model.nir_reduce, model.tir_reduce)):
        module.register_forward_hook(reduction_hook(index))
    return model
