"""Frozen M1 utility stages, using the actual trained pooling and untouched deployment."""
import types

import torch
from torch.nn import functional as F

import diagnose_common_outlet_scale as framework
from common_outlet_axis import build
from shared_identity_axis import metric_descriptor


STAGES = (*framework.STAGES, 'M_aux', 'F_aux')
METADATA = {}


def build_trained(args, cfg, classes, cameras):
    METADATA['pooling'] = args.pooling
    return build(args, cfg, classes, cameras)


def attach_probe(model):
    captured, auxiliary, handles = {}, {}, []
    for name, module in (('M', model.modality_projection), ('F', model.frequency_projection),
                         ('I', model.interaction_projection)):
        def capture(module, inputs, output, name=name):
            assert name not in captured
            captured[name] = output.detach()
        handles.append(module.register_forward_hook(capture))
    original_forward, original_fuse = model.forward, model.fuse
    original_mread, original_fread = model.modality_expert.read, model.frequency_expert.read

    def mread(self, values, eligible, structured, condition=None):
        result = original_mread(values, eligible, structured, condition)
        if condition is None:
            assert 'M_aux' not in auxiliary
            auxiliary['M_aux'] = self.pool(result, eligible).mean(1).detach()
        return result

    def fread(self, values, available, structured, condition=None):
        result = original_fread(values, available, structured, condition)
        if condition is None:
            assert 'F_aux' not in auxiliary
            auxiliary['F_aux'] = self.pool(result, available, structured).detach()
        return result

    def forward(self, *args, **kwargs):
        assert not self.training
        auxiliary.clear()
        return original_forward(*args, **kwargs)

    def inspected(self, base, modality, frequency, gates, use_m, use_f, eligible, relation_mass=None):
        captured.clear()
        result = original_fuse(base, modality, frequency, gates, use_m, use_f, eligible, relation_mass)
        assert use_m and use_f and set(captured) == {'M', 'F', 'I'}
        assert set(auxiliary) == {'M_aux', 'F_aux'}
        with torch.autocast('cuda', enabled=False):
            base, modality, frequency, gates = (v.float() for v in (base, modality, frequency, gates))
            batch, _, dim = modality.shape
            count = eligible.sum(1, keepdim=True)
            anchor = base[:, 1536:5120].reshape(batch, 7, dim).norm(dim=-1, keepdim=True)
            shared = base[:, 5120:]
            shared_norm = shared.norm(dim=1, keepdim=True)
            assert (shared_norm > 0).all()
            weights = relation_mass * count
            pre = (F.normalize(modality, dim=-1) * anchor * eligible[..., None]) * weights[..., None]
            post = (F.normalize(captured['M'].float(), dim=-1) * anchor * eligible[..., None]) * weights[..., None]
            irows = F.normalize(captured['I'].float(), dim=1)[:, None] * anchor * eligible[..., None] * weights[..., None]
            df = self.residual_scale[1] * gates[:, 1:2] * F.normalize(captured['F'].float(), dim=1) * shared_norm
            coefficients = self.last_outlet_weights.float()
            if self.pooling == 'original_mean':
                dm = self.residual_scale[0] * gates[:, :1] * post.flatten(1)
                di = self.residual_scale[2] * gates[:, 2:] * irows.flatten(1)
                dm, di = dm.reshape(batch, 7, dim), di.reshape(batch, 7, dim)
                aggregate = lambda value: value.mean(1)
            else:
                dm = self.residual_scale[0] * gates[:, :1, None] * post
                di = self.residual_scale[2] * gates[:, 2:, None] * irows
                aggregate = lambda value: (value * coefficients[..., None]).sum(1)
            mi = aggregate(dm + di)
            reconstructed = metric_descriptor(torch.cat((base[:, :5120], shared + (df + mi)), 1))
            assert torch.equal(reconstructed, result), 'probe must preserve the production operation association'
            assert torch.count_nonzero(anchor * ~eligible[..., None]) == 0
            assert torch.count_nonzero(modality * ~eligible[..., None]) == 0
            weighted_anchor = (anchor.squeeze(-1) * weights).sum(1)
            readout_anchor = (anchor.squeeze(-1) * weights * coefficients).sum(1)
            assert (weighted_anchor > 0).all() and (readout_anchor > 0).all()
            self.outlet_stages = dict(deployed=result, base_common=shared,
                M_pre=aggregate(pre), M_post=aggregate(post), F_pre=frequency,
                F_post=captured['F'].float(), **auxiliary)
            self.outlet_scalars = dict(legal_relations=count[:, 0].float(), shared_norm=shared_norm[:, 0],
                private_norm=base[:, :5120].norm(dim=1), weighted_anchor=weighted_anchor,
                weighted_anchor_readout=readout_anchor,
                weighted_anchor_relative_to_shared=readout_anchor / shared_norm[:, 0],
                M_common_increment_relative_to_shared=aggregate(dm).norm(dim=1) / shared_norm[:, 0],
                F_common_increment_relative_to_shared=df.norm(dim=1) / shared_norm[:, 0],
                I_common_increment_relative_to_shared=aggregate(di).norm(dim=1) / shared_norm[:, 0],
                MI_common_increment_relative_to_shared=mi.norm(dim=1) / shared_norm[:, 0],
                total_increment_relative_to_shared=(df + mi).norm(dim=1) / shared_norm[:, 0],
                M_cancellation_ratio=aggregate(post).norm(dim=1) / readout_anchor,
                common_cosine_change=1 - F.cosine_similarity(shared, shared + (df + mi), dim=1),
                M_pre_norm=aggregate(pre).norm(dim=1), M_post_norm=aggregate(post).norm(dim=1),
                F_pre_norm=frequency.norm(dim=1), F_post_norm=captured['F'].float().norm(dim=1),
                gate_M=gates[:, 0], gate_F=gates[:, 1], gate_I=gates[:, 2])
            for index in range(7):
                for name, values in (('anchor', anchor[:, :, 0]), ('relation_mass', relation_mass),
                                     ('eligible', eligible.float()), ('outlet_coefficient', coefficients)):
                    self.outlet_scalars[name + '_' + str(index)] = values[:, index]
        return result

    model.forward = types.MethodType(forward, model)
    model.modality_expert.read = types.MethodType(mread, model.modality_expert)
    model.frequency_expert.read = types.MethodType(fread, model.frequency_expert)
    model.fuse = types.MethodType(inspected, model)
    return handles


def write_result(path, value):
    value['pooling'] = METADATA['pooling']
    value['stages'] = list(STAGES)
    value['scope'] = 'Frozen M1; source-identical auxiliary representations versus actual routed evidence and projection. No optimizer or deployment change.'
    if path.name == 'result.json':
        value['protocol'] = 'Actual trained outlet aggregation retained for both M_pre/M_post. F_pre/F_post differ only by PF. M_aux/F_aux are exactly the existing independent auxiliary-loss representations of the same input availability. All stages use normalized512D retrieval except deployed5632D.'
        value['limits'] = 'Single development split/seed; stage comparisons diagnose empirical identity utility, not information destruction, strict disentanglement or causality.'
    ORIGINAL_WRITE(path, value)


ORIGINAL_WRITE = framework.write_json
framework.build = build_trained
framework.attach_probe = attach_probe
framework.STAGES = STAGES
framework.write_json = write_result

if __name__ == '__main__':
    framework.main()
