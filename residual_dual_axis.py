"""Preserve DeMo identity features; add modest band-conditioned residuals."""
import torch
from torch import nn

from dual_axis import AugmentedDeMo, Collaboration, RELATIONS


class ResidualCollaboration(Collaboration):
    def __init__(self, dim):
        super().__init__(dim)
        self.message_scale = nn.Parameter(torch.tensor([.1, .1]))

    def joint(self, u, v, interaction=True):
        a = self.relation_score(u.mean(2)).squeeze(-1)
        c = self.band_score(v.mean(1)).squeeze(-1)
        psi = self.interaction(torch.cat((u, v), -1)).squeeze(-1)
        logits = a[:, :, None] + c[:, None, :] + (psi if interaction else 0)
        pi = logits.flatten(1).softmax(-1).reshape_as(logits)
        gates = self.outer_gate(torch.cat((u.mean((1, 2)), v.mean((1, 2))), -1)).sigmoid()
        conditional_band = pi / pi.sum(2, keepdim=True)
        modality = ((u + self.message_scale[0] * gates[:, None, None, :1] * torch.tanh(self.to_modality(v))) * conditional_band[..., None]).sum(2)
        frequency = ((v + self.message_scale[1] * gates[:, None, None, 1:] * torch.tanh(self.to_frequency(u))) * pi[..., None]).sum((1, 2))
        return modality, frequency, pi, gates

    def ordinary(self, evidence):
        x = evidence.mean(1)
        context = x.mean(1, keepdim=True).expand_as(x)
        gates = self.outer_gate(torch.cat((x.mean(1), context.mean(1)), -1)).sigmoid()
        value = x + self.message_scale[0] * gates[:, None, :1] * torch.tanh(self.to_modality(x)) + self.message_scale[1] * gates[:, None, 1:] * torch.tanh(self.to_frequency(context))
        score = self.relation_score(x) + self.band_score(x) + self.interaction(torch.cat((x, context), -1))
        weights = score.squeeze(-1).softmax(1)
        return (value * weights[..., None]).sum(1), weights, gates


class ResidualAugmentedDeMo(AugmentedDeMo):
    def __init__(self, classes, cfg, cameras, variant):
        super().__init__(classes, cfg, cameras, variant)
        d = self.feat_dim
        # Both controls use precisely these same active parameters.
        self.frequency.values = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 64), nn.GELU(), nn.Linear(64, d), nn.GELU())
            for _ in range(3)])
        self.band_adapter = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 64), nn.GELU(), nn.Linear(64, d))
        nn.init.normal_(self.band_adapter[-1].weight, std=.001)
        nn.init.zeros_(self.band_adapter[-1].bias)
        self.band_scale = nn.Parameter(torch.tensor(.1))
        self.frequency_scale = nn.Parameter(torch.tensor(.05))
        self.collaboration = ResidualCollaboration(d)
        # Original HDM/PIFE objectives keep their weights; extra heads are modest.
        self.loss_weights = [ .25, 1., .1 ] + ([1.] if self.direct else [1., 1., 1.])

    def forward(self, x, label=None, cam_label=None, view_label=None, return_pattern=3):
        patches, globals_ = [], []
        for key, reduce in zip(('RGB', 'NI', 'TI'), (self.rgb_reduce, self.nir_reduce, self.tir_reduce)):
            cash, global_ = self.BACKBONE(x[key], cam_label=cam_label, view_label=view_label)
            patches.append(cash)
            local = self.pool(cash.permute(0, 2, 1)).squeeze(-1)
            globals_.append(reduce(torch.cat((global_, local), -1)))
        original = torch.cat(globals_, -1)
        bands = self.bands(torch.stack(patches, 1))
        _, frequency_by_source = self.frequency(bands)
        relation_slots = self.generalFusion.forward_HDM(*patches, *globals_)
        if self.variant == 'ordinary':
            result = self.generalFusion.forward_ATM(*relation_slots)
            moe = result[0] if self.training else result
            frequency_by_source = frequency_by_source + self.band_scale * self.band_adapter(frequency_by_source)
            freq, route, gates = self.collaboration.ordinary(frequency_by_source)
        else:
            full_relation = torch.stack([slot.reshape(original.shape[0], self.feat_dim) for slot in relation_slots], 1)
            v = torch.stack([frequency_by_source[:, subset].mean(1) for subset in RELATIONS], 1)
            # Full-frequency identity relation is the anchor; each band supplies a residual.
            u = full_relation[:, :, None] + self.band_scale * self.band_adapter(v)
            relation, freq, route, gates = self.collaboration.joint(u, v)
            result = self.generalFusion.forward_ATM(*relation.unbind(1))
            moe = result[0] if self.training else result
        fused = torch.cat((original, moe, self.frequency_scale * freq), -1)
        self.last_route, self.last_gates = route.detach(), gates.detach()
        if not self.training:
            return fused
        output = [self.fused_classifier(self.fused_neck(fused)), fused,
                  self.classifier_moe(self.bottleneck_moe(moe)), moe,
                  self.frequency_classifier(self.frequency_neck(freq)), freq]
        if self.direct:
            output.extend((self.classifier(self.bottleneck(original)), original))
        else:
            for feat, classifier, neck in zip(globals_, (self.classifier_r, self.classifier_n, self.classifier_t), (self.bottleneck_r, self.bottleneck_n, self.bottleneck_t)):
                output.extend((classifier(neck(feat)), feat))
        return tuple(output)
