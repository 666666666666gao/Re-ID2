"""Matched DeMo + frequency and dual-axis models; no contribution loss yet."""
import torch
from torch import nn
from torch.nn import functional as F
from modeling.make_model import DeMo, __factory_T_type as factory
from modeling.meta_arch import weights_init_classifier, weights_init_kaiming

RELATIONS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))


class SpatialBands(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.height, self.width = height, width
        fy = torch.fft.fftfreq(height)[:, None]
        fx = torch.fft.fftfreq(width)[None, :]
        radius = (fy.square() + fx.square()).sqrt()
        masks = torch.stack((radius <= .125, (radius > .125) & (radius <= .25), radius > .25))
        self.register_buffer('masks', masks)

    def forward(self, patches):
        # patches B,M,N,D have an actual two-dimensional spatial layout.
        b, m, n, d = patches.shape
        assert n == self.height * self.width
        with torch.autocast('cuda', enabled=False):
            grid = patches.float().transpose(-1, -2).reshape(b, m, d, self.height, self.width)
            spectrum = torch.fft.fft2(grid, norm='ortho')
            bands = torch.fft.ifft2(spectrum[:, :, None] * self.masks[None, None, :, None], norm='ortho').real
            return bands.flatten(-2).transpose(-1, -2)  # B,M,band,N,D


class FrequencyEvidence(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.values = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU()) for _ in range(3)])
        self.source = nn.Parameter(torch.randn(3, dim) * .02)
        self.query = nn.Parameter(torch.randn(3, 1, dim) * .02)
        self.attention = nn.ModuleList([nn.MultiheadAttention(dim, 8, batch_first=True) for _ in range(3)])
        self.cross_band = nn.MultiheadAttention(dim, 8, batch_first=True)

    def forward(self, bands):
        b, m, _, n, d = bands.shape
        tokens, evidence = [], []
        for band in range(3):
            # Nonlinearity precedes pooling: signed high-pass responses do not cancel.
            value = self.values[band](bands[:, :, band])
            tokens.append(value)
            provenance = value + self.source[None, :, None]
            q = self.query[band].expand(b * m, -1, -1)
            pooled = self.attention[band](q, provenance.reshape(b * m, n, d), provenance.reshape(b * m, n, d), need_weights=False)[0]
            evidence.append(pooled.reshape(b, m, d))
        evidence = torch.stack(evidence, dim=2)
        by_source = evidence.reshape(b * m, 3, d)
        evidence = (by_source + self.cross_band(by_source, by_source, by_source, need_weights=False)[0]).reshape(b, m, 3, d)
        return tokens, evidence


class Collaboration(nn.Module):
    """Same active parameter set: frequency-only refinement or conditional dual-axis routing."""
    def __init__(self, dim):
        super().__init__()
        self.to_modality = nn.Linear(dim, dim)
        self.to_frequency = nn.Linear(dim, dim)
        self.relation_score = nn.Linear(dim, 1, bias=False)
        self.band_score = nn.Linear(dim, 1, bias=False)
        self.interaction = nn.Sequential(nn.Linear(2 * dim, 64), nn.GELU(), nn.Linear(64, 1, bias=False))
        self.outer_gate = nn.Linear(2 * dim, 2)

    def joint(self, u, v, interaction=True):
        # a_S and c_b are computed independently; psi is the non-factorizable term.
        a = self.relation_score(u.mean(2)).squeeze(-1)
        c = self.band_score(v.mean(1)).squeeze(-1)
        psi = self.interaction(torch.cat((u, v), -1)).squeeze(-1)
        logits = a[:, :, None] + c[:, None, :] + (psi if interaction else 0)
        pi = logits.flatten(1).softmax(-1).reshape_as(logits)
        message_m = torch.tanh(self.to_modality(v))
        message_f = torch.tanh(self.to_frequency(u))
        gates = self.outer_gate(torch.cat((u.mean((1, 2)), v.mean((1, 2))), -1)).sigmoid()
        # Conditional band pooling retains seven relation slots for DeMo ATMoE.
        conditional_band = pi / pi.sum(2, keepdim=True)
        modality = ((u + gates[:, None, None, :1] * message_m) * conditional_band[..., None]).sum(2)
        frequency = ((v + gates[:, None, None, 1:] * message_f) * pi[..., None]).sum((1, 2))
        return modality, frequency, pi, gates

    def ordinary(self, evidence):
        # Parameter-matching capacity lives entirely inside the ordinary frequency branch.
        # It never reads HDM evidence or sends a message back to DeMo.
        x = evidence.mean(1)
        context = x.mean(1, keepdim=True).expand_as(x)
        gates = self.outer_gate(torch.cat((x.mean(1), context.mean(1)), -1)).sigmoid()
        value = x + gates[:, None, :1] * torch.tanh(self.to_modality(x)) + gates[:, None, 1:] * torch.tanh(self.to_frequency(context))
        score = self.relation_score(x) + self.band_score(x) + self.interaction(torch.cat((x, context), -1))
        weights = score.squeeze(-1).softmax(1)
        return (value * weights[..., None]).sum(1), weights, gates


class AugmentedDeMo(DeMo):
    def __init__(self, classes, cfg, cameras, variant):
        super().__init__(classes, cfg, cameras, 0, factory)
        assert variant in ('ordinary', 'dual')
        self.variant = variant
        d = self.feat_dim
        self.bands = SpatialBands(cfg.INPUT.SIZE_TRAIN[0] // 16, cfg.INPUT.SIZE_TRAIN[1] // 16)
        self.frequency = FrequencyEvidence(d)
        self.collaboration = Collaboration(d)
        self.frequency_neck = nn.BatchNorm1d(d)
        self.frequency_neck.bias.requires_grad_(False)
        self.frequency_classifier = nn.Linear(d, classes, bias=False)
        self.fused_neck = nn.BatchNorm1d(11 * d)
        self.fused_neck.bias.requires_grad_(False)
        self.fused_classifier = nn.Linear(11 * d, classes, bias=False)
        self.frequency_neck.apply(weights_init_kaiming)
        self.fused_neck.apply(weights_init_kaiming)
        self.frequency_classifier.apply(weights_init_classifier)
        self.fused_classifier.apply(weights_init_classifier)

    def forward(self, x, label=None, cam_label=None, view_label=None, return_pattern=3):
        patches, globals_ = [], []
        for key, reduce in zip(('RGB', 'NI', 'TI'), (self.rgb_reduce, self.nir_reduce, self.tir_reduce)):
            cash, global_ = self.BACKBONE(x[key], cam_label=cam_label, view_label=view_label)
            patches.append(cash)
            local = self.pool(cash.permute(0, 2, 1)).squeeze(-1)
            globals_.append(reduce(torch.cat((global_, local), -1)))
        original = torch.cat(globals_, -1)
        bands = self.bands(torch.stack(patches, 1))
        band_tokens, frequency_by_source = self.frequency(bands)
        if self.variant == 'ordinary':
            result = self.generalFusion(*patches, *globals_)
            moe = result[0] if self.training else result
            freq, route, gates = self.collaboration.ordinary(frequency_by_source)
        else:
            relation_by_band = []
            for band in range(3):
                values = band_tokens[band]
                slots = self.generalFusion.forward_HDM(*values.unbind(1), *values.mean(2).unbind(1))
                relation_by_band.append(torch.stack([slot.reshape(values.shape[0], self.feat_dim) for slot in slots], 1))
            u = torch.stack(relation_by_band, 2)
            # Each V_S,b sees ONLY modalities belonging to S.
            v = torch.stack([frequency_by_source[:, subset].mean(1) for subset in RELATIONS], 1)
            relation, freq, route, gates = self.collaboration.joint(u, v)
            result = self.generalFusion.forward_ATM(*relation.unbind(1))
            moe = result[0] if self.training else result
        fused = torch.cat((original, moe, freq), -1)
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
