"""Source-constrained dual axes, one conditional update, and retrieval calibration."""
import torch
from torch import nn
from torch.nn import functional as F

from dual_axis import RELATIONS, SpatialBands
from modeling.make_model import DeMo, __factory_T_type as factory
from modeling.meta_arch import weights_init_classifier, weights_init_kaiming


class ModalityAxisExpert(nn.Module):
    def __init__(self, dim, rank=64):
        super().__init__()
        self.value = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, rank), nn.GELU())
        self.source = nn.Parameter(torch.randn(3, rank) * .02)
        self.query = nn.Parameter(torch.randn(7, rank) * .02)
        self.band = nn.Parameter(torch.randn(3, rank) * .02)
        self.attention = nn.MultiheadAttention(rank, 8, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(rank), nn.Linear(rank, dim), nn.GELU())
        self.pool_score = nn.Linear(dim, 1, bias=False)

    def prepare(self, tokens, available):
        values = self.value(tokens) + self.source[None, :, None, None]
        return values * available[:, :, None, None, None]

    def read(self, values, eligible, structured, condition=None):
        batch, _, _, _, rank = values.shape
        rows = []
        for relation, subset in enumerate(RELATIONS):
            columns = []
            for band in range(3):
                if structured:
                    value = values[:, subset, band].flatten(1, 2)
                else:
                    value = values[:, :, band].flatten(1, 2)
                query = (self.query[relation] + self.band[band]).expand(batch, 1, rank)
                if condition is not None:
                    query = query + condition[:, relation, band, None]
                feature = self.attention(query, value, value, need_weights=False)[0].squeeze(1)
                columns.append(self.output(feature))
            rows.append(torch.stack(columns, 1))
        return torch.stack(rows, 1) * eligible[:, :, None, None]

    def pool(self, evidence, eligible):
        weights = self.pool_score(evidence).squeeze(-1).softmax(2)
        return (evidence * weights[..., None]).sum(2) * eligible[:, :, None]


class FrequencyAxisExpert(nn.Module):
    def __init__(self, dim, rank=64):
        super().__init__()
        self.values = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, rank), nn.GELU()) for _ in range(3)])
        self.source = nn.Parameter(torch.randn(3, rank) * .02)
        self.query = nn.Parameter(torch.randn(3, rank) * .02)
        self.attention = nn.ModuleList([nn.MultiheadAttention(rank, 8, batch_first=True) for _ in range(3)])
        self.cross_band = nn.MultiheadAttention(rank, 8, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(rank), nn.Linear(rank, dim), nn.GELU())
        self.pool_score = nn.Linear(dim, 1, bias=False)

    def prepare(self, tokens, available, structured):
        batch, _, _, _, _ = tokens.shape
        pooled = []
        for band in range(3):
            value = self.values[band](tokens[:, :, band]) + self.source[None, :, None]
            value = value * available[:, :, None, None]
            if not structured:
                value = value.flatten(1, 2)[:, None].expand(-1, 3, -1, -1)
            length, rank = value.shape[-2:]
            query = (self.query[band] + self.source).expand(batch, -1, -1).reshape(batch * 3, 1, rank)
            value = value.reshape(batch * 3, length, rank)
            feature = self.attention[band](query, value, value, need_weights=False)[0]
            pooled.append(feature.reshape(batch, 3, rank))
        return torch.stack(pooled, 2)

    def read(self, pooled, available, structured, condition=None):
        batch, groups, bands, rank = pooled.shape
        values = pooled.reshape(batch * groups, bands, rank)
        query = values if condition is None else values + condition.reshape_as(values)
        evidence = values + self.cross_band(query, values, values, need_weights=False)[0]
        evidence = self.output(evidence).reshape(batch, groups, bands, -1)
        if structured:
            evidence = evidence * available[:, :, None, None]
        return evidence

    def pool(self, evidence, available, structured):
        logits = self.pool_score(evidence).squeeze(-1)
        if structured:
            logits = logits.masked_fill(~available[:, :, None], -torch.inf)
        weights = logits.flatten(1).softmax(1).reshape_as(logits)
        return (evidence * weights[..., None]).sum((1, 2))


class ConditionalRouter(nn.Module):
    def __init__(self, dim, rank=64):
        super().__init__()
        self.to_modality_query = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, rank), nn.Tanh())
        self.to_frequency_query = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, rank), nn.Tanh())
        self.relation_score = nn.Linear(dim, 1, bias=False)
        self.band_score = nn.Linear(dim, 1, bias=False)
        self.psi = nn.Sequential(nn.Linear(2 * dim, rank), nn.GELU(), nn.Linear(rank, 1, bias=False))

    @staticmethod
    def by_relation(frequency):
        return torch.stack([frequency[:, subset].mean(1) for subset in RELATIONS], 1)

    def conditions(self, modality, frequency):
        # Actual native-AMP smoke lost the small frequency-condition LN gradients.
        # Keep this lightweight query projection in FP32; the experts remain AMP.
        with torch.autocast('cuda', enabled=False):
            modality, frequency = modality.float(), frequency.float()
            v = self.by_relation(frequency)
            to_m = .1 * self.to_modality_query(v)
            to_f = torch.stack([modality[:, [s for s, subset in enumerate(RELATIONS) if m in subset]].mean(1)
                                for m in range(3)], 1)
            return to_m, .1 * self.to_frequency_query(to_f)

    def route(self, modality, frequency, eligible, interaction=True):
        v = self.by_relation(frequency)
        a = self.relation_score(modality.mean(2)).squeeze(-1)
        c = self.band_score(v.mean(1)).squeeze(-1)
        psi = self.psi(torch.cat((modality, v), -1)).squeeze(-1)
        logits = a[:, :, None] + c[:, None, :] + (psi if interaction else 0)
        logits = logits.masked_fill(~eligible[:, :, None], -torch.inf)
        pi = logits.flatten(1).softmax(1).reshape_as(logits)
        # Avoid dividing an invalid relation's zero mass by zero.
        mass = pi.sum(2, keepdim=True)
        conditional = pi / torch.where(eligible[:, :, None], mass, torch.ones_like(mass))
        m = (modality * conditional[..., None]).sum(2)
        f = (v * pi[..., None]).sum((1, 2))
        return m, f, pi


class ContributionCalibrator(nn.Module):
    def __init__(self, dim, rank=64):
        super().__init__()
        self.predictor = nn.Sequential(nn.LayerNorm(3 * dim), nn.Linear(3 * dim, rank), nn.GELU(), nn.Linear(rank, 3))
        nn.init.normal_(self.predictor[-1].weight, std=.001)
        nn.init.zeros_(self.predictor[-1].bias)

    def forward(self, base, modality, frequency):
        context = torch.cat((base.reshape(base.shape[0], 11, -1).mean(1), modality.mean(1), frequency), -1)
        prediction = self.predictor(context.detach())
        return prediction, (20 * prediction).sigmoid()

    @staticmethod
    @torch.no_grad()
    def targets(states, labels):
        # One detached reference gallery and identical positive/negative indices
        # are used for all four states. No dataset-level mAP enters training.
        reference = F.normalize(states['00'].float(), dim=1)
        with torch.autocast('cuda', enabled=False):
            similarity = reference @ reference.T
        same = labels[:, None].eq(labels[None])
        positive = same & ~torch.eye(len(labels), device=labels.device, dtype=torch.bool)
        negative = ~same
        assert positive.any(1).all() and negative.any(1).all()
        pos = similarity.masked_fill(~positive, torch.inf).argmin(1)
        neg = similarity.masked_fill(~negative, -torch.inf).argmax(1)
        direction = reference[pos] - reference[neg]
        scores = {key: (F.normalize(value.float(), dim=1) * direction).sum(1) for key, value in states.items()}
        target = torch.stack((scores['11'] - scores['01'], scores['11'] - scores['10'],
                              scores['11'] - scores['10'] - scores['01'] + scores['00']), 1)
        return target, pos, neg, scores


class AxisCollaborationDeMo(DeMo):
    def __init__(self, classes, cfg, cameras, variant):
        super().__init__(classes, cfg, cameras, 0, factory)
        assert variant in ('axis_collaboration', 'plain_twins', 'ordinary_frequency')
        self.variant = variant
        self.structured = variant == 'axis_collaboration'
        self.frequency_only = variant == 'ordinary_frequency'
        dim = self.feat_dim
        self.bands = SpatialBands(cfg.INPUT.SIZE_TRAIN[0] // 16, cfg.INPUT.SIZE_TRAIN[1] // 16)
        self.modality_expert = ModalityAxisExpert(dim)
        self.frequency_expert = FrequencyAxisExpert(dim)
        self.router = ConditionalRouter(dim)
        self.calibrator = ContributionCalibrator(dim)
        self.modality_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, dim))
        self.frequency_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, dim))
        self.interaction_projection = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, 64), nn.GELU(), nn.Linear(64, dim))
        for projection in (self.modality_projection, self.frequency_projection, self.interaction_projection):
            nn.init.normal_(projection[-1].weight, std=.001)
            nn.init.zeros_(projection[-1].bias)
        self.residual_scale = nn.Parameter(torch.tensor([.1, .05, .05]))
        self.joint_interaction = True
        self.contribution_loss_weight = .05
        for name, size in (('fused', 11 * dim), ('modality', dim), ('frequency', dim)):
            neck = nn.BatchNorm1d(size)
            neck.bias.requires_grad_(False)
            neck.apply(weights_init_kaiming)
            classifier = nn.Linear(size, classes, bias=False)
            classifier.apply(weights_init_classifier)
            setattr(self, name + '_neck', neck)
            setattr(self, name + '_classifier', classifier)
        self.loss_weights = [.25, 1., .1, .1] + ([1.] if self.direct else [1., 1., 1.])

    def fuse(self, base, modality, frequency, gates, use_m, use_f, eligible):
        batch, _, dim = modality.shape
        delta_m = torch.zeros_like(base[:, 3 * dim:10 * dim])
        delta_f = torch.zeros_like(frequency)
        if use_m:
            projected = self.modality_projection(modality) * eligible[:, :, None]
            delta_m = self.residual_scale[0] * gates[:, :1] * projected.flatten(1)
        if use_f:
            delta_f = self.residual_scale[1] * gates[:, 1:2] * self.frequency_projection(frequency)
        if use_m and use_f:
            interaction = self.interaction_projection(torch.cat((modality.mean(1), frequency), -1))
            projected = interaction[:, None].expand(batch, 7, dim) * eligible[:, :, None]
            delta_m = delta_m + self.residual_scale[2] * gates[:, 2:] * projected.flatten(1)
        if self.frequency_only:
            # All matching capacity stays inside the added frequency branch.
            # The original DeMo descriptor is the first 5120 coordinates.
            delta_f = delta_f + delta_m.reshape(batch, 7, dim).mean(1)
            delta_m = torch.zeros_like(delta_m)
        return base + torch.cat((torch.zeros_like(base[:, :3 * dim]), delta_m, delta_f), -1)

    def controlled_states(self, base, m0, f0, eligible, available, full):
        # Neither conditional message nor psi is evaluated in states 10/01.
        modality = self.modality_expert.pool(m0, eligible)
        frequency = self.frequency_expert.pool(f0, available, self.structured)
        _, gm = self.calibrator(base, modality, torch.zeros_like(frequency))
        _, gf = self.calibrator(base, torch.zeros_like(modality), frequency)
        return {'00': base, '10': self.fuse(base, modality, torch.zeros_like(frequency), gm, True, False, eligible),
                '01': self.fuse(base, torch.zeros_like(modality), frequency, gf, False, True, eligible), '11': full}

    def forward(self, x, label=None, cam_label=None, view_label=None, return_pattern=3):
        keys = ('RGB', 'NI', 'TI')
        # The benchmark represents missing sensors by exactly zero normalized input.
        available = torch.stack([x[key].flatten(1).ne(0).any(1) for key in keys], 1)
        assert available.any(1).all(), 'benchmark conditions retain at least one modality'
        patches, globals_ = [], []
        for key, reduce in zip(keys, (self.rgb_reduce, self.nir_reduce, self.tir_reduce)):
            patch, global_ = self.BACKBONE(x[key], cam_label=cam_label, view_label=view_label)
            patches.append(patch)
            local = self.pool(patch.permute(0, 2, 1)).squeeze(-1)
            globals_.append(reduce(torch.cat((global_, local), -1)))
        original = torch.cat(globals_, -1)
        result = self.generalFusion(*patches, *globals_)
        base_moe = result[0] if self.training else result
        base = torch.cat((original, base_moe, torch.zeros_like(globals_[0])), -1)
        spatial = torch.stack(patches, 1)
        if self.structured or self.frequency_only:
            tokens = self.bands(spatial)
        else:
            tokens = spatial[:, :, None].expand(-1, -1, 3, -1, -1)
        if self.structured:
            eligible = torch.stack([available[:, subset].all(1) for subset in RELATIONS], 1)
        else:
            # Two ordinary experts read the full unfiltered modality/position set.
            # Their three streams and seven slots have no source/band restriction.
            eligible = torch.ones((len(available), 7), device=available.device, dtype=torch.bool)
        mvalues = self.modality_expert.prepare(tokens, available)
        fvalues = self.frequency_expert.prepare(tokens, available, self.structured)
        m0 = self.modality_expert.read(mvalues, eligible, self.structured)
        f0 = self.frequency_expert.read(fvalues, available, self.structured)
        # Simultaneous one-round update: only the independent evidence supplies
        # query conditions; each expert continues to read its own values.
        cm, cf = self.router.conditions(m0, f0)
        m1 = self.modality_expert.read(mvalues, eligible, self.structured, cm)
        f1 = self.frequency_expert.read(fvalues, available, self.structured, cf)
        modality, frequency, route = self.router.route(m1, f1, eligible, self.joint_interaction)
        prediction, gates = self.calibrator(base, modality, frequency)
        fused = self.fuse(base, modality, frequency, gates, True, True, eligible)
        self.last_route, self.last_gates = route.detach(), gates.detach()
        self.last_contribution_prediction = prediction.detach()
        if not self.training:
            return fused
        states = self.controlled_states(base, m0, f0, eligible, available, fused)
        target, pos, neg, scores = self.calibrator.targets(states, label)
        contribution = F.smooth_l1_loss(prediction.float(), target)
        self.contribution_audit = {'target_requires_grad': target.requires_grad,
                                   'reference': 'one stopped-gradient current-batch base gallery; same indices for all states, no EMA',
                                   'positive_indices': pos.tolist(), 'negative_indices': neg.tolist(),
                                   'targets': target.tolist(), 'predictions': prediction.detach().float().tolist(),
                                   'scores': {key: value.tolist() for key, value in scores.items()},
                                   'loss': float(contribution.detach()), 'closed_state_conditions': False,
                                   'inference_counterfactuals': False}
        standalone_m = self.modality_expert.pool(m0, eligible).mean(1)
        standalone_f = self.frequency_expert.pool(f0, available, self.structured)
        output = [self.fused_classifier(self.fused_neck(fused)), fused,
                  self.classifier_moe(self.bottleneck_moe(base_moe)), base_moe,
                  self.modality_classifier(self.modality_neck(standalone_m)), standalone_m,
                  self.frequency_classifier(self.frequency_neck(standalone_f)), standalone_f]
        if self.direct:
            output.extend((self.classifier(self.bottleneck(original)), original))
        else:
            for feature, classifier, neck in zip(globals_, (self.classifier_r, self.classifier_n, self.classifier_t), (self.bottleneck_r, self.bottleneck_n, self.bottleneck_t)):
                output.extend((classifier(neck(feature)), feature))
        output.append(self.contribution_loss_weight * contribution)
        return tuple(output)
