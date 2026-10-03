"""Retain the independent frequency AUX and supervise actual routed retrieval."""
import torch
from torch.nn import functional as F
from dual_axis import RELATIONS
from metric_mass_axis_collaboration import MetricMassAxisCollaborationDeMo


class RoutedSupervisionMetricDeMo(MetricMassAxisCollaborationDeMo):
    def __init__(self,classes,cfg,cameras):
        super().__init__(classes,cfg,cameras)
        # Fixed total F auxiliary budget .1: independent .05 + routed .05.
        # The original pool_score remains trained on its independent path.
        self.loss_weights[3]=.05
        self.loss_weights.insert(4,.05)

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
            fused = self.fuse(base, modality, frequency, gates, True, True, eligible, route.sum(2))
            self.last_route, self.last_gates = route.detach(), gates.detach()
            self.last_contribution_prediction = prediction.detach()
            if not self.training:
                return fused
            states = self.controlled_states(base, m0, f0, eligible, available, fused)
            target, pos, neg, scores = self.calibrator.targets(states, label)
            contribution = F.smooth_l1_loss(prediction.float(), target)
            self.contribution_audit = {'target_requires_grad': target.requires_grad,
                                       'reference': 'one stopped-gradient current-batch full11 gallery; same indices for all four states, no EMA',
                                       'positive_indices': pos.tolist(), 'negative_indices': neg.tolist(),
                                       'targets': target.tolist(), 'predictions': prediction.detach().float().tolist(),
                                       'scores': {key: value.tolist() for key, value in scores.items()},
                                       'loss': float(contribution.detach()), 'closed_state_conditions': False,
                                       'inference_counterfactuals': False}
            standalone_m = self.modality_expert.pool(m0, eligible).mean(1)
            standalone_f = self.frequency_expert.pool(f0, available, self.structured)
            # Supervise the same jointly routed PF direction used by retrieval.
            # Reuse the existing frequency classifier; no new trainable head.
            with torch.autocast('cuda', enabled=False):
                routed_f = F.normalize(self.frequency_projection(frequency.float()), dim=1)
                routed_score = 32. * F.linear(routed_f, F.normalize(self.frequency_classifier.weight.float(), dim=1))

            output = [self.fused_classifier(self.fused_neck(fused)), fused,
                      self.classifier_moe(self.bottleneck_moe(base_moe)), base_moe,
                      self.modality_classifier(self.modality_neck(standalone_m)), standalone_m,
                      self.frequency_classifier(self.frequency_neck(standalone_f)), standalone_f]
            output.extend((routed_score, routed_f))
            if self.direct:
                output.extend((self.classifier(self.bottleneck(original)), original))
            else:
                for feature, classifier, neck in zip(globals_, (self.classifier_r, self.classifier_n, self.classifier_t), (self.bottleneck_r, self.bottleneck_n, self.bottleneck_t)):
                    output.extend((classifier(neck(feature)), feature))
            output.append(self.contribution_loss_weight * contribution)
            return tuple(output)
