"""P1-B: source-specific F residuals in DeMo identity relation coordinates."""
import torch
from torch.nn import functional as F
from dual_axis import RELATIONS
from mass_axis_collaboration import MassAxisCollaborationDeMo


class RelationFrequencyInterfaceDeMo(MassAxisCollaborationDeMo):
    def fuse(self,base,modality,frequency,gates,use_m,use_f,eligible,relation_mass=None,relation_frequency=None):
        original=super().fuse(base,modality,frequency,gates,use_m,use_f,eligible,relation_mass)
        if not use_f:return original
        assert relation_frequency is not None and relation_mass is not None
        with torch.autocast('cuda',enabled=False):
            batch,relations,dim=relation_frequency.shape
            anchor=base[:,3*dim:10*dim].float().reshape(batch,relations,dim).norm(dim=-1,keepdim=True).detach()
            direction=F.normalize(self.frequency_projection(relation_frequency.float()),dim=-1)
            weights=relation_mass.float()*eligible.sum(1,keepdim=True)
            residual=self.residual_scale[1]*gates.float()[:,1:2,None]*direction*anchor*weights[...,None]*eligible[...,None]
            identity=original[:,:10*dim].float()
            corrected=torch.cat((identity[:,:3*dim],identity[:,3*dim:]+residual.flatten(1)),1)
            return torch.cat((corrected,torch.zeros_like(base[:,-dim:])),1)

    def controlled_states(self,base,m0,f0,eligible,available,full):
        modality=self.modality_expert.pool(m0,eligible)
        frequency=self.frequency_expert.pool(f0,available,self.structured)
        _,gm=self.calibrator(base,modality,torch.zeros_like(frequency))
        _,gf=self.calibrator(base,torch.zeros_like(modality),frequency)
        # State01 sees only independent frequency evidence from each legal S.
        # It never reads M conditions, relation_score(M), full route, psi or I.
        v=self.router.by_relation(f0)
        with torch.autocast('cuda',enabled=False):
            v=v.float()
            band_weights=self.frequency_expert.pool_score(v).squeeze(-1).softmax(2)
            relation_frequency=(v*band_weights[...,None]).sum(2)
            logits=self.router.band_score(relation_frequency).squeeze(-1)
            relation_mass=logits.masked_fill(~eligible,-torch.inf).softmax(1)
        return {'00':base,
            '10':self.fuse(base,modality,torch.zeros_like(frequency),gm,True,False,eligible),
            '01':self.fuse(base,torch.zeros_like(modality),frequency,gf,False,True,eligible,relation_mass,relation_frequency),
            '11':full}

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
            v = self.router.by_relation(f1)
            mass = route.sum(2, keepdim=True)
            conditional = route / torch.where(eligible[:, :, None], mass, torch.ones_like(mass))
            relation_frequency = (v * conditional[..., None]).sum(2)
            fused = self.fuse(base, modality, frequency, gates, True, True, eligible, route.sum(2), relation_frequency)
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
