"""Keep the strong MS private/public reference; correct only its seven private relations."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from modeling.meta_arch import weights_init_classifier,weights_init_kaiming
from relation_local_identity_outlet import RelationLocalPIAxis
from shared_identity_axis import encode_available,base_supervision
from original_identity_anchor import assert_anchor_unchanged
from msvr_stronger_private_teacher import install as install_strong_teacher
import run_r201y_MS_K4 as y


class StrongFullRelationAxis(RelationLocalPIAxis):
    def __init__(self,classes,cfg,cameras,variant,seed,bypass):
        super().__init__(classes,cfg,cameras,variant,seed,bypass)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+30000)
            self.fused_neck=nn.BatchNorm1d(5632)
            self.fused_neck.bias.requires_grad_(False)
            self.fused_neck.apply(weights_init_kaiming)
            self.fused_classifier=nn.Linear(5632,classes,bias=False)
            self.fused_classifier.apply(weights_init_classifier)

    def train(self,mode=True):
        super().train(mode)
        self.shared_projection.eval()
        return self

    def complete_descriptor(self,base,private):
        with torch.autocast('cuda',enabled=False):
            assert base.shape[1]==5632 and private.shape[1]==5120
            public=F.normalize(base[:,5120:].float(),dim=1)
            return torch.cat((math.sqrt(.75)*private.float(),math.sqrt(.25)*public),1)

    def controlled_states(self,base,m0,f0,eligible,available,full):
        states=super().controlled_states(base,m0,f0,eligible,available,full)
        states={key:self.complete_descriptor(base,value) if key!='11' else value for key,value in states.items()}
        assert all(value.shape==full.shape for value in states.values())
        assert all(torch.equal(value[:,5120:],full[:,5120:]) for value in states.values())
        return states

    def forward(self,x,label=None,cam_label=None,view_label=None,partial=False,return_states=False):
        patches,globals_,base,eligible,available,common=encode_available(self,x,cam_label,view_label)
        spatial=torch.stack(patches,1);tokens=self.bands(spatial)
        mvalues=self.modality_expert.prepare(tokens,available)
        fvalues=self.frequency_expert.prepare(tokens,available,self.structured)
        m0=self.modality_expert.read(mvalues,eligible,self.structured)
        f0=self.frequency_expert.read(fvalues,available,self.structured)
        cm,cf=self.router.conditions(m0,f0)
        m1=self.modality_expert.read(mvalues,eligible,self.structured,cm)
        f1=self.frequency_expert.read(fvalues,available,self.structured,cf)
        modality,frequency,route=self.router.route(m1,f1,eligible,self.joint_interaction)
        prediction,gates=self.calibrator(base,modality,frequency)
        v=self.router.by_relation(f1);mass=route.sum(2,keepdim=True)
        conditional=route/torch.where(eligible[:,:,None],mass,torch.ones_like(mass))
        relation_frequency=(v*conditional[...,None]).sum(2)
        private=self.fuse(base,modality,frequency,gates,True,True,eligible,route.sum(2),relation_frequency)
        fused=self.complete_descriptor(base,private)
        self.last_route,self.last_gates=route.detach(),gates.detach()
        if not self.training and not return_states:return fused
        if self.training and partial:return self.fused_classifier(self.fused_neck(fused)),fused
        states=self.controlled_states(base,m0,f0,eligible,available,fused)
        if not self.training:return states
        target,pos,neg,scores=self.calibrator.targets(states,label)
        contribution=F.smooth_l1_loss(prediction.float(),target)
        self.contribution_audit=dict(target_requires_grad=target.requires_grad,positive_indices=pos.tolist(),negative_indices=neg.tolist(),
            targets=target.tolist(),predictions=prediction.detach().float().tolist(),scores={key:value.tolist() for key,value in scores.items()},loss=float(contribution.detach()))
        standalone_m=self.modality_expert.pool(m0,eligible).mean(1)
        standalone_f=self.frequency_expert.pool(f0,available,self.structured)
        output=[self.fused_classifier(self.fused_neck(fused)),fused,
            self.modality_classifier(self.modality_neck(standalone_m)),standalone_m,
            self.frequency_classifier(self.frequency_neck(standalone_f)),standalone_f]
        output.extend(base_supervision(self,globals_,base,common));output.append(self.contribution_loss_weight*contribution)
        return tuple(output)


def build(args,cfg,classes,cameras):
    assert args.dataset=='MSVR310' and args.seed==42 and args.freeze_identity_encoder==1
    # Process-local class dispatch through the existing K/Y builders; no old source edit.
    y.m.k.RelationLocalPIAxis=StrongFullRelationAxis
    model=install_strong_teacher(y.build(args,cfg,classes,cameras),args,classes,cameras)
    model.shared_projection.requires_grad_(False)
    model.retained_public_state={key:value.detach().cpu().clone() for key,value in model.shared_projection.state_dict().items()}
    model.anchor_record.update(method='AH retain complete strong5632 reference with relation-local private corrections',
        descriptor_dim=5632,deployed_descriptor_dim=5632,
        complete_reference=dict(private_dimension=5120,public_dimension=512,metric_weights=[.75,.25],
            public_projection_frozen=True,public_projection_parameters=527872,fresh_fused_head_dimension=5632,
            private_correction_bound=.10,complete_correction_bound=math.sqrt(.75)*.10,
            limits='Full strong teacher gains are inherited baseline ability. No new public expert or public correction. '
                'Versus AE private fixture, retaining public metric also requires a5632 fused head and freezing public projection; '
                'not claimed an isolated one-parameter or equal-trainable-count change versus AE. Ordinary/axis matched within AH.'))
    model.anchor_record['strong_teacher']['private_reference']='AH complete5632 .75private/.25public; both retained teacher projections'
    model.anchor_record['strong_teacher']['limits']='Strong full teacher49.426/69.036 is baseline credit, not an expert gain. AH00 requires full official warm parity before training.'
    model.anchor_record['final_unit_descriptor_bound']['placement']='Private5120 after full M/F/I aggregation; retained public512 appended at .25 metric weight'
    model.anchor_record['final_unit_descriptor_bound']['complete_descriptor_bound']=math.sqrt(.75)*.10
    model.anchor_record['M_F_auxiliary_outlet']['inference']='Hooks/pool replacement inactive in eval; AH retains full5632 .75private/.25public descriptor'
    model.train();assert_retained(model)
    return model


@torch.no_grad()
def assert_retained(model):
    assert_anchor_unchanged(model)
    assert not model.shared_projection.training and all(not p.requires_grad for p in model.shared_projection.parameters())
    assert all(torch.equal(value.detach().cpu(),model.retained_public_state[key]) for key,value in model.shared_projection.state_dict().items())
