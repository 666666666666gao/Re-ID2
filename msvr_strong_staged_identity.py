"""Strong private reference, outlet training, then limited DeMo relation adaptation."""
from copy import copy

import torch
from torch import nn
from torch.nn import functional as F

from experiment_data import seed_all
from modeling.meta_arch import weights_init_classifier, weights_init_kaiming
from msvr_stronger_private_teacher import install as install_strong_teacher
from original_identity_anchor import IDENTITY_MODULES
from shared_identity_axis import SharedIdentityDeMo, encode_available, base_supervision
import shared_identity_axis as shared
import original_identity_anchor as anchor
import run_r201y_MS_K4 as y

CONTROLS=('axis_frozen','axis_staged','frequency_staged','demo_staged')
HEAD_EPOCHS=5
RELATION_LR_FACTOR=.1


class StrongPrivateDeMo(SharedIdentityDeMo):
    """No new M/F experts or routing; the same 5120D private retrieval interface."""
    def __init__(self, classes, cfg, cameras, seed):
        super().__init__(classes,cfg,cameras,seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+30000)
            self.fused_neck=nn.BatchNorm1d(5120)
            self.fused_neck.bias.requires_grad_(False)
            self.fused_neck.apply(weights_init_kaiming)
            self.fused_classifier=nn.Linear(5120,classes,bias=False)
            self.fused_classifier.apply(weights_init_classifier)
        self.freeze_identity_encoder=True
        self.loss_weights=[1.,1.,.25]+([1.] if self.direct else [1.,1.,1.])
        for name in IDENTITY_MODULES:getattr(self,name).requires_grad_(False)

    def train(self,mode=True):
        super().train(mode)
        for name in IDENTITY_MODULES:getattr(self,name).eval()
        return self

    def forward(self,x,label=None,cam_label=None,view_label=None,partial=False,return_states=False):
        _,globals_,base,_,_,common=encode_available(self,x,cam_label,view_label)
        with torch.autocast('cuda',enabled=False):
            descriptor=F.normalize(base[:,:5120].float(),dim=1)
        if not self.training:
            return {key:descriptor for key in ('00','10','01','11')} if return_states else descriptor
        output=[self.fused_classifier(self.fused_neck(descriptor)),descriptor]
        if not partial:output.extend(base_supervision(self,globals_,base,common))
        return tuple(output)


def build(args,cfg,classes,cameras):
    assert args.dataset=='MSVR310' and args.seed==42 and args.variant in CONTROLS
    assert args.freeze_identity_encoder==1 and cfg.DATALOADER.NUM_INSTANCE==4
    if args.variant=='demo_staged':
        shared.available_base_fusion=anchor.fp32_base_fusion
        seed_all(args.seed)
        model=StrongPrivateDeMo(classes,cfg,cameras,args.seed).float().cuda()
        model.anchor_record={}
        model.normal_priority_smoke=args.mode=='smoke'
        model.train()
    else:
        loading=copy(args)
        loading.variant='frequency_shared' if args.variant=='frequency_staged' else 'axis_shared'
        model=y.build(loading,cfg,classes,cameras)
    model=install_strong_teacher(model,args,classes,cameras)
    model.control=args.variant
    model.staged_adaptation=args.variant!='axis_frozen'
    model.stage_epoch=0
    model.native_attempt=0
    model.adaptation_phase='outlets'
    model.freeze_identity_encoder=not model.staged_adaptation
    model.anchor_record.update(method='AE strong private reference and five outlet epochs then limited relations',
        anchor_training_epochs=50,new_stage_epochs=50,descriptor_dim=5120,
        staged_adaptation=dict(control=args.variant,outlet_epochs=HEAD_EPOCHS,
            relation_epochs=45 if model.staged_adaptation else 0,
            adapted_module='generalFusion',relation_LR_factor=RELATION_LR_FACTOR,
            original_encoder_eval_modes_fixed=True,original_buffers_retained=True,
            single_Adam_and_scheduler=True,parameter_groups_registered_before_stage1=True,
            native='Staged smoke: one successful outlet update then two relation updates; not five mature epochs',
            scope='Outlet stage also updates existing auxiliary heads/shared projection; not an exact linear probe. '
                '00 after relation adaptation differs from the teacher; cap.10 bounds current base-to-final only. '
                'Noexpert control has fewer parameters and no M/F computation. Ordinary/axis staged counts match. '
                'Strong private initialization is inherited baseline ability, not expert gain.'))
    model.train()
    assert_retained(model)
    return model


def select_phase(model):
    if model.normal_priority_smoke:
        model.native_attempt+=1
        adapt=model.staged_adaptation and model.native_attempt>1
    else:
        assert 1<=model.stage_epoch<=50
        adapt=model.staged_adaptation and model.stage_epoch>HEAD_EPOCHS
    model.generalFusion.requires_grad_(adapt)
    model.adaptation_phase='relations' if adapt else 'outlets'
    assert not any(module.training for module in model.generalFusion.modules())


@torch.no_grad()
def assert_retained(model):
    current=model.state_dict()
    for name in IDENTITY_MODULES:
        module=getattr(model,name)
        assert not any(child.training for child in module.modules())
        expected=name=='generalFusion' and model.adaptation_phase=='relations'
        assert all(p.requires_grad==expected for p in module.parameters())
    buffers={'generalFusion.'+name for name,_ in model.generalFusion.named_buffers()}
    kept={key:value for key,value in model.anchor_encoder_state.items()
          if model.adaptation_phase=='outlets' or not key.startswith('generalFusion.') or key in buffers}
    assert all(torch.equal(current[key].detach().cpu(),value) for key,value in kept.items())


def noexpert_step(model,batch,optimizer,scaler,loss_fn,xent,retained):
    # Reuse the exact G full/partial forwards, BN2, zero partial losses and two backwards.
    # G's variant argument is only its source fixture assertion, not a model selection.
    from run_r201g_normal_priority import step as full_partial_step
    from layers.triplet_loss import TripletLoss
    calls=0
    hinge=TripletLoss(margin=.6)
    primary={}
    def loss(score,feature,target,target_cam):
        nonlocal calls
        calls+=1
        if calls!=1:return loss_fn(score,feature,target,target_cam)
        assert feature.shape==(64,5120)
        metric,pos,neg=hinge(feature,target)
        primary.update(primary_full_metric='unit5120_margin06_batch_hard_triplet',
            primary_margin=.6,primary_triplet_loss=float(metric.detach()),
            primary_margin_violations=int((pos.detach()-neg.detach()+.6>0).sum()),
            primary_positive_distance_mean=float(pos.detach().mean()),
            primary_negative_distance_mean=float(neg.detach().mean()))
        return .05*xent(score,target)+metric
    result=full_partial_step(model,batch,optimizer,scaler,loss,xent,retained,'frequency_shared')
    assert calls==len(model.loss_weights)
    result.update(primary,no_new_experts=True,M_F_auxiliary_weight=0.,contribution_regression_weight=0.)
    return result


def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    assert variant==model.control
    select_phase(model)
    gf_ids={id(p) for p in model.generalFusion.parameters()}
    groups=[group for group in optimizer.param_groups if id(group['params'][0]) in gf_ids]
    if model.staged_adaptation:
        assert {id(group['params'][0]) for group in groups}==gf_ids
    else:assert not groups
    nominal=[group['lr'] for group in groups]
    if model.adaptation_phase=='relations':
        for group,rate in zip(groups,nominal):group['lr']=RELATION_LR_FACTOR*rate
    if variant=='demo_staged':
        result=noexpert_step(model,batch,optimizer,scaler,loss_fn,xent,retained)
    else:
        underlying='frequency_shared' if variant=='frequency_staged' else 'axis_shared'
        result=y.step(model,batch,optimizer,scaler,loss_fn,xent,retained,underlying)
    for group,rate in zip(groups,nominal):group['lr']=rate
    if model.adaptation_phase=='outlets':
        assert all(p.grad is None for p in model.generalFusion.parameters())
        assert all(p not in optimizer.state for p in model.generalFusion.parameters())
    if model.normal_priority_smoke:
        assert result['optimizer_updated']
        assert_retained(model)
        result['generalFusion_native_gradients']={name:bool(p.grad is not None and
            torch.isfinite(p.grad).all() and p.grad.abs().sum()>0)
            for name,p in model.generalFusion.named_parameters()}
        if model.adaptation_phase=='relations':assert all(result['generalFusion_native_gradients'].values())
    result.update(control=variant,adaptation_phase=model.adaptation_phase,stage_epoch=model.stage_epoch,
        freeze_identity_encoder=model.freeze_identity_encoder,
        generalFusion_registered_parameter_count=sum(p.numel() for group in groups for p in group['params']),
        generalFusion_effective_LR_factor=RELATION_LR_FACTOR if model.adaptation_phase=='relations' else 0.,
        nominal_group_LR_restored=True,core_eval_modes_unchanged=True)
    return result
