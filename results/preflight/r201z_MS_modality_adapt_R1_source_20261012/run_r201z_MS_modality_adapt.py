"""Z: let the existing DeMo modality relations adapt to the unchanged Y experts."""
from copy import copy
import torch
import run_r201y_MS_K4 as y
from adapt_existing_modality_fusion import install,assert_retained,ADAPTED_PARAMETERS

configuration=y.configuration
VARIANTS=y.VARIANTS
REVISION=('Z MS-only existing-modality-parameter gradient factor versus completed Y: '
    'enable existing generalFusion HDM/ATMoE9958912 parameter gradients while every original module remains eval; '
    'BACKBONE/rgb_reduce/nir_reduce/tir_reduce remain frozen and exact, all generalFusion BN buffers/dropout modes unchanged. '
    'No new parameters/experts/outputcoords, same5120/FFT/twoaxes/PI/router/gates/mean512/MFaux.001/mainCE.05/reg0/hinge.6/cap.10/P16K4/BN2/partial0/two backwards, '
    'same originalDeMo42 teacher50 verified frozen load thenfreshheads/newAdam/additional50/705updates. '
    'Both ordinary/axis adapt identically; new native3 allactivegradients/strictreload/retained4+buffers before50. '
    'Effective trainables increase3122822->13081734, total100821638 unchanged; Y is not equal-trainable-budget. '
    '00 now changes with modality parameters, and cap.10 references current base, not total distance to original teacher. '
    'Full official1032train/591Q/1055G/normalbenchmarkmAP-best earliestties/samecheckpoint six/CMC/perquery/groups/best-last. '
    'CurrentW/Y negative complete evidence motivates compatibility test but doesnot prove frozenrelations causal; known finetuning control notnewinnovation or guaranteedgain. '
    'No201NN/new49/missing selection/failedseedsearch, X original55293 GPU2/3 unchanged.')
TRIPLET_CLASS=y.TripletLoss

def build(args,cfg,classes,cameras):
    assert args.dataset=='MSVR310' and args.freeze_identity_encoder==0
    # Y intentionally only accepts a frozen loading stage; complete that verified load unchanged.
    loading=copy(args);loading.freeze_identity_encoder=1
    model=install(y.build(loading,cfg,classes,cameras))
    model.anchor_record['method']=REVISION
    return model

def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    probes={};xent_calls=0
    representatives=(model.generalFusion.rnt.out_proj.weight,model.generalFusion.moe.experts[0].expertHead[0].mlp[0].weight)
    names=('generalFusion.rnt.out_proj.weight','generalFusion.moe.experts.0.expertHead.0.mlp.0.weight')
    def record(key,loss):
        with torch.autocast('cuda',enabled=False):
            probes[key]=tuple(g.detach().float() for g in torch.autograd.grad(loss,representatives,retain_graph=True))
    def hinge_factory(margin):
        assert margin==.6
        hinge=TRIPLET_CLASS(margin=margin)
        def observed(feature,target):
            out=hinge(feature,target);record('main_hinge',out[0]);return out
        return observed
    def observed_xent(score,target):
        nonlocal xent_calls
        xent_calls+=1;loss=xent(score,target)
        if xent_calls==1:record('main_CE',.05*loss)
        return loss
    def observed_aux(score,feature,target,target_cam):
        loss=loss_fn(score,feature,target,target_cam)
        if feature.shape==(64,3584):
            assert 'base_aux' not in probes
            record('base_aux',model.loss_weights[3]*loss)
        return loss
    if model.normal_priority_smoke:
        y.TripletLoss=hinge_factory
        detail=y.step(model,batch,optimizer,scaler,observed_aux,observed_xent,retained,variant)
        y.TripletLoss=TRIPLET_CLASS
    else:
        detail=y.step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant)
    assert_retained(model)
    detail.update(freeze_identity_encoder=False,generalFusion_parameter_adaptation=True,
        generalFusion_trainable_parameters=ADAPTED_PARAMETERS,generalFusion_eval_buffers_and_frozen4_retained=True,
        bound_reference='current_adapting_base_not_original_teacher')
    if model.normal_priority_smoke:
        # Observational mirror of the existing third-step gate; no new first/second-step gate.
        gradients={n:bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0)
                   for n,p in model.generalFusion.named_parameters()}
        current=model.state_dict();changed=[k for k,v in model.anchor_encoder_state.items() if k.startswith('generalFusion.') and not torch.equal(current[k].detach().cpu(),v)]
        detail['generalFusion_native_gradient_audit']=dict(parameters=gradients,changed_tensors=changed,all_active_nonzero=all(gradients.values()),eval_mode=True)
        assert xent_calls==2 and set(probes)=={'main_hinge','main_CE','base_aux'}
        reports=[]
        for index,name in enumerate(names):
            main=probes['main_hinge'][index]+probes['main_CE'][index];aux=probes['base_aux'][index]
            assert torch.isfinite(main).all() and torch.isfinite(aux).all()
            norm=main.norm()*aux.norm()
            reports.append(dict(parameter=name,main_hinge_norm=float(probes['main_hinge'][index].norm()),main_CE05_norm=float(probes['main_CE'][index].norm()),
                main_combined_norm=float(main.norm()),weighted_base_aux_norm=float(aux.norm()),base_aux_weight=model.loss_weights[3],
                cosine=float((main*aux).sum()/norm) if norm>0 else None,
                actual_full_unscaled_norm=float(representatives[index].grad.float().norm())))
        detail['generalFusion_main_base_aux_native']=reports
    return detail

if __name__=='__main__':
    runner=y.w.runner;runner.configuration=configuration;runner.assert_anchor_unchanged=assert_retained
    runner.write_json=y.w.s.p.parent.parent.parent.k.parent.g.record
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Unchanged Y partial0/BN2/two backwards. Only generalFusion gradients enabled; original modes and buffers still fixed.')
