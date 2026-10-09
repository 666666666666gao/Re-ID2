"""M: weak-dataset margin control; retain the selected K RGBNT201 model."""
from collections import Counter
import torch
import run_full_official_frozen_anchor_experiment as runner
import run_r201k_relation_local_pi as k
import run_r201l_uniform_k8 as l
from layers.triplet_loss import TripletLoss
from original_identity_anchor import assert_anchor_unchanged
from rgbnt201_projection_probe import VARIANTS

REVISION=('R201M single final-primary margin factor relative to uniform L: '
    'final full-unit5120 batch-hard hinge margin0.3 becomes0.6, with truthful '
    'telemetry and same-feature margin0.3 diagnostic only. FusedCE0.25, '
    'auxiliary original soft Triplets/CE/contribution, partial0 forward/BN2/'
    'mining/two backwards, K shared relation-local PI/PM/PF/router/gates/'
    'residuals, frozen respective originalDeMo42 teacher50, fresh heads/Adam/'
    'lr/AMP/seed42/B64/P8K8/additional50 and5120D remain. Do not change '
    'MODEL.NO_MARGIN or global SOLVER.MARGIN. Train MSVR310/RGBNT100 only; '
    'preserve K201 margin0.3 best, explicitly dataset-specific training parameters. '
    'Full official train/query/gallery, '
    'normal mAP-best earliestties benchmark selected, samebest allsix/later paper-six masks. '
    'Known loss control, not new architecture, universal inactivity cause '
    'or guaranteed +1/generalization/calibration/causal mechanism. Native '
    'expert-gradient comparisons and detached route/gate statistics are diagnostics only.')


configuration=l.configuration


def build(args,cfg,classes,cameras):
    assert cfg.MODEL.NO_MARGIN and cfg.SOLVER.MARGIN==.3
    assert cfg.DATALOADER.NUM_INSTANCE==8 and args.dataset in ('RGBNT100','MSVR310')
    model=k.build(args,cfg,classes,cameras)
    model.anchor_record['method']=REVISION
    return model


def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    counts=Counter(batch[1].tolist());assert len(counts)==8 and set(counts.values())=={8}
    calls,metric,primary_expert_gradients=0,{},{}
    hinge=TripletLoss(margin=.6)
    if model.normal_priority_smoke:model.relation_pi_native_audit={}

    def losses(score,feature,target,target_cam):
        nonlocal calls
        calls+=1
        if calls!=1:
            auxiliary=loss_fn(score,feature,target,target_cam)
            if model.normal_priority_smoke and calls in (2,3):
                name='M' if calls==2 else 'F'
                expert=model.modality_expert if name=='M' else model.frequency_expert
                gradient=torch.autograd.grad(model.loss_weights[calls-1]*auxiliary,
                    expert.output[1].weight,retain_graph=True)[0].detach().float()
                primary=primary_expert_gradients[name]
                assert torch.isfinite(gradient).all()
                denominator=primary.norm()*gradient.norm()
                metric['primary_vs_aux_'+name]=dict(
                    parameter=name+'.output.1.weight',auxiliary_weight=model.loss_weights[calls-1],
                    primary_norm=float(primary.norm()),weighted_auxiliary_norm=float(gradient.norm()),
                    cosine=float((primary*gradient).sum()/denominator) if denominator>0 else None)
            return auxiliary
        assert feature.shape==(64,5120)
        loss,positive,negative=hinge(feature,target)
        gap=positive-negative+.6
        baseline_gap=positive-negative+.3
        baseline_loss=baseline_gap.clamp_min(0).mean()
        metric.update(primary_full_metric='unit5120_margin06_batch_hard_triplet',
            primary_metric_coefficient=1.,primary_margin=.6,primary_triplet_loss=float(loss.detach()),
            primary_margin_violations=int((gap.detach()>0).sum()),
            primary_positive_distance_mean=float(positive.detach().mean()),
            primary_negative_distance_mean=float(negative.detach().mean()),
            primary03_same_feature_loss=float(baseline_loss.detach()),
            primary03_same_feature_violations=int((baseline_gap.detach()>0).sum()),
            auxiliary_original_soft_triplet=True)
        route=model.last_route.float()
        relations,bands=route.sum(2),route.sum(1)
        logs=route.clamp_min(1e-12).log()
        metric.update(route_entropy_mean=float(-(route*logs).sum((1,2)).mean()),
            route_nonindependence_mean=float((route*(logs-
                (relations[:,:,None]*bands[:,None,:]).clamp_min(1e-12).log())).sum((1,2)).mean()),
            route_relation_mass_mean=relations.mean(0).cpu().tolist(),
            route_band_mass_mean=bands.mean(0).cpu().tolist(),
            full_gate_mean=model.last_gates.float().mean(0).cpu().tolist(),
            residual_scales=model.residual_scale.detach().float().cpu().tolist())
        if model.normal_priority_smoke:
            assert abs(float(loss.detach())-float(gap.detach().clamp_min(0).mean()))<1e-6
            with torch.autocast('cuda',enabled=False):
                gradient=torch.autograd.grad(loss,feature,retain_graph=True)[0]
                baseline_gradient=torch.autograd.grad(baseline_loss,feature,retain_graph=True)[0]
                representatives=(model.modality_projection[-1].weight,
                    model.frequency_projection[-1].weight,model.interaction_projection[-1].weight)
                experts=(model.modality_expert.output[1].weight,model.frequency_expert.output[1].weight)
                parameter_gradients=torch.autograd.grad(loss,representatives+experts,retain_graph=True)
                primary_expert_gradients.update({n:g.detach().float() for n,g in zip(('M','F'),parameter_gradients[3:])})
            assert torch.isfinite(gradient).all() and torch.isfinite(baseline_gradient).all()
            assert all(torch.isfinite(g).all() for g in parameter_gradients)
            if loss.detach()>0:assert gradient.abs().sum()>0
            metric.update(primary_margin_definition_equal=True,
                primary_feature_gradient_finite=True,primary_feature_gradient_l1=float(gradient.abs().sum()),
                primary03_same_feature_gradient_l1=float(baseline_gradient.abs().sum()),
                primary_per_anchor_positive=positive.detach().cpu().tolist(),
                primary_per_anchor_negative=negative.detach().cpu().tolist(),
                primary_parameter_gradient_l1={n:float(g.abs().sum()) for n,g in zip(('PM','PF','PI','M','F'),parameter_gradients)})
        return .25*xent(score,target)+loss

    detail=k.parent.g.step(model,batch,optimizer,scaler,losses,xent,retained,variant)
    assert calls==len(model.loss_weights)
    detail.update(metric,sampling_identities=8,sampling_instances=8,sampling_batch=64,PI_outlet='shared_relation_local')
    if model.normal_priority_smoke:
        audit=model.relation_pi_native_audit
        assert audit['pi_input_shape']==[64,7,1024] and audit['pi_output_shape']==[64,7,512]
        assert audit['per_relation_outside_slot_errors']==[0.]*7
        assert audit['state_10_parent_equal'] and audit['state_01_parent_equal']
        detail['relation_PI_native']=dict(audit)
    return detail


if __name__=='__main__':
    runner.configuration=configuration
    runner.assert_anchor_unchanged=assert_anchor_unchanged
    runner.write_json=k.parent.g.record
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Unchanged L partial0 forward/BN2/mining/two backwards and auxiliarysoft; only final-primary hinge margin0.6.')
