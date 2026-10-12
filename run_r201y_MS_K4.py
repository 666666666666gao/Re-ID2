"""Y: exact W MS recipe, with only original DeMo P16/K4 sampling restored."""
from collections import Counter
import torch
import run_r201w_reduced_auxiliary as w
import run_r201m_primary_margin06 as m
from layers.triplet_loss import TripletLoss

VARIANTS=w.VARIANTS
REVISION='Y MS-only sampling control versus completed WMS: NUM_INSTANCE8->4, B64/P8K8->P16K4. This changes positive multiplicity/negative identities and sampler length/formal updates1000->705 perarm; same50epochs not equalupdates. Source CPU actual W1032records/155IDs/98under8 versus46under4 motivates, not proves cause. Hold W actual11 projected-unit relation mean512/frozenobsolete1024score params/MFaux.001/mainCE.05/reg0/hinge.6/cap.10/PI/FFT/experts/route/gates/5120D, frozen originalDeMo42 teacher50/freshheads42/newAdam/lr/AMP/BN2/partial0/two backwards. Both ordinary/axis identical K4 order/budget/native3/allgradient strict beforefresh50. Full officialtrain/Q/G/benchmarkmAP-best earliestties/samecheckpoint six/CMC/perquery/groups/best-last/fullcurves. No201NN/49/missing selection/failedseedsearch, sourceknown samplercontrol notnovelmechanism or guaranteedgain. K201protected.'

def configuration(args):
    assert args.dataset=='MSVR310'
    cfg=w.configuration(args);cfg.defrost();cfg.DATALOADER.NUM_INSTANCE=4;cfg.freeze()
    assert cfg.SOLVER.IMS_PER_BATCH==64 and cfg.SOLVER.MAX_EPOCHS==50
    return cfg

def margin_build(args,cfg,classes,cameras):
    assert cfg.MODEL.NO_MARGIN and cfg.SOLVER.MARGIN==.3
    assert args.dataset=='MSVR310' and cfg.DATALOADER.NUM_INSTANCE==4
    model=m.k.build(args,cfg,classes,cameras)
    model.anchor_record['method']=REVISION
    return model

def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    counts=Counter(batch[1].tolist());assert len(counts)==16 and set(counts.values())=={4}
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

    detail=m.k.parent.g.step(model,batch,optimizer,scaler,losses,xent,retained,variant)
    assert calls==len(model.loss_weights)
    detail.update(metric,sampling_identities=16,sampling_instances=4,sampling_batch=64,PI_outlet='shared_relation_local')
    if model.normal_priority_smoke:
        audit=model.relation_pi_native_audit
        assert audit['pi_input_shape']==[64,7,1024] and audit['pi_output_shape']==[64,7,512]
        assert audit['per_relation_outside_slot_errors']==[0.]*7
        assert audit['state_10_parent_equal'] and audit['state_01_parent_equal']
        detail['relation_PI_native']=dict(audit)
    return detail


margin_step=step

def build(args,cfg,classes,cameras):
    # Process-local dispatch only: original module sources and separate X process unchanged.
    m.build=margin_build;m.step=margin_step
    model=w.build(args,cfg,classes,cameras)
    model.anchor_record.update(method=REVISION,sampler_control=dict(parent_K=8,current_K=4,batch_size=64,identities_per_batch=16,
        old_updates_per_arm=1000,new_updates_per_arm=705,same_epochs_not_equal_updates=True))
    return model

def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    detail=w.step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant)
    assert detail['sampling_identities']==16 and detail['sampling_instances']==4 and detail['sampling_batch']==64
    return detail

if __name__=='__main__':
    runner=w.runner;runner.configuration=configuration;runner.assert_anchor_unchanged=w.assert_anchor_unchanged
    runner.write_json=w.s.p.parent.parent.parent.k.parent.g.record
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='W MS losses/outlets/BN2/partial0 unchanged, only actual K4 sampler; new705updates not1000.')
