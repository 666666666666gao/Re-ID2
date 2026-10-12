"""AH MS full strong identity metric, same current relation experts and normal objectives."""
from collections import Counter
import torch
from layers.triplet_loss import TripletLoss
from shared_identity_axis import partial_gallery_triplet
from run_experiment import write_json
from msvr_full_identity_reference import build as full_build,assert_retained
import run_r201y_MS_K4 as y

configuration=y.configuration
VARIANTS=y.VARIANTS
REVISION=('AH MSVR310 retain strong teacher37 full5632 identity reference at .75private/.25public, '
    'freeze existing learned public projection along with five core modules, while K relation-local '
    'M/F/I correct only private5120 with cap.10. Fresh fused classifier/BN become5632 so the actual '
    'final deployed descriptor receives mainCE.05 and hinge.6; same other Y objectives '
    'actual11 PM/PF uniform-unitmean512 auxiliaries.001/frozen obsolete1024scorers, '
    'reg0/same-state targets, P16K4/B64/teacher50+extra50/newAdam/heads42/BN2/partial0/two backwards. '
    'No new branch or public correction, no visual/GF adaptation, preserve Fourier/two axes/ '
    'one condition/joint route/7relations. Full reference must reload all1032train591Q1055G '
    'with installedGT and selected teacher fullnormal perquery parity before native3 thenformal50. '
    'Ordinary/axis matched within AH; retaining full interface requires wider fusedhead and '
    'frozen public projection, so AE versus AH is not claimed one-tensor or equal-capacity ablation. '
    'OldM7 common-residual and AE staged-private negative outcomes retained, not rerun. '
    'Inherited strongteacher gains are baseline credit; require actual11minus00 and matched '
    'ordinary/axis comparison. One seed42/benchmarkmAP selection earliestties/all6oneweight. '
    'Only physicalGPU23/max2 one/card after AG actualclosed; no201NN/missing专项/new49/seedhunt/T-P.')


def full_step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    assert variant in VARIANTS
    images, target, cam, scene, names = batch
    images = {key: value.cuda(non_blocking=True) for key, value in images.items()}
    target, cam, scene = target.cuda(), cam.cuda(), scene.cuda()
    optimizer.zero_grad(set_to_none=True)
    bn_before = int(model.fused_neck.num_batches_tracked) if model.normal_priority_smoke else None
    with torch.autocast('cuda'):
        output = model(images, label=target, cam_label=cam, view_label=scene)
        end = len(output) - len(output) % 2
        assert len(model.loss_weights) == end // 2
        assert output[1].shape == (64, 5632)
        full_loss = sum(model.loss_weights[i // 2] * loss_fn(output[i], output[i + 1], target, cam)
                        for i in range(0, end, 2))
        if len(output) % 2:
            full_loss = full_loss + output[-1]
        gallery = output[1].detach()
    assert torch.isfinite(full_loss)
    scaler.scale(full_loss).backward()
    del output
    partial = {key: value if index in retained else torch.zeros_like(value)
               for index, (key, value) in enumerate(images.items())}
    with torch.autocast('cuda'):
        score, query = model(partial, label=target, cam_label=cam, view_label=scene, partial=True)
        partial_ce = xent(score, target)
        cross_triplet, pos, neg = partial_gallery_triplet(query, gallery, target)
        partial_loss = 0. * partial_ce + 0. * cross_triplet
    assert torch.isfinite(partial_ce) and torch.isfinite(cross_triplet) and partial_loss == 0
    scaler.scale(partial_loss).backward()
    scaler.unscale_(optimizer)
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    detail = dict(loss=float(full_loss.detach() + partial_loss.detach()), full_loss=float(full_loss.detach()),
        partial_ce=float(partial_ce.detach()), cross_triplet=float(cross_triplet.detach()),
        partial_loss_effective=float(partial_loss.detach()), partial_CE_weight=0., partial_triplet_weight=0.,
        partial_set=''.join('RNT'[i] for i in retained), positive_indices=pos.tolist(), negative_indices=neg.tolist(),
        reference_requires_grad=gallery.requires_grad, optimizer_updated=scaler.get_scale() >= previous_scale,
        amp_scale=scaler.get_scale(), names=list(names), primary_full_metric='unit5632_soft_triplet',
        freeze_identity_encoder=True, added_losses=0, descriptor_dim=5632)
    assert not detail['reference_requires_grad']
    if model.normal_priority_smoke:
        detail['fused_BN_calls'] = int(model.fused_neck.num_batches_tracked) - bn_before
        assert detail['fused_BN_calls'] == 2
        detail['full_unit_max_error'] = float((gallery.norm(dim=1) - 1).abs().max())
        assert detail['full_unit_max_error'] < 1e-6
    return detail


def margin_step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
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
        assert feature.shape==(64,5632)
        loss,positive,negative=hinge(feature,target)
        gap=positive-negative+.6
        baseline_gap=positive-negative+.3
        baseline_loss=baseline_gap.clamp_min(0).mean()
        metric.update(primary_full_metric='unit5632_margin06_batch_hard_triplet',
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

    detail=full_step(model,batch,optimizer,scaler,losses,xent,retained,variant)
    assert calls==len(model.loss_weights)
    detail.update(metric,sampling_identities=16,sampling_instances=4,sampling_batch=64,PI_outlet='shared_relation_local')
    if model.normal_priority_smoke:
        audit=model.relation_pi_native_audit
        assert audit['pi_input_shape']==[64,7,1024] and audit['pi_output_shape']==[64,7,512]
        assert audit['per_relation_outside_slot_errors']==[0.]*7
        assert audit['state_10_parent_equal'] and audit['state_01_parent_equal']
        detail['relation_PI_native']=dict(audit)
    return detail

def build(args,cfg,classes,cameras):
    y.margin_step=margin_step
    model=full_build(args,cfg,classes,cameras)
    model.anchor_record['method']=REVISION
    assert sum(p.numel() for p in model.parameters())==100902022
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==2674822
    return model


def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    detail=y.step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant)
    assert detail['descriptor_dim']==5632 and detail['primary_full_metric']=='unit5632_margin06_batch_hard_triplet'
    private=detail['final_descriptor_bound_full']
    detail.update(private_bound_dimension=5120,retained_public_dimension=512,retained_public_projection_frozen=True,
        complete_correction_bound=.75**.5*private['epsilon'],main_loss_deployed_dimension=5632)
    if model.normal_priority_smoke:assert_retained(model)
    return detail


def record(path,value):
    if 'descriptor_dim' in value:
        assert value['descriptor_dim']==5632
    if 'pretraining' in value:value['pretraining']='Selected full official MSVR310 sharedDeMo50 teacher37, complete5632 retained; additional50/freshmatched5632head/newAdam, private encoder andpublic projection fixed'
    write_json(path,value)


if __name__=='__main__':
    runner=y.w.runner;runner.configuration=configuration;runner.assert_anchor_unchanged=assert_retained;runner.write_json=record
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Existing Y sixpartial0 forwards/mining/BN2/two backwards retained, same partial sets andorders. Full and partial descriptors now5632, all nonzero mainnormal objectives use actual deployed metric. Missing performance not claimed.')
