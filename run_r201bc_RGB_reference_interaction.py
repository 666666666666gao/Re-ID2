"""BC: AV plus one identity-reference-conditioned existing PI bottleneck."""
import torch
import run_r201av_RGB_frequency_value_decay as parent
from identity_reference_interaction import install_identity_reference_interaction

configuration=parent.configuration
VARIANTS=parent.VARIANTS
make_optimizer=parent.make_optimizer
REVISION=('BC RGBNT100 returns to stronger AV, not failed BA initialization or AZ corrected-state metric. '
    'Only existing relation-local PI hidden64 receives multiplicative 2sigmoid(Wb unit(stopgrad(base_relation512))). '
    'One zero-initialized shared512->64 matrix32768 parameters, same ordinary and structured models; '
    'zero gate logits preserve initial AV output without changing existing tensors or RNG. '
    'Original PM/PF and auxiliary outlets, conditional experts/joint routing, final5120 and cap.10 remain. '
    'Known reference-conditioned modulation, not new frequency evidence or established cause. '
    'AV teacher42/fresh heads42/newAdam/extra50/full8675train1715Q8575G/hardP8K8/B64/T2/'
    'CE.25/hinge.6/reg.05/MFaux.001 first11pureunit3584+scored512/BN2/partial0/two backwards retained. '
    'Normal benchmarkmAP earliesttie/allsix samebest/CMC50/perquery/groups/bestlast/eightstates. '
    'Keep201K/AV/MSstrong; only26physicalGPU2/3, noT-P/new201NN/missing49/failedseedhunt. '
    'Complete source review/native/paired full50/evaluation required; prepared source is not an actual result.')


def build(args,cfg,classes,cameras):
    model=parent.build(args,cfg,classes,cameras)
    existing={name:(p,p._version) for name,p in model.named_parameters()}
    model=install_identity_reference_interaction(model)
    current=dict(model.named_parameters())
    assert set(current)-set(existing)=={'interaction_projection.identity_reference_weight'}
    assert all(current[n] is p and current[n]._version==version for n,(p,version) in existing.items())
    model.anchor_record['identity_reference_interaction']['existing_parameter_objects_versions_exact']=True
    model.anchor_record['method']=REVISION
    return model


def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    parameter=model.interaction_projection.identity_reference_weight
    before=float(parameter.detach().float().norm())
    detail=parent.step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant)
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    detail['identity_reference_interaction']=dict(parameters_added=32768,
        weight_norm_before=before,weight_norm_after=float(parameter.detach().float().norm()),
        unscaled_task_gradient_norm=float(parameter.grad.detach().float().norm()),
        final_descriptor_dim=5120,base_reference_stop_gradient=True,
        placement='Existing per-relation PI hidden64; same gate for controlled ordinary and structured expert models')
    if model.normal_priority_smoke:
        assert parameter.grad.detach().abs().sum()>0
    return detail


if __name__=='__main__':
    import run_full_official_frozen_anchor_experiment as runner
    assert runner.make_optimizer is parent.parent.parent.original_make_optimizer
    runner.make_optimizer=make_optimizer
    runner.configuration=configuration
    runner.assert_anchor_unchanged=parent.parent.parent.parent.parent.parent.parent.x.w.assert_anchor_unchanged
    runner.write_json=parent.parent.parent.parent.parent.parent.parent.x.w.s.p.parent.parent.parent.k.parent.g.record
    runner.make_loader=parent.parent.parent.parent.parent.parent.parent.make_loader
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Same AV BN2/partial0/two backwards; one32768-parameter identity reference gate inside existing PI only.')
