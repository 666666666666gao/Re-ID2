"""AV: remove coupled decay only from existing mid/high F input Linear weights."""
import math
import torch
import run_r201as_RGB_query_normalization as parent

configuration=parent.configuration
VARIANTS=parent.VARIANTS
TARGETS=('frequency_expert.values.1.1.weight','frequency_expert.values.2.1.weight')
REVISION=('AV RGBNT100 changes only Adam weight_decay1e-4->0 for the existing middle/high '
    'frequency value-encoder Linear(512,64) WEIGHTS. Low-band value Linear, all biases, '
    'LayerNorms, shared M value encoder and remaining optimizer groups retain AS policy. '
    'AQ five packed QKV weights already decay0; AS only M projectedQ per-head L2*sqrt8 '
    'retained. No new network, parameter, loss, forward, frequency transform, descriptor '
    'or inference normalization. AT exact fullQG shows preserved raw/LN band variation '
    'but F Linear attenuation; AU sampled four matched training batches finds mid/high '
    'selected weight norms .014-.028 and L2 gradients larger than full task in all four '
    'batches of both controls. Local observations motivate a test, not decay causality '
    'or a guarantee of identity/retrieval gain. Same original teacher42 frozen/fresh '
    'expertheads42/newAdam/extra50/hardP8K8/B64/T2/cap.10/mainCE.25/hinge.6/reg.05/'
    'MFaux.001 actualunit3584+scored512/BN2/partial0/two backwards/5120. Both ordinary '
    'mixed-source and structured controls receive identical optimizer change. Full '
    'official8675train/1715Q8575G/50epochs/normalbenchmarkmAP earliesttie/allsixsamebest '
    'and8frozenstates. Only2026physicalGPU2/3/max2/onepercard; no201NN/missing49/seeds/T-P.')


def build(args,cfg,classes,cameras):
    assert args.dataset=='RGBNT100'
    model=parent.build(args,cfg,classes,cameras)
    model.anchor_record.update(method=REVISION,frequency_value_decay_targets=list(TARGETS),
        frequency_value_decay_old=1e-4,frequency_value_decay_new=0.,frequency_value_parameters_added=0,
        frequency_value_decay_limit='Only two middle/high input Linear weights; low band and every other AS policy retained. Not a new identity mechanism or established cause.')
    return model


def make_optimizer(cfg,model,center):
    optimizer,center_optimizer=parent.parent.make_optimizer(cfg,model,center)
    named=dict(model.named_parameters());target_ids={id(named[n]):n for n in TARGETS}
    assert len(target_ids)==2 and all(named[n].requires_grad and named[n].shape==(64,512) for n in TARGETS)
    changed=[]
    for group in optimizer.param_groups:
        assert len(group['params'])==1
        p=group['params'][0]
        if id(p) in target_ids:
            assert group['weight_decay']==1e-4
            group['weight_decay']=0.;changed.append(target_ids[id(p)])
    assert set(changed)==set(TARGETS)
    model.frequency_value_decay_parameters=[named[n] for n in TARGETS]
    return optimizer,center_optimizer


def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    parameters=model.frequency_value_decay_parameters
    before=[float(p.detach().float().norm()) for p in parameters]
    detail=parent.step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant)
    assert all(p.grad is not None for p in parameters)
    after=[float(p.detach().float().norm()) for p in parameters]
    gradients=[float(p.grad.detach().float().norm()) for p in parameters]
    assert all(math.isfinite(x) for x in before+after+gradients)
    detail['frequency_value_input_decay']=dict(old=1e-4,new=0.,targets=list(TARGETS),parameters_added=0,
        weight_norm_before=before,weight_norm_after=after,unscaled_task_gradient_norm=gradients,
        counterfactual_old_L2_gradient_norm=[1e-4*x for x in before],
        low_band_shared_M_bias_LN_policy_unchanged=True,
        limit='After real parent unscale/Adam step. Counterfactual L2 magnitude only, not an old-Adam replay; no added forward/loss/update.')
    return detail


if __name__=='__main__':
    import run_full_official_frozen_anchor_experiment as runner
    assert runner.make_optimizer is parent.parent.original_make_optimizer
    runner.make_optimizer=make_optimizer
    runner.configuration=configuration
    runner.assert_anchor_unchanged=parent.parent.parent.parent.parent.parent.x.w.assert_anchor_unchanged
    runner.write_json=parent.parent.parent.parent.parent.parent.x.w.s.p.parent.parent.parent.k.parent.g.record
    runner.make_loader=parent.parent.parent.parent.parent.parent.make_loader
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Same AS BN2/partial0/two backwards; only two F middle/high value Linear weight decays removed, no missing design.')
