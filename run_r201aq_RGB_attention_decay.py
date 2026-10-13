"""AQ: remove decay only from five packed new attention QKV weight parameters."""
import math
import torch
import run_r201al_RGB_route_temperature as parent
from solver.make_optimizer import make_optimizer as original_make_optimizer

configuration=parent.configuration
VARIANTS=parent.VARIANTS
TARGETS=('modality_expert.attention.in_proj_weight',
    'frequency_expert.attention.0.in_proj_weight','frequency_expert.attention.1.in_proj_weight',
    'frequency_expert.attention.2.in_proj_weight','frequency_expert.cross_band.in_proj_weight')
REVISION=('AQ RGBNT100 tests weight_decay1e-4->0 only for the five packed new M/F/crossband '
    'attention in_proj_weight parameters. Each packed parameter contains Q,K AND V; this is '
    'not a Q/K-only intervention. Biases, query/source/band embeddings, output projections, '
    'router/calibrator/all heads and all other optimizer groups keep existing decay/LR. '
    'Same fresh AL seed42/newAdam/expert50/original teacher50 frozen/T2/cap.10/hardP8K8/B64/'
    'mainCE.25/hinge.6/reg.05/MFaux.001/unit3584/scored512/BN2/partial0/two backwards/5120. '
    'Both ordinary mixed-source and structured controls receive identical optimizer policy. '
    'AP full10290 exact-output diagnostics measured uniform M/F/crossband reads; selectedAL '
    'Q/K norms1.7-1.9e-6 versus heterogeneous V. This motivates a decay test, does not prove '
    'decay caused collapse or predicts retrieval gains. No new network/loss/teacher prior/parameter. '
    'Complete8675train/1715Q/8575G,50 epochs, same benchmarkmAP/earliesttie/allsix selectedweight; '
    'only2026GPU2/3/max2/onepercard, no201NN/missing专项/49/seedhunt/T-P.')


def build(args,cfg,classes,cameras):
    model=parent.build(args,cfg,classes,cameras)
    model.anchor_record.update(method=REVISION,attention_decay_targets=list(TARGETS),
        attention_decay_parameters_added=0,attention_decay_old=1e-4,attention_decay_new=0.,
        attention_decay_limit='Packed QKV includes V. Remaining decay/LR/model/loss and source/band access are unchanged; no QK-exclusive causal claim.')
    return model


def make_optimizer(cfg,model,center):
    assert cfg.SOLVER.OPTIMIZER_NAME=='Adam' and cfg.SOLVER.WEIGHT_DECAY==1e-4
    optimizer,center_optimizer=original_make_optimizer(cfg,model,center)
    named=dict(model.named_parameters());target_ids={id(named[n]):n for n in TARGETS}
    assert len(target_ids)==5 and all(named[n].requires_grad for n in TARGETS)
    changed=[]
    for group in optimizer.param_groups:
        assert len(group['params'])==1
        parameter=group['params'][0]
        if id(parameter) in target_ids:
            assert group['weight_decay']==1e-4 and parameter.shape==(192,64)
            group['weight_decay']=0.;changed.append(target_ids[id(parameter)])
    assert set(changed)==set(TARGETS)
    model.attention_decay_parameters=[named[n] for n in TARGETS]
    return optimizer,center_optimizer


def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    parameters=model.attention_decay_parameters
    before=torch.stack([part.float().norm() for value in parameters for part in value.detach().chunk(3)]).cpu().tolist()
    detail=parent.step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant)
    assert all(value.grad is not None for value in parameters)
    after=torch.stack([part.float().norm() for value in parameters for part in value.detach().chunk(3)]).cpu().tolist()
    gradient=torch.stack([part.float().norm() for value in parameters for part in value.grad.detach().chunk(3)]).cpu().tolist()
    assert all(math.isfinite(v) for v in before+after+gradient)
    detail['attention_projection_decay']=dict(old=1e-4,new=0.,packed_includes_V=True,targets=list(TARGETS),
        weight_norm_before=before,weight_norm_after=after,unscaled_task_gradient_norm=gradient,
        counterfactual_old_L2_gradient_norm=[1e-4*v for v in before],
        ordering='TARGETS order, Q/K/V per target. Gradients after actual unscale and parent optimizer step; no added forward, loss or optimizer step. Counterfactual old L2 is magnitude only, not a replay.')
    if model.normal_priority_smoke:
        detail['attention_projection_decay_native_nonzero_QK_task_gradients']=sum(gradient[i]>0 for i in range(15) if i%3!=2)
    return detail


if __name__=='__main__':
    runner=parent.parent.parent.parent.x.w.runner
    assert runner.make_optimizer is original_make_optimizer
    runner.make_optimizer=make_optimizer
    runner.configuration=configuration
    runner.assert_anchor_unchanged=parent.parent.parent.parent.x.w.assert_anchor_unchanged
    runner.write_json=parent.parent.parent.parent.x.w.s.p.parent.parent.parent.k.parent.g.record
    runner.make_loader=parent.parent.parent.parent.make_loader
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Same AL BN2/partial0/two backwards. Only five packed new attention weight optimizer groups lose decay; no missing training design.')
