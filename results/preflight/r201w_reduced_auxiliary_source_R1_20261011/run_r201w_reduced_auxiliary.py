"""W: reduce only M/F auxiliary identity weights on completed T MS/S RGB recipes."""
import run_r201t_projected_auxiliary_mean as t
import run_r201s_projected_auxiliary as s
import run_full_official_frozen_anchor_experiment as runner
from original_identity_anchor import assert_anchor_unchanged

PARENTS={'MSVR310':t,'RGBNT100':s}
PREVIOUS={'MSVR310':.01,'RGBNT100':.05}
COEFFICIENT=.001
configuration=s.configuration
VARIANTS=s.VARIANTS
REVISION=(
    'R201W single M/F auxiliary-identity-weight factor relative to completed T MS/S RGB: '
    'both M/F CE+softTriplet weights become .001 from MS .01/RGB .05. '
    'Keep exact T uniform projected-unit mean512/obsolete pool_score freeze1024 forMS '
    'and S existing-score projected-unit pool512 forRGB; no new outlet/projection '
    'or normal/missing inference policy. Same parent mainCE MS.05/RGB.25, '
    'contribution regression MS0/RGB.05, primary unit5120 hinge.6, PI relation-local, '
    'bound.10, FFT/experts/router/gates/frozen originalteacher42/freshheads42, '
    'P8K8/full officialtrain/Q/G/newAdam/additional50/partial0/BN2/two backwards. '
    'Closed native one-parameter early mainhinge-versus-weightedaux ratios motivate '
    'the test but omit mainCE/allparameters/fulltraining and do not prove causality. '
    'Ordinary/axis use identical dataset recipes/budget/selection and new native3 each. '
    'Normal benchmark mAP-best earliestties, allsix metrics samecheckpoint; '
    'no artificial split/201NN/new49/missing selection/failedseedsearch. '
    'Preserve K201best and all completed negative evidence; no guaranteedgain.')


def build(args,cfg,classes,cameras):
    model=PARENTS[args.dataset].build(args,cfg,classes,cameras)
    assert model.loss_weights[1:3]==[PREVIOUS[args.dataset]]*2
    model.loss_weights=list(model.loss_weights)
    model.loss_weights[1:3]=[COEFFICIENT]*2
    model.anchor_record.update(method=REVISION,
        M_F_auxiliary_weight_previous=PREVIOUS[args.dataset],
        M_F_auxiliary_weight=COEFFICIENT,
        reduced_auxiliary_parent={'MSVR310':'T','RGBNT100':'S'}[args.dataset])
    return model


def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    dataset=model.anchor_record['projected_auxiliary_parent_dataset']
    assert model.loss_weights[1:3]==[COEFFICIENT]*2
    detail=PARENTS[dataset].step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant)
    detail.update(M_F_auxiliary_weight_previous=PREVIOUS[dataset],
        M_F_auxiliary_weight=COEFFICIENT,
        reduced_auxiliary_parent=model.anchor_record['reduced_auxiliary_parent'])
    return detail


if __name__=='__main__':
    runner.configuration=configuration;runner.assert_anchor_unchanged=assert_anchor_unchanged
    runner.write_json=s.p.parent.parent.parent.k.parent.g.record
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Unchanged TMS/SRGB partial0 forwards/BN2/two backwards; only M/F auxiliary weights reduced to .001.')
