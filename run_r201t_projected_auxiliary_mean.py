"""S: place unchanged auxiliary objectives on the actual11 projected pools."""
import torch
import run_r201p_auxiliary_balance as p
import run_r201r_primary_ce as r
import run_full_official_frozen_anchor_experiment as runner
from original_identity_anchor import assert_anchor_unchanged
from routed_projected_auxiliary_mean import install_projected_auxiliary

PARENTS={'MSVR310':r,'RGBNT100':p}
configuration=p.configuration
VARIANTS=p.VARIANTS
REVISION=('R201T MS-only recovery after actual S ordinary native pool_score gradient failure: '
    'replace weighted projected relation pools with uniform means of actual first11 '
    'PM/PF per512 unit directions, reuse existing512 auxiliary heads/CE/softTriplet. '
    'Freeze exactly modality_expert.pool_score.weight and frequency_expert.pool_score.weight '
    '(1024 parameters), since they now appear only in stopped controlled10/01 targets, '
    'not normal11/eval or any actual loss path. All-trainable native gate is retained '
    'for remaining parameters, matchedordinary/axis total and effective counts agree. '
    'No inferred FP16 diagnosis, no artificial gradients or stale S axis native reuse. '
    'SameMS parentR CE.05/reg0/aux.01/mainunit5120hinge.6/PI/cap.10/FFT/router, '
    'P8K8/seed42/originalteacher50 frozen/freshheads/Adam/extra50/partial0/BN2/two backwards. '
    'No new params/neck/projection/attention calls; first actual11 only. '
    'Full officialtrain/Q/G, benchmarkmAP-best earliestties/fixedbest sixmetrics. '
    'Keep live S RGB100 source/controller/receiver unchanged on2/3; TMS uses0/1 only. '
    'KeepK201best/no201NN/no49/failedseed search. Source-known auxiliary supervision '
    'interface test, not newloss or guaranteed retrievalgain.')


def build(args,cfg,classes,cameras):
    assert args.dataset=='MSVR310'
    model=install_projected_auxiliary(PARENTS[args.dataset].build(args,cfg,classes,cameras))
    model.anchor_record['method']=REVISION
    model.anchor_record['projected_auxiliary_parent_dataset']=args.dataset
    model.anchor_record['parent_main_CE_weight']={'MSVR310':.05,'RGBNT100':.25}[args.dataset]
    model.anchor_record['parent_contribution_regression_weight']=model.contribution_loss_weight
    return model


def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    calls=0;xent_calls=0;direct={};main={}
    def counted_xent(score,target):
        nonlocal xent_calls
        xent_calls+=1
        value=xent(score,target)
        if xent_calls==1:main['main_CE_unweighted']=float(value.detach())
        return value
    def auxiliary(score,feature,target,target_cam):
        nonlocal calls
        calls+=1
        loss=loss_fn(score,feature,target,target_cam)
        # Parent M handles the first/final hinge itself; these are incoming M/F.
        if calls in (1,2):
            name=('M','F')[calls-1]
            assert torch.equal(feature,model.projected_auxiliary_features[calls-1])
            if model.normal_priority_smoke:
                projection=model.modality_projection if name=='M' else model.frequency_projection
                gradient=torch.autograd.grad(model.loss_weights[calls]*loss,projection[-1].weight,retain_graph=True)[0].detach().float()
                assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
                direct[name]=dict(parameter=name+'projection.last.weight',weighted_auxiliary_norm=float(gradient.norm()),weighted_auxiliary_l1=float(gradient.abs().sum()),finite=True)
        return loss
    detail=PARENTS[model.anchor_record['projected_auxiliary_parent_dataset']].step(model,batch,optimizer,scaler,auxiliary,counted_xent,retained,variant)
    assert calls==len(model.loss_weights)-1 and xent_calls==2
    detail['M_F_auxiliary_outlet']='actual_first11_projected_unit_relation_mean512'
    detail['projected_auxiliary_forward_contract']=dict(model.projected_auxiliary_forward_audit)
    detail.update(main,parent_main_CE_weight=model.anchor_record['parent_main_CE_weight'],
        main_CE_effective=model.anchor_record['parent_main_CE_weight']*main['main_CE_unweighted'],
        parent_contribution_regression_weight=model.contribution_loss_weight,xent_counted_calls=xent_calls)
    if model.normal_priority_smoke:
        assert set(direct)=={'M','F'}
        detail['direct_PM_PF_auxiliary_gradient']=direct
    model.projected_auxiliary_features=None
    return detail


if __name__=='__main__':
    runner.configuration=configuration;runner.assert_anchor_unchanged=assert_anchor_unchanged
    runner.write_json=p.parent.parent.parent.k.parent.g.record
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Parent dataset-specific P/R partial0 computation/BN2/two backwards retained; only M/F auxiliary feature outlet redirected to first main11 PM/PF unit means, old auxiliary scorers explicitly frozen.')
