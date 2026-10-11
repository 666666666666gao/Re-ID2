"""S: place unchanged auxiliary objectives on the actual11 projected pools."""
import torch
import run_r201p_auxiliary_balance as p
import run_r201r_primary_ce as r
import run_full_official_frozen_anchor_experiment as runner
from original_identity_anchor import assert_anchor_unchanged
from routed_projected_auxiliary_coordinate_metric import install_projected_auxiliary

PARENTS={'MSVR310':r,'RGBNT100':p}
configuration=p.configuration
VARIANTS=p.VARIANTS
REVISION=('R201U single M/F auxiliary metric-feature factor relative to complete T: '
    'auxiliary CE scores remain actual first11 PM/PF unit-direction relation mean512 '
    'through the same original neck/classifier, but existing auxiliary softTriplet '
    'receives those first11 per512 unit directions flattened and whole-normalized '
    '3584D, matching the relation-coordinate evidence retained in deployment. '
    'No new loss, head, projection, attention, backbone or parameter; two additional '
    'auxiliary feature normalizations only. Keep T obsolete scorer freeze1024, '
    'same total100821638/trainable3122822, M/Faux.01/mainCE.05/reg0/hinge.6/cap.10, '
    'PI/FFT/router/teacher42 frozen/freshhead/Adam/P8K8/extra50/partial0/BN2/two backwards. '
    'New matchedordinary/axis native3 each required before formal50; no reused T native. '
    'Fullofficialtrain/Q/G normalmAP-best earliestties fixedsixmetrics/curves. '
    'Mean512 versus unit3584 current fullGT outlet gap motivates an interface test, '
    'not a proven failure cause or guaranteed gain. PreserveK201/live S RGB2/3/no49/seedsearch.')


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
        metric_feature=feature
        if calls in (1,2):
            assert torch.equal(feature,model.projected_auxiliary_features[calls-1])
            metric_feature=model.projected_auxiliary_metric_features[calls-1]
            assert metric_feature.shape==(len(target),3584) and metric_feature.requires_grad
            assert torch.isfinite(metric_feature).all() and (metric_feature.norm(dim=1)-1).abs().max()<1e-6
        loss=loss_fn(score,metric_feature,target,target_cam)
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
    detail['M_F_auxiliary_outlet']='actual_first11_projected_unit_relation_mean512_CE_unit3584_Triplet'
    detail['M_F_auxiliary_metric_outlet']='actual_first11_unit_relation_coordinates3584'
    detail['projected_auxiliary_forward_contract']=dict(model.projected_auxiliary_forward_audit)
    detail.update(main,parent_main_CE_weight=model.anchor_record['parent_main_CE_weight'],
        main_CE_effective=model.anchor_record['parent_main_CE_weight']*main['main_CE_unweighted'],
        parent_contribution_regression_weight=model.contribution_loss_weight,xent_counted_calls=xent_calls)
    if model.normal_priority_smoke:
        assert set(direct)=={'M','F'}
        detail['direct_PM_PF_auxiliary_gradient']=direct
    model.projected_auxiliary_features=None
    model.projected_auxiliary_metric_features=None
    return detail


if __name__=='__main__':
    runner.configuration=configuration;runner.assert_anchor_unchanged=assert_anchor_unchanged
    runner.write_json=p.parent.parent.parent.k.parent.g.record
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Parent dataset-specific P/R partial0 computation/BN2/two backwards retained; Only incoming M/F auxiliary Triplet feature becomes actualfirst11 unit3584; CE mean512/Tcallcounts/obsolete1024freeze unchanged.')
