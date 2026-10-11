"""S: place unchanged auxiliary objectives on the actual11 projected pools."""
import torch
import run_r201p_auxiliary_balance as p
import run_r201r_primary_ce as r
import run_full_official_frozen_anchor_experiment as runner
from original_identity_anchor import assert_anchor_unchanged
from routed_projected_auxiliary import install_projected_auxiliary

PARENTS={'MSVR310':r,'RGBNT100':p}
configuration=p.configuration
VARIANTS=p.VARIANTS
REVISION=('R201S single auxiliary-outlet factor: independent raw M/F pools512 become '
    'existing-score weighted pools of actual first normal11 routed PM/PF per-relation unit512 directions. '
    'Existing auxiliary classifier/neck and CE/softTriplet/weights reused without '
    'additional neck/projection/attention calls or parameters. Existing pool_score '
    'now scores seven actual projected relations instead of raw band/source-band '
    'evidence, keeping these parameters trainable. Controlled first pools remain '
    'unchanged, not later10/01 or partial projections. MS R mainCE.05/reg0/MFaux.01; '
    'RGB100 P mainCE.25/reg.05/MFaux.05; both keep mainunit5120hinge.6, '
    'relation-localPI/wholeunitcap.10/Fourier/experts/router/gates, '
    'P8K8/seed42/originalteacher50 frozen/freshheads/Adam/extra50, '
    'partial0forward/BN2/two backwards and normalbenchmarkmAP-best earliesttie. '
    'Auxiliary values now use FP32 unit directions in the same512 head dimension; '
    'normal eval graph and5120 output unchanged. Native observes direct PM/PF '
    'auxiliary gradients, unchanged callcounts and actual3updates beforeformal. '
    'Full officialtrain/Q/G, matchedordinary/axis; bothweaknormal beforepaper-six. '
    'K201best preserved/no201NN/no49/failedseed search. Known supervision principle '
    'tested previously at different V6/V8 interfaces, not new novelty or guaranteedgain.')


def build(args,cfg,classes,cameras):
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
    detail['M_F_auxiliary_outlet']='actual_first11_projected_unit_relation_score_pool512'
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
        partial_training='Parent dataset-specific P/R partial0 computation/BN2/two backwards retained; only M/F auxiliary feature outlet redirected to first main11 PM/PF units.')
