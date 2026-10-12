"""AG: test RGB actual relation-coordinate metric supervision, on the exact AD hard recipe."""
import torch
import run_r201ad_RGB_hard as parent
import run_r201s_projected_auxiliary as shared_auxiliary
from routed_projected_auxiliary_rgb_coordinate_metric import install_projected_auxiliary

# The existing AD builder calls S.build inside this isolated experiment process.
# Replace its installer, keeping AD's FP32 scorers and hard sampler untouched.
shared_auxiliary.install_projected_auxiliary=install_projected_auxiliary
configuration=parent.configuration
VARIANTS=parent.VARIANTS
REVISION=('AG RGBNT100 single auxiliary metric-feature factor versus completed AD hard: '
    'same scored512 actual first11 M/F CE outlets and FP32 scorers, but auxiliary softTriplet '
    'uses the same first11 per512 unit directions flattened and whole-normalized3584. '
    'No parameter/head/RNG/attention/neck/normal inference change. Preserve original teacher42 frozen, '
    'freshheads42/newAdam/additional50/full official8675train1715Q8575G, P8K8/same hard schedule, '
    'mainCE.25/reg.05/MFaux.001/hinge.6/cap.10/BN2/partial0/two backwards/5120. '
    'Both ordinary/axis require fresh native3 before formal50; selected best normal benchmarkmAP '
    'earliest ties/all6sameweight and curves. AF oracle branch coverage motivates a controlled '
    'interface test, not a proved cause or gain. OldU MS unit metric failed and is not rerun. '
    'OnlyGPU23/no201NN/missing专项/49/seedhunt; preserve K201 and strongMS5632 baseline.')


def build(args,cfg,classes,cameras):
    model=parent.build(args,cfg,classes,cameras)
    model.anchor_record.update(method=REVISION,auxiliary_metric_feature='actual_first11_per512_unit_relation_coordinates3584',
        auxiliary_CE_feature='unchanged_existing_score_FP32_pool512',auxiliary_metric_parameters_added=0)
    return model


def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    calls=0
    def coordinate_loss(score,feature,target,target_cam):
        nonlocal calls
        calls+=1
        metric_feature=feature
        if calls in (1,2):
            assert torch.equal(feature,model.projected_auxiliary_features[calls-1])
            metric_feature=model.projected_auxiliary_metric_features[calls-1]
            assert metric_feature.shape==(len(target),3584) and metric_feature.requires_grad
            assert torch.isfinite(metric_feature).all() and (metric_feature.norm(dim=1)-1).abs().max()<1e-6
        return loss_fn(score,metric_feature,target,target_cam)
    detail=parent.step(model,batch,optimizer,scaler,coordinate_loss,xent,retained,variant)
    assert calls==len(model.loss_weights)-1
    detail.update(M_F_auxiliary_Triplet_outlet='actual_first11_per512_unit_relation_coordinates3584',
        M_F_auxiliary_CE_outlet='unchanged_existing_score_FP32_pool512',auxiliary_metric_feature_dimension=3584,
        auxiliary_CE_feature_dimension=512,auxiliary_coordinate_loss_calls=calls)
    model.projected_auxiliary_metric_features=None
    return detail


if __name__=='__main__':
    runner=parent.parent.x.w.runner
    runner.configuration=configuration
    runner.assert_anchor_unchanged=parent.parent.x.w.assert_anchor_unchanged
    runner.write_json=parent.parent.x.w.s.p.parent.parent.parent.k.parent.g.record
    runner.make_loader=parent.parent.make_loader
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='AD hard M/FscorerFP32/BN2/partial0/two backwards unchanged; only incoming M/F metric feature becomes actual first11 unit3584 while scored512 CE is retained.')
