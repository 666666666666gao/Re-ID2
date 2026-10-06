"""R201I: change only G's final full-unit primary metric to existing margin0.3 Triplet."""
import torch
import run_full_official_frozen_anchor_experiment as runner
import run_r201g_normal_priority as g
from layers.triplet_loss import TripletLoss
from original_identity_anchor import assert_anchor_unchanged
from rgbnt201_projection_probe import VARIANTS

REVISION=('R201I single primary-metric factor relative to completed G: only the first '
    'final full-unit5120 batch-hard soft Triplet becomes existing margin0.3 hinge Triplet. '
    'FusedCE.25, auxiliary soft Triplets/CE/contribution, partial0 weights and forwards/BN2/'
    'mining/two backward calls, frozen originalDeMo50 anchor, fresh experts/heads/Adam, '
    'seed42/B64/additional50 and existing G per-dataset sampler '
    '(RGBNT201 K8, MSVR310 K4, RGBNT100 K16), all architecture/5120D inference unchanged. '
    'Do not globally change MODEL.NO_MARGIN. Entire official train/query/gallery; '
    'normal highest mAP/earliest tie, benchmark-selected, same fixed best for later49. '
    'Matched ordinary frequency control, then three normal datasets before missing. '
    'Known hinge loss, not new architecture or guaranteed +2/stability/causal claim.')


def build(args,cfg,classes,cameras):
    assert cfg.MODEL.NO_MARGIN and cfg.SOLVER.MARGIN==.3
    model=g.build(args,cfg,classes,cameras)
    model.anchor_record['method']=REVISION
    return model


def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    calls,metric=0,{}
    hinge=TripletLoss(margin=.3)

    def losses(score,feature,target,target_cam):
        nonlocal calls
        calls+=1
        if calls!=1:
            return loss_fn(score,feature,target,target_cam)
        assert feature.shape==(64,5120)
        loss,positive,negative=hinge(feature,target)
        gap=positive-negative+.3
        metric.update(primary_full_metric='unit5120_margin03_batch_hard_triplet',
            primary_metric_coefficient=1.,primary_margin=.3,primary_triplet_loss=float(loss.detach()),
            primary_margin_violations=int((gap.detach()>0).sum()),
            primary_positive_distance_mean=float(positive.detach().mean()),
            primary_negative_distance_mean=float(negative.detach().mean()),
            auxiliary_original_soft_triplet=True)
        if model.normal_priority_smoke:
            assert abs(float(loss.detach())-float(gap.detach().clamp_min(0).mean()))<1e-6
            with torch.autocast('cuda',enabled=False):
                gradient=torch.autograd.grad(loss,feature,retain_graph=True)[0]
            assert torch.isfinite(gradient).all()
            if loss.detach()>0:
                assert gradient.abs().sum()>0
            metric.update(primary_margin_definition_equal=True,
                primary_feature_gradient_finite=True,primary_feature_gradient_l1=float(gradient.abs().sum()))
        return .25*xent(score,target)+loss

    detail=g.step(model,batch,optimizer,scaler,losses,xent,retained,variant)
    assert calls==len(model.loss_weights)
    detail.update(metric)
    return detail


if __name__=='__main__':
    runner.assert_anchor_unchanged=assert_anchor_unchanged
    runner.write_json=g.record
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='G partial objectives remain zero; same forwards, fusedBN2, mining and two backwards. Only primary full unit margin0.3 Triplet changes.')
