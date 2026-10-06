"""J: retain I primary hinge; change only standalone M/F auxiliary Triplets."""
import torch

import run_full_official_frozen_anchor_experiment as runner
import run_r201g_normal_priority as g
import run_r201i_primary_margin as i
from layers.triplet_loss import TripletLoss
from original_identity_anchor import assert_anchor_unchanged
from rgbnt201_projection_probe import VARIANTS

REVISION = ('J single factor relative to completed I: standalone independent M/F raw512 '
    'auxiliary batch-hard soft Triplets become existing margin0.3 hinge. Primary unit5120 '
    'hinge0.3 unchanged; same raw M/F features, weights, CE.25, base/common soft losses, '
    'contribution, zero partial targets/forward/BN/mining/two backwards, frozen original '
    'DeMo42, new Adam, B64, additional50, existing per-dataset PK sampling and5120D '
    'inference. Matched ordinary/axis on full official train/query/gallery, no holdout. '
    'Highest normal benchmark mAP/earliest tie, all six metrics same checkpoint. '
    'Known-loss diagnostic, not a new module or proven cause/stability/+2 advantage.')


def build(args, cfg, classes, cameras):
    model = i.build(args, cfg, classes, cameras)
    model.anchor_record['method'] = REVISION
    return model


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    calls, metrics = 0, {}
    hinge = TripletLoss(margin=.3)

    def losses(score, feature, target, target_cam):
        nonlocal calls
        calls += 1
        if calls > 3:
            return loss_fn(score, feature, target, target_cam)
        branch = ('primary', 'auxiliary_M', 'auxiliary_F')[calls - 1]
        assert feature.shape == (64, 5120 if calls == 1 else 512)
        loss, positive, negative = hinge(feature, target)
        gap = positive - negative + .3
        metrics[branch] = dict(margin=.3, loss=float(loss.detach()),
            violations=int((gap.detach() > 0).sum()),
            positive_mean=float(positive.detach().mean()),
            negative_mean=float(negative.detach().mean()),
            feature_norm_mean=float(feature.detach().float().norm(dim=1).mean()))
        if model.normal_priority_smoke:
            assert abs(float(loss.detach()) - float(gap.detach().clamp_min(0).mean())) < 1e-6
            with torch.autocast('cuda', enabled=False):
                gradient = torch.autograd.grad(loss, feature, retain_graph=True)[0]
            assert torch.isfinite(gradient).all()
            if loss.detach() > 0:
                assert gradient.abs().sum() > 0
            metrics[branch].update(definition_equal=True, gradient_finite=True,
                feature_gradient_l1=float(gradient.abs().sum()))
        return .25 * xent(score, target) + loss

    detail = g.step(model, batch, optimizer, scaler, losses, xent, retained, variant)
    assert calls == len(model.loss_weights)
    detail.update(primary_margin=.3, primary_metric_coefficient=1.,
        primary_full_metric='unit5120_margin03_batch_hard_triplet',
        primary_triplet_loss=metrics['primary']['loss'],
        primary_margin_violations=metrics['primary']['violations'],
        primary_positive_distance_mean=metrics['primary']['positive_mean'],
        primary_negative_distance_mean=metrics['primary']['negative_mean'],
        auxiliary_original_soft_triplet=False, auxiliary_M_F_margin=.3,
        remaining_base_common_original_soft=True, metric_calls=calls,
        J_branch_metrics=metrics)
    return detail


if __name__ == '__main__':
    runner.assert_anchor_unchanged = assert_anchor_unchanged
    runner.write_json = g.record
    runner.main(builder=build, update=step, variants=VARIANTS, method_revision=REVISION,
        partial_training='I partial zero objectives and forwards/BN retained; only independent M/F auxiliary metrics change.')
