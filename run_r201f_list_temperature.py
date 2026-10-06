"""R201F: change only the existing Smooth-AP temperature.01 to.05."""
import torch

import run_full_official_frozen_anchor_experiment as runner
from identity_coordinate_three_dataset import build as coordinate_build
from list_retrieval_objective import smooth_ap
from original_identity_anchor import assert_anchor_unchanged
from rgbnt201_projection_probe import VARIANTS
from run_rgbnt201_identity_outlet import record
from run_shared_identity_experiment import step as shared_step

REVISION = ('R201F list objective only relative to R201C narrow/unit: same original RGBNT201 E28 frozen '
    'identity encoder, fresh experts/heads, source-aware axes, route, gate, projections, contribution '
    'targets, unit5120 inference, full/partial CE and auxiliary triplets. Relative to completed R201E.01 ONLY its temperature changes to.05. The first full-fused '
    'soft Triplet term is replaced by label-defined Smooth-AP, temperature.05, metric coefficient1; '
    'all other loss weights unchanged. Every B64 training observation still participates. List '
    'positives exclude the query itself, use training identity labels, and are camera-agnostic; '
    'official installed-GT evaluation retains its camera exclusions. No new parameters or claim '
    'that Smooth-AP is novel. Entire3951/836/836, no holdout, seed42/P8K8/B64, original50 plus new50 '
    'and fresh Adam. Earliest full official mAP-best selected; not an untouched test.')


def build(args, cfg, classes, cameras):
    assert args.dataset == 'RGBNT201'
    assert cfg.MODEL.ID_LOSS_WEIGHT == .25 and cfg.MODEL.TRIPLET_LOSS_WEIGHT == 1
    assert cfg.MODEL.IF_LABELSMOOTH == 'on'
    args.primary_full_metric = 'label_masked_smooth_AP_unit5120'
    model = coordinate_build(args, cfg, classes, cameras)
    model.list_smoke = args.mode == 'smoke'
    model.anchor_record['method'] = 'R201C narrow encoder/model unchanged; primary full-view Smooth-AP temperature.05 only relative to completed R201E.01'
    return model


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    assert variant in VARIANTS and len(batch[-1]) == len(set(batch[-1])) == 64
    calls, metric = 0, {}

    def losses(score, feature, target, target_cam):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert isinstance(feature, torch.Tensor) and feature.shape == (64, 5120)
            ap_loss = smooth_ap(feature, target, temperature=.05)
            metric['smooth_ap_loss'] = float(ap_loss.detach())
            if model.list_smoke:
                gradient_amp = torch.autograd.grad(ap_loss, feature, retain_graph=True)[0]
                with torch.autocast('cuda', enabled=False):
                    gradient = torch.autograd.grad(ap_loss, feature, retain_graph=True)[0]
                assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
                metric['AP_feature_gradient_l1'] = float(gradient.detach().abs().sum())
                metric['AP_feature_gradient_AMP_finite'] = bool(torch.isfinite(gradient_amp).all())
                metric['AP_feature_gradient_AMP_l1'] = float(gradient_amp.detach().abs().sum())
            return .25 * xent(score, target) + ap_loss
        return loss_fn(score, feature, target, target_cam)

    detail = shared_step(model, batch, optimizer, scaler, losses, xent, retained)
    assert calls == len(model.loss_weights)
    detail.update(metric, primary_full_metric='label_masked_smooth_AP_unit5120',
                  temperature=.05, metric_coefficient=1., added_losses=0,
                  descriptor_dim=5120, freeze_identity_encoder=True)
    return detail


if __name__ == '__main__':
    runner.assert_anchor_unchanged = assert_anchor_unchanged
    runner.write_json = record
    runner.main(builder=build, update=step, variants=VARIANTS, method_revision=REVISION)
