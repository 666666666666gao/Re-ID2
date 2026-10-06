"""One training-interface factor; original50 plus matched additional50."""
import run_full_official_frozen_anchor_experiment as runner
from original_identity_anchor import assert_anchor_unchanged
from r201d_full_triplet import build
from rgbnt201_projection_probe import VARIANTS
from run_rgbnt201_identity_outlet import record
from run_shared_identity_experiment import step as shared_step


REVISION = ('R201D full-triplet interface only: keep R201C narrow experts, unit fused CE, '
    'unit partial query/full-gallery objective, contribution targets, all losses/weights, '
    'frozen original identity encoder, routing, sampling and normalized5120 retrieval. '
    'Only the full fused feature consumed by the existing unnormalized soft-triplet '
    'is restored to its pre-normalization coordinates. No new parameters or loss. '
    'RGBNT201 entire3951/836/836, originalbestE28 plus fresh50/newAdam/seed42/B64/P8K8, '
    'paired ordinary-frequency/dual-axis. Benchmark-selected earliest mAP best, not untouched test. '
    'Training loss divergence is observed, but this interface is not proven to cause it.')


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    assert variant in VARIANTS
    detail = shared_step(model, batch, optimizer, scaler, loss_fn, xent, retained)
    detail.update(full_metric=model.full_metric_audit, added_losses=0, descriptor_dim=5120,
                  freeze_identity_encoder=True, full_triplet_feature='raw_fused_only')
    assert detail['full_metric']['unit_norm_max_error'] < 1e-5
    assert detail['full_metric']['normalized_raw_max_error'] < 1e-5
    return detail


if __name__ == '__main__':
    runner.assert_anchor_unchanged = assert_anchor_unchanged
    runner.write_json = record
    runner.main(builder=build, update=step, variants=VARIANTS, method_revision=REVISION)
