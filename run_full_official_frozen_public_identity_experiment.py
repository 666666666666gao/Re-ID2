"""M8: the existing public-outlet CE under M7's unchanged frozen identity anchor."""
from frozen_identity_anchor_axis import build as anchor_build
from public_outlet_identity_training import step as public_update
from run_full_official_frozen_anchor_experiment import main as anchor_main


VARIANTS = ('axis_shared', 'frequency_shared', 'twins_shared')
PUBLIC_IDENTITY_WEIGHT = .1


def build(args, cfg, classes, cameras):
    assert args.variant in VARIANTS and args.freeze_identity_encoder == 1
    model = anchor_build(args, cfg, classes, cameras)
    args.public_identity_weight = PUBLIC_IDENTITY_WEIGHT
    model.public_identity_weight = PUBLIC_IDENTITY_WEIGHT
    return model


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    detail = public_update(model, batch, optimizer, scaler, loss_fn, xent, retained, variant)
    detail.update(freeze_identity_encoder=True, added_losses=1,
                  initialization='same full-official shared-DeMo50 selected anchor')
    return detail


if __name__ == '__main__':
    anchor_main(builder=build, update=step, variants=VARIANTS,
        method_revision='M8 changes only M7 fixed-expert training: reuse the existing M6 public-outlet identity CE, total weight0.1 split0.05 full/0.05 partial, on the actual fused5632D descriptor slice5120:. Same existing shared_neck/classifier, two additional BN/head calls and no extra backbone pass or parameters. All six identity encoder modules remain frozen with exact weights/buffers/modes checked. Original M4 losses, sampling, CLIP anchor, optimizer/AMP512, descriptor, inference and checkpoint selection are unchanged. This does not claim the earlier M6 loss was untried or successful; only its combination with a fixed identity anchor is new. Same50 anchor plus additional50/new optimizer; no untouched final test or three-dataset success.')
