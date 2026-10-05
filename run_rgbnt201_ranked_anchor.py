"""R201B: the protected R201A outlet with one deployment-ranking objective."""
import run_full_official_frozen_anchor_experiment as runner
from original_identity_anchor import build, assert_anchor_unchanged, VARIANTS
from run_experiment import write_json
from run_rgbnt201_original_anchor import PRETRAINING
from rgbnt201_ranked_outlet import step as ranked_step, TEMPERATURE, RANK_WEIGHT


REVISION = ('R201B changes only the objective relative to the completed R201A fixed control: '
    'GT-aware smooth AP on the actual normalized 5632D full and partial descriptors, '
    'temperature .05 and weight1 each. Original DeMo50 selectedE28 plus additional50/new Adam. '
    'Original identity encoder fixed in eval mode; same FP32 base fusion, shared outlet, '
    '.75/.25 metric, full ReID and partial CE/triplet, sampling and BN calls. '
    'Same-ID/same-camera junk excluded; queries without a legal cross-camera positive '
    'skip only the ranking term and are counted. Not a new expert or dual-axis novelty claim. '
    'Entire official3951/836/836; benchmark-selected highest mAP/earliest ties; fixed best for all49.')


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    assert variant == 'demo_shared'
    detail = ranked_step(model, batch, optimizer, scaler, loss_fn, xent, retained)
    detail.update(freeze_identity_encoder=bool(model.freeze_identity_encoder), added_losses=2,
                  initialization='same full-official original DeMo50 selected anchor plus identical fresh shared/fused heads')
    return detail


def record(path, value):
    if 'pretraining' in value:
        value['pretraining'] = PRETRAINING
        value['ranking_objective'] = dict(temperature=TEMPERATURE, weight=RANK_WEIGHT,
            descriptor='Actual normalized full/partial fused5632D; stopped full gallery for partial',
            junk='same identity and same camera', positive='same identity and different camera')
    write_json(path, value)


if __name__ == '__main__':
    runner.assert_anchor_unchanged = assert_anchor_unchanged
    runner.write_json = record
    runner.main(builder=build, update=step, variants=VARIANTS, method_revision=REVISION)
