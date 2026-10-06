"""Repeat the reviewed I objective with distinct expert seeds and one fixed DeMo42 anchor."""
import run_full_official_frozen_anchor_experiment as runner
import run_r201g_normal_priority as g
import run_r201i_primary_margin as i
from identity_coordinate_expert_seeds import build as coordinate_build
from original_identity_anchor import assert_anchor_unchanged

REVISION = ('Unchanged R201I final unit5120 primary hinge margin.3 and all G auxiliary '
    'soft/CE/contribution, partial0 forwards/BN2/mining/backwards. RGBNT201 only: '
    'expert seeds42/43/44 with fixed original DeMo seed42 identity anchor. Full official '
    'train/query/gallery, B64 P8K8, newAdam/additional50, mAP-best earliest tie, '
    'same-checkpoint six metrics, benchmark selected. Conditional expert-stage repeats, '
    'not independent whole-pipeline seeds. Both ordinary mixed-source control and axis '
    'use each seed; seed42 completed I pair is reused, no seed42 retraining. '
    'Report mean/sampleSD, paired differences and separate Best-of3; no result splicing.')


def build(args, cfg, classes, cameras):
    assert cfg.MODEL.NO_MARGIN and cfg.SOLVER.MARGIN == .3
    assert cfg.MODEL.ID_LOSS_WEIGHT == .25 and cfg.MODEL.TRIPLET_LOSS_WEIGHT == 1
    assert cfg.SOLVER.SEED == args.seed and cfg.DATALOADER.NUM_INSTANCE == 8
    model = coordinate_build(args, cfg, classes, cameras)
    model.normal_priority_smoke = args.mode == 'smoke'
    model.anchor_record['method'] = REVISION
    return model


if __name__ == '__main__':
    runner.assert_anchor_unchanged = assert_anchor_unchanged
    runner.write_json = g.record
    runner.main(builder=build, update=i.step, variants=('frequency_shared','axis_shared'),
        method_revision=REVISION,
        partial_training='Unchanged I: zero partial objectives; same partial forwards/BN2/mining/two backward calls.')
