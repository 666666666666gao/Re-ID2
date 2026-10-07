"""Repeat the unchanged K graph/objective on the five predeclared expert seeds."""
import run_full_official_frozen_anchor_experiment as runner
import run_r201g_normal_priority as g
import run_r201k_relation_local_pi as k
from identity_coordinate_k_expert_seeds import build as coordinate_build
from original_identity_anchor import assert_anchor_unchanged

REVISION = (k.REVISION + ' RGBNT201 conditional expert-stage N5: predeclared seeds42/43/44/45/46, '
    'fixed original DeMo42 teacher, existing K42 pair reused without retraining. '
    'Each additional seed has ordinary and axis, unchanged B64/P8K8/additional50, '
    'same-checkpoint normal mAP-best earliest tie/six metrics. Report every seed, '
    'mean/sampleSD/paired delta and separate Best-of5; benchmark-selected, '
    'not five end-to-end pipelines. No graph/loss/route/parameter/dimension factor.')


def build(args, cfg, classes, cameras):
    assert cfg.MODEL.NO_MARGIN and cfg.SOLVER.MARGIN == .3
    assert cfg.MODEL.ID_LOSS_WEIGHT == .25 and cfg.MODEL.TRIPLET_LOSS_WEIGHT == 1
    assert cfg.SOLVER.SEED == args.seed and cfg.DATALOADER.NUM_INSTANCE == 8
    model = coordinate_build(args, cfg, classes, cameras)
    model.normal_priority_smoke = args.mode == 'smoke'
    model.relation_pi_native_audit = {}
    model.anchor_record['method'] = REVISION
    return model


if __name__ == '__main__':
    runner.assert_anchor_unchanged = assert_anchor_unchanged
    runner.write_json = g.record
    runner.main(builder=build, update=k.step, variants=('frequency_shared','axis_shared'),
        method_revision=REVISION,
        partial_training='Unchanged K/I: zero partial objectives, same partial forwards/BN2/mining/two backward calls.')
