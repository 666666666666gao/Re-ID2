"""L: unchanged K experts/objective with one uniform P8/K8 sampling policy."""
from collections import Counter
import run_full_official_frozen_anchor_experiment as runner
import run_r201k_relation_local_pi as k
from original_identity_anchor import assert_anchor_unchanged
from rgbnt201_projection_probe import VARIANTS

BASE_CONFIGURATION = runner.configuration
REVISION = (k.REVISION + ' L single training-sampler factor: set NUM_INSTANCE8 on all datasets, '
    'B64/P8K8. RGBNT201 alreadyK8: exact K42 ordinary/axis complete results reused, no new201 '
    'training. MSVR310 K4/P16 becomesK8/P8; RGBNT100 K16/P4 becomesK8/P8. This changes both '
    'positive multiplicity and negative identities plus sampler length; only paired L arms '
    'have equal batch orders, not L versus oldK. Report actual updates: new1000/6563 perarm '
    'for MSVR/100 atseed42 and all50epochs, not old705/6357. Same originalDeMo42 teacher50, '
    'fresh experts/heads/Adam/additional50/lr/AMP/primaryhinge.3/auxsoft/CE/contribution/'
    'partial0-forwardBN2/5120D/parameters and normal mAP-best earliestties. '
    'Normal three-dataset coverage before fixed49; no test identities in training, no '
    'newarchitecture or guaranteed improvement/all3+2/calibration/causal claim.')


def configuration(args):
    cfg = BASE_CONFIGURATION(args)
    cfg.defrost()
    cfg.DATALOADER.NUM_INSTANCE = 8
    cfg.freeze()
    assert cfg.SOLVER.IMS_PER_BATCH == 64 and cfg.SOLVER.MAX_EPOCHS == 50
    return cfg


def build(args, cfg, classes, cameras):
    assert args.dataset in ('MSVR310', 'RGBNT100') and cfg.DATALOADER.NUM_INSTANCE == 8
    model = k.build(args, cfg, classes, cameras)
    model.anchor_record['method'] = REVISION
    return model


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    counts = Counter(batch[1].tolist())
    assert len(counts) == 8 and set(counts.values()) == {8}
    detail = k.step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant)
    detail.update(sampling_identities=8, sampling_instances=8, sampling_batch=64)
    return detail


if __name__ == '__main__':
    runner.configuration = configuration
    runner.assert_anchor_unchanged = assert_anchor_unchanged
    runner.write_json = k.parent.g.record
    runner.main(builder=build, update=step, variants=VARIANTS, method_revision=REVISION,
        partial_training='Unchanged K partial0 forwards/BN2/mining/two backwards; only PK sampling is uniform8.')
