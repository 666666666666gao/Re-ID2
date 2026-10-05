"""Reuse the full-official M7 training loop; keep running M7/M8 sources immutable."""
import run_full_official_frozen_anchor_experiment as runner
from original_identity_anchor import build, assert_anchor_unchanged, VARIANTS
from run_experiment import write_json
from run_shared_identity_experiment import step as shared_step


PRETRAINING = 'Full-official original RGBNT201 DeMo50 selected best; new shared/fused heads; additional50 with new Adam/scheduler/AMP512'
REVISION = ('R201A isolates original-identity-encoder freezing versus updating, from the same original DeMo bestE28. '
    'Identical existing shared outlet, .75/.25 metric blocks, original full and partial-query/full-gallery losses, '
    'seed42/B64 and additional50/new optimizer. Fixed control freezes BACKBONE/reducers/generalFusion in eval mode, '
    'Base fusion is FP32 in both controls after the actual same-batch gradient probe; backbone/reducers/heads retain AMP. '
    'including buffers; shared projection and classification heads remain trainable. No new expert, loss, gate or '
    'frequency transform. This is a staged reference/outlet diagnosis, not unique dual-axis gain or a total50 comparison. '
    'Entire3951train/836query/836gallery, benchmark-selected highest mAP/earliest ties; all49 use that one best.')


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    assert variant == 'demo_shared'
    detail = shared_step(model, batch, optimizer, scaler, loss_fn, xent, retained)
    detail.update(freeze_identity_encoder=bool(model.freeze_identity_encoder), added_losses=0,
                  initialization='same full-official original DeMo50 selected anchor plus identical fresh shared/fused heads')
    return detail


def record(path, value):
    # The reused M7 loop hardcodes a shared-DeMo lineage. Replace that metadata
    # in this process only; original runtime sources used by M8 stay untouched.
    if 'pretraining' in value:
        value['pretraining'] = PRETRAINING
    write_json(path, value)


if __name__ == '__main__':
    runner.assert_anchor_unchanged = assert_anchor_unchanged
    runner.write_json = record
    runner.main(builder=build, update=step, variants=VARIANTS, method_revision=REVISION)
