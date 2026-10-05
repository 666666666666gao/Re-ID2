"""Run the same narrow R201C loss and normal full-official loop on each dataset."""
import run_full_official_frozen_anchor_experiment as runner
from identity_coordinate_three_dataset import build, VARIANTS
from original_identity_anchor import assert_anchor_unchanged
from run_rgbnt201_identity_outlet import step as r201c_step
from run_experiment import write_json


def record(path, value):
    if 'descriptor_dim' in value:
        value['descriptor_dim'] = 5120
    if 'pretraining' in value:
        value['pretraining'] = 'Corresponding original DeMo official50 best plus additional50/freshAdam; identity encoder fixed'
    write_json(path, value)


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    detail = r201c_step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant)
    detail['initialization'] = 'corresponding original bestE' + str(model.anchor_record['original_best_epoch']) + '; identical fresh expert/outlet heads'
    return detail


if __name__ == '__main__':
    runner.assert_anchor_unchanged = assert_anchor_unchanged
    runner.write_json = record
    runner.main(builder=build, update=step, variants=VARIANTS,
        method_revision='User order: three datasets normal full-official results first; missing evaluation afterwards. '
            'Same R201C narrow5120 expert/outlet/loss/augmentation across datasets, original50 anchor plus extra50. '
            'No dataset-specific method or checkpoint rule; benchmark maximum mAP/earliest ties. '
            'RGBNT201 normal narrow two controls already completed; remaining MSVR310/RGBNT100 two controls.')
