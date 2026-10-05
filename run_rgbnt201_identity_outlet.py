"""Use the established full-official loop, with explicit plain shared-step losses."""
import argparse
import sys

import run_full_official_frozen_anchor_experiment as runner
from original_identity_anchor import assert_anchor_unchanged
from rgbnt201_identity_outlet import build, VARIANTS
from run_experiment import write_json
from run_shared_identity_experiment import step as shared_step


REVISION = ('R201C common fixture: original RGBNT201 E28 private encoder fixed/eval, '
    'M/F/I increments share the original seven identity relation slots, no forcedpublic25 retrieval block, '
    '5120D output equal to original DeMo. Ordinary-frequency and dual-axis use the same outlet, '
    'eligible sources, original FFT/full-partial losses/seed42/B64/P8K8/extra50 budget. '
    'Only PM/PF P(x) versus x+P(x) varies within this fixture; introduction of this whole interface '
    'versus R201B/probe is a coupled setup change. Public head remains auxiliary/context. '
    'No additional loss or new calibrated-gate claim. Full3951/836/836, no holdout, '
    'benchmark-selected best mAP with earliest ties, that fixed best for all49.')


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    assert variant in VARIANTS
    detail = shared_step(model, batch, optimizer, scaler, loss_fn, xent, retained)
    detail.update(freeze_identity_encoder=True, bypass=model.bypass,
        added_losses=0, descriptor_dim=5120, initialization='original E28; identical fresh expert/outlet heads')
    return detail


def record(path, value):
    if 'descriptor_dim' in value:
        value['descriptor_dim'] = 5120
    if 'pretraining' in value:
        value['pretraining'] = 'Full-official original RGBNT201 DeMo50 bestE28 plus additional50/new Adam; original identity encoder fixed'
    write_json(path, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--bypass', type=int, choices=(0, 1), required=True)
    option, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    def selected_builder(args, cfg, classes, cameras):
        args.bypass = option.bypass
        return build(args, cfg, classes, cameras)
    runner.assert_anchor_unchanged = assert_anchor_unchanged
    runner.write_json = record
    runner.main(builder=selected_builder, update=step, variants=VARIANTS, method_revision=REVISION)
