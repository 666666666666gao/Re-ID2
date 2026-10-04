"""Full official M3a/M3b comparison; no historical fit/dev identity holdout."""
import argparse
import sys

from independent_control_axis import build
from run_full_official_experiment import main
from run_shared_identity_experiment import step as shared_step


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    assert variant == model.variant
    model.alignment_names = tuple(batch[-1])
    detail = shared_step(model, batch, optimizer, scaler, loss_fn, xent, retained)
    detail.update(model.relation_audit)
    detail.update(model.alignment_audit)
    return detail


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gate-gradient-mode', choices=('measurement_only', 'independent_control'), required=True)
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    def builder(args, cfg, classes, cameras):
        args.gate_gradient_mode = options.gate_gradient_mode
        args.pooling = 'original_mean'
        args.relation_weight, args.alignment_weight, args.alignment_temperature = .1, .1, .07
        return build(args, cfg, classes, cameras)

    main(builder=builder, update=step,
        variants=('axis_shared', 'frequency_shared', 'twins_shared'),
        method_revision='Full-official M3a versus M3b: fixed FFT, experts, joint routing, common identity outlet and all M2b losses. M3a regression-only predictor gates; M3b adds only a separate rank16 task-trained control head with zero output initialization, detached evidence and detached prediction inputs. No historical research weights.',
        partial_training='One uniformly sampled proper modality set per batch against stopped full-view gallery; same .25 CE/.5 triplet as full official shared-identity DeMo. No new availability distribution or loss.')
