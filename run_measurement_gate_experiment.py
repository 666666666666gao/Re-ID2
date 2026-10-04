"""M3a fresh50: M2b forward/loss values retained, task-to-predictor gradient cut."""
import argparse
import sys

from measurement_gate_axis import build
import run_shared_identity_experiment as training


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--relation-weight', type=float, choices=(.1,), required=True)
    parser.add_argument('--alignment-weight', type=float, choices=(.1,), required=True)
    parser.add_argument('--alignment-temperature', type=float, choices=(.07,), required=True)
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    original_step = training.step

    def step(*args, **kwargs):
        args[0].alignment_names = tuple(args[1][-1])
        detail = original_step(*args, **kwargs)
        detail.update(args[0].relation_audit)
        detail.update(args[0].alignment_audit)
        detail['gate_gradient_mode'] = 'measurement_only'
        return detail

    def builder(args, cfg, classes, cameras):
        args.pooling = 'original_mean'
        args.relation_weight, args.alignment_weight = options.relation_weight, options.alignment_weight
        args.alignment_temperature = options.alignment_temperature
        return build(args, cfg, classes, cameras)

    training.step = step
    training.main(builder,
        method_revision='M3a only detaches the gates used in fusion; predictor learns unchanged contribution regression, no full/partial retrieval-task gradient. M2b relation0.1/alignment0.1/tau0.07 and every forward/loss value at identical weights retained. No independent control head is added.',
        metric='Unchanged5632D sqrt(.75) private5120 plus sqrt(.25) common512; same predictor parameters, gates, residual scales and routing')
