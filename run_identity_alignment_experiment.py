"""M2b fresh50: one actual-PF/public cross-observation identity objective."""
import argparse
import sys

from identity_alignment_axis import build
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
        return detail

    def builder(args, cfg, classes, cameras):
        args.pooling = 'original_mean'
        args.relation_weight, args.alignment_weight = options.relation_weight, options.alignment_weight
        args.alignment_temperature = options.alignment_temperature
        return build(args, cfg, classes, cameras)

    training.step = step
    training.main(builder,
        method_revision='M2b single added factor: actual routed PF query to stopped base_common identity gallery, distinct-observation same-training-ID positives; weight0.1 temperature0.07; M2 relation0.1/all other training unchanged',
        metric='Unchanged5632D sqrt(.75) private5120 plus sqrt(.25) common512; no new parameters or gate/amplitude intervention')
