"""M2 fresh50: one fixed relation objective, otherwise identical M1 training."""
import argparse
import sys

from frequency_relation_axis import build
import run_shared_identity_experiment as training


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--relation-weight', type=float, choices=(.1,), required=True)
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    original_step = training.step

    def step(*args, **kwargs):
        detail = original_step(*args, **kwargs)
        detail.update(args[0].relation_audit)
        return detail

    def builder(args, cfg, classes, cameras):
        args.pooling, args.relation_weight = 'original_mean', options.relation_weight
        return build(args, cfg, classes, cameras)

    training.step = step
    training.main(builder,
        method_revision='M2 single factor: weight0.1 normalized pair-distance preservation from stopped actual routed F_pre to PF output, full view only; no extra head/parameters; original M1 losses/partial sampling/gates unchanged',
        metric='Unchanged5632D sqrt(.75) private5120 plus sqrt(.25) common512; original_mean M/I outlet, actualPF F increment unchanged')
