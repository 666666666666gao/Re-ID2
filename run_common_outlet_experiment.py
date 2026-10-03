"""M1 fresh50 with unchanged V12 learning; pooling is persisted in run arguments."""
import argparse
import sys

from common_outlet_axis import POOLING, build
from run_shared_identity_experiment import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pooling', choices=POOLING, required=True)
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    def model_builder(args, cfg, classes, cameras):
        args.pooling = options.pooling
        return build(args, cfg, classes, cameras)

    main(model_builder,
         method_revision='M1 common_outlet: original_mean versus eligible_mean versus one shared seed query; identical active query residual, V12 experts/FFT/router/losses/sampling; no relation distillation or new cross-set training',
         metric='5632D sqrt(.75) private5120 plus sqrt(.25) common512; seven routed M/I relations use the selected outlet; F remains its existing common increment')
