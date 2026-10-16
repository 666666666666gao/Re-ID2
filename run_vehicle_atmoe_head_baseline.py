"""Full-official vehicle baseline with only the existing ATMoE count changed."""
import argparse
import sys

import run_full_official_experiment as runner
from vehicle_atmoe_head_configuration import configuration, verify_partition


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--atmoe-heads', type=int, choices=(8, 16, 32), required=True)
    selected, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    def configure(args):
        return configuration(args, selected.atmoe_heads)

    def build(args, cfg, classes, cameras):
        model = runner.build(args, cfg, classes, cameras)
        verify_partition(model, selected.atmoe_heads)
        return model

    runner.configuration = configure
    runner.main(builder=build, variants=('demo', 'demo_shared'),
        method_revision=(f'Existing ATMoE partition heads={selected.atmoe_heads}; '
            'fresh full-official identity baseline, not a dual-axis model. '
            'Only MODEL.HEAD changes; original head8 references are historical. '
            'RGBNT100 paper target87.2/98.6; MSVR31050.2/60.8. '
            'No RGBNT201 training or missing-modality evaluation.'))


if __name__ == '__main__':
    main()
