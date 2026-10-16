"""Existing DeMo with the paper's RGBNT100 B128/K16/50-epoch training batch."""
import argparse
import sys

import run_full_official_experiment as runner
from vehicle_atmoe_head_configuration import configuration as head_configuration, verify_partition


def configuration(args, heads):
    assert args.dataset == 'RGBNT100' and args.variant == 'demo' and heads in (8, 32)
    cfg = head_configuration(args, heads)
    assert cfg.SOLVER.IMS_PER_BATCH == 64 and cfg.DATALOADER.NUM_INSTANCE == 16
    assert cfg.SOLVER.MAX_EPOCHS == 50
    cfg.defrost()
    cfg.SOLVER.IMS_PER_BATCH = 128
    cfg.freeze()
    args.training_batch = 128
    return cfg


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--atmoe-heads', type=int, choices=(8, 32), required=True)
    selected, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    def configure(args):
        return configuration(args, selected.atmoe_heads)

    def build(args, cfg, classes, cameras):
        model = runner.build(args, cfg, classes, cameras)
        verify_partition(model, selected.atmoe_heads)
        return model

    runner.configuration = configure
    runner.main(builder=build, variants=('demo',),
        method_revision=(f'RGBNT100 existing DeMo ATMoE heads={selected.atmoe_heads}; '
            'paper B128/K16/50epochs, public CLIP fresh baseline, seed42. '
            'Relative to the same head B64 candidate only training batch changes. '
            'P8K16 differs from prior frozen-expert P8K8 and changes update count. '
            'All8675 training records/full1715Q8575G; normal benchmark mAP best earliest ties. '
            'Actual attempts/updates/skips disclosed, no equal-update claim or expert gain. '
            'Finite heads8/32 comparison, no new module/loss, RGBNT201 or missing evaluation. '
            'Paper acceptance87.2mAP/98.6Rank-1; all six metrics from one checkpoint.'))


if __name__ == '__main__':
    main()
