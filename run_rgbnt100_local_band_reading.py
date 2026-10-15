"""Full-official existing training loop with one F-only internal reading factor."""
import argparse
import sys

from rgbnt100_expert_seed_recipe import configure_runner
from rgbnt100_local_band_recipe import build_local, revision


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--local-parent-recipe', choices=('AV', 'BC'), required=True)
    parser.add_argument('--local-reading-grid', type=int, choices=(1, 2), required=True)
    selected, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    key, grid = selected.local_parent_recipe, selected.local_reading_grid
    runner, parent = configure_runner(key)

    def build(args, cfg, classes, cameras):
        return build_local(key, grid, args, cfg, classes, cameras)

    runner.main(builder=build, update=parent.step,
        variants=('frequency_shared', 'axis_shared'), method_revision=revision(key, grid),
        partial_training='Same selected parent BN2/partial0/mining/two backwards; no missing-specialist training.')


if __name__ == '__main__':
    main()
