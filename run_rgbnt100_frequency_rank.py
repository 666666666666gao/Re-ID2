"""Full-official matched AV42 training for one F internal-width candidate."""
import argparse
import sys

from rgbnt100_expert_seed_recipe import configure_runner
from rgbnt100_frequency_rank import build_rank, make_optimizer, revision


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--frequency-rank', type=int, required=True)
    selected, remaining = parser.parse_known_args()
    assert selected.frequency_rank == 128
    sys.argv = [sys.argv[0], *remaining]
    runner, parent = configure_runner('AV')
    assert runner.make_optimizer is parent.make_optimizer
    runner.make_optimizer = make_optimizer

    def build(args, cfg, classes, cameras):
        return build_rank(selected.frequency_rank, args, cfg, classes, cameras)

    def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
        detail = parent.step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant)
        detail['frequency_hidden_width'] = dict(model.anchor_record['frequency_hidden_width'])
        return detail

    runner.main(builder=build, update=step, variants=('frequency_shared', 'axis_shared'),
        method_revision=revision(selected.frequency_rank),
        partial_training='Unchanged AV BN2/partial0/two backwards; no missing-specialist target.')


if __name__ == '__main__':
    main()
