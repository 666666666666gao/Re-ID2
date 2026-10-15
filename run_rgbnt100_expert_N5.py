"""Run one admitted expert seed with the separately locked existing AV or BC policy."""
import argparse
import sys

from rgbnt100_expert_seed_recipe import build_selected, configure_runner, revision


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--expert-recipe',choices=('AV','BC'),required=True)
    selected,remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0],*remaining]
    key = selected.expert_recipe
    runner,parent = configure_runner(key)

    def build(args,cfg,classes,cameras):
        return build_selected(key,args,cfg,classes,cameras)

    runner.main(builder=build,update=parent.step,variants=('frequency_shared','axis_shared'),
        method_revision=revision(key),
        partial_training='Selected existing AV/BC partial0, identical partial forward/BN2/mining/two backwards. No missing-specialist training.')


if __name__ == '__main__':
    main()
