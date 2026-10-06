"""Unchanged four-state evaluator, with actual extra learner seeds on the same DeMo42."""
import evaluate_identity_coordinate_missing49_stream as evaluator
from identity_coordinate_three_dataset import build as seed42_build
from identity_coordinate_expert_seeds import build as extra_seed_build


def build(args,cfg,classes,cameras):
    assert args.seed in (42,43,44)
    if args.seed==42:
        return seed42_build(args,cfg,classes,cameras)
    assert args.dataset=='RGBNT201'
    return extra_seed_build(args,cfg,classes,cameras)


if __name__=='__main__':
    evaluator.build=build
    evaluator.main()
