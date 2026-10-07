"""Fixed Best-of-five ordinary winner45; unchanged K graph and evaluator."""
import evaluate_identity_coordinate_missing49_stream as evaluator
from relation_local_identity_outlet import RelationLocalPIAxis
from run_r201k_expert_N5 import REVISION, build as expert_build


def build(args, cfg, classes, cameras):
    assert args.dataset == 'RGBNT201' and args.variant == 'frequency_shared'
    assert args.seed == 45
    model = expert_build(args, cfg, classes, cameras)
    assert type(model) is RelationLocalPIAxis
    assert model.anchor_record['method'] == REVISION
    assert model.anchor_record['anchor_seed'] == 42 and model.anchor_record['expert_seed'] == 45
    return model


if __name__ == '__main__':
    evaluator.build = build
    evaluator.main()
