"""Bind the unchanged full49 protocol to the actual relation-local K forward."""
import evaluate_identity_coordinate_missing49_stream as evaluator
from relation_local_identity_outlet import RelationLocalPIAxis
from run_r201k_relation_local_pi import REVISION, build as relation_local_build


def build(args, cfg, classes, cameras):
    assert args.seed == 42
    model = relation_local_build(args, cfg, classes, cameras)
    assert type(model) is RelationLocalPIAxis
    assert model.anchor_record['method'] == REVISION
    return model


if __name__ == '__main__':
    evaluator.build = build
    evaluator.main()
