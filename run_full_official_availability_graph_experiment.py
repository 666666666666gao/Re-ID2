"""Fresh full-official trials with one cross-availability public identity objective."""
import os

from availability_graph_training import step
from gpu_thermal_execute import check_limits
from run_full_official_experiment import main
from run_full_official_modality_outlet_experiment import builder as m4_builder
from run_shared_identity_experiment import build as shared_builder


VARIANTS = ('axis_shared', 'frequency_shared', 'twins_shared', 'demo_shared')


def builder(args, cfg, classes, cameras):
    if args.variant == 'demo_shared':
        check_limits(int(os.environ['CUDA_VISIBLE_DEVICES']))
        model = shared_builder(args, cfg, classes, cameras)
    else:
        model = m4_builder(args, cfg, classes, cameras)
    args.graph_weight, args.graph_temperature = .1, .07
    args.graph_reference_cycle = ['R', 'N', 'T', 'RN', 'RT', 'NT', 'RNT']
    model.graph_weight, model.graph_temperature, model.graph_step = .1, .07, 0
    return model


if __name__ == '__main__':
    main(builder=builder, update=step, variants=VARIANTS,
        method_revision='One extra .1 public identity graph objective: .05 full-to-selected-source reference plus .05 partial-to-same-reference. References reuse detached real per-modality raw features from the existing full training pass, shared projection stopped; distinct observations with same train identity are positives. Reference availability cycles seven sets without new RNG/backbone passes. All original full, partial CE/triplet and M4 expert objectives unchanged; no new parameters or inference changes. Applied also to shared-identity DeMo and matched ordinary controls.',
        partial_training='Original one uniform proper subset and stopped full fused gallery .25CE/.5triplet retained. Added public-only identity supervision compares both full and partial queries with a cyclic selected-source public reference. No dev/test identities; no artificial train holdout.')

