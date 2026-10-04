"""Fresh full-official M4 parents plus direct shared-head public identity CE."""
import os

from gpu_thermal_execute import check_limits
from public_outlet_identity_training import step
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
    args.public_identity_weight = .1
    model.public_identity_weight = .1
    return model


if __name__ == '__main__':
    main(builder=builder, update=step, variants=VARIANTS,
        method_revision='One added .1 direct public identity CE factor versus M4 parents: .05 on full and .05 on the existing uniformly sampled partial view, using the actual deployed fused public512 and the same existing shared BN/classifier. No graph-reference objective. All original full/partial/expert objectives retained; no new parameters, backbone passes, source sampling or inference changes. Shared BN receives two explicitly additional training calls. Apply identically to shared-identity DeMo and matched ordinary controls.',
        partial_training='Original one uniform proper subset and stopped full fused gallery .25CE/.5triplet retained. Added label-smoothed identity CE directly supervises the actual partial public outlet through the same existing head as the full public outlet; only official training identities, no artificial holdout.')
