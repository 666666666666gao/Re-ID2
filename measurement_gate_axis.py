"""M3a single factor: contribution prediction receives only its regression gradient."""
import os

from torch import nn

from axis_collaboration import ContributionCalibrator
from experiment_data import seed_all
from gpu_thermal_execute import check_limits
from identity_alignment_axis import IdentityAlignmentAxis


class MeasurementOnlyCalibrator(ContributionCalibrator):
    def __init__(self, original):
        nn.Module.__init__(self)
        self.predictor = original.predictor
        self.targets = original.targets

    def forward(self, base, modality, frequency):
        prediction, gates = super().forward(base, modality, frequency)
        return prediction, gates.detach()


class MeasurementGateAxis(IdentityAlignmentAxis):
    def __init__(self, *args):
        super().__init__(*args)
        self.calibrator = MeasurementOnlyCalibrator(self.calibrator)


def build(args, cfg, classes, cameras):
    assert args.pooling == 'original_mean' and args.variant in ('axis_shared', 'frequency_shared', 'twins_shared')
    check_limits(int(os.environ['CUDA_VISIBLE_DEVICES']))
    seed_all(args.seed)
    args.gate_gradient_mode = 'measurement_only'
    return MeasurementGateAxis(classes, cfg, cameras, args.variant, args.seed,
        args.relation_weight, args.alignment_weight, args.alignment_temperature).float().cuda()
