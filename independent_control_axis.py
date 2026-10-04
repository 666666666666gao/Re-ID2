"""M3b: a separate task-trained gate, with regression-only contribution prediction."""
import os

import torch
from torch import nn

from experiment_data import seed_all
from gpu_thermal_execute import check_limits
from measurement_gate_axis import MeasurementGateAxis, MeasurementOnlyCalibrator


class IndependentControlCalibrator(MeasurementOnlyCalibrator):
    def __init__(self, original):
        super().__init__(original)
        context_dim = self.predictor[0].normalized_shape[0]
        self.control = nn.Sequential(nn.LayerNorm(context_dim + 3),
            nn.Linear(context_dim + 3, 16), nn.GELU(), nn.Linear(16, 3))
        nn.init.zeros_(self.control[-1].weight)
        nn.init.zeros_(self.control[-1].bias)

    def forward(self, base, modality, frequency):
        context = torch.cat((base.reshape(base.shape[0], 11, -1).mean(1),
            modality.mean(1), frequency), -1).detach()
        prediction = self.predictor(context)
        control = self.control(torch.cat((context, prediction.detach()), -1))
        return prediction, (20 * prediction.detach() + control).sigmoid()


class IndependentControlAxis(MeasurementGateAxis):
    def __init__(self, *args):
        super().__init__(*args)
        self.calibrator = IndependentControlCalibrator(self.calibrator)


def build(args, cfg, classes, cameras):
    assert args.variant in ('axis_shared', 'frequency_shared', 'twins_shared')
    assert args.gate_gradient_mode in ('measurement_only', 'independent_control')
    assert args.pooling == 'original_mean'
    assert (args.relation_weight, args.alignment_weight, args.alignment_temperature) == (.1, .1, .07)
    check_limits(int(os.environ['CUDA_VISIBLE_DEVICES']))
    seed_all(args.seed)
    model_type = MeasurementGateAxis if args.gate_gradient_mode == 'measurement_only' else IndependentControlAxis
    return model_type(classes, cfg, cameras, args.variant, args.seed,
        args.relation_weight, args.alignment_weight, args.alignment_temperature).float().cuda()
