"""CPU value and derivative check of the exact pre-normalization recovery."""
import json
import torch
from torch import nn
from torch.nn import functional as F
from r201d_full_triplet import RawFullTripletAxis


torch.set_num_threads(4)
torch.manual_seed(42)
model = RawFullTripletAxis.__new__(RawFullTripletAxis)
nn.Module.__init__(model)
model.modality_projection = nn.Identity()
model.frequency_projection = nn.Identity()
model.interaction_projection = nn.Linear(1024, 512, bias=False)
model.residual_scale = nn.Parameter(torch.tensor([.1, .05, .05]))
base = torch.randn(8, 5120)
modality = torch.randn(8, 7, 512, requires_grad=True)
frequency = torch.randn(8, 512, requires_grad=True)
relation_frequency = torch.randn(8, 7, 512, requires_grad=True)
gates = torch.rand(8, 3, requires_grad=True)
mass = torch.rand(8, 7).softmax(1)
eligible = torch.ones(8, 7, dtype=torch.bool)
unit = model.fuse(base, modality, frequency, gates, True, True, eligible, mass, relation_frequency)
raw = model.full_metric_feature
anchor = base[:, 1536:].reshape(8, 7, 512).norm(dim=-1, keepdim=True)
delta = model.residual_scale[0] * gates[:, :1, None] * F.normalize(modality, dim=-1) * anchor
delta = delta + model.residual_scale[1] * gates[:, 1:2, None] * F.normalize(relation_frequency, dim=-1) * anchor
interaction = F.normalize(model.interaction_projection(torch.cat((modality.mean(1), frequency), 1)), dim=1)
delta = delta + model.residual_scale[2] * gates[:, 2:, None] * interaction[:, None] * anchor
expected = torch.cat((base[:, :1536], base[:, 1536:] + (delta * (7 * mass[..., None])).flatten(1)), 1)
assert torch.allclose(raw, expected, rtol=2e-6, atol=2e-6)
assert torch.allclose(F.normalize(raw, dim=1), unit, rtol=2e-6, atol=2e-6)
weights = torch.randn_like(expected)
inputs = (modality, relation_frequency, gates, model.residual_scale, model.interaction_projection.weight)
left = torch.autograd.grad((raw * weights).sum(), inputs, retain_graph=True)
right = torch.autograd.grad((expected * weights).sum(), inputs)
assert all(torch.allclose(a, b, rtol=2e-5, atol=2e-5) for a, b in zip(left, right))
print(json.dumps(dict(status='PASS_RAW_FUSION_VALUE_AND_DERIVATIVES',
    raw_value_max_error=float((raw.detach() - expected.detach()).abs().max()),
    derivative_max_error=max(float((a-b).abs().max()) for a,b in zip(left,right)),
    raw_norm_mean=float(raw.detach().norm(dim=1).mean()), optimizer_updates=0, CUDA_calls=0)))
