"""Isolated RGBNT100 joint-logit temperature change; existing models stay unchanged."""
from types import MethodType

import torch

from axis_collaboration import ConditionalRouter
import run_r201ag_RGB_coordinate_metric as parent

TEMPERATURE = 2.


def routing(router, modality, frequency, eligible, interaction, temperature):
    v = router.by_relation(frequency)
    a = router.relation_score(modality.mean(2)).squeeze(-1)
    c = router.band_score(v.mean(1)).squeeze(-1)
    psi = router.psi(torch.cat((modality, v), -1)).squeeze(-1)
    logits = (a[:, :, None] + c[:, None, :] + (psi if interaction else 0)) / temperature
    logits = logits.masked_fill(~eligible[:, :, None], -torch.inf)
    pi = logits.flatten(1).softmax(1).reshape_as(logits)
    mass = pi.sum(2, keepdim=True)
    conditional = pi / torch.where(eligible[:, :, None], mass, torch.ones_like(mass))
    m = (modality * conditional[..., None]).sum(2)
    f = (v * pi[..., None]).sum((1, 2))
    return m, f, pi


def route(self, modality, frequency, eligible, interaction=True):
    if self.capture_temperature_native:
        self.temperature_native_inputs = (modality.detach(), frequency.detach(), eligible.detach(), interaction)
    return routing(self, modality, frequency, eligible, interaction, TEMPERATURE)


def build(args, cfg, classes, cameras):
    model = parent.build(args, cfg, classes, cameras)
    assert args.dataset == 'RGBNT100' and model.freeze_identity_encoder
    assert model.router.route.__func__ is ConditionalRouter.route
    model.router.route = MethodType(route, model.router)
    model.router.capture_temperature_native = False
    model.router.temperature_native_inputs = None
    model.anchor_record.update(routing_temperature=TEMPERATURE, routing_parameters_added=0,
        routing_change='All existing joint a_S+c_b+psi logits divided by2 before legal-relation masking and softmax. All other routing, conditioning, projections, gates, losses and frozen encoder behavior retain AG.')
    return model


@torch.no_grad()
def native_probes(router):
    modality, frequency, eligible, interaction = router.temperature_native_inputs
    with torch.autocast('cuda', enabled=False):
        modality, frequency = modality.float(), frequency.float()
        old = ConditionalRouter.route(router, modality, frequency, eligible, interaction)
        one = routing(router, modality, frequency, eligible, interaction, 1.)
        two = routing(router, modality, frequency, eligible, interaction, TEMPERATURE)
        independent = routing(router, modality, frequency, eligible, False, TEMPERATURE)[2]
        v = router.by_relation(frequency)
        a = router.relation_score(modality.mean(2)).squeeze(-1).masked_fill(~eligible, -torch.inf)
        c = router.band_score(v.mean(1)).squeeze(-1)
        product = (a / TEMPERATURE).softmax(1)[:, :, None] * (c / TEMPERATURE).softmax(1)[:, None, :]
        equal = all(torch.equal(x, y) for x, y in zip(old, one))
        error = float((independent - product).abs().max())
        difference = float((two[2] - one[2]).abs().max())
        assert equal and error < 1e-6 and difference > 0
    return dict(temperature1_exact_parent=equal, no_psi_product_max_error=error,
        temperature2_vs1_max_probability_difference=difference,
        scope='Routing-only FP32 probes using cached native inputs and current parameters; no additional backbone forward, BN calls, random draws or optimizer updates.')
