"""Keep the AG framework; adapt only the last existing visual block after warmup."""
from types import MethodType
import torch

import run_r201ag_RGB_coordinate_metric as parent
from original_identity_anchor import IDENTITY_MODULES

HEAD_EPOCHS = 5
VISUAL_LR_FACTOR = .1
PREFIX = 'BACKBONE.base.transformer.resblocks.11.'
CONTROLS = ('frequency_shared', 'axis_shared')


def tail(model):
    blocks = model.BACKBONE.base.transformer.resblocks
    assert len(blocks) == 12
    return blocks[11]


def build(args, cfg, classes, cameras):
    assert args.dataset == 'RGBNT100' and args.variant in CONTROLS
    assert args.seed == 42 and args.freeze_identity_encoder == 1
    assert not cfg.MODEL.FROZEN and not cfg.MODEL.PROMPT and not cfg.MODEL.ADAPTER
    model = parent.build(args, cfg, classes, cameras)
    original_train = model.train

    def train(self, mode=True):
        original_train(mode)
        # Learning weights does not require changing any original core mode/BN.
        for name in IDENTITY_MODULES:
            getattr(self, name).eval()
        return self

    model.train = MethodType(train, model)
    model.freeze_identity_encoder = False
    model.stage_epoch = 0
    model.native_attempt = 0
    model.adaptation_phase = 'outlets'
    model.tail_parameter_count = sum(p.numel() for p in tail(model).parameters())
    assert model.tail_parameter_count == 7087872
    model.anchor_record.update(
        freeze_identity_encoder=False,
        retained_identity_scope='All original five modules except BACKBONE.base.transformer.resblocks.11 parameters',
        staged_adaptation=dict(outlet_epochs=5, visual_epochs=45,
            adapted_prefix=PREFIX, visual_LR_factor=.1,
            nominal_visual_LR=5e-6, native_constructor_warmup_effective_LR=3.5e-6,
            scheduled_visual_LR='0.1 of current scheduled group LR, including existing warmup/cosine noise; not a fixed maximum',
            parameters_added=0, registered_visual_parameters=model.tail_parameter_count,
            original_eval_modes_and_all_buffers_fixed=True,
            original_other11_visual_blocks_and_reducers_HDM_ATMoE_fixed=True,
            single_optimizer_and_scheduler=True, visual_groups_registered_before_outlet_phase=True,
            native='One actual successful outlet update, then two last-visual-block updates; not five completed warmup epochs',
            limit='Currentbase00 can change after epoch5. Cap.10 bounds correction from the current base, not drift from the original teacher.'))
    model.anchor_record['final_unit_descriptor_bound']['inactive_output'] = (
        'Exact current candidate bytes; 00 is the current identity path and may adapt after epoch5')
    model.train()
    assert_retained(model)
    return model


def select_phase(model):
    if model.normal_priority_smoke:
        model.native_attempt += 1
        adapt = model.native_attempt > 1
    else:
        assert 1 <= model.stage_epoch <= 50
        adapt = model.stage_epoch > HEAD_EPOCHS
    tail(model).requires_grad_(adapt)
    model.adaptation_phase = 'visual_tail' if adapt else 'outlets'


@torch.no_grad()
def assert_retained(model):
    current = model.state_dict()
    names = {id(p) for p in tail(model).parameters()}
    for name in IDENTITY_MODULES:
        module = getattr(model, name)
        assert not any(child.training for child in module.modules())
        assert all(p.requires_grad == (id(p) in names and model.adaptation_phase == 'visual_tail')
                   for p in module.parameters())
    buffers = {name for name, _ in model.named_buffers()}
    kept = {key: value for key, value in model.anchor_encoder_state.items()
            if model.adaptation_phase == 'outlets' or not key.startswith(PREFIX) or key in buffers}
    assert all(torch.equal(current[key].detach().cpu(), value) for key, value in kept.items())


def step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant):
    select_phase(model)
    parameters = tuple(tail(model).parameters())
    ids = {id(p) for p in parameters}
    groups = [g for g in optimizer.param_groups if id(g['params'][0]) in ids]
    assert {id(g['params'][0]) for g in groups} == ids
    nominal = [g['lr'] for g in groups]
    if model.adaptation_phase == 'visual_tail':
        for group, rate in zip(groups, nominal):
            group['lr'] = VISUAL_LR_FACTOR * rate
    actual_rates = [g['lr'] for g in groups]
    result = parent.step(model, batch, optimizer, scaler, loss_fn, xent, retained, variant)
    for group, rate in zip(groups, nominal):
        group['lr'] = rate
    if model.adaptation_phase == 'outlets':
        assert all(p.grad is None and p not in optimizer.state for p in parameters)
    if model.normal_priority_smoke:
        assert result['optimizer_updated']
        assert_retained(model)
        gradients = {name: bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
                     for name, p in tail(model).named_parameters()}
        if model.adaptation_phase == 'visual_tail':
            assert all(gradients.values())
        result['last_visual_block_native_gradients'] = gradients
    result.update(adaptation_phase=model.adaptation_phase, stage_epoch=model.stage_epoch,
        freeze_identity_encoder=False, adapted_prefix=PREFIX,
        last_visual_block_registered_parameters=model.tail_parameter_count,
        last_visual_block_effective_LR_factor=VISUAL_LR_FACTOR if model.adaptation_phase == 'visual_tail' else 0.,
        last_visual_block_actual_LR=actual_rates, nominal_group_LR_restored=True,
        original_core_eval_modes_fixed=True, original_core_buffers_retained=True)
    return result
