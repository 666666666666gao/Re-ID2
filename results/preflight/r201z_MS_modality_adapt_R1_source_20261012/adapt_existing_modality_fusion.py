"""Allow gradients only in the existing DeMo HDM/ATMoE, keeping eval modes."""
import torch
from original_identity_anchor import IDENTITY_MODULES

FROZEN_MODULES=tuple(n for n in IDENTITY_MODULES if n!='generalFusion')
ADAPTED_PARAMETERS=9958912

@torch.no_grad()
def assert_retained(model):
    assert model.freeze_identity_encoder is False
    current=model.state_dict()
    for name in FROZEN_MODULES:
        module=getattr(model,name)
        assert not module.training and all(not p.requires_grad for p in module.parameters())
    assert not any(m.training for m in model.generalFusion.modules())
    assert all(p.requires_grad for p in model.generalFusion.parameters())
    buffers={'generalFusion.'+n for n,_ in model.generalFusion.named_buffers()}
    retained={k:v for k,v in model.anchor_encoder_state.items() if k.split('.')[0] in FROZEN_MODULES or k in buffers}
    assert all(torch.equal(current[k].detach().cpu(),v) for k,v in retained.items())

def install(model):
    assert model.freeze_identity_encoder and model.feat_dim==512 and model.head==8
    assert sum(p.numel() for p in model.generalFusion.parameters())==ADAPTED_PARAMETERS
    model.generalFusion.requires_grad_(True)
    model.freeze_identity_encoder=False
    # Existing OriginalProjectionProbe.train() always puts all five original modules in eval mode.
    model.train()
    model.anchor_record.update(freeze_identity_encoder=False,partial_identity_adaptation=dict(
        adapted_module='generalFusion',adapted_parameters=ADAPTED_PARAMETERS,frozen_modules=list(FROZEN_MODULES),
        adapted_module_training=False,BN_and_dropout_modes_unchanged=True,original_generalFusion_buffers_unchanged=True,
        parameters_added=0,loader_stage='Unchanged frozen Y initialization loads and verifies original teacher first; then only generalFusion parameter gradients enabled',
        scope='Original full encoder equality is not claimed; four frozen modules and all generalFusion buffers retained. 00 uses adapting relations, not the original DeMo.'))
    model.anchor_record['final_unit_descriptor_bound'].update(
        inactive_output='Exact current candidate bytes; 00 uses current adapting DeMo relation path',
        base_reference='Current adapting base, not the fixed original teacher; cap.10 does not bound teacher-to-final movement')
    assert_retained(model)
    return model
