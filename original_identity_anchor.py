"""RGBNT201: learn the existing shared outlet from the stronger original DeMo."""
import json
import os
from pathlib import Path

import torch

from experiment_data import seed_all
from gpu_thermal_execute import check_limits
from official_training_data import full_records
from shared_identity_axis import SharedIdentityDeMo
import shared_identity_axis as shared


IDENTITY_MODULES = ('BACKBONE', 'rgb_reduce', 'nir_reduce', 'tir_reduce', 'generalFusion')
NEW_MODULES = ('shared_projection', 'shared_neck', 'shared_classifier', 'fused_neck', 'fused_classifier')
VARIANTS = ('demo_shared',)
BASE_FUSION = shared.available_base_fusion


def fp32_base_fusion(module, patches, globals_, available):
    """Warm original HDM query gradients are zero under AMP in the actual probe."""
    with torch.autocast('cuda', enabled=False):
        return BASE_FUSION(module, [value.float() for value in patches],
                           [value.float() for value in globals_], available)


class OriginalAnchoredDeMo(SharedIdentityDeMo):
    def __init__(self, *args, freeze_identity_encoder):
        super().__init__(*args)
        self.freeze_identity_encoder = freeze_identity_encoder
        if freeze_identity_encoder:
            for name in IDENTITY_MODULES:
                getattr(self, name).requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_identity_encoder:
            for name in IDENTITY_MODULES:
                getattr(self, name).eval()
        return self


@torch.no_grad()
def assert_anchor_unchanged(model):
    assert model.freeze_identity_encoder
    for name in IDENTITY_MODULES:
        module = getattr(model, name)
        assert not module.training and all(not p.requires_grad for p in module.parameters())
    current = model.state_dict()
    assert all(torch.equal(current[key].detach().cpu(), value)
               for key, value in model.anchor_encoder_state.items())


def build(args, cfg, classes, cameras):
    assert args.dataset == 'RGBNT201' and args.variant == 'demo_shared'
    assert args.freeze_identity_encoder in (0, 1) and args.seed == 42
    check_limits(int(os.environ['CUDA_VISIBLE_DEVICES']))
    shared.available_base_fusion = fp32_base_fusion
    seed_all(args.seed)
    model = OriginalAnchoredDeMo(classes, cfg, cameras, args.seed,
        freeze_identity_encoder=bool(args.freeze_identity_encoder))
    anchor = Path(args.anchor_run_dir)
    trained = json.loads((anchor / 'result.json').read_text())
    assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50
    assert trained['arguments']['dataset'] == args.dataset and trained['arguments']['variant'] == 'demo'
    assert trained['arguments']['seed'] == args.seed and trained['descriptor_dim'] == 5120
    assert trained['optimizer_steps'] == 2647 and trained['amp_skipped_steps'] == 0
    assert trained['training_heldout_identities'] == 0
    assert trained['classes'] == classes and trained['camera_embeddings'] == cameras
    assert json.loads((anchor.parent / (anchor.name + '_exit.json')).read_text())['exit_code'] == 0
    _, _, _, _, _, manifest = full_records(args.data_root, args.dataset)
    assert json.loads((anchor / 'official_split_manifest.json').read_text()) == manifest
    saved = torch.load(anchor / 'best.pth', map_location='cpu', weights_only=True)
    current = model.state_dict()
    assert set(saved) <= set(current)
    assert {key.split('.')[0] for key in set(current) - set(saved)} == set(NEW_MODULES)
    assert all(current[key].shape == value.shape and current[key].dtype == value.dtype for key, value in saved.items())
    current.update(saved)
    model.load_state_dict(current, strict=True)
    assert all(torch.equal(model.state_dict()[key].detach().cpu(), value) for key, value in saved.items())
    model.anchor_encoder_state = {key: value.clone() for key, value in saved.items()
        if key.split('.')[0] in IDENTITY_MODULES} if args.freeze_identity_encoder else {}
    model.anchor_record = dict(run_dir=str(anchor), selected_epoch=trained['best']['epoch'],
        anchor_training_epochs=50, anchor_model='demo', initialized_state_tensors=len(saved),
        new_modules=list(NEW_MODULES), freeze_identity_encoder=bool(args.freeze_identity_encoder),
        identity_modules=list(IDENTITY_MODULES), shared_projection_frozen=False,
        selected_anchor_full_metrics=trained['full_metrics'],
        fusion_precision='FP32 base fusion in both controls; backbone/reducers/heads retain existing AMP; actual same-batch precision probe restored all six warm HDM query gradients',
        optimizer_schedule='Original DeMo50 selected best plus additional50/new Adam/scheduler/AMP512; no optimizer resume',
        limits='Only original private encoder is fixed in the protected control. Newly initialized shared/fused heads remain trainable; the .75/.25 fused descriptor is not claimed equal to the original 5120D descriptor.')
    model = model.float().cuda()
    model.train()
    if model.freeze_identity_encoder:
        assert_anchor_unchanged(model)
    return model
