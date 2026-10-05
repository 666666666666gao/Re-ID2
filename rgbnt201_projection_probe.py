"""Bounded RGBNT201 outlet diagnosis; no formal training or new metric rule."""
import json
from pathlib import Path

import torch
from torch import nn

from common_coordinate_axis import CommonCoordinateAxis
from experiment_data import seed_all
from original_identity_anchor import IDENTITY_MODULES, fp32_base_fusion, assert_anchor_unchanged
from official_training_data import full_records
import shared_identity_axis as shared


VARIANTS = ('frequency_shared', 'axis_shared')
NEW_MODULES = ('bands', 'modality_expert', 'frequency_expert', 'router', 'calibrator',
    'modality_projection', 'frequency_projection', 'interaction_projection', 'residual_scale',
    'fused_neck', 'fused_classifier', 'modality_neck', 'modality_classifier',
    'frequency_neck', 'frequency_classifier', 'shared_projection', 'shared_neck', 'shared_classifier')


class EvidenceBypass(nn.Sequential):
    """Keep the same projection tensors and state keys; add only its input."""
    def forward(self, value):
        return value + super().forward(value)


class OriginalProjectionProbe(CommonCoordinateAxis):
    def __init__(self, classes, cfg, cameras, variant, seed, bypass):
        super().__init__(classes, cfg, cameras, variant, seed)
        self.freeze_identity_encoder = True
        self.bypass = bypass
        if bypass:
            self.modality_projection = EvidenceBypass(*self.modality_projection.children())
            self.frequency_projection = EvidenceBypass(*self.frequency_projection.children())
        for name in IDENTITY_MODULES:
            getattr(self, name).requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        for name in IDENTITY_MODULES:
            getattr(self, name).eval()
        return self


def build(args, cfg, classes, cameras):
    assert args.dataset == 'RGBNT201' and args.variant in VARIANTS and args.seed == 42
    assert args.bypass in (0, 1)
    shared.available_base_fusion = fp32_base_fusion
    seed_all(args.seed)
    model = OriginalProjectionProbe(classes, cfg, cameras, args.variant, args.seed, bool(args.bypass))
    anchor = Path(args.anchor_run_dir)
    trained = json.loads((anchor / 'result.json').read_text())
    assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50
    assert trained['arguments']['dataset'] == 'RGBNT201' and trained['arguments']['variant'] == 'demo'
    assert trained['arguments']['seed'] == 42 and trained['best']['epoch'] == 28
    assert trained['descriptor_dim'] == 5120 and trained['optimizer_steps'] == 2647
    assert trained['amp_skipped_steps'] == trained['training_heldout_identities'] == 0
    assert trained['classes'] == classes and trained['camera_embeddings'] == cameras
    assert json.loads((anchor.parent / (anchor.name + '_exit.json')).read_text())['exit_code'] == 0
    manifest = full_records(args.data_root, args.dataset)[-1]
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
        if key.split('.')[0] in IDENTITY_MODULES}
    model.anchor_record = dict(original_run=str(anchor), original_best_epoch=28,
        identity_modules=list(IDENTITY_MODULES), public_projection='fresh, trainable; same seed and tensors in every control',
        metric='unchanged common-coordinate .75 private/.25 public; all expert types use the common outlet',
        factor='PM/PF P(x) versus x+P(x); interaction projection unchanged',
        scope='Native diagnosis only. Introduction of experts/common outlet is shared setup, not a one-factor change from R201B. '
              '00 is not claimed equal to original DeMo; full private block is. All7 compares the availability-aware frozen reference.')
    model = model.float().cuda()
    model.train()
    assert_anchor_unchanged(model)
    return model
