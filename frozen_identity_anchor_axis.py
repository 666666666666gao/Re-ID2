"""Initialize from the complete shared-DeMo identity anchor; optionally keep its encoder fixed."""
import json
import os
from pathlib import Path

import torch

from experiment_data import seed_all
from gpu_thermal_execute import check_limits
from modality_outlet_alignment_axis import ModalityOutletAlignmentAxis
from official_training_data import full_records
from shared_identity_axis import SharedIdentityDeMo


IDENTITY_MODULES = ('BACKBONE', 'rgb_reduce', 'nir_reduce', 'tir_reduce', 'generalFusion', 'shared_projection')
VARIANTS = ('demo_shared', 'axis_shared', 'frequency_shared', 'twins_shared')


class AnchoredAxis(ModalityOutletAlignmentAxis):
    def __init__(self, *args, freeze_identity_encoder, **kwargs):
        super().__init__(*args, **kwargs)
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
    current = model.state_dict()
    for name in IDENTITY_MODULES:
        module = getattr(model, name)
        assert not module.training and all(not p.requires_grad for p in module.parameters())
    assert all(torch.equal(current[key].detach().cpu(), value)
               for key, value in model.anchor_encoder_state.items())


def build(args, cfg, classes, cameras):
    assert args.variant in VARIANTS and args.freeze_identity_encoder in (0, 1)
    check_limits(int(os.environ['CUDA_VISIBLE_DEVICES']))
    seed_all(args.seed)
    if args.variant == 'demo_shared':
        assert args.freeze_identity_encoder == 0
        model = SharedIdentityDeMo(classes, cfg, cameras, args.seed)
        model.freeze_identity_encoder = False
    else:
        args.gate_gradient_mode, args.pooling = 'measurement_only', 'original_mean'
        args.relation_weight, args.alignment_weight, args.alignment_temperature = .1, .1, .07
        args.modality_alignment_weight = .1
        model = AnchoredAxis(classes, cfg, cameras, args.variant, args.seed,
                             .1, .1, .07, modality_alignment_weight=.1,
                             freeze_identity_encoder=bool(args.freeze_identity_encoder))
    anchor = Path(args.anchor_run_dir)
    trained = json.loads((anchor / 'result.json').read_text())
    assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50 and trained['training_heldout_identities'] == 0
    assert trained['arguments']['variant'] == 'demo_shared' and trained['arguments']['dataset'] == args.dataset
    assert trained['arguments']['seed'] == args.seed and trained['descriptor_dim'] == 5632
    assert trained['classes'] == classes and trained['camera_embeddings'] == cameras
    _, _, _, _, _, manifest = full_records(args.data_root, args.dataset)
    assert manifest == json.loads((anchor / 'official_split_manifest.json').read_text())
    saved = torch.load(anchor / 'best.pth', map_location='cpu', weights_only=True)
    current = model.state_dict()
    mapping = {key: ('shared_projection.projection.' + key.removeprefix('shared_projection.')
                     if args.variant != 'demo_shared' and key.startswith('shared_projection.') else key)
               for key in saved}
    assert len(set(mapping.values())) == len(saved)
    assert set(mapping.values()) <= set(current)
    assert all(current[mapping[key]].shape == value.shape and current[mapping[key]].dtype == value.dtype
               for key, value in saved.items())
    current.update({mapping[key]: value for key, value in saved.items()})
    if args.variant != 'demo_shared':
        current['shared_projection.query'] = torch.zeros_like(current['shared_projection.query'])
    model.load_state_dict(current, strict=True)
    model.anchor_encoder_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                                  if key.split('.')[0] in IDENTITY_MODULES} if args.freeze_identity_encoder else {}
    model.anchor_record = dict(run_dir=str(anchor), selected_epoch=trained['best']['epoch'],
                               anchor_training_epochs=50, anchor_model='demo_shared',
                               initialized_state_tensors=len(mapping),
                               explicit_projection_key_mapping=6 if args.variant != 'demo_shared' else 0,
                               extra_query_initialization='zero' if args.variant != 'demo_shared' else 'absent',
                               freeze_identity_encoder=bool(args.freeze_identity_encoder),
                               identity_modules=list(IDENTITY_MODULES),
                               selected_anchor_full_metrics=trained['full_metrics'],
                               optimizer_schedule='New Adam/scheduler/AMP512 for the additional50; not an optimizer-state resume')
    model = model.float().cuda()
    model.train()
    if args.freeze_identity_encoder:
        assert_anchor_unchanged(model)
    return model
