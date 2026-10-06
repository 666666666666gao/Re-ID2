"""RGBNT201 I expert seeds 42/43/44 with the same frozen original DeMo seed42 anchor."""
import json
from pathlib import Path

import torch

from experiment_data import seed_all
from official_training_data import full_records
from original_identity_anchor import IDENTITY_MODULES, assert_anchor_unchanged, fp32_base_fusion
from rgbnt201_identity_outlet import IdentityCoordinateAxis
from rgbnt201_projection_probe import NEW_MODULES, VARIANTS
import shared_identity_axis as shared


def build(args, cfg, classes, cameras):
    assert args.dataset == 'RGBNT201' and args.variant in ('frequency_shared', 'axis_shared')
    assert args.seed in (42, 43, 44) and args.freeze_identity_encoder == 1
    shared.available_base_fusion = fp32_base_fusion
    seed_all(args.seed)
    model = IdentityCoordinateAxis(classes, cfg, cameras, args.variant, args.seed, False)
    anchor = Path(args.anchor_run_dir)
    trained = json.loads((anchor / 'result.json').read_text())
    assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50 and trained['descriptor_dim'] == 5120
    assert trained['arguments']['dataset'] == args.dataset and trained['arguments']['variant'] == 'demo'
    assert trained['arguments']['seed'] == 42 and trained['training_heldout_identities'] == 0
    assert trained['optimizer_steps'] == trained['steps'] and trained['amp_skipped_steps'] == 0
    assert trained['classes'] == classes and trained['camera_embeddings'] == cameras
    assert json.loads((anchor.parent / (anchor.name + '_exit.json')).read_text())['exit_code'] == 0
    assert json.loads((anchor / 'official_split_manifest.json').read_text()) == full_records(args.data_root, args.dataset)[-1]
    saved = torch.load(anchor / 'best.pth', map_location='cpu', weights_only=True)
    current = model.state_dict()
    assert set(saved) <= set(current)
    assert {key.split('.')[0] for key in set(current) - set(saved)} == set(NEW_MODULES)
    assert all(current[key].shape == value.shape and current[key].dtype == value.dtype for key, value in saved.items())
    current.update(saved)
    model.load_state_dict(current, strict=True)
    assert all(torch.equal(model.state_dict()[key].detach().cpu(), value) for key, value in saved.items())
    model.anchor_encoder_state = {key: value.clone() for key, value in saved.items() if key.split('.')[0] in IDENTITY_MODULES}
    model.anchor_record = dict(original_run=str(anchor), original_best_epoch=trained['best']['epoch'],
        anchor_training_epochs=50, new_stage_epochs=50, anchor_seed=42, expert_seed=args.seed, identity_modules=list(IDENTITY_MODULES),
        descriptor_dim=5120, bypass=False, selected_anchor_full_metrics=trained['full_metrics'],
        method='Exactly the R201C narrow5120 identity-coordinate expert; no new loss, metric or gate',
        optimizer_schedule=f'Original50/seed42 plus additional50/freshAdam/expert-seed{args.seed}',
        limits='Common context remains auxiliary; coupled gate is not claimed calibrated. '
               'Twelve disjoint-source cases lack shared identity coordinates; missing evaluation deferred.')
    model = model.float().cuda()
    model.train()
    assert_anchor_unchanged(model)
    return model
