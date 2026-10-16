"""Existing AH experts on the selected full5632 MSVR310 head16 identity baseline."""
import hashlib
import json
from pathlib import Path

import torch

from experiment_data import seed_all
from official_training_data import full_records
from original_identity_anchor import IDENTITY_MODULES, fp32_base_fusion
from run_experiment import write_json
from msvr_full_identity_reference import StrongFullRelationAxis, assert_retained
import identity_coordinate_three_dataset as coordinate
import shared_identity_axis as shared
import run_r201g_normal_priority as g
import run_r201ah_MS_full_identity as ah

VARIANTS = ah.VARIANTS
TEACHER_SHA256 = 'a126eb25b42daf9c0b731b731242895a12a0918852361caf893ce71e636f02c9'
REVISION = (
    'Same AH expert model, normal objectives and full5632 .75private/.25public metric; '
    'initialize both controls directly from the independently audited MSVR310 head16 '
    'shared-DeMo50 bestE32. No intermediate head8 load. Five identity modules and '
    'learned public projection frozen; same fresh matched5632 fused head, extra50/newAdam, '
    'seed42/P16K4/B64/GradScaler512/BN2/partial0/two backwards. Existing FFT/M/F/rank64/ '
    'one condition/joint routing/relation-local PI/private cap.10 unchanged. '
    'Full official1032train591Q1055G, benchmark mAP-best earliest ties and same-checkpoint '
    'six metrics. Baseline51.430265/68.358714 is inherited credit, not an expert gain. '
    'Require complete00 teacher parity/native3 per control before formal50. '
    'Only physicalGPU3 serialized while original RGBNT100 stays onGPU2; no new missing '
    'evaluation, RGBNT201 training, seed search or temperature/power queries.')


def configuration(args):
    cfg = ah.configuration(args)
    cfg.defrost()
    cfg.MODEL.HEAD = 16
    cfg.freeze()
    return cfg


def reference_build(args, cfg, classes, cameras):
    assert args.dataset == 'MSVR310' and args.variant in VARIANTS
    assert args.seed == 42 and args.freeze_identity_encoder == 1 and cfg.MODEL.HEAD == 16
    anchor = Path(args.anchor_run_dir)
    trained = json.loads((anchor/'result.json').read_text())
    assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50 and trained['descriptor_dim'] == 5632
    assert trained['arguments']['dataset'] == 'MSVR310' and trained['arguments']['variant'] == 'demo_shared'
    assert trained['arguments']['seed'] == 42 and trained['arguments']['atmoe_heads'] == 16
    assert trained['classes'] == classes == 155 and trained['camera_embeddings'] == cameras
    assert trained['steps'] == trained['optimizer_steps'] == 705 and trained['amp_skipped_steps'] == 0
    assert trained['training_heldout_identities'] == 0 and trained['best']['epoch'] == 32
    assert json.loads((anchor.parent/(anchor.name+'_exit.json')).read_text())['exit_code'] == 0
    assert json.loads((anchor/'official_split_manifest.json').read_text()) == full_records(args.data_root, 'MSVR310')[-1]
    assert hashlib.sha256((anchor/'best.pth').read_bytes()).hexdigest() == TEACHER_SHA256
    shared.available_base_fusion = fp32_base_fusion
    seed_all(args.seed)
    model = coordinate.IdentityCoordinateAxis(classes, cfg, cameras, args.variant, args.seed, False)
    saved = torch.load(anchor/'best.pth', map_location='cpu', weights_only=True)
    excluded = {'fused_neck', 'fused_classifier'}
    selected = {key: value for key, value in saved.items() if key.split('.')[0] not in excluded}
    current = model.state_dict()
    assert set(saved) <= set(current)
    assert all(current[key].shape == value.shape and current[key].dtype == value.dtype for key, value in saved.items())
    assert {key for key in selected if key.split('.')[0] in IDENTITY_MODULES} == {
        key for key in current if key.split('.')[0] in IDENTITY_MODULES}
    fresh = {key: value.detach().cpu().clone() for key, value in current.items() if key not in selected}
    current.update(selected)
    model.load_state_dict(current, strict=True)
    assert all(torch.equal(model.state_dict()[key].detach().cpu(), value) for key, value in selected.items())
    assert all(torch.equal(model.state_dict()[key].detach().cpu(), value) for key, value in fresh.items())
    model.anchor_encoder_state = {key: value.clone() for key, value in selected.items()
                                  if key.split('.')[0] in IDENTITY_MODULES}
    model.anchor_record = dict(
        original_run=str(anchor), original_best_epoch=32, anchor_training_epochs=50, new_stage_epochs=50,
        identity_modules=list(IDENTITY_MODULES), descriptor_dim=5632, bypass=False,
        selected_anchor_full_metrics=trained['full_metrics'], teacher_variant='demo_shared',
        teacher_descriptor_dim=5632, deployed_descriptor_dim=5632,
        optimizer_schedule='Head16 shared-DeMo50 bestE32 plus additional50/freshAdam; fixed seed42',
        strong_teacher=dict(weight_sha256=TEACHER_SHA256, identity_heads=16,
            copied_tensors=len(selected), excluded_teacher_prefixes=sorted(excluded), fresh_student_tensors=len(fresh),
            private_reference='Complete5632 .75private/.25public; both trained teacher projections retained',
            limits='Teacher performance is baseline credit. Require full00 parity and matched ordinary/axis incremental utility.'))
    return model.float().cuda()


def build(args, cfg, classes, cameras):
    ah.y.m.k.RelationLocalPIAxis = StrongFullRelationAxis
    g.coordinate_build = reference_build
    ah.y.margin_step = ah.margin_step
    model = ah.y.build(args, cfg, classes, cameras)
    model.shared_projection.requires_grad_(False)
    model.retained_public_state = {key: value.detach().cpu().clone()
                                   for key, value in model.shared_projection.state_dict().items()}
    model.anchor_record.update(method=REVISION, complete_reference=dict(
        private_dimension=5120, public_dimension=512, metric_weights=[.75, .25],
        public_projection_frozen=True, public_projection_parameters=527872, fresh_fused_head_dimension=5632,
        private_correction_bound=.10, complete_correction_bound=.75**.5*.10))
    model.anchor_record['final_unit_descriptor_bound']['placement'] = 'Private5120 after M/F/I; retained public512 at .25 metric weight'
    model.anchor_record['final_unit_descriptor_bound']['complete_descriptor_bound'] = .75**.5*.10
    model.anchor_record['M_F_auxiliary_outlet']['inference'] = 'Eval hooks inactive; full5632 .75private/.25public descriptor'
    assert sum(p.numel() for p in model.parameters()) == 100787334
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 2674822
    model.train()
    assert_retained(model)
    return model


step = ah.step


def record(path, value):
    if 'descriptor_dim' in value:
        assert value['descriptor_dim'] == 5632
    if 'arguments' in value:
        value['arguments']['atmoe_heads'] = 16
    if 'pretraining' in value:
        value['pretraining'] = 'Full official MSVR310 head16 shared-DeMo50 bestE32, complete5632 retained; additional50/fresh matched fused head/newAdam'
    write_json(path, value)


if __name__ == '__main__':
    runner = ah.y.w.runner
    runner.configuration = configuration
    runner.assert_anchor_unchanged = assert_retained
    runner.write_json = record
    runner.main(builder=build, update=step, variants=VARIANTS, method_revision=REVISION,
        partial_training='Same AH partial0 forwards/mining/BN2/two backwards. Only the selected frozen identity reference changes to head16/E32; missing performance not claimed.')
