"""R201C common fixture: expert corrections in original identity coordinates."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiment_data import seed_all
from modeling.meta_arch import weights_init_classifier, weights_init_kaiming
from official_training_data import full_records
from original_identity_anchor import IDENTITY_MODULES, assert_anchor_unchanged, fp32_base_fusion
from rgbnt201_projection_probe import NEW_MODULES, OriginalProjectionProbe, VARIANTS
from shared_identity_axis import encode_available, base_supervision
import shared_identity_axis as shared


class IdentityCoordinateAxis(OriginalProjectionProbe):
    def __init__(self, classes, cfg, cameras, variant, seed, bypass):
        super().__init__(classes, cfg, cameras, variant, seed, bypass)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 30000)
            self.fused_neck = nn.BatchNorm1d(5120)
            self.fused_neck.bias.requires_grad_(False)
            self.fused_neck.apply(weights_init_kaiming)
            self.fused_classifier = nn.Linear(5120, classes, bias=False)
            self.fused_classifier.apply(weights_init_classifier)

    def fuse(self, base, modality, frequency, gates, use_m, use_f, eligible,
             relation_mass=None, relation_frequency=None):
        with torch.autocast('cuda', enabled=False):
            base, modality, frequency, gates = (value.float() for value in (base, modality, frequency, gates))
            batch, relations, dim = modality.shape
            anchor = base[:, 1536:5120].reshape(batch, relations, dim).norm(dim=-1, keepdim=True).detach()
            if relation_mass is None:
                assert use_m and not use_f
                logits = self.router.relation_score(modality).squeeze(-1)
                relation_mass = logits.masked_fill(~eligible, -torch.inf).softmax(1)
            weights = relation_mass.float() * eligible.sum(1, keepdim=True)
            delta = torch.zeros_like(base[:, 1536:5120]).reshape(batch, relations, dim)
            if use_m:
                direction = F.normalize(self.modality_projection(modality), dim=-1)
                delta = delta + self.residual_scale[0] * gates[:, :1, None] * direction * anchor
            if use_f:
                assert relation_frequency is not None
                direction = F.normalize(self.frequency_projection(relation_frequency.float()), dim=-1)
                delta = delta + self.residual_scale[1] * gates[:, 1:2, None] * direction * anchor
            if use_m and use_f:
                interaction = F.normalize(self.interaction_projection(torch.cat((modality.mean(1), frequency), -1)), dim=1)
                delta = delta + self.residual_scale[2] * gates[:, 2:, None] * interaction[:, None] * anchor
            delta = delta * weights[..., None] * eligible[..., None]
            private = torch.cat((base[:, :1536], base[:, 1536:5120] + delta.flatten(1)), 1)
            return F.normalize(private, dim=1)

    def controlled_states(self, base, m0, f0, eligible, available, full):
        modality = self.modality_expert.pool(m0, eligible)
        frequency = self.frequency_expert.pool(f0, available, self.structured)
        _, gm = self.calibrator(base, modality, torch.zeros_like(frequency))
        _, gf = self.calibrator(base, torch.zeros_like(modality), frequency)
        # Independent F-only relation evidence; no M message, full route, psi or I.
        with torch.autocast('cuda', enabled=False):
            v = self.router.by_relation(f0.float())
            band_weights = self.frequency_expert.pool_score(v).squeeze(-1).softmax(2)
            relation_frequency = (v * band_weights[..., None]).sum(2)
            logits = self.router.band_score(relation_frequency).squeeze(-1)
            mass = logits.masked_fill(~eligible, -torch.inf).softmax(1)
        return {'00': F.normalize(base[:, :5120].float(), dim=1),
            '10': self.fuse(base, modality, torch.zeros_like(frequency), gm, True, False, eligible),
            '01': self.fuse(base, torch.zeros_like(modality), frequency, gf, False, True, eligible, mass, relation_frequency),
            '11': full}

    def forward(self, x, label=None, cam_label=None, view_label=None, partial=False, return_states=False):
        patches, globals_, base, eligible, available, common = encode_available(self, x, cam_label, view_label)
        spatial = torch.stack(patches, 1)
        tokens = self.bands(spatial)
        mvalues = self.modality_expert.prepare(tokens, available)
        fvalues = self.frequency_expert.prepare(tokens, available, self.structured)
        m0 = self.modality_expert.read(mvalues, eligible, self.structured)
        f0 = self.frequency_expert.read(fvalues, available, self.structured)
        cm, cf = self.router.conditions(m0, f0)
        m1 = self.modality_expert.read(mvalues, eligible, self.structured, cm)
        f1 = self.frequency_expert.read(fvalues, available, self.structured, cf)
        modality, frequency, route = self.router.route(m1, f1, eligible, self.joint_interaction)
        prediction, gates = self.calibrator(base, modality, frequency)
        v = self.router.by_relation(f1)
        mass = route.sum(2, keepdim=True)
        conditional = route / torch.where(eligible[:, :, None], mass, torch.ones_like(mass))
        relation_frequency = (v * conditional[..., None]).sum(2)
        fused = self.fuse(base, modality, frequency, gates, True, True, eligible, route.sum(2), relation_frequency)
        self.last_route, self.last_gates = route.detach(), gates.detach()
        if not self.training and not return_states:
            return fused
        if self.training and partial:
            return self.fused_classifier(self.fused_neck(fused)), fused
        states = self.controlled_states(base, m0, f0, eligible, available, fused)
        if not self.training:
            return states
        target, pos, neg, scores = self.calibrator.targets(states, label)
        contribution = F.smooth_l1_loss(prediction.float(), target)
        self.contribution_audit = dict(target_requires_grad=target.requires_grad,
            positive_indices=pos.tolist(), negative_indices=neg.tolist(), targets=target.tolist(),
            predictions=prediction.detach().float().tolist(), scores={key: value.tolist() for key, value in scores.items()},
            loss=float(contribution.detach()))
        standalone_m = self.modality_expert.pool(m0, eligible).mean(1)
        standalone_f = self.frequency_expert.pool(f0, available, self.structured)
        output = [self.fused_classifier(self.fused_neck(fused)), fused,
            self.modality_classifier(self.modality_neck(standalone_m)), standalone_m,
            self.frequency_classifier(self.frequency_neck(standalone_f)), standalone_f]
        output.extend(base_supervision(self, globals_, base, common))
        output.append(self.contribution_loss_weight * contribution)
        return tuple(output)


def build(args, cfg, classes, cameras):
    assert args.dataset == 'RGBNT201' and args.variant in VARIANTS and args.seed == 42
    assert args.freeze_identity_encoder == 1 and args.bypass in (0, 1)
    shared.available_base_fusion = fp32_base_fusion
    seed_all(args.seed)
    model = IdentityCoordinateAxis(classes, cfg, cameras, args.variant, args.seed, bool(args.bypass))
    anchor = Path(args.anchor_run_dir)
    trained = json.loads((anchor / 'result.json').read_text())
    assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50 and trained['best']['epoch'] == 28
    assert trained['arguments']['dataset'] == 'RGBNT201' and trained['arguments']['variant'] == 'demo'
    assert trained['arguments']['seed'] == 42 and trained['descriptor_dim'] == 5120
    assert trained['optimizer_steps'] == 2647 and trained['amp_skipped_steps'] == trained['training_heldout_identities'] == 0
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
    model.anchor_record = dict(original_run=str(anchor), original_best_epoch=28,
        anchor_training_epochs=50, new_stage_epochs=50, identity_modules=list(IDENTITY_MODULES),
        descriptor_dim=5120, bypass=bool(args.bypass),
        metric='Original globals1536 + seven identity relation slots3584; normalized after M/F/I corrections; no public25 block',
        common_setup_changes='Original E28 frozen identity plus matched relation-coordinate M/F/I outlet and5120 fused head. '
            'This whole setup is not a one-factor continuation of R201B/probe. Only PM/PF bypass is isolated within this fixture.',
        common_context='Existing shared projection/head remains trainable auxiliary/context; it is not an appended retrieval block',
        optimizer_schedule='Original50 plus additional50 with fresh Adam/scheduler/AMP512; no optimizer resume',
        control_limit='Ordinary-frequency slots mix source inputs and do not have structured physical-source provenance. '
            'Both controls use identical eligible masks, parameters, losses, output dimensions and outlet placement. '
            'Unchanged coupled contribution predictor is a task gate, not claimed calibrated.')
    model = model.float().cuda()
    model.train()
    assert_anchor_unchanged(model)
    return model
