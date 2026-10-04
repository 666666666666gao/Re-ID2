"""Real CUDA zero-weight parity, PM-loss gradients and stopped-reference checks."""
import argparse
import gc
from pathlib import Path

import torch

from experiment_data import make_loader, seed_all
from independent_control_axis import build as parent_build
from modality_outlet_alignment_axis import build
from official_training_data import full_records
from run_experiment import configuration, write_json
from run_shared_identity_experiment import PARTIAL_SETS


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--variant', choices=('axis_shared', 'frequency_shared', 'twins_shared'), required=True)
    args = parser.parse_args()
    args.dataset, args.seed, args.pooling, args.gate_gradient_mode = 'MSVR310', 42, 'original_mean', 'measurement_only'
    args.relation_weight, args.alignment_weight, args.alignment_temperature = .1, .1, .07
    args.modality_alignment_weight = 0
    torch.set_num_threads(4)
    out = Path(args.output); out.mkdir(exist_ok=False)
    cfg = configuration(args)
    train, query, gallery, classes, cameras, _ = full_records(args.data_root, args.dataset)
    assert (len(train), len(query), len(gallery)) == (1032, 591, 1055)
    parent = parent_build(args, cfg, classes, cameras).eval()
    model = build(args, cfg, classes, cameras).eval()
    assert set(model.state_dict()) == set(parent.state_dict())
    assert all(torch.equal(value, parent.state_dict()[key]) for key, value in model.state_dict().items())
    counts = dict(parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), descriptor_dim=5632)
    assert counts['parameters'] == sum(p.numel() for p in parent.parameters())
    assert counts['trainable_parameters'] == sum(p.numel() for p in parent.parameters() if p.requires_grad)
    images, _, cam, scene, _ = next(iter(make_loader(query[:8], cfg, False, args.seed)))
    images = {key: value.cuda() for key, value in images.items()}
    cam, scene = cam.cuda(), scene.cuda()
    checks = []
    for retained in (*PARTIAL_SETS, (0, 1, 2)):
        partial = {key: value if i in retained else torch.zeros_like(value) for i, (key, value) in enumerate(images.items())}
        for amp in (False, True):
            with torch.no_grad(), torch.autocast('cuda', enabled=amp):
                expected = parent(partial, cam_label=cam, view_label=scene, return_states=True)
                actual = model(partial, cam_label=cam, view_label=scene, return_states=True)
            assert set(actual) == set(expected) == {'00', '10', '01', '11'}
            assert all(torch.equal(actual[key], expected[key]) for key in actual)
            checks.append(dict(retained=retained, AMP=amp, all_four_states_exact=True))
    images, labels, cam, scene, names = next(iter(make_loader(train, cfg, True, args.seed)))
    images = {key: value.cuda() for key, value in images.items()}
    labels, cam, scene = labels.cuda(), cam.cuda(), scene.cuda()
    parent.train(); model.train()
    parent.alignment_names = model.alignment_names = tuple(names)
    with torch.no_grad(), torch.autocast('cuda'):
        seed_all(1234); expected = parent(images, label=labels, cam_label=cam, view_label=scene)
        seed_all(1234); actual = model(images, label=labels, cam_label=cam, view_label=scene)
    assert len(expected) == len(actual) and all(torch.equal(a, b) for a, b in zip(actual, expected))
    del expected, actual, parent
    gc.collect(); torch.cuda.empty_cache()
    model.modality_alignment_weight = .1
    with torch.autocast('cuda'):
        result = model(images, label=labels, cam_label=cam, view_label=scene)
    loss = model.last_modality_alignment_loss
    assert loss is not None and loss.requires_grad and torch.isfinite(loss)
    assert model.last_modality_outlet.shape == (64, 512) and torch.isfinite(model.last_modality_outlet).all()
    projection = list(model.modality_projection.parameters())
    reference = list(model.shared_projection.parameters())
    predictor = list(model.calibrator.predictor.parameters())
    gradients = torch.autograd.grad(loss, projection + reference + predictor, allow_unused=True)
    assert all(value is not None and torch.isfinite(value).all() and value.abs().sum() > 0 for value in gradients[:len(projection)])
    assert all(value is None for value in gradients[len(projection):])
    assert not model.modality_alignment_audit['identity_alignment_reference_requires_grad']
    assert model.modality_alignment_audit['identity_alignment_positive_observations_distinct']
    assert model.modality_alignment_audit['identity_alignment_valid_anchors'] > 0
    write_json(out / 'result.json', dict(status='PASS_FULL_OFFICIAL_PM_OUTLET_CONTRACT', variant=args.variant,
        counts=counts, eval_checks=checks, zero_weight_full_training_outputs_exact=True,
        state_dict_identical_to_M3a=True, added_parameters=0, PM_loss_finite_nonzero_gradients=True,
        public_projection_no_alignment_gradient=True, predictor_no_alignment_gradient=True,
        training_heldout_identities=0, optimizer_updates=0, new_weights=0,
        alignment=model.modality_alignment_audit,
        scope='Numerical real-CUDA contract only;8 query parity and one64 training batch, not benchmark performance.'))
    print('FULL_OFFICIAL_PM_OUTLET_CONTRACT_PASS', args.variant, flush=True)


if __name__ == '__main__':
    main()
