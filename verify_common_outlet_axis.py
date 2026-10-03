"""CUDA evidence for M1; source preparation does not substitute for this check."""
import argparse
import gc
import json
import os
from pathlib import Path

import torch

from common_coordinate_axis import CommonCoordinateAxis
from common_outlet_axis import POOLING, SharedReadout, build, outlet_weights
from dual_axis import RELATIONS
from experiment_data import make_loader, seed_all, split_records
from gpu_thermal_execute import check_limits
from run_mass_experiment import configuration, write_json
from run_shared_identity_experiment import PARTIAL_SETS
from shared_identity_axis import KEYS, partial_gallery_triplet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    check_limits(int(os.environ['CUDA_VISIBLE_DEVICES']))
    args.dataset, args.seed, args.variant = 'MSVR310', 42, 'axis_shared'
    torch.set_num_threads(4)
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    cfg = configuration(args)
    _, dev, _, classes, cameras = split_records(args.data_root, args.dataset)
    images, _, cam, scene, _ = next(iter(make_loader(dev[:8], cfg, False, args.seed)))
    images = {key: value.cuda() for key, value in images.items()}
    cam, scene = cam.cuda(), scene.cuda()
    reference, common = None, None
    counts, checks = {}, []
    for pooling in POOLING:
        variants = ('axis_shared', 'frequency_shared', 'twins_shared', 'demo_shared') if pooling == 'original_mean' else ('axis_shared', 'frequency_shared', 'twins_shared')
        for variant in variants:
            args.pooling, args.variant = pooling, variant
            model = build(args, cfg, classes, cameras)
            state = model.state_dict()
            if reference is None:
                reference = {name: value.cpu().clone() for name, value in state.items()}
                common = {name: value for name, value in reference.items() if name.startswith('shared_')}
            elif variant != 'demo_shared':
                assert set(state) == set(reference)
                assert all(torch.equal(value.cpu(), reference[name]) for name, value in state.items())
            else:
                assert all(torch.equal(state[name].cpu(), value) for name, value in common.items())
            counts[pooling + '/' + variant] = dict(parameters=sum(p.numel() for p in model.parameters()),
                                                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
            model.eval()
            versions = {name: value._version for name, value in state.items()}
            for retained in (*PARTIAL_SETS, (0, 1, 2)):
                partial = {key: value if index in retained else torch.zeros_like(value)
                           for index, (key, value) in enumerate(images.items())}
                with torch.no_grad():
                    states = model(partial, cam_label=cam, view_label=scene, return_states=True)
                eligible = torch.tensor([all(index in retained for index in subset) for subset in RELATIONS], device='cuda')
                for name, feature in states.items():
                    assert feature.shape == (8, 5632) and torch.isfinite(feature).all()
                    assert torch.equal(feature[:, :5120], states['00'][:, :5120])
                    blocks = feature[:, :5120].reshape(8, 10, 512)
                    assert all(torch.count_nonzero(blocks[:, index]) == 0 for index in range(3) if index not in retained)
                    assert torch.count_nonzero(blocks[:, 3:][:, ~eligible]) == 0
                    assert torch.allclose(feature[:, 5120:].square().sum(1), torch.full((8,), .25, device='cuda'), atol=1e-6, rtol=0)
                if variant != 'demo_shared':
                    coefficients = model.last_outlet_weights
                    assert coefficients.shape == (8, 7) and torch.count_nonzero(coefficients[:, ~eligible]) == 0
                    expected = len([i for i in eligible.tolist() if i]) / 7 if pooling == 'original_mean' else 1
                    assert torch.allclose(coefficients.sum(1), torch.full((8,), expected, device='cuda'), atol=1e-6, rtol=0)
                checks.append(dict(pooling=pooling, variant=variant, retained=''.join('RNT'[i] for i in retained),
                                   finite=True, invalid_coordinates_zero=True, unchanged_private=True))
            assert versions == {name: value._version for name, value in model.state_dict().items()}
            del model, state, states
            gc.collect()
            torch.cuda.empty_cache()
    augmented = [value for name, value in counts.items() if not name.endswith('/demo_shared')]
    assert len(augmented) == 9 and all(value == augmented[0] for value in augmented)
    # A local generator must not perturb the existing sampling/initialization RNG.
    generator_before = torch.random.get_rng_state().clone()
    readout = SharedReadout(torch.nn.Identity(), 42)
    assert torch.equal(generator_before, torch.random.get_rng_state())
    readout = readout.cuda()
    value = torch.randn(8, 512, device='cuda', requires_grad=True)
    readout(value).square().sum().backward()
    assert readout.query.grad is not None and readout.query.grad.abs().sum() > 0
    modality = torch.randn(8, 7, 512, device='cuda', requires_grad=True)
    valid = torch.tensor([[True, False, True, False, True, False, True]] * 8, device='cuda')
    attention = outlet_weights('seed_query', readout.query, modality, valid)
    (attention * torch.arange(7, device='cuda')).sum().backward()
    assert modality.grad.abs().sum() > 0 and torch.count_nonzero(modality.grad[:, ~valid[0]]) == 0
    assert torch.count_nonzero(attention[:, ~valid[0]]) == 0
    # Same raw shared-readout seed in the original-mean reference; original V12 sources stay untouched.
    seed_all(42)
    baseline = CommonCoordinateAxis(classes, cfg, cameras, 'axis_shared', 42)
    baseline.shared_projection = SharedReadout(baseline.shared_projection, 42)
    baseline = baseline.float().cuda().eval()
    args.pooling, args.variant = 'original_mean', 'axis_shared'
    current = build(args, cfg, classes, cameras).eval()
    for retained in (*PARTIAL_SETS, (0, 1, 2)):
        partial = {key: value if index in retained else torch.zeros_like(value)
                   for index, (key, value) in enumerate(images.items())}
        with torch.no_grad():
            previous = baseline(partial, cam_label=cam, view_label=scene, return_states=True)
            actual = current(partial, cam_label=cam, view_label=scene, return_states=True)
        assert all(torch.equal(actual[key], previous[key]) for key in actual)
    query = torch.randn(8, 12, device='cuda', requires_grad=True)
    gallery = torch.randn(8, 12, device='cuda', requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device='cuda')
    loss, pos, neg = partial_gallery_triplet(query, gallery, labels)
    loss.backward()
    assert gallery.grad is None and query.grad.abs().sum() > 0
    assert labels[pos].eq(labels).all() and labels[neg].ne(labels).all()
    assert pos.ne(torch.arange(8, device='cuda')).all()
    report = dict(status='PASS_COMMON_OUTLET_TENSOR_CONTRACT', checks=checks, parameter_counts=counts,
                  augmented_initial_states_exact=True, shared_initial_state_exact=True, active_shared_query=True,
                  seeded_readout_original_mean_parity=True, local_query_rng_unchanged=True,
                  partial_gallery_detached=True, optimizer_updates=0, official_test_uses=0)
    write_json(output / 'result.json', report)
    print('COMMON_OUTLET_TENSOR_PASS', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
