"""Actual CUDA contracts before matched fresh50; no official-test data."""
import argparse
import gc
import json
from pathlib import Path

import torch

from dual_axis import RELATIONS
from experiment_data import make_loader, split_records
from run_mass_experiment import configuration, write_json
from run_shared_identity_experiment import PARTIAL_SETS, build
from shared_identity_axis import KEYS, available_base_fusion, metric_descriptor, partial_gallery_triplet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    args.dataset, args.variant, args.seed = 'MSVR310', 'axis_shared', 42
    torch.set_num_threads(4)
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    cfg = configuration(args)
    _, dev, _, classes, cameras = split_records(args.data_root, args.dataset)
    images, _, cam, scene, _ = next(iter(make_loader(dev[:8], cfg, False, args.seed)))
    images = {key: value.cuda() for key, value in images.items()}
    cam, scene = cam.cuda(), scene.cuda()
    reference, common, sizes, records = None, None, {}, []
    for variant in ('axis_shared', 'frequency_shared', 'twins_shared', 'demo_shared'):
        args.variant = variant
        model = build(args, cfg, classes, cameras)
        state = model.state_dict()
        if reference is None:
            reference = {name: value.cpu().clone() for name, value in state.items()}
            common = {name: value for name, value in reference.items() if name.startswith('shared_')}
        elif variant != 'demo_shared':
            assert set(reference) == set(state)
            assert all(torch.equal(value.cpu(), reference[name]) for name, value in state.items())
        else:
            assert all(torch.equal(state[name].cpu(), value) for name, value in common.items())
        sizes[variant] = {'parameters': sum(p.numel() for p in model.parameters()),
                          'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
        model.eval()
        versions = {name: value._version for name, value in state.items()}
        for retained in (*PARTIAL_SETS, (0, 1, 2)):
            partial = {key: value if index in retained else torch.zeros_like(value)
                       for index, (key, value) in enumerate(images.items())}
            with torch.no_grad():
                states = model(partial, cam_label=cam, view_label=scene, return_states=True)
            valid = torch.tensor([all(index in retained for index in subset) for subset in RELATIONS], device='cuda')
            for name, feature in states.items():
                assert feature.shape == (8, 5632) and torch.isfinite(feature).all()
                blocks = feature[:, :5120].reshape(8, 10, 512)
                assert all(torch.count_nonzero(blocks[:, index]) == 0 for index in range(3) if index not in retained)
                assert torch.count_nonzero(blocks[:, 3:][:, ~valid]) == 0
                assert torch.allclose(feature[:, 5120:].square().sum(1), torch.full((8,), .25, device='cuda'), atol=1e-6, rtol=0)
            records.append({'variant': variant, 'retained': ''.join('RNT'[i] for i in retained),
                            'descriptor_dim': 5632, 'finite': True, 'invalid_private_coordinates_zero': True,
                            'shared_metric_mass': states['11'][:, 5120:].square().sum(1).tolist(),
                            'full_vs_base_feature_change': float((states['11'] - states['00']).abs().max())})
        assert versions == {name: value._version for name, value in model.state_dict().items()}
        # Existing DeMo expert BN must not ingest zeros for invalid relations.
        fusion = model.generalFusion
        invalid = [i for i, subset in enumerate(RELATIONS) if not all(m == 0 for m in subset)]
        before = {name: value.clone() for name, value in fusion.state_dict().items()
                  if any(name.startswith('moe.experts.' + str(h) + '.expertHead.' + str(i) + '.mlp.2.')
                         for h in range(8) for i in invalid)}
        fusion.train()
        available = torch.tensor([[True, False, False]] * 8, device='cuda')
        patches = [torch.randn(8, 8, 512, device='cuda'), torch.zeros(8, 8, 512, device='cuda'), torch.zeros(8, 8, 512, device='cuda')]
        globals_ = [torch.randn(8, 512, device='cuda'), torch.zeros(8, 512, device='cuda'), torch.zeros(8, 512, device='cuda')]
        result = available_base_fusion(fusion, patches, globals_, available)
        assert torch.isfinite(result).all()
        assert before and all(torch.equal(value, fusion.state_dict()[name]) for name, value in before.items())
        del model, state, fusion, result, before, patches, globals_, states
        gc.collect()
        torch.cuda.empty_cache()
    assert sizes['axis_shared'] == sizes['frequency_shared'] == sizes['twins_shared']
    query = torch.randn(8, 12, device='cuda', requires_grad=True)
    gallery = torch.randn(8, 12, device='cuda', requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device='cuda')
    loss, pos, neg = partial_gallery_triplet(query, gallery, labels)
    loss.backward()
    assert gallery.grad is None and torch.isfinite(query.grad).all() and query.grad.abs().sum() > 0
    assert labels[pos].eq(labels).all() and labels[neg].ne(labels).all() and pos.ne(torch.arange(8, device='cuda')).all()
    # Source-disjoint private coordinates now share a measurable common metric.
    raw = torch.zeros(2, 5632, device='cuda')
    raw[0, :512], raw[1, 512:1024], raw[:, 5120:] = 1, 1, 1
    feature = metric_descriptor(raw)
    common_similarity = float((feature[0] * feature[1]).sum())
    assert abs(common_similarity - .25) < 1e-6
    report = {'status': 'PASS_SHARED_IDENTITY_TENSOR_CONTRACT', 'parameter_counts': sizes,
              'augmented_initial_states_exact': True, 'all_models_shared_initial_state_exact': True,
              'availability_checks': records, 'frozen_tensor_versions_unchanged': True,
              'invalid_expert_BN_unchanged': True, 'partial_gallery_detached': True,
              'positive_self_excluded': True, 'ground_truth_positive_negative_indices_valid': True,
              'disjoint_source_common_similarity': common_similarity,
              'optimizer_updates': 0, 'official_test_uses': 0}
    write_json(output / 'result.json', report)
    print('SHARED_IDENTITY_TENSOR_PASS', json.dumps(sizes), flush=True)


if __name__ == '__main__':
    main()
