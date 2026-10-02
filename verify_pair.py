"""Actual common-initialization, routing, spectral and compute witnesses."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity
from run_experiment import configuration, build, write_json
from experiment_data import split_records, make_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--pretrained', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    args.seed = 42
    cfg = configuration(args)
    fit, dev, queries, classes, cameras = split_records(args.data_root, args.dataset)
    args.variant = 'ordinary'
    ordinary = build(args, cfg, classes, cameras).eval()
    args.variant = 'dual'
    dual = build(args, cfg, classes, cameras).eval()
    a, b = ordinary.state_dict(), dual.state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    count = sum(p.numel() for p in ordinary.parameters())
    assert count == sum(p.numel() for p in dual.parameters())
    patches = torch.randn(2, 3, 128, 512, device='cuda')
    bands = dual.bands(patches)
    assert torch.allclose(bands.sum(2), patches, atol=2e-6)
    with torch.no_grad():
        values, _ = dual.frequency(bands)
        zero_high = dual.frequency.values[2](torch.zeros_like(bands[:, :, 2]))
    high_response = float((values[2].mean(2) - zero_high.mean(2)).abs().max())
    assert high_response > 0
    u, v = torch.randn(2, 7, 3, 512, device='cuda'), torch.randn(2, 7, 3, 512, device='cuda')
    with torch.no_grad():
        _, _, independent, _ = dual.collaboration.joint(u, v, interaction=False)
        _, _, joint, _ = dual.collaboration.joint(u, v)
    product = independent.sum(2)[:, :, None] * independent.sum(1)[:, None, :]
    independent_error = float((independent - product).abs().max())
    joint_error = float((joint - joint.sum(2)[:, :, None] * joint.sum(1)[:, None, :]).abs().max())
    assert independent_error < 1e-6 and joint_error > 1e-6
    images, _, cam, scene, _ = next(iter(make_loader(dev[:8], cfg, False, 42)))
    images = {k: x.cuda() for k, x in images.items()}
    cam, scene = cam.cuda(), scene.cuda()
    compute = {}
    for name, model in [('ordinary', ordinary), ('dual', dual)]:
        with torch.no_grad():
            for _ in range(2):
                output = model(images, cam_label=cam, view_label=scene)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(5):
                output = model(images, cam_label=cam, view_label=scene)
            torch.cuda.synchronize()
            seconds = (time.time() - start) / (5 * len(cam))
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
                output = model(images, cam_label=cam, view_label=scene)
        assert output.shape == (len(cam), 5632) and torch.isfinite(output).all()
        compute[name] = {'fp32_inference_seconds_per_triplet_batch8': seconds,
                         'profiler_supported_flops_per_triplet': sum(e.flops for e in prof.key_averages()) / len(cam),
                         'flops_boundary': 'PyTorch profiler supported operators only; FFT and some attention operations may be omitted'}
    result = {'status': 'PAIR_PASS', 'dataset': args.dataset, 'parameters_each': count, 'descriptor_dim_each': 5632,
              'all_initial_state_tensors_equal': True, 'band_reconstruction_max_error': float((bands.sum(2) - patches).abs().max()),
              'high_band_input_dependent_pool_max_difference': high_response, 'independent_route_product_max_error': independent_error,
              'joint_route_product_max_error': joint_error, 'compute': compute}
    write_json(Path(args.output) / 'pair.json', result)
    print('PAIR_PASS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
