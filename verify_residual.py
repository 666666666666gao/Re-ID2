"""Real-batch V2 parameter/init equality and preserved DeMo anchor witness."""
import argparse
from functools import partial
import json
from pathlib import Path

import torch

from experiment_data import split_records, make_loader
from run_experiment import build, configuration, write_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['RGBNT201', 'RGBNT100', 'MSVR310'], required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    args.seed = 42
    torch.set_num_threads(4)
    cfg = configuration(args)
    _, dev, _, classes, cameras = split_records(args.data_root, args.dataset)
    args.variant = 'demo'
    baseline = build(args, cfg, classes, cameras).eval()
    args.variant = 'ordinary_residual'
    ordinary = build(args, cfg, classes, cameras).eval()
    args.variant = 'dual_residual'
    dual = build(args, cfg, classes, cameras).eval()
    osd, dsd = ordinary.state_dict(), dual.state_dict()
    assert osd.keys() == dsd.keys() and all(torch.equal(osd[k], dsd[k]) for k in osd)
    assert all(torch.equal(value, dsd[k]) for k, value in baseline.state_dict().items())
    assert sum(p.numel() for p in ordinary.parameters()) == sum(p.numel() for p in dual.parameters())
    images, _, cam, scene, _ = next(iter(make_loader(dev[:8], cfg, False, args.seed)))
    images = {key: value.cuda() for key, value in images.items()}
    kw = {'cam_label': cam.cuda(), 'view_label': scene.cuda()}
    reference = baseline(images, **kw)
    ordinary_feature = ordinary(images, **kw)
    dual_feature = dual(images, **kw)
    assert reference.shape == (len(cam), 5120)
    assert ordinary_feature.shape == dual_feature.shape == (len(cam), 5632)
    assert torch.isfinite(ordinary_feature).all() and torch.isfinite(dual_feature).all()
    route = dual.last_route
    product = route.sum(2, keepdim=True) * route.sum(1, keepdim=True)
    joint_difference = float((route - product).abs().sum((1, 2)).mean())
    assert joint_difference > 1e-8
    original_joint = dual.collaboration.joint
    dual.collaboration.joint = partial(original_joint, interaction=False)
    dual(images, **kw)
    independent = dual.last_route
    independent_error = float((independent - independent.sum(2, keepdim=True) * independent.sum(1, keepdim=True)).abs().sum((1, 2)).mean())
    assert independent_error < 1e-6
    dual.collaboration.joint = original_joint
    dual.band_scale.zero_()
    dual.frequency_scale.zero_()
    dual.collaboration.message_scale.zero_()
    anchor = dual(images, **kw)
    max_error = float((anchor[:, :5120] - reference).abs().max())
    assert torch.allclose(anchor[:, :5120], reference, atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(anchor[:, 5120:]) == 0
    assert torch.equal(ordinary_feature[:, :5120], reference)
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    receipt = {'status': 'PASS', 'dataset': args.dataset, 'batch': len(cam), 'seed': 42,
               'parameters_each': sum(p.numel() for p in ordinary.parameters()),
               'matched_full_initial_state': True, 'original_DeMo_shared_initial_state_equal': True,
               'descriptor_dim': 5632, 'ordinary_base_bitwise_equal': True,
               'dual_zero_residual_anchor_max_abs_error': max_error,
               'joint_minus_marginal_product_L1': joint_difference, 'optimizer_updates': 0,
               'psi_off_marginal_product_L1': independent_error,
               'limits': 'Eight actual dev samples, initialization/structure witness; not retrieval utility or complete dataset parity.'}
    write_json(out/'verification.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
