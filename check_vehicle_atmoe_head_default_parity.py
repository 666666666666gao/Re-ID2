"""Actual head8 default parity, to be run only after the current jobs close."""
import argparse
import os
from pathlib import Path
import time

import torch

from experiment_data import make_loader
from official_training_data import full_records
from run_experiment import configuration as original_configuration, write_json
from run_full_official_experiment import build
from vehicle_atmoe_head_configuration import configuration, verify_partition


@torch.no_grad()
def outputs(model, images, cameras, scenes):
    model.eval()
    full = model(images, cam_label=cameras, view_label=scenes).float().cpu()
    with torch.autocast('cuda'):
        mixed = model(images, cam_label=cameras, view_label=scenes).float().cpu()
    return full, mixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('RGBNT100', 'MSVR310'), required=True)
    parser.add_argument('--variant', choices=('demo', 'demo_shared'), required=True)
    for key in ('data-root', 'pretrained', 'output'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    args.seed = 42
    assert int(os.environ['CUDA_VISIBLE_DEVICES']) in (2, 3)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = original_configuration(args)
    selected = configuration(args, 8)
    assert selected.dump() == cfg.dump()
    train, query, gallery, classes, cameras, _ = full_records(args.data_root, args.dataset)
    images, _, camera, scene, _ = next(iter(make_loader(query[:8], cfg, False, args.seed)))
    images = {key: value.cuda() for key, value in images.items()}
    camera, scene = camera.cuda(), scene.cuda()
    original = build(args, cfg, classes, cameras)
    verify_partition(original, 8)
    saved = {name: tensor.detach().cpu().clone() for name, tensor in original.state_dict().items()}
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    expected = outputs(original, images, camera, scene)
    del original
    configured = build(args, selected, classes, cameras)
    verify_partition(configured, 8)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    state = configured.state_dict()
    assert state.keys() == saved.keys()
    assert all(torch.equal(tensor.cpu(), saved[name]) for name, tensor in state.items())
    actual = outputs(configured, images, camera, scene)
    assert all(torch.equal(first, second) for first, second in zip(expected, actual))
    dimension = 5120 if args.variant == 'demo' else 5632
    assert expected[0].shape == expected[1].shape == (8, dimension)
    write_json(out/'result.json', dict(
        status='ACTUAL_VEHICLE_ATMOE_HEAD8_DEFAULT_CONFIG_STATE_RNG_FP32_AMP_PARITY_PASS',
        arguments=vars(args), original_head=8, configured_head=8, config_dump_exact=True,
        state_tensors_exact=True, CPU_CUDA_RNG_exact=True,
        fp32_output_exact=True, autocast_output_exact=True,
        train_records=len(train), query_records=len(query), gallery_records=len(gallery),
        parameters=sum(value.numel() for value in configured.parameters()),
        descriptor_dim=dimension, forward_record_executions=32,
        optimizer_updates=0, checkpoint_writes=0, finished=time.time()))
    print('VEHICLE_ATMOE_HEAD8_DEFAULT_PARITY_PASS', args.dataset, flush=True)


if __name__ == '__main__':
    main()
