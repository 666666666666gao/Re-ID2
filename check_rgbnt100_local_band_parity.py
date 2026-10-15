"""Actual grid1 model parity gate when admitted; no optimizer or retrieval score."""
import argparse
import gc
import json
from pathlib import Path

import torch

from experiment_data import make_loader
from official_training_data import full_records
from original_identity_anchor import assert_anchor_unchanged
import identity_coordinate_rgbnt100_expert_seeds as seed_factory
from rgbnt100_expert_seed_recipe import build_selected, recipe_module, PARAMETERS
from rgbnt100_local_band_recipe import build_local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-parent-recipe', choices=('AV', 'BC'), required=True)
    parser.add_argument('--variant', choices=('frequency_shared', 'axis_shared'), required=True)
    for key in ('data-root', 'pretrained', 'anchor-run-dir', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    args.dataset, args.seed, args.freeze_identity_encoder, args.mode = 'RGBNT100', 42, 1, 'smoke'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    key = args.local_parent_recipe
    parent = recipe_module(key)
    cfg = parent.configuration(args)
    train, query, gallery, classes, cameras, _ = full_records(args.data_root, args.dataset)
    assert (len(train), len(query), len(gallery), classes) == (8675, 1715, 8575, 50)
    images, _, cam, scene, _ = next(iter(make_loader(query[:8], cfg, False, 42)))
    images = {name: value.cuda(non_blocking=True) for name, value in images.items()}
    cam, scene = cam.cuda(), scene.cuda()
    seed_factory.install_fixed_teacher_expert_seed_builder()
    original = build_selected(key, args, cfg, classes, cameras)
    original.eval()
    assert_anchor_unchanged(original)
    expected = {name: value.detach().cpu().clone() for name, value in original.state_dict().items()}
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    outputs = {}
    for amp in (False, True):
        with torch.no_grad(), torch.autocast('cuda', enabled=amp):
            states = original(images, cam_label=cam, view_label=scene, return_states=True)
        assert set(states) == {'00', '10', '01', '11'}
        assert all(value.shape == (8, 5120) and torch.isfinite(value).all() for value in states.values())
        outputs[amp] = {name: value.detach().cpu().clone() for name, value in states.items()}
        assert torch.equal(torch.get_rng_state(), cpu_rng) and torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert all(torch.equal(value.detach().cpu(), expected[name]) for name, value in original.state_dict().items())
    assert_anchor_unchanged(original)
    del original, states
    gc.collect()
    torch.cuda.empty_cache()
    model = build_local(key, 1, args, cfg, classes, cameras)
    assert set(model.state_dict()) == set(expected)
    assert all(torch.equal(value.detach().cpu(), expected[name]) for name, value in model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), cpu_rng) and torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert (sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad)) == PARAMETERS[key]
    model.eval()
    assert_anchor_unchanged(model)
    for amp in (False, True):
        with torch.no_grad(), torch.autocast('cuda', enabled=amp):
            actual = model(images, cam_label=cam, view_label=scene, return_states=True)
        assert set(actual) == set(outputs[amp])
        assert all(torch.equal(actual[name].cpu(), outputs[amp][name]) for name in actual)
        assert torch.equal(torch.get_rng_state(), cpu_rng) and torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert all(torch.equal(value.detach().cpu(), expected[name]) for name, value in model.state_dict().items())
    assert_anchor_unchanged(model)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status='ACTUAL_LOCAL_GRID1_CONSTRUCTOR_RNG_FP32_AMP_FOUR_STATE_PARITY_PASS',
        parent_recipe=key, variant=args.variant, teacher_seed=42, expert_seed=42,
        grid=1, state_entries=len(expected), state_tensors_exact=True,
        CPU_CUDA_RNG_exact=True, fp32_four_states_exact=True, autocast_four_states_exact=True,
        identity_reference_unchanged=True, parameters=PARAMETERS[key][0],
        trainable_parameters=PARAMETERS[key][1], descriptor_dim=5120,
        model_forward_batches=4, forward_record_executions=32,
        optimizer_updates=0, checkpoint_writes=0,
        limits='Eight official query records, two constructors, FP32 and autocast evaluation only. Not whole-Q/G equality, a trained trajectory equivalence, grid2 gradient/native Adam, a retrieval score or launch authorization. No current experiment is inspected or modified.')
    (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(record['status'], key, args.variant, flush=True)


if __name__ == '__main__':
    main()
