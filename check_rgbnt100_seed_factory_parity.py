"""Actual native seed42 construction/RNG/four-state parity; no optimizer or metric."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert-recipe',choices=('AV','BC'),required=True)
    parser.add_argument('--variant',choices=('frequency_shared','axis_shared'),required=True)
    for key in ('data-root','pretrained','anchor-run-dir','output'):
        parser.add_argument('--'+key,required=True)
    args = parser.parse_args()
    args.dataset='RGBNT100';args.seed=42;args.freeze_identity_encoder=1;args.mode='smoke'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    parent = recipe_module(args.expert_recipe)
    cfg = parent.configuration(args)
    train,query,gallery,classes,cameras,_ = full_records(args.data_root,args.dataset)
    assert (len(train),len(query),len(gallery),classes)==(8675,1715,8575,50)
    images,_,cam,scene,_ = next(iter(make_loader(query[:8],cfg,False,42)))
    images={key:value.cuda(non_blocking=True) for key,value in images.items()}
    cam,scene=cam.cuda(),scene.cuda()
    original = parent.build(args,cfg,classes,cameras)
    original.eval();assert_anchor_unchanged(original)
    expected={name:value.detach().cpu().clone() for name,value in original.state_dict().items()}
    cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
    with torch.no_grad():
        original_states=original(images,cam_label=cam,view_label=scene,return_states=True)
    assert set(original_states)=={'00','10','01','11'}
    outputs={name:value.detach().cpu().clone() for name,value in original_states.items()}
    assert torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),cuda_rng)
    assert all(torch.equal(value.detach().cpu(),expected[name]) for name,value in original.state_dict().items())
    del original,original_states
    gc.collect();torch.cuda.empty_cache()
    seed_factory.install_fixed_teacher_expert_seed_builder()
    model=build_selected(args.expert_recipe,args,cfg,classes,cameras)
    assert set(model.state_dict())==set(expected)
    assert all(torch.equal(value.detach().cpu(),expected[name]) for name,value in model.state_dict().items())
    assert torch.equal(torch.get_rng_state(),cpu_rng) and torch.equal(torch.cuda.get_rng_state(),cuda_rng)
    assert (sum(p.numel() for p in model.parameters()),sum(p.numel() for p in model.parameters() if p.requires_grad))==PARAMETERS[args.expert_recipe]
    model.eval();assert_anchor_unchanged(model)
    with torch.no_grad():
        actual=model(images,cam_label=cam,view_label=scene,return_states=True)
    assert set(actual)==set(outputs)
    assert all(torch.equal(actual[name].cpu(),outputs[name]) for name in outputs)
    assert all(torch.equal(value.detach().cpu(),expected[name]) for name,value in model.state_dict().items())
    assert_anchor_unchanged(model)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    record=dict(status='ACTUAL_SEED42_NATIVE_CONSTRUCTOR_RNG_FOUR_STATE_PARITY_PASS',
        recipe=args.expert_recipe,variant=args.variant,teacher_seed=42,expert_seed=42,
        state_entries=len(expected),state_tensors_exact=True,CPU_CUDA_RNG_exact=True,
        four_states_exact=True,identity_reference_unchanged=True,parameters=PARAMETERS[args.expert_recipe][0],
        trainable_parameters=PARAMETERS[args.expert_recipe][1],descriptor_dim=5120,
        model_forward_batches=2,forward_record_executions=16,optimizer_updates=0,checkpoint_writes=0,
        limits='Native construction and eight-record four-state equality only, not full inference, training, a new metric, a successful seed43 run, or whole-pipeline repeat evidence.')
    (output/'result.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(record['status'],args.expert_recipe,args.variant,flush=True)


if __name__ == '__main__':
    main()
