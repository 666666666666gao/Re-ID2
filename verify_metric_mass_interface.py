"""Actual tensor contract for the single-factor metric interface."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiment_data import make_loader, split_records
from run_mass_experiment import build as build_v5, configuration, write_json
from run_metric_mass_experiment import build


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--pretrained',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    args.dataset='MSVR310';args.variant='axis_metric_fullref';args.seed=42;args.contribution_weight=.05
    output=Path(args.output);output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    cfg=configuration(args)
    _,dev,_,classes,cameras=split_records(args.data_root,args.dataset)
    baseline_args=argparse.Namespace(**{**vars(args),'variant':'axis_mass_fullref'})
    baseline=build_v5(baseline_args,cfg,classes,cameras)
    model=build(args,cfg,classes,cameras)
    old,new=baseline.state_dict(),model.state_dict()
    assert set(old)==set(new)
    changed=[name for name in old if not torch.equal(old[name],new[name])]
    assert changed==['residual_scale'] and torch.equal(old['residual_scale'][[0,2]],new['residual_scale'][[0,2]])
    assert sum(p.numel() for p in baseline.parameters())==sum(p.numel() for p in model.parameters())
    # The F scalar does not enter the original identity coordinates or gates.
    # Compare those coordinates exactly, then check the new frequency rule.
    images,_,cam,scene,_=next(iter(make_loader(dev[:8],cfg,False,args.seed)))
    images={key:value.cuda() for key,value in images.items()}
    baseline.eval();model.eval()
    capture={}
    hook=model.calibrator.register_forward_hook(lambda module,inputs,result:capture.update(gates=result[1]))
    with torch.no_grad():
        before=baseline(images,cam_label=cam.cuda(),view_label=scene.cuda())
        after=model(images,cam_label=cam.cuda(),view_label=scene.cuda())
    hook.remove()
    assert before.shape==after.shape==(8,5632)
    assert torch.equal(before[:,:5120],after[:,:5120])
    weights=model.residual_scale[1].sigmoid()*capture['gates'][:,1]
    normalized=F.normalize(after.float(),dim=1)
    measured=normalized[:,-512:].square().sum(1)
    error=float((measured-weights).abs().max())
    assert error<1e-7 and torch.isfinite(after).all() and (weights>0).all() and (weights<1).all()
    write_json(output/'result.json',dict(status='PASS_METRIC_INTERFACE_TENSOR_CONTRACT',parameters=sum(p.numel() for p in model.parameters()),
        descriptor_dim=5632,initial_state_changed_only_F_scalar=True,identity_coordinates_exactly_equal=True,
        initial_frequency_budget=float(model.residual_scale[1].sigmoid()),metric_weight_max_error=error,
        actual_sample_weights=weights.tolist(),optimizer_updates=0,official_test_uses=0))
    print('METRIC_INTERFACE_TENSOR_PASS',json.dumps(dict(error=error,weights=weights.tolist())),flush=True)


if __name__=='__main__':main()
