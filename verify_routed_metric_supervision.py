"""Check unchanged inference and supervision of the actual routed PF direction."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiment_data import make_loader, seed_all, split_records
from layers.make_loss import make_loss
from run_metric_mass_experiment import build as build_metric
from run_routed_metric_experiment import build, configuration, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--pretrained',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    args.dataset='MSVR310';args.variant='axis_metric_routed_fullref';args.seed=42;args.contribution_weight=.05
    output=Path(args.output);output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    cfg=configuration(args)
    fit,dev,_,classes,cameras=split_records(args.data_root,args.dataset)
    baseline_args=argparse.Namespace(**{**vars(args),'variant':'axis_metric_fullref'})
    baseline=build_metric(baseline_args,cfg,classes,cameras)
    model=build(args,cfg,classes,cameras)
    old,new=baseline.state_dict(),model.state_dict()
    assert set(old)==set(new) and all(torch.equal(old[name],new[name]) for name in old)
    parameters=sum(p.numel() for p in model.parameters())
    assert parameters==sum(p.numel() for p in baseline.parameters())
    assert model.loss_weights[:5]==[.25,1.,.1,.05,.05]
    assert model.loss_weights[5:]==baseline.loss_weights[4:]
    assert sum(model.loss_weights[3:5])==baseline.loss_weights[3]==.1
    images,_,cam,scene,_=next(iter(make_loader(dev[:8],cfg,False,args.seed)))
    images={key:value.cuda() for key,value in images.items()}
    baseline.eval();model.eval()
    with torch.no_grad():
        before=baseline(images,cam_label=cam.cuda(),view_label=scene.cuda())
        after=model(images,cam_label=cam.cuda(),view_label=scene.cuda())
    assert before.shape==after.shape==(8,5632) and torch.equal(before,after)
    del old,new,baseline,before,after,images
    torch.cuda.empty_cache()
    seed_all(args.seed)
    images,target,cam,scene,_=next(iter(make_loader(fit,cfg,True,args.seed)))
    images={key:value.cuda() for key,value in images.items()}
    target,cam,scene=target.cuda(),cam.cuda(),scene.cuda()
    capture=[]
    hook=model.frequency_projection.register_forward_hook(lambda module,inputs,result:capture.append(result))
    versions={name:value._version for name,value in model.state_dict().items()}
    model.train()
    with torch.autocast('cuda'):
        values=model(images,label=target,cam_label=cam,view_label=scene)
    hook.remove()
    routed_score,routed_f=values[8:10]
    assert routed_f.shape==(64,512) and routed_score.shape==(64,classes)
    # First PF call is full11's actual frequency projection in inherited fuse.
    direction_error=float((F.normalize(capture[0].float(),dim=1)-routed_f).abs().max())
    expected_score=32.*F.linear(routed_f,F.normalize(model.frequency_classifier.weight.float(),dim=1))
    assert direction_error==0 and torch.equal(expected_score,routed_score)
    assert torch.isfinite(routed_score).all() and torch.isfinite(routed_f).all()
    assert torch.allclose(routed_f.norm(dim=1),torch.ones(64,device='cuda'),atol=1e-6,rtol=0)
    loss_fn,_=make_loss(cfg,classes)
    with torch.autocast('cuda'):
        routed_loss=.05*loss_fn(routed_score,routed_f,target,cam)
    pf=list(model.frequency_projection.parameters())
    gradients=torch.autograd.grad(routed_loss,pf)
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients)
    changed=[name for name,value in model.state_dict().items() if value._version!=versions[name]]
    write_json(output/'result.json',dict(status='PASS_ROUTED_METRIC_SUPERVISION_TENSOR_CONTRACT',parameters=parameters,
        descriptor_dim=5632,initial_state_exactly_equal=True,initial_eval_exactly_equal=True,
        F_auxiliary_weights=model.loss_weights[3:5],F_auxiliary_total=.1,cosine_classifier_scale=32.,
        actual_full11_PF_direction_max_error=direction_error,routed_PF_gradient_norms=[float(g.norm()) for g in gradients],
        temporary_training_state_version_changes=changed,temporary_training_state_discarded=True,
        optimizer_updates=0,official_test_uses=0))
    print('ROUTED_METRIC_SUPERVISION_TENSOR_PASS',json.dumps(dict(direction_error=direction_error,parameters=parameters)),flush=True)


if __name__=='__main__':main()
