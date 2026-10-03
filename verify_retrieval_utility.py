"""Check unchanged descriptors and genuine stopped-reference gain gradients."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F

from experiment_data import make_loader, split_records
from run_relation_frequency_experiment import build as build_b
from run_retrieval_utility_experiment import build, configuration, write_json
from retrieval_utility_axis import joint_retrieval_gain


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--pretrained',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    args.dataset='MSVR310';args.variant='axis_retrieval_utility_fullref';args.seed=42;args.contribution_weight=.05
    output=Path(args.output);output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    cfg=configuration(args)
    fit,dev,_,classes,cameras=split_records(args.data_root,args.dataset)
    baseline=build_b(argparse.Namespace(**{**vars(args),'variant':'axis_relation_frequency_fullref'}),cfg,classes,cameras)
    model=build(args,cfg,classes,cameras)
    old,new=baseline.state_dict(),model.state_dict()
    assert set(old)==set(new) and all(torch.equal(old[key],new[key]) for key in old)
    assert model.loss_weights==baseline.loss_weights
    parameters=sum(p.numel() for p in model.parameters())
    assert parameters==sum(p.numel() for p in baseline.parameters())
    images,_,cam,scene,_=next(iter(make_loader(dev[:8],cfg,False,args.seed)))
    images={key:value.cuda() for key,value in images.items()};cam,scene=cam.cuda(),scene.cuda()
    baseline.eval();model.eval()
    versions={key:value._version for key,value in model.state_dict().items()}
    with torch.no_grad():
        before=baseline(images,cam_label=cam,view_label=scene)
        after=model(images,cam_label=cam,view_label=scene)
    assert torch.equal(before,after) and after.shape==(8,5632) and after[:,-512:].eq(0).all()
    assert versions=={key:value._version for key,value in model.state_dict().items()}
    images,labels,cam,scene,_=next(iter(make_loader(fit,cfg,True,args.seed)))
    images={key:value.cuda() for key,value in images.items()};labels,cam,scene=labels.cuda(),cam.cuda(),scene.cuda()
    baseline.train();model.train()
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    with torch.no_grad(),torch.autocast('cuda'):
        old_output=baseline(images,label=labels,cam_label=cam,view_label=scene)
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
    with torch.no_grad(),torch.autocast('cuda'):
        new_output=model(images,label=labels,cam_label=cam,view_label=scene)
    assert len(old_output)==len(new_output) and all(torch.equal(a,b) for a,b in zip(old_output[:-1],new_output[:-1]))
    assert torch.isclose(new_output[-1].float()-old_output[-1].float(),torch.tensor(model.gain_weight*model.gain_audit['loss'],device='cuda'),atol=1e-7,rtol=1e-5)
    assert not hasattr(model,'_gain_states')
    # Actual differentiable residual coordinates, with a separate base leaf.
    base=torch.randn(8,32,device='cuda',requires_grad=True)
    delta=(.001*torch.randn_like(base)).detach().requires_grad_()
    labels=torch.arange(4,device='cuda').repeat_interleave(2)
    states={'00':base,'10':base,'01':base,'11':base+delta}
    loss,audit=joint_retrieval_gain(states,labels)
    grad_base,grad_delta=torch.autograd.grad(loss,(base,delta))
    assert torch.isfinite(loss) and loss>0
    assert grad_base.eq(0).all() and torch.isfinite(grad_delta).all() and grad_delta.abs().sum()>0
    with torch.no_grad():
        reference=F.normalize(base,dim=1);similarity=reference@reference.T
        same=labels[:,None].eq(labels[None]);positive=same & ~torch.eye(8,device='cuda',dtype=torch.bool)
        pos=similarity.masked_fill(~positive,torch.inf).argmin(1)
        neg=similarity.masked_fill(same,-torch.inf).argmax(1)
        direction=reference[pos]-reference[neg]
        r00=(reference*direction).sum(1);r11=(F.normalize(base+delta,dim=1)*direction).sum(1)
        weights=(-r00/.1).sigmoid();expected=(weights*(r00+.02-r11).relu()).sum()/weights.sum()
        assert torch.equal(loss.detach(),expected) and audit['positive_indices']==pos.tolist() and audit['negative_indices']==neg.tolist()
    write_json(output/'result.json',dict(status='PASS_RETRIEVAL_UTILITY_CONTRACT',parameters=parameters,
        descriptor_dim=5632,active_descriptor_dim=5120,initial_all_state_and_loss_weights_equal_B=True,
        eval_descriptor_bitwise_equal_B=True,train_all_original_output_pairs_bitwise_equal_B=True,
        only_new_scalar_gain_added=True,direct_base_gain_gradient_exact_zero=True,
        actual_residual_gain_gradient_finite_nonzero=True,legal_fixed_reference_indices_exact=True,
        no_retained_gain_graph=True,optimizer_updates=0,official_test_uses=0,gain_sanity=audit))
    print('RETRIEVAL_UTILITY_TENSOR_PASS',flush=True)


if __name__=='__main__':main()
