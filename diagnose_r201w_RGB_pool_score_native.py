"""Observe the exact failed W RGB ordinary native3; no formal training or rescue PASS."""
import argparse
import io
import json
from pathlib import Path
import time

import torch

from experiment_data import make_loader, seed_all
from layers.make_loss import make_loss
from layers.softmax_loss import CrossEntropyLabelSmooth
from official_training_data import full_records
from original_identity_anchor import assert_anchor_unchanged
import run_r201w_reduced_auxiliary as w
from run_shared_identity_experiment import PARTIAL_SETS
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler


def gradient_record(parameter):
    gradient=parameter.grad
    return dict(present=gradient is not None,
                finite=bool(gradient is not None and torch.isfinite(gradient).all()),
                l1=float(gradient.detach().float().abs().sum()) if gradient is not None else None,
                norm=float(gradient.detach().float().norm()) if gradient is not None else None,
                nonzero_elements=int(torch.count_nonzero(gradient)) if gradient is not None else None)


def main():
    parser=argparse.ArgumentParser()
    for name in ('data-root','pretrained','anchor-run-dir','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    args.dataset='RGBNT100';args.variant='frequency_shared';args.freeze_identity_encoder=1;args.seed=42;args.mode='smoke'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=w.configuration(args)
    train,query,gallery,classes,cameras,manifest=full_records(args.data_root,args.dataset)
    model=w.build(args,cfg,classes,cameras)
    loss_fn,center=make_loss(cfg,classes)
    optimizer,_=make_optimizer(cfg,model,center)
    scheduler=create_scheduler(cfg,optimizer)
    scaler=torch.amp.GradScaler('cuda',init_scale=512)
    xent=CrossEntropyLabelSmooth(classes)
    parameter=model.frequency_expert.pool_score.weight
    forward_calls=[];backward_calls=[];local_vjps=[];projected={}
    context={};original_scale=scaler.scale

    def tagged_scale(outputs):
        context['scale_calls']+=1
        assert context['scale_calls'] in (1,2)
        context['phase']='scaled_full_backward' if context['scale_calls']==1 else 'scaled_partial_zero_backward'
        return original_scale(outputs)

    scaler.scale=tagged_scale

    def event(kind):
        context['event_id']+=1
        return dict(event=kind,event_id=context['event_id'],phase=context['phase'],
            scale=float(scaler.get_scale()),pool_backward_id=context['pool_backward_id'])

    def observe_forward(module,inputs,output):
        value=inputs[0].detach().float();scores=output.detach().float()
        forward_calls.append(dict(input_shape=list(value.shape),input_dtype=str(inputs[0].dtype),output_dtype=str(output.dtype),
            candidate_spread_max=float((value-value.mean(-2,keepdim=True)).abs().max()),
            score_spread_max=float((scores-scores.mean(-2,keepdim=True)).abs().max()),finite=bool(torch.isfinite(scores).all())))
        if value.ndim==3 and value.shape[1]==7:
            projected['Z']=value
            projected['scorer_weight']=module.weight.detach().float().clone()

    def observe_backward(gradient):
        value=gradient.detach().float()
        backward_calls.append(dict(**event('native_scorer_weight_gradient'),dtype=str(gradient.dtype),finite=bool(torch.isfinite(value).all()),l1=float(value.abs().sum()),norm=float(value.norm()),nonzero_elements=int(torch.count_nonzero(value))))

    def observe_neck(module,inputs):
        pooled=inputs[0];assert pooled.shape==(64,512) and pooled.requires_grad
        z=projected['Z'];weight=projected['scorer_weight']
        def local_fp32_vjp(incoming):
            context['pool_backward_id']+=1
            tag=event('same_Z_weight_incoming_FP32_local_VJP')
            with torch.enable_grad(),torch.autocast('cuda',enabled=False):
                probe=weight.detach().clone().requires_grad_(True)
                scores=torch.nn.functional.linear(z,probe).squeeze(-1)
                feature=(z*scores.softmax(1)[...,None]).sum(1)
                derivative=torch.autograd.grad(feature,probe,grad_outputs=incoming.detach().float())[0].detach()
            local_vjps.append(dict(**tag,incoming_dtype=str(incoming.dtype),incoming_norm=float(incoming.detach().float().norm()),
                fp32_weight_VJP_finite=bool(torch.isfinite(derivative).all()),fp32_weight_VJP_l1=float(derivative.abs().sum()),
                fp32_weight_VJP_norm=float(derivative.norm()),fp32_weight_VJP_nnz=int(torch.count_nonzero(derivative)),
                same_Z_weight_upstream=True,updates=0))
        pooled.register_hook(local_fp32_vjp)

    handles=[model.frequency_expert.pool_score.register_forward_hook(observe_forward),parameter.register_hook(observe_backward),model.frequency_neck.register_forward_pre_hook(observe_neck)]
    seed_all(args.seed);model.train();details=[];attempts=[]
    for i,batch in enumerate(make_loader(train,cfg,True,args.seed)):
        context.clear();context.update(phase='unscaled_auxiliary_autograd_probes',scale_calls=0,event_id=0,pool_backward_id=0)
        forward_calls.clear();backward_calls.clear();local_vjps.clear();projected.clear();before=parameter.detach().float().clone()
        detail=w.step(model,batch,optimizer,scaler,loss_fn,xent,PARTIAL_SETS[i%6],args.variant)
        assert context['scale_calls']==2
        assert_anchor_unchanged(model)
        gradients={name:gradient_record(p) for name,p in model.named_parameters() if p.requires_grad}
        row=dict(attempt=i+1,detail=detail,gradients=gradients,frequency_pool_score_forward=list(forward_calls),frequency_pool_score_scaled_backward=list(backward_calls),
            frequency_pool_score_parameter_delta_l1=float((parameter.detach().float()-before).abs().sum()),AMP_scale=float(scaler.get_scale()),
            same_input_weight_upstream_FP32_local_VJPs=list(local_vjps))
        attempts.append(row);details.append(detail)
        (out/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
        if sum(d['optimizer_updated'] for d in details)==3:break
    assert sum(d['optimizer_updated'] for d in details)==3
    for handle in handles:handle.remove()
    last=attempts[-1]['gradients']
    rejected=[name for name,record in last.items() if not(record['present'] and record['finite'] and record['l1']>0)]
    model.eval()
    images,_,cam,scene,_=next(iter(make_loader(query[:8],cfg,False,args.seed)))
    images={key:value.cuda() for key,value in images.items()}
    with torch.no_grad():before=model(images,cam_label=cam.cuda(),view_label=scene.cuda())
    buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
    model.load_state_dict(torch.load(buffer,map_location='cuda',weights_only=True),strict=True)
    with torch.no_grad():after=model(images,cam_label=cam.cuda(),view_label=scene.cuda())
    assert torch.equal(before,after) and before.shape==(8,5120) and torch.isfinite(before).all()
    record=dict(status='DIAGNOSTIC_COMPLETE_NOT_NATIVE_ACCEPTANCE',finished=time.time(),arguments=vars(args),official_split_manifest=manifest,
        parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        successful_updates=3,attempts=len(attempts),final_native_rejected_parameters=rejected,last_gradient_predicate_passed=not rejected,formal_training_admitted=False,
        strict_reload_equal=True,descriptor_shape=list(before.shape),source_factor='Exact W RGB ordinary native3 plus observation-only hooks/gradient snapshots; no model/loss/precision/optimizer/seed change',
        limitation='Finite/zero histories explain this controlled replay only; no formal50, benchmark selection, new model gain, or automatic rescue pass.',weights_saved=0)
    (out/'diagnostic.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
