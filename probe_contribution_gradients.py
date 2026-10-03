"""One training-mode gradient diagnostic, without optimizer or file mutation."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiment_data import make_loader, seed_all, split_records
from layers.make_loss import make_loss
from run_mass_experiment import build as build_v5, configuration, write_json
from run_projected_mass_experiment import build as build_v6


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run,output=Path(args.run_dir),Path(args.output)
    assert not output.exists()
    exit_file=run.parent/(run.name+'_exit.json')
    terminal=json.loads((run/'result.json').read_text())
    assert terminal['status']=='COMPLETE' and terminal['epochs']==50
    assert json.loads(exit_file.read_text())['exit_code']==0
    arguments=argparse.Namespace(**terminal['arguments'])
    assert arguments.dataset=='MSVR310' and arguments.seed==42
    builders={'axis_mass_fullref':build_v5,'axis_mass_projected_fullref':build_v6}
    builder=builders[arguments.variant]
    inputs=(run/'best.pth',run/'best_dev_arrays.npz',run/'result.json',exit_file)
    proofs={str(path):{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in inputs}
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    cfg=configuration(arguments)
    fit,_,_,classes,cameras=split_records(arguments.data_root,arguments.dataset)
    model=builder(arguments,cfg,classes,cameras)
    model.load_state_dict(torch.load(run/'best.pth',map_location='cuda',weights_only=True),strict=True)
    versions={name:value._version for name,value in model.state_dict().items()}
    seed_all(arguments.seed)
    images,labels,cam,scene,names=next(iter(make_loader(fit,cfg,True,arguments.seed)))
    image_sha={key:hashlib.sha256(value.numpy().tobytes()).hexdigest() for key,value in images.items()}
    assert len(labels)==64 and len(labels.unique())>1
    images={key:value.cuda(non_blocking=True) for key,value in images.items()}
    labels,cam,scene=labels.cuda(),cam.cuda(),scene.cuda()
    loss_fn,_=make_loss(cfg,classes)
    model.train()
    with torch.autocast('cuda'):
        values=model(images,label=labels,cam_label=cam,view_label=scene)
        assert len(values)%2==1
        end=len(values)-1
        assert len(model.loss_weights)==end//2
        task=sum(model.loss_weights[i//2]*loss_fn(values[i],values[i+1],labels,cam) for i in range(0,end,2))
        contribution=values[-1]
    assert torch.isfinite(task) and torch.isfinite(contribution)
    assert not model.contribution_audit['target_requires_grad']
    named=list(model.calibrator.predictor.named_parameters())
    parameters=[parameter for _,parameter in named]
    # Match native AMP's established scale without constructing an optimizer.
    scale=512.0
    task_grad=torch.autograd.grad(task*scale,parameters,retain_graph=True)
    contribution_grad=torch.autograd.grad(contribution*scale,parameters)
    task_vector=torch.cat([value.detach().float().flatten()/scale for value in task_grad])
    contribution_vector=torch.cat([value.detach().float().flatten()/scale for value in contribution_grad])
    assert torch.isfinite(task_vector).all() and torch.isfinite(contribution_vector).all()
    task_norm=float(task_vector.norm());contribution_norm=float(contribution_vector.norm())
    assert task_norm>0 and contribution_norm>0
    per_parameter=[]
    for (name,_),left,right in zip(named,task_grad,contribution_grad):
        per_parameter.append(dict(name=name,task_gradient_L2=float(left.float().norm()/scale),weighted_contribution_gradient_L2=float(right.float().norm()/scale)))
    audit=model.contribution_audit
    target=torch.tensor(audit['targets'],dtype=torch.float32)
    prediction=torch.tensor(audit['predictions'],dtype=torch.float32)
    assert target.shape==prediction.shape==(64,3)
    changed=[name for name,value in model.state_dict().items() if value._version!=versions[name]]
    for path in inputs:
        proof=proofs[str(path)]
        assert path.stat().st_size==proof['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==proof['sha256']
    write_json(output,dict(status='PASS_ACTUAL_GRADIENT_DIAGNOSTIC',dataset=arguments.dataset,variant=arguments.variant,
        selected_epoch=terminal['best']['epoch'],seed=arguments.seed,batch_size=64,names=list(names),labels=labels.cpu().tolist(),
        cameras=cam.cpu().tolist(),scenes=scene.cpu().tolist(),image_tensor_sha256=image_sha,original_input_files=proofs,
        contribution_weight=model.contribution_loss_weight,task_loss=float(task.detach()),weighted_contribution_loss=float(contribution.detach()),
        native_autocast=True,autograd_scale=scale,optimizer_updates=0,optimizer_constructed=False,official_test_uses=0,
        task_predictor_gradient_L2=task_norm,weighted_contribution_predictor_gradient_L2=contribution_norm,
        task_to_weighted_contribution_gradient_L2_ratio=task_norm/contribution_norm,
        gradient_cosine=float(torch.dot(task_vector,contribution_vector)/(task_vector.norm()*contribution_vector.norm())),
        per_parameter=per_parameter,target_mean=target.mean(0).tolist(),target_std=target.std(0,unbiased=False).tolist(),
        prediction_mean=prediction.mean(0).tolist(),prediction_std=prediction.std(0,unbiased=False).tolist(),
        prediction_target_MAE=(prediction-target).abs().mean(0).tolist(),target_requires_grad=False,
        reference=audit['reference'],positive_indices=audit['positive_indices'],negative_indices=audit['negative_indices'],
        in_memory_state_versions_changed=changed,peak_memory_bytes=torch.cuda.max_memory_allocated(),
        scope='One identical seeded training batch through each dev-selected checkpoint, task loss exactly follows existing step() without the contribution term. Gradient comparison concerns only predictor parameters. No optimizer, checkpoint write, evaluation inference or candidate training.',
        limits='Training mode updates temporary in-memory BatchNorm buffers; those buffers are discarded and original checkpoint files stay byte-identical. One-batch gradient ratios are a local diagnostic, not a convergence, generalization or causal conclusion.'))
    print('CONTRIBUTION_GRADIENT_PROBE_PASS',arguments.variant,task_norm/contribution_norm,flush=True)


if __name__=='__main__':
    main()
