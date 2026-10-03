"""P0: split task gradients in shared CLIP and actual retrieval projections."""
import argparse
import hashlib
from pathlib import Path
import time

import torch

from experiment_data import make_loader, seed_all
from layers.make_loss import make_loss
from probe_axis_retrieval_utility import load_checkpoint, unchanged_files
from run_mass_experiment import write_json


def cosine(left,right):
    ln,rn=left.norm(),right.norm()
    return float(torch.dot(left,right)/(ln*rn)) if ln>0 and rn>0 else None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output);assert not output.exists()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    started=time.time()
    model,original,cfg,fit,_,_,classes,terminal,proofs=load_checkpoint(Path(args.run_dir))
    named=[(name,value) for name,value in model.named_parameters() if value.requires_grad and
        name.startswith(('BACKBONE.','modality_projection.','frequency_projection.'))]
    parameters=[value for _,value in named]
    assert named and all(any(name.startswith(prefix) for name,_ in named) for prefix in ('BACKBONE.','modality_projection.','frequency_projection.'))
    groups={'shared_CLIP':[i for i,(name,_) in enumerate(named) if name.startswith('BACKBONE.')],
        'PM':[i for i,(name,_) in enumerate(named) if name.startswith('modality_projection.')],
        'PF':[i for i,(name,_) in enumerate(named) if name.startswith('frequency_projection.')]}
    versions={name:value._version for name,value in model.state_dict().items()}
    seed_all(original.seed)
    loader=iter(make_loader(fit,cfg,True,original.seed))
    loss_fn,_=make_loss(cfg,classes)
    model.train()
    batches=[]
    # Predeclared first three training batches, identical across V5/V6 MSVR.
    # No development/test identities are used for any loss or reference.
    for batch_index in range(3):
        images,labels,cam,scene,names=next(loader)
        image_sha={key:hashlib.sha256(value.numpy().tobytes()).hexdigest() for key,value in images.items()}
        images={key:value.cuda(non_blocking=True) for key,value in images.items()}
        labels,cam,scene=labels.cuda(),cam.cuda(),scene.cuda()
        assert len(labels)==64
        with torch.autocast('cuda'):
            values=model(images,label=labels,cam_label=cam,view_label=scene)
            assert len(values)%2==1 and len(model.loss_weights)==(len(values)-1)//2
            terms=[model.loss_weights[i//2]*loss_fn(values[i],values[i+1],labels,cam) for i in range(0,len(values)-1,2)]
            losses={'base':terms[1]+sum(terms[4:]),'fused':terms[0],'M_aux':terms[2],'F_aux':terms[3]}
        assert all(torch.isfinite(value) for value in losses.values())
        gradients,per_loss={},{}
        for index,(key,loss) in enumerate(losses.items()):
            raw=torch.autograd.grad(loss*512.,parameters,retain_graph=index<3,allow_unused=True)
            # None means this loss has no path to a parameter in the existing
            # graph (e.g. V5 AUX -> PM/PF). Treat it as a mathematical zero
            # only for vector comparisons and explicitly report the disconnection.
            parts=[value.detach().float().cpu().flatten()/512. if value is not None else torch.zeros(parameter.numel())
                for value,parameter in zip(raw,parameters)]
            assert all(torch.isfinite(value).all() for value in parts)
            gradients[key]={group:torch.cat([parts[i] for i in indices]) for group,indices in groups.items()}
            per_loss[key]={group:dict(gradient_L2=float(gradients[key][group].norm()),
                disconnected_parameters=[named[i][0] for i in indices if raw[i] is None],
                connected_parameter_count=sum(raw[i] is not None for i in indices)) for group,indices in groups.items()}
            del raw,parts
        comparisons={group:dict(base_M_aux=cosine(gradients['base'][group],gradients['M_aux'][group]),
            base_F_aux=cosine(gradients['base'][group],gradients['F_aux'][group]),
            fused_M_plus_F_aux=cosine(gradients['fused'][group],gradients['M_aux'][group]+gradients['F_aux'][group]),
            base_fused=cosine(gradients['base'][group],gradients['fused'][group])) for group in groups}
        batches.append(dict(batch_index=batch_index,names=list(names),labels=labels.cpu().tolist(),cameras=cam.cpu().tolist(),
            scenes=scene.cpu().tolist(),image_tensor_sha256=image_sha,weighted_losses={key:float(value.detach()) for key,value in losses.items()},
            gradient_groups=per_loss,cosines=comparisons))
        print('SPLIT_TASK_GRADIENT_BATCH',original.dataset,batch_index,comparisons['shared_CLIP'],flush=True)
        del gradients,values,losses,terms
    unchanged_files(proofs)
    changed=[name for name,value in model.state_dict().items() if value._version!=versions[name]]
    write_json(output,dict(status='PASS_ACTUAL_SPLIT_TASK_GRADIENT_DIAGNOSTIC',dataset=original.dataset,variant=original.variant,
        selected_epoch=terminal['best']['epoch'],seed=original.seed,batch_size=64,training_batches=3,batches=batches,
        optimizer_constructed=False,optimizer_updates=0,official_test_uses=0,native_autocast=True,autograd_scale=512.,
        original_input_files=proofs,in_memory_state_versions_changed=changed,
        parameter_groups={group:dict(parameter_names=[named[i][0] for i in indices],
            scalar_count=sum(named[i][1].numel() for i in indices)) for group,indices in groups.items()},
        loss_definition='Weighted original CE+Triplet heads. Base=base-MoE plus all original global heads; fused=.25 fused head; M/F=.1 corresponding existing auxiliary head. These four terms sum to the original task loss excluding contribution calibration.',
        scope='First three seeded training batches through a fixed dev-selected checkpoint, no optimizer or weight writes. Temporary training-mode BatchNorm state is recorded and discarded.',
        limits='Three local batches describe current gradient directions, not sustained conflict during training or a causal explanation. Zero/disconnected projection gradients do not imply the projection lacks gradients from the fused loss.',
        peak_memory_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.time()-started))


if __name__=='__main__': main()
