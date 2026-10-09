"""Train-only diagnosis of selected M checkpoints; no optimizer or checkpoint writes."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiment_data import make_loader,seed_all
from layers.make_loss import make_loss
from layers.softmax_loss import CrossEntropyLabelSmooth
from layers.triplet_loss import TripletLoss
from official_training_data import full_records
from original_identity_anchor import assert_anchor_unchanged
from probe_mature_identity_gradient import PREFIXES,SCALE,gradient,norm,compare
from run_r201m_primary_margin06 import build,configuration

BATCHES=4


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run,out=Path(args.run_dir),Path(args.output)
    assert not out.exists()
    trained=json.loads((run/'result.json').read_text())
    assert trained['status']=='COMPLETE' and trained['epochs']==50 and trained['descriptor_dim']==5120
    assert trained['arguments']['dataset'] in ('MSVR310','RGBNT100') and trained['arguments']['variant'] in ('frequency_shared','axis_shared')
    assert trained['arguments']['seed']==42 and trained['amp_skipped_steps']==trained['training_heldout_identities']==0
    assert trained['method_revision'].startswith('R201M single final-primary margin factor')
    assert json.loads((run.parent/(run.name+'_exit.json')).read_text())['exit_code']==0
    assert json.loads((run/'normal_cpu_audit.json').read_text())['status']=='PASS'
    protected=(run/'best.pth',run/'result.json',run/'official_split_manifest.json')
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    arguments=argparse.Namespace(**trained['arguments'])
    cfg=configuration(arguments)
    assert cfg.MODEL.NO_MARGIN and cfg.MODEL.ID_LOSS_WEIGHT==.25 and cfg.MODEL.TRIPLET_LOSS_WEIGHT==1
    assert cfg.DATALOADER.NUM_INSTANCE==8 and not cfg.MODEL.DIRECT
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    train,_,_,classes,cameras,manifest=full_records(arguments.data_root,arguments.dataset)
    assert manifest==json.loads((run/'official_split_manifest.json').read_text())
    assert len(train)==trained['train_records']
    model=build(arguments,cfg,classes,cameras)
    model.load_state_dict(torch.load(run/'best.pth',map_location='cuda',weights_only=True),strict=True)
    assert not model.normal_priority_smoke
    loss_fn,_=make_loss(cfg,classes)
    xent=CrossEntropyLabelSmooth(classes);hinge=TripletLoss(margin=.6)
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad and n.split('.')[0] in PREFIXES]
    parameters=[p for _,p in named]
    groups={g:[i for i,(n,_) in enumerate(named) if n.split('.')[0]==g] for g in PREFIXES}
    assert all(groups.values()) and len(groups['residual_scale'])==1
    parameter_versions={n:p._version for n,p in model.named_parameters()}
    buffer_versions={n:p._version for n,p in model.named_buffers()}
    learned_scales=model.residual_scale.detach().float().cpu().tolist()
    seed_all(43);model.train();rows=[]
    for index,batch in enumerate(make_loader(train,cfg,True,43),1):
        images,labels,cam,scene,names=batch
        counts=Counter(labels.tolist());assert len(names)==64 and len(counts)==8 and set(counts.values())=={8}
        images={k:v.cuda(non_blocking=True) for k,v in images.items()}
        labels,cam,scene=labels.cuda(),cam.cuda(),scene.cuda()
        with torch.autocast('cuda'):
            output=model(images,label=labels,cam_label=cam,view_label=scene)
            assert len(output)==17 and output[1].shape==(64,5120) and model.loss_weights[:3]==[1.,.1,.1]
            components=dict(primary_CE=.25*xent(output[0],labels),primary_hinge=hinge(output[1],labels)[0],
                modality_aux=model.loss_weights[1]*loss_fn(output[2],output[3],labels,cam),
                frequency_aux=model.loss_weights[2]*loss_fn(output[4],output[5],labels,cam),
                other_full=sum(model.loss_weights[i//2]*loss_fn(output[i],output[i+1],labels,cam) for i in range(6,16,2)),
                contribution=output[-1])
        assert all(torch.isfinite(v) for v in components.values())
        grads={k:gradient(v,parameters,True) for k,v in components.items()}
        grads['fused_primary']=gradient(components['primary_CE']+components['primary_hinge'],parameters,True)
        grads['M_F_aux']=gradient(components['modality_aux']+components['frequency_aux'],parameters,True)
        grads['full_total']=gradient(sum(components.values()),parameters,False)
        base=F.normalize(torch.cat((output[11],output[13],output[15],output[7]),dim=1).detach().float(),dim=1)
        fused=output[1].detach().float()
        assert base.shape==fused.shape and (fused.norm(dim=1)-1).abs().max()<1e-6
        angular=(base*fused).sum(1).clamp(-1,1).acos()*180/torch.pi
        roles={}
        for group,indices in groups.items():
            roles[group]=dict(gradient_l2={k:norm(g,indices) for k,g in grads.items()},
                unused_tensors={k:sum(g[i] is None for i in indices) for k,g in grads.items()},
                primary_CE_vs_hinge=compare(grads['primary_CE'],grads['primary_hinge'],indices),
                fused_primary_vs_M_aux=compare(grads['fused_primary'],grads['modality_aux'],indices),
                fused_primary_vs_F_aux=compare(grads['fused_primary'],grads['frequency_aux'],indices),
                fused_primary_vs_M_F_aux=compare(grads['fused_primary'],grads['M_F_aux'],indices),
                fused_primary_vs_contribution=compare(grads['fused_primary'],grads['contribution'],indices))
        scale_index=groups['residual_scale'][0]
        rows.append(dict(batch=index,names=list(names),training_labels=labels.cpu().tolist(),
            losses={k:float(v.detach()) for k,v in components.items()},deploy_parameter_roles=roles,
            residual_scale_gradients={k:None if g[scale_index] is None else g[scale_index].tolist() for k,g in grads.items()},
            gates=model.last_gates.float().mean(0).cpu().tolist(),
            route_relation_mass=model.last_route.float().sum(2).mean(0).cpu().tolist(),
            route_band_mass=model.last_route.float().sum(1).mean(0).cpu().tolist(),
            normalized_descriptor_shift_l2_mean=float((fused-base).norm(dim=1).mean()),
            normalized_descriptor_angle_degrees_mean=float(angular.mean())))
        assert_anchor_unchanged(model)
        assert all(p.grad is None for p in model.parameters())
        assert parameter_versions=={n:p._version for n,p in model.named_parameters()}
        print('M_SELECTED_GRADIENT_BATCH',arguments.dataset,arguments.variant,index,flush=True)
        del output,components,grads,base,fused
        if index==BATCHES:break
    assert len(rows)==BATCHES and all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in protected)
    result=dict(status='ACTUAL_M_SELECTED_TRAIN_ONLY_FULL_TASK_GRADIENT_DIAGNOSIS',
        completed_at=datetime.now().astimezone().isoformat(timespec='seconds'),source_run=str(run),
        dataset=arguments.dataset,variant=arguments.variant,selected_epoch=trained['best']['epoch'],
        observation_seed=43,training_records=len(train),batches=BATCHES,neural_forwards=BATCHES,
        optimizer_constructed=False,optimizer_updates=0,checkpoint_writes=0,official_query_gallery_neural_uses=0,
        parameters_unchanged=True,frozen_identity_unchanged=True,protected_inputs=hashes,gradient_scale=SCALE,
        learned_residual_scales_at_load=learned_scales,selected_loss_weights=list(model.loss_weights),
        temporary_buffer_version_changes=[n for n,p in model.named_buffers() if p._version!=buffer_versions[n]],
        roles={g:[named[i][0] for i in indices] for g,indices in groups.items()},rows=rows,
        limits='Four full-view training forwards at a fixed benchmark-selected checkpoint, both actual final CE and hinge plus weighted auxiliaries. The zero-weight partial forward/BN call is not replayed here; this is a local full-objective gradient observation, not the entire two-forward optimizer step. No optimizer; temporary train-mode BN statistics are discarded and never saved; buffer-version changes are not a complete value-change detector. None/zero cosines explicitly reflect disconnected gradient paths, not a fallback. Does not establish sustained training conflict, late-checkpoint gradients, causal mechanism or new retrieval gain. Descriptor-angle measurements use normalized final-minus-base, not pre-normalization radial residual energy.')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(result['status'],flush=True)


if __name__=='__main__':main()
