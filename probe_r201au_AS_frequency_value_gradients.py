"""Selected AS full-objective F value-Linear gradients; no optimizer or scores."""
import argparse,json,math
from collections import Counter
from datetime import datetime
from pathlib import Path
import torch

from experiment_data import seed_all
from layers.make_loss import make_loss
from layers.softmax_loss import CrossEntropyLabelSmooth
from layers.triplet_loss import TripletLoss
from official_training_data import full_records
from original_identity_anchor import assert_anchor_unchanged
from probe_mature_identity_gradient import compare,norm
from run_r201ac_RGB_hard_identity import make_loader
from run_r201as_RGB_query_normalization import build,configuration

SCALE=65536.
TARGETS=tuple(f'frequency_expert.values.{b}.1.weight' for b in range(3))+('modality_expert.value.1.weight',)


def gradient(loss,parameters,retain):
    values=torch.autograd.grad(SCALE*loss,parameters,retain_graph=retain,allow_unused=True)
    assert all(x is None or torch.isfinite(x).all() for x in values)
    return [None if x is None else x.detach().float().cpu()/SCALE for x in values]


def main():
    parser=argparse.ArgumentParser()
    for key in ('run-dir','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--batches',type=int,choices=(1,4),required=True)
    args=parser.parse_args();run,out=Path(args.run_dir),Path(args.output);assert not out.exists()
    trained=json.loads((run/'result.json').read_text());arguments=argparse.Namespace(**trained['arguments'])
    assert trained['status']=='COMPLETE' and trained['epochs']==50 and trained['optimizer_steps']==6563
    assert arguments.dataset=='RGBNT100' and arguments.variant in ('frequency_shared','axis_shared') and arguments.seed==42
    assert trained['best']['epoch']==(23 if arguments.variant=='axis_shared' else 22)
    assert trained['descriptor_dim']==5120 and trained['amp_skipped_steps']==trained['training_heldout_identities']==0
    assert json.loads((run/'normal_cpu_audit.json').read_text())['status']=='PASS'
    cfg=configuration(arguments);assert cfg.SOLVER.OPTIMIZER_NAME=='Adam' and cfg.SOLVER.WEIGHT_DECAY==1e-4
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    train,_,_,classes,cameras,manifest=full_records(arguments.data_root,'RGBNT100')
    assert len(train)==8675 and classes==50 and manifest==json.loads((run/'official_split_manifest.json').read_text())
    seed_all(42);model=build(arguments,cfg,classes,cameras)
    named=dict(model.named_parameters());initial={n:float(named[n].detach().float().norm()) for n in TARGETS}
    model.load_state_dict(torch.load(run/'best.pth',map_location='cuda',weights_only=True),strict=True)
    assert not model.normal_priority_smoke and model.loss_weights[:3]==[1.,.001,.001]
    parameters=[named[n] for n in TARGETS];assert all(p.requires_grad and p.shape==(64,512) for p in parameters)
    weights=[p.detach().float().cpu().clone() for p in parameters]
    versions={n:p._version for n,p in model.named_parameters()}
    loss_fn,_=make_loss(cfg,classes);xent=CrossEntropyLabelSmooth(classes);hinge=TripletLoss(margin=.6)
    seed_all(43);model.train();rows=[]
    for index,batch in enumerate(make_loader(train,cfg,True,43),1):
        images,labels,cam,scene,names=batch;counts=Counter(labels.tolist())
        assert len(names)==64 and len(counts)==8 and set(counts.values())=={8}
        images={k:x.cuda(non_blocking=True) for k,x in images.items()};labels,cam,scene=labels.cuda(),cam.cuda(),scene.cuda()
        with torch.autocast('cuda'):
            output=model(images,label=labels,cam_label=cam,view_label=scene)
            assert len(output)==17 and output[1].shape==(64,5120)
            m,f=model.projected_auxiliary_metric_features;assert m.shape==f.shape==(64,3584)
            components=dict(primary_CE=.25*xent(output[0],labels),primary_hinge=hinge(output[1],labels)[0],
                modality_aux=.001*loss_fn(output[2],m,labels,cam),frequency_aux=.001*loss_fn(output[4],f,labels,cam),
                other_full=sum(model.loss_weights[i//2]*loss_fn(output[i],output[i+1],labels,cam) for i in range(6,16,2)),contribution=output[-1])
        assert all(torch.isfinite(x) for x in components.values())
        grads={k:gradient(x,parameters,True) for k,x in components.items()}
        grads['fused_primary']=gradient(components['primary_CE']+components['primary_hinge'],parameters,True)
        grads['M_F_aux']=gradient(components['modality_aux']+components['frequency_aux'],parameters,True)
        grads['full_total']=gradient(sum(components.values()),parameters,False)
        assert all(x is not None for x in grads['full_total'])
        l2=[1e-4*w for w in weights];targets={}
        for i,name in enumerate(TARGETS):
            task=grads['full_total'][i];task_norm=norm(grads['full_total'],[i]);decay_norm=norm(l2,[i])
            targets[name]=dict(selected_weight_norm=norm(weights,[i]),fresh_build_seed42_weight_norm=initial[name],
                existing_coupled_Adam_weight_decay=1e-4,L2_gradient_norm=decay_norm,
                component_gradient_norm={k:norm(g,[i]) for k,g in grads.items()},
                component_disconnected={k:g[i] is None for k,g in grads.items()},
                full_task_vs_L2=compare(grads['full_total'],l2,[i]),
                primary_vs_F_aux=compare(grads['fused_primary'],grads['frequency_aux'],[i]),
                full_task_vs_weight=compare(grads['full_total'],weights,[i]),
                L2_to_full_task_norm_ratio=None if task_norm==0 else decay_norm/task_norm,
                full_task_gradient_nonzero_elements=int(torch.count_nonzero(task)),total_elements=task.numel())
        rows.append(dict(batch=index,names=list(names),training_labels=labels.cpu().tolist(),losses={k:float(x.detach()) for k,x in components.items()},targets=targets))
        assert_anchor_unchanged(model)
        assert all(p.grad is None for p in model.parameters()) and versions=={n:p._version for n,p in model.named_parameters()}
        print('AU_TRAIN_ONLY_VALUE_GRADIENT_BATCH',arguments.variant,index,flush=True)
        del output,components,grads,m,f
        if index==args.batches:break
    assert len(rows)==args.batches and len(model.modality_expert.attention.projected_query_unit_audit)==42*args.batches
    result=dict(status='SMOKE_PASS' if args.batches==1 else 'COMPLETE',completed_at=datetime.now().astimezone().isoformat(timespec='seconds'),
        source_run=str(run),variant=arguments.variant,dataset='RGBNT100',selected_epoch=trained['best']['epoch'],training_records=8675,observation_seed=43,
        batches=args.batches,neural_forwards=args.batches,gradient_scale=SCALE,targets=list(TARGETS),rows=rows,
        optimizer_constructed=False,optimizer_updates=0,checkpoint_writes=0,official_query_gallery_neural_uses=0,parameters_unchanged=True,frozen_identity_unchanged=True,
        limits='Four sampled training batches per fixed selected AS control, not full-dataset gradient coverage. Same actual full objectives/AMP/augmented hardP8K8, but no zero-weight partial forward or full optimizer replay. Scale65536 is a fixed diagnostic gradient scale, not the saved dynamic GradScaler state; finite gradients checked. Temporary train-mode new BN statistics discarded, no saved parameter changes. Existing coupled Adam L2 is wd*selected weight; norm/cosine is not an actual Adam update or sustained cause. Fresh-build norm uses explicit seed42 and current build, not claimed byte-identical initial formal optimizer state. None ratios/cosines denote actual zero/disconnected gradients. Spatial attenuation and these local gradients do not prove erased identity information or predict retrieval gains.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(result['status'],flush=True)


if __name__=='__main__':main()
