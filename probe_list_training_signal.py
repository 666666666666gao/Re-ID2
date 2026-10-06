"""Fresh anchor training observations; feature-gradient diagnosis, zero updates."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import torch

from experiment_data import make_loader, seed_all
from layers.softmax_loss import CrossEntropyLabelSmooth
from list_retrieval_objective import smooth_ap
from official_training_data import full_records
from original_identity_anchor import assert_anchor_unchanged
from run_experiment import configuration
from run_r201e_list_objective import build

TEMPERATURES = (.01,.02,.05,.1)
BATCHES = 16
GRADIENT_SCALE = 512.


def stats(gradient):
    assert torch.isfinite(gradient).all()
    return dict(l1=float(gradient.abs().sum()),l2=float(gradient.norm()),
                maximum=float(gradient.abs().max()),nonzero=int(torch.count_nonzero(gradient)))


def main():
    parser = argparse.ArgumentParser()
    for name in ('data-root','pretrained','anchor-run-dir','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--variant',choices=('frequency_shared','axis_shared'),required=True)
    args = parser.parse_args()
    path = Path(args.output)
    assert not path.exists()
    args.dataset,args.seed,args.freeze_identity_encoder,args.mode = 'RGBNT201',42,1,'diagnostic'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    cfg = configuration(args)
    train,query,gallery,classes,cameras,_ = full_records(args.data_root,args.dataset)
    assert (len(train),len(query),len(gallery))==(3951,836,836)
    model = build(args,cfg,classes,cameras)
    xent = CrossEntropyLabelSmooth(classes)
    seed_all(43)
    model.train()
    rows = []
    for index,batch in enumerate(make_loader(train,cfg,True,43),1):
        images,target,cam,scene,names = batch
        assert len(names)==len(set(names))==64
        images = {key:value.cuda() for key,value in images.items()}
        target,cam,scene = target.cuda(),cam.cuda(),scene.cuda()
        with torch.autocast('cuda'):
            output = model(images,label=target,cam_label=cam,view_label=scene)
            feature = output[1]
            ce = .25*xent(output[0],target)
        assert feature.dtype==torch.float32 and feature.shape==(64,5120)
        with torch.autocast('cuda',enabled=False):
            ce_gradient = torch.autograd.grad(GRADIENT_SCALE*ce,feature,retain_graph=True)[0]/GRADIENT_SCALE
            unit = torch.nn.functional.normalize(feature,dim=1)
            scores = unit @ unit.T
            same = target[:,None].eq(target[None,:])
            eye = torch.eye(64,device=target.device,dtype=torch.bool)
            positive,negative = same & ~eye, ~same
            assert positive.sum(1).eq(7).all() and negative.sum(1).eq(56).all()
            margin = scores.masked_fill(~positive,torch.inf).min(1).values-scores.masked_fill(~negative,-torch.inf).max(1).values
            ce_stats = stats(ce_gradient)
            assert ce_stats['l2']>0
            values = []
            for temperature in TEMPERATURES:
                ap = smooth_ap(feature,target,temperature=temperature)
                gradient = torch.autograd.grad(GRADIENT_SCALE*ap,feature,retain_graph=True)[0]/GRADIENT_SCALE
                grad_stats = stats(gradient)
                values.append(dict(temperature=temperature,loss=float(ap.detach()),gradient=grad_stats,
                    AP_to_primary_CE_gradient_l2_ratio=grad_stats['l2']/ce_stats['l2'],
                    gradient_dot_primary_CE=float((gradient*ce_gradient).sum())))
        assert_anchor_unchanged(model)
        assert all(p.grad is None for p in model.parameters())
        rows.append(dict(batch=index,names=list(names),training_labels=target.tolist(),
            camera_labels=cam.tolist(),identity_counts=torch.unique(target,return_counts=True)[1].tolist(),
            primary_CE_loss=float(ce.detach()),primary_CE_gradient=ce_stats,grid=values,
            hard_positive_minus_negative_cosine_margin=dict(minimum=float(margin.min()),mean=float(margin.mean()),maximum=float(margin.max()))))
        if index==BATCHES:
            break
    assert len(rows)==BATCHES
    aggregate = []
    for i,temperature in enumerate(TEMPERATURES):
        values = [row['grid'][i] for row in rows]
        aggregate.append(dict(temperature=temperature,mean_AP_loss=math.fsum(v['loss'] for v in values)/BATCHES,
            exact_zero_displayed_losses=sum(v['loss']==0 for v in values),
            zero_feature_gradients=sum(v['gradient']['l2']==0 for v in values),
            mean_AP_to_primary_CE_gradient_l2_ratio=math.fsum(v['AP_to_primary_CE_gradient_l2_ratio'] for v in values)/BATCHES,
            maximum_AP_to_primary_CE_gradient_l2_ratio=max(v['AP_to_primary_CE_gradient_l2_ratio'] for v in values)))
    result = dict(status='ACTUAL_TRAIN_ONLY_LIST_PRIMARY_CE_FEATURE_GRADIENT_GRID',
        completed_at=datetime.now().isoformat(timespec='seconds'),variant=args.variant,
        dataset='RGBNT201',anchor=model.anchor_record,seed=42,observation_seed=43,
        neural_forwards=BATCHES,optimizer_updates=0,formal_runs=0,checkpoint_writes=0,
        train_query_gallery_counts=[3951,836,836],observed_training_identities=len({y for r in rows for y in r['training_labels']}),
        temperatures=list(TEMPERATURES),gradient_scale=GRADIENT_SCALE,
        parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        descriptor_dim=5120,rows=rows,aggregate=aggregate,
        limits='Fresh expert/head initialization on the original selected E28 frozen identity anchor; 16 fixed training batches only, not a trained checkpoint or full-training derivative census. CE comparison is the weighted first fused CE only, not total/partial/aux gradients. Autograd feature derivatives use scale512 then divide512 to match the current formal loss scale. Four temperatures use the same forward; no optimizer, query/gallery inference, temperature choice by benchmark, or retrieval gain claim. Training-mode head BN statistics remain private in this diagnostic process and are not saved.')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],variant=args.variant,forwards=BATCHES,optimizer_updates=0,aggregate=aggregate)),flush=True)


if __name__=='__main__':
    main()
