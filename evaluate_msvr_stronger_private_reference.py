"""Reload the strong teacher's private block in both existing expert variants."""
import argparse
import csv
import io
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiment_data import make_loader
from full_evaluation import distance, extract, full_metrics
from msvr_stronger_private_teacher import install
from official_training_data import full_records, metadata
from original_identity_anchor import assert_anchor_unchanged
import run_r201y_MS_K4 as y

METRICS=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')


class OnlyBaseState(nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model

    def forward(self,images,cam_label=None,view_label=None):
        return self.model(images,cam_label=cam_label,view_label=view_label,return_states=True)['00']


def main():
    parser=argparse.ArgumentParser()
    for key in ('data-root','pretrained','anchor-run-dir','output'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args();args.dataset='MSVR310';args.seed=42;args.freeze_identity_encoder=1;args.mode='train'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    train,query,gallery,classes,cameras,manifest=full_records(args.data_root,args.dataset)
    assert (len(train),len(query),len(gallery),classes)==(1032,591,1055,155)
    installed=metadata(query,'query')|metadata(gallery,'gallery')
    teacher=Path(args.anchor_run_dir).parent/'MSVR310_demo_shared_s42'
    with np.load(teacher/'best_official_arrays.npz',allow_pickle=False) as arrays:
        cached={k:arrays[k].copy() for k in arrays.files}
    assert all(np.array_equal(cached[k],v) for k,v in installed.items())
    reference={s:F.normalize(torch.from_numpy(cached[s+'_features'][:,:5120].copy()),dim=1) for s in ('query','gallery')}
    assert reference['query'].shape==(591,5120) and reference['gallery'].shape==(1055,5120)
    reference_metrics=full_metrics(distance(reference['query'],reference['gallery']),cached['query_ids'],cached['gallery_ids'],cached['query_scenes'],cached['gallery_scenes'],cached['query_names'],cached['query_cameras'],cached['query_scenes'],out/'cached_private_reference')
    variants={};started=time.time()
    for variant in ('frequency_shared','axis_shared'):
        args.variant=variant;cfg=y.configuration(args)
        model=install(y.build(args,cfg,classes,cameras),args,classes,cameras);model.eval()
        wrapper=OnlyBaseState(model);wrapper.eval()
        images,_,cam,scene,_=next(iter(make_loader(query[:8],cfg,False,args.seed)))
        images={k:v.cuda() for k,v in images.items()}
        with torch.no_grad():before=wrapper(images,cam_label=cam.cuda(),view_label=scene.cuda())
        buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
        model.load_state_dict(torch.load(buffer,map_location='cuda',weights_only=True),strict=True)
        with torch.no_grad():after=wrapper(images,cam_label=cam.cuda(),view_label=scene.cuda())
        assert torch.equal(before,after) and before.shape==(8,5120) and torch.isfinite(before).all()
        buffer.close()
        features,runtime=extract(wrapper,query+gallery,cfg,args.seed)
        assert features.shape==(1646,5120) and torch.isfinite(features).all()
        values={s:features[:591] if s=='query' else features[591:] for s in ('query','gallery')}
        error={s:float((values[s]-reference[s]).abs().max()) for s in values}
        assert all(v<3e-5 for v in error.values())
        distances=distance(values['query'],values['gallery'])
        metrics=full_metrics(distances,cached['query_ids'],cached['gallery_ids'],cached['query_scenes'],cached['gallery_scenes'],cached['query_names'],cached['query_cameras'],cached['query_scenes'],out/variant)
        assert all(abs(metrics[k]-reference_metrics[k])<1e-8 for k in METRICS)
        with (out/(variant+'.csv')).open(encoding='utf-8',newline='') as f:actual=list(csv.DictReader(f))
        with (out/'cached_private_reference.csv').open(encoding='utf-8',newline='') as f:expected=list(csv.DictReader(f))
        assert actual==expected
        assert_anchor_unchanged(model)
        np.savez_compressed(out/(variant+'_normal_arrays.npz'),**installed,query_features=values['query'].numpy(),gallery_features=values['gallery'].numpy(),distances=distances)
        variants[variant]=dict(metrics=metrics,feature_max_abs_difference=error,per_query_equal=True,strict_reload_equal=True,descriptor_dim=5120,full_query=591,full_gallery=1055,optimizer_updates=0,anchor=model.anchor_record,runtime=runtime)
        del wrapper,model,features,before,after;torch.cuda.empty_cache()
    record=dict(status='COMPLETE_STRONG_TEACHER_PRIVATE5120_RELOAD_FULL_OFFICIAL_PARITY',finished=time.time(),seconds=time.time()-started,official_split_manifest=manifest,variants=variants,reference_metrics=reference_metrics,optimizer_updates=0,new_weights=0,training_admitted=False,scope='Both existing expert models use the exact stronger teacher core; actual00 equals cached private reference on all official query/gallery and per-query rows. No expert increment or adaptation is measured.',limits='Full shared teacher5632 49.426/69.036 remains an existing baseline. This5120 initialization is only its private block, selected by the teacher full benchmark mAP. Historical teacher missing objectives and benchmark checkpoint selection are inherited, not new method gains.')
    (out/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=record['status'],metrics={v:{k:r['metrics'][k] for k in METRICS} for v,r in variants.items()},full_queries=591,full_gallery=1055,new_weights=0,training_admitted=False)),flush=True)


if __name__=='__main__':main()
