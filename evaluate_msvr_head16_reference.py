"""Before head16 expert training, verify full00 against the new selected head16 teacher."""
import argparse,csv,io,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiment_data import make_loader
from full_evaluation import distance,extract,full_metrics
from official_training_data import full_records,metadata
from run_msvr_head16_experts import build,configuration,assert_retained

METRICS=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')


class OnlyBaseState(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,images,cam_label=None,view_label=None):
        return self.model(images,cam_label=cam_label,view_label=view_label,return_states=True)['00']


def main():
    parser=argparse.ArgumentParser()
    for key in ('data-root','pretrained','anchor-run-dir','output','variant'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();args.dataset='MSVR310';args.seed=42;args.freeze_identity_encoder=1;args.mode='train'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    train,query,gallery,classes,cameras,manifest=full_records(args.data_root,args.dataset)
    assert (len(train),len(query),len(gallery),classes)==(1032,591,1055,155)
    installed=metadata(query,'query')|metadata(gallery,'gallery')
    teacher=Path(args.anchor_run_dir)
    selected=json.loads((teacher/'result.json').read_text())
    with np.load(teacher/'best_official_arrays.npz',allow_pickle=False) as arrays:cached={k:arrays[k].copy() for k in arrays.files}
    assert all(np.array_equal(cached[k],v) for k,v in installed.items())
    reference={s:torch.from_numpy(cached[s+'_features'].copy()) for s in ('query','gallery')}
    assert reference['query'].shape==(591,5632) and reference['gallery'].shape==(1055,5632)
    reference_metrics=full_metrics(cached['distances'],cached['query_ids'],cached['gallery_ids'],cached['query_scenes'],cached['gallery_scenes'],
        cached['query_names'],cached['query_cameras'],cached['query_scenes'],out/'cached_full_reference')
    assert all(abs(reference_metrics[k]-selected['full_metrics'][k])<1e-8 for k in METRICS)
    cfg=configuration(args);model=build(args,cfg,classes,cameras);model.eval();wrapper=OnlyBaseState(model);wrapper.eval()
    images,_,cam,scene,_=next(iter(make_loader(query[:8],cfg,False,args.seed)));images={k:v.cuda() for k,v in images.items()}
    with torch.no_grad():before=wrapper(images,cam_label=cam.cuda(),view_label=scene.cuda())
    buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
    model.load_state_dict(torch.load(buffer,map_location='cuda',weights_only=True),strict=True)
    with torch.no_grad():after=wrapper(images,cam_label=cam.cuda(),view_label=scene.cuda())
    assert torch.equal(before,after) and before.shape==(8,5632);buffer.close()
    features,runtime=extract(wrapper,query+gallery,cfg,args.seed)
    assert features.shape==(1646,5632) and torch.isfinite(features).all()
    values=dict(query=features[:591],gallery=features[591:])
    error={s:float((values[s]-reference[s]).abs().max()) for s in values};assert all(v<3e-5 for v in error.values())
    metrics=full_metrics(distance(values['query'],values['gallery']),cached['query_ids'],cached['gallery_ids'],cached['query_scenes'],cached['gallery_scenes'],
        cached['query_names'],cached['query_cameras'],cached['query_scenes'],out/'actual00')
    assert all(abs(metrics[k]-reference_metrics[k])<1e-8 for k in METRICS)
    rows=lambda path:list(csv.DictReader(path.open(encoding='utf-8',newline='')))
    assert rows(out/'actual00.csv')==rows(out/'cached_full_reference.csv')
    assert_retained(model)
    record=dict(status='COMPLETE_HEAD16_FULL5632_FROZEN00_OFFICIAL_TEACHER_PARITY',variant=args.variant,finished=time.time(),
        train_records=1032,query_records=591,gallery_records=1055,descriptor_dim=5632,strict_reload_equal=True,
        per_query_equal=True,feature_max_abs_difference=error,metrics={k:metrics[k] for k in METRICS},anchor=model.anchor_record,runtime=runtime,
        optimizer_updates=0,new_weights=0,scope='Actual full5632 frozen00 equals the existing selected teacher fullnormal six metrics and all per-query rows. No expert gain or training admission is implied.')
    (out/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=record['status'],variant=args.variant,metrics=record['metrics'])),flush=True)


if __name__=='__main__':main()
