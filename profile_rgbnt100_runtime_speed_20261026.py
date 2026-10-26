"""Disposable runtime profiling; never selects a checkpoint or changes formal recipes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiment_data import Triplets, seed_all
from full_evaluation import distance, full_metrics
from official_training_data import full_records, metadata
from original_identity_anchor import assert_anchor_unchanged
from run_experiment import write_json
from run_full_official_frozen_anchor_experiment import make_loss, CrossEntropyLabelSmooth, PARTIAL_SETS
from solver.scheduler_factory import create_scheduler
import run_rgbnt100_B128_reference_experts as recipe
import run_r201ac_RGB_hard_identity as sampling

METRICS = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
INFER_ARMS = ((4, 4, 64), (4, 4, 128), (4, 4, 256), (4, 4, 512),
              (1, 4, 256), (8, 4, 256), (4, 8, 256), (4, 4, 64))


def cpu():
    return list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))


class Samples:
    def __init__(self, gpu):
        self.gpu = gpu
        self.rows = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        while not self.stop.is_set():
            raw = subprocess.check_output(['nvidia-smi', '-i', str(self.gpu),
                '--query-gpu=memory.used,utilization.gpu,utilization.memory',
                '--format=csv,noheader,nounits'], text=True, timeout=15)
            values = list(map(int, raw.strip().split(',')))
            self.rows.append(dict(epoch=time.time(),memory_used_MiB=values[0],
                gpu_util_percent=values[1],memory_util_percent=values[2]))
            self.stop.wait(2)

    def start(self):
        self.before = cpu()
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join()
        delta = [a-b for a,b in zip(cpu(),self.before)]
        total = sum(delta)
        return dict(samples=self.rows,host_CPU_busy_percent=100*(total-delta[3]-delta[4])/total,
            host_CPU_iowait_percent=100*delta[4]/total)


def loader(records,cfg,training,seed,workers,batch):
    sampler = sampling.experiment_data.RandomIdentitySampler(records,batch,cfg.DATALOADER.NUM_INSTANCE) if training else None
    return DataLoader(Triplets(records,cfg,training),batch_size=batch,sampler=sampler,
        shuffle=False,num_workers=workers,drop_last=training,
        generator=torch.Generator().manual_seed(seed),pin_memory=True)


def training(args,cfg,train,classes,cameras,out):
    seed_all(42)
    model = recipe.build(args,cfg,classes,cameras)
    loss_fn,center = make_loss(cfg,classes)
    optimizer,_ = recipe.make_optimizer(cfg,model,center)
    create_scheduler(cfg,optimizer).step(1)
    scaler = torch.amp.GradScaler('cuda',init_scale=512)
    xent = CrossEntropyLabelSmooth(classes)
    model.train()
    seed_all(43)
    generator = torch.Generator().manual_seed(1043)
    sampling.CURRENT.clear()
    sampling.CURRENT['seed'] = 43
    iterator = iter(loader(train,cfg,True,43,args.workers,64))
    details=[]
    for _ in range(3):
        batch=next(iterator)
        retained=PARTIAL_SETS[int(torch.randint(6,(1,),generator=generator))]
        details.append(recipe.step(model,batch,optimizer,scaler,loss_fn,xent,retained,args.variant))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    sample=Samples(args.gpu);sample.start()
    started=time.perf_counter();data_wait=0.
    for _ in range(24):
        before=time.perf_counter();batch=next(iterator);data_wait+=time.perf_counter()-before
        retained=PARTIAL_SETS[int(torch.randint(6,(1,),generator=generator))]
        details.append(recipe.step(model,batch,optimizer,scaler,loss_fn,xent,retained,args.variant))
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    resources=sample.finish()
    assert len(details)==27 and all(d['optimizer_updated'] for d in details)
    assert all(d['partial_loss_effective']==0 for d in details)
    assert_anchor_unchanged(model)
    value=dict(status='COMPLETE_DISPOSABLE_TRAIN_SPEED_NOT_METHOD_RESULT',threads=args.threads,
        workers=args.workers,batch=64,P=8,K=8,warm_updates=3,timed_updates=24,
        successful_updates=27,amp_skips=0,timed_seconds=seconds,
        main_records_per_second=24*64/seconds,main_plus_partial_records_per_second=24*128/seconds,
        data_wait_seconds=data_wait,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),resources=resources,
        order=[dict(names=d['names'],partial_set=d['partial_set']) for d in details],
        scalar_losses=[d['loss'] for d in details],train_records_available=len(train),
        checkpoint_writes=0,raw_array_writes=0,identity_anchor_exact=True,
        limitation='27 disposable real updates, not full training or candidate result; fixed B64/P8K8/AMP/loss/architecture, original immutable weights never written.')
    del iterator
    write_json(out/'result.json',value)


def rows(path):
    with path.open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


@torch.no_grad()
def inference(args,cfg,query,gallery,classes,cameras,out):
    model=recipe.build(args,cfg,classes,cameras)
    model.load_state_dict(torch.load(Path(args.run_dir)/'best.pth',map_location='cuda',weights_only=True),strict=True)
    model.eval();assert_anchor_unchanged(model)
    versions={k:v._version for k,v in model.state_dict().items()}
    stored=json.loads((Path(args.run_dir)/'result.json').read_text())
    original=rows(Path(args.run_dir)/'best_per_query.csv')
    records=query+gallery;arrays=metadata(query,'query')|metadata(gallery,'gallery')
    arms=[]
    for index,(threads,workers,batchsize) in enumerate(INFER_ARMS):
        torch.set_num_threads(threads)
        seed_all(42)
        first=next(iter(loader(records[:64],cfg,False,42,workers,64)))
        images={k:v.cuda(non_blocking=True) for k,v in first[0].items()}
        model(images,cam_label=first[2].cuda(),view_label=first[3].cuda())
        del first,images
        torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        features=[];names=[];events=[];wait=0.
        sample=Samples(args.gpu);sample.start();start=time.perf_counter()
        iterator=iter(loader(records,cfg,False,42,workers,batchsize))
        for _ in range(len(iterator)):
            before=time.perf_counter();images,_,cam,scene,batch_names=next(iterator);wait+=time.perf_counter()-before
            images={k:v.cuda(non_blocking=True) for k,v in images.items()}
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
            value=F.normalize(model(images,cam_label=cam.cuda(),view_label=scene.cuda()).float(),dim=1)
            b.record();features.append(value.cpu());events.append((a,b));names.extend(batch_names)
            if hasattr(model,'last_route'):
                model.last_route.cpu();model.last_gates.cpu()
        torch.cuda.synchronize();extract_seconds=time.perf_counter()-start
        resources=sample.finish();allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
        del iterator,images,value
        assert names==list(map(str,arrays['query_names']))+list(map(str,arrays['gallery_names']))
        joined=torch.cat(features);assert joined.shape==(10290,5120) and torch.isfinite(joined).all()
        # Keep metric arithmetic at original four CPU threads, independently of NN thread sweep.
        torch.set_num_threads(4)
        metric_start=time.perf_counter();d=distance(joined[:1715],joined[1715:])
        prefix=out/('arm_'+str(index))
        values=full_metrics(d,arrays['query_ids'],arrays['gallery_ids'],arrays['query_cameras'],arrays['gallery_cameras'],
            arrays['query_names'],arrays['query_cameras'],arrays['query_scenes'],prefix)
        metric_seconds=time.perf_counter()-metric_start
        current=rows(prefix.with_suffix('.csv'))
        assert len(current)==len(original)==1715
        changes={k:sum(a[k]!=b[k] for a,b in zip(current,original)) for k in ('AP','INP','first_match','last_match')}
        assert all(a['name']==b['name'] and a['identity']==b['identity'] and a['camera']==b['camera'] for a,b in zip(current,original))
        difference={k:values[k]-stored['full_metrics'][k] for k in METRICS}
        arm=dict(arm=index,threads=threads,workers=workers,batch=batchsize,
            full_official_triplets=10290,query_records=1715,gallery_records=8575,
            extract_seconds=extract_seconds,triplets_per_second=10290/extract_seconds,
            metric_seconds=metric_seconds,total_seconds=extract_seconds+metric_seconds,
            data_wait_seconds=wait,metadata_H2D_model_normalize_CUDA_event_seconds=sum(a.elapsed_time(b) for a,b in events)/1000,
            peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,resources=resources,
            metrics={k:values[k] for k in METRICS},metric_difference_from_original=difference,
            per_query_changes=changes,all_six_exact=all(x==0 for x in difference.values()),
            per_query_rank_exact=changes['first_match']==changes['last_match']==0,
            warm_triplets=64,inference_precision='Original FP32; no autocast/TF32/compile changes',
            checkpoint_writes=0,raw_array_writes=0)
        write_json(out/('arm_'+str(index)+'_timing.json'),arm);arms.append(arm)
        del features,joined,d,events
    assert versions=={k:v._version for k,v in model.state_dict().items()};assert_anchor_unchanged(model)
    write_json(out/'result.json',dict(status='COMPLETE_FULL_QG_RUNTIME_SCREEN_NOT_NEW_METHOD_RESULT',arms=arms,
        selected_run=args.run_dir,total_full_QG_passes=8,checkpoint_writes=0,raw_array_writes=0,
        selection_rule='Speed only among arms preserving all six exact metrics and per-query ranks; runtime screen once per nonbaseline setting; first/last baseline expose drift, not isolated repeated proof.',
        limitation='Dataset/current fixed model only. Inference batch changes may alter floating-point ties; every arm has full installed official GT ranking, changes are retained, no checkpoint or seed selection. No training batch changes.'))


def main():
    parser=argparse.ArgumentParser()
    for key in ('run-dir','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--mode',choices=('train','inference'),required=True)
    parser.add_argument('--gpu',type=int,choices=(2,3),required=True)
    parser.add_argument('--threads',type=int,choices=(1,4,8),default=4)
    parser.add_argument('--workers',type=int,choices=(4,8),default=4)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    trained=json.loads((Path(args.run_dir)/'result.json').read_text())
    original=argparse.Namespace(**trained['arguments'])
    for key in ('data_root','pretrained','anchor_run_dir','dataset','variant','seed','freeze_identity_encoder'):
        setattr(args,key,getattr(original,key))
    args.mode_actual=args.mode;args.mode='train';torch.set_num_threads(args.threads);torch.backends.cudnn.benchmark=False
    assert args.dataset=='RGBNT100' and args.variant=='axis_shared' and args.seed==42
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(args.gpu)
    checkpoint=Path(args.run_dir)/'best.pth';before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cfg=recipe.configuration(args)
    train,query,gallery,classes,cameras,manifest=full_records(args.data_root,args.dataset)
    assert (len(train),len(query),len(gallery),classes)==(8675,1715,8575,50)
    if args.mode_actual=='train':training(args,cfg,train,classes,cameras,out)
    else:inference(args,cfg,query,gallery,classes,cameras,out)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before
    assert not list(out.rglob('*.pth')) and not list(out.rglob('*.npz'))
    print('RUNTIME_PROFILE_COMPLETE',args.mode_actual,flush=True)


if __name__=='__main__':main()
