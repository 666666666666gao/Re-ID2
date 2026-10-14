"""Disposable BA workload for throughput; not a method run or selected result."""
import argparse,json,time
from pathlib import Path
import torch
from experiment_data import seed_all
from official_training_data import full_records
from run_full_official_frozen_anchor_experiment import make_loss,CrossEntropyLabelSmooth,PARTIAL_SETS,evaluate
from original_identity_anchor import assert_anchor_unchanged
from solver.scheduler_factory import create_scheduler
from run_experiment import write_json
import run_r201ba_RGB_teacher_head as recipe
from run_r201ac_RGB_hard_identity import make_loader


def main():
    parser=argparse.ArgumentParser()
    for key in ('data-root','pretrained','anchor-run-dir','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    args.dataset='RGBNT100';args.variant='axis_shared';args.seed=42
    args.freeze_identity_encoder=1;args.mode='train'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    cfg=recipe.configuration(args)
    train,query,gallery,classes,cameras,manifest=full_records(args.data_root,args.dataset)
    assert (len(train),len(query),len(gallery),classes)==(8675,1715,8575,50)
    model=recipe.build(args,cfg,classes,cameras)
    assert not model.normal_priority_smoke
    loss_fn,center=make_loss(cfg,classes);optimizer,_=recipe.make_optimizer(cfg,model,center)
    scheduler=create_scheduler(cfg,optimizer);scheduler.step(1)
    scaler=torch.amp.GradScaler('cuda',init_scale=512);xent=CrossEntropyLabelSmooth(classes)
    initialization_seconds=time.perf_counter()-started
    seed_all(args.seed+1);generator=torch.Generator().manual_seed(args.seed+1001)
    model.train();iterator=iter(make_loader(train,cfg,True,args.seed+1));details=[]
    def update():
        batch=next(iterator);retained=PARTIAL_SETS[int(torch.randint(6,(1,),generator=generator))]
        d=recipe.step(model,batch,optimizer,scaler,loss_fn,xent,retained,args.variant)
        assert d['optimizer_updated'] and d['loss']==d['full_loss'] and d['partial_loss_effective']==0
        details.append(d)
    warm_start=time.perf_counter()
    for _ in range(3):update()
    torch.cuda.synchronize();warm_seconds=time.perf_counter()-warm_start
    train_start=time.perf_counter()
    for _ in range(16):update()
    torch.cuda.synchronize();train_seconds=time.perf_counter()-train_start
    del iterator
    assert_anchor_unchanged(model)
    evaluation_start=time.perf_counter()
    values,runtime=evaluate(model,(query,gallery),cfg,args,out/'per_query')
    torch.cuda.synchronize();evaluation_seconds=time.perf_counter()-evaluation_start
    assert_anchor_unchanged(model)
    assert not list(out.rglob('*.pth')) and not list(out.rglob('*.npz'))
    (out/'training_details.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in details),encoding='utf-8')
    write_json(out/'result.json',dict(status='COMPLETE_BENCHMARK_NOT_METHOD_RESULT',arguments=vars(args),
        initialization_seconds=initialization_seconds,warm_seconds=warm_seconds,training_seconds=train_seconds,
        evaluation_seconds=evaluation_seconds,worker_seconds_excludes_import=time.perf_counter()-started,
        warm_updates=3,timed_updates=16,successful_updates=19,amp_skipped_steps=0,
        training_main_forward_records=19*64,training_partial_forward_records=19*64,
        query_gallery_forward_records=10290,checkpoint_writes=0,raw_distance_writes=0,
        parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        descriptor_dim=5120,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        full_official_manifest=manifest,full_evaluation_runtime=runtime,benchmark_only_GT_metrics=values,
        anchor=model.anchor_record,model_normal_priority_smoke=False,identity_anchor_exact=True,
        limits='Disposable19-update speed workload, not complete training/coverage or candidate score. Each arm has two identical jobs: fresh initialization+3warm+16timed updates+ALL official normal QG ranking. No weights saved, no score-based choice. Uses actual BA formal step/AMP/BN2/partial0/two backwards and original frozen teacher. Phase end-to-end timing in controller includes imports, initialization, warm, loader teardown, evaluation and evidence writes. Micro-workload does not prove full50 scaling. No temperature/power read.'))
    print('BB_DISPOSABLE_JOB_COMPLETE',flush=True)


if __name__=='__main__':main()
