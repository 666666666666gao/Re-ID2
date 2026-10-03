"""Clean FP32 inference costs on identical installed development triplets."""
import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from experiment_data import make_loader, seed_all, split_records
from mass_axis_collaboration import MassAxisCollaborationDeMo
from run_experiment import build, configuration, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--pretrained',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    args.seed,args.contribution_weight=42,.05
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    output=Path(args.output);output.mkdir(exist_ok=False)
    variants=('demo','axis_scaled_fullref','plain_scaled_fullref','frequency_scaled_fullref','axis_mass_fullref')
    rows=[]
    for dataset in ('MSVR310','RGBNT201','RGBNT100'):
        args.dataset=dataset
        cfg=configuration(args)
        _,dev,_,classes,cameras=split_records(args.data_root,dataset)
        images,_,cam,scene,names=next(iter(make_loader(dev[:8],cfg,False,args.seed)))
        images={key:value.cuda() for key,value in images.items()};cam,scene=cam.cuda(),scene.cuda()
        for variant in variants:
            args.variant=variant
            seed_all(args.seed)
            if variant=='axis_mass_fullref':
                model=MassAxisCollaborationDeMo(classes,cfg,cameras).float().cuda()
            else:
                model=build(args,cfg,classes,cameras)
            model.eval()
            assert all(not module.training for module in model.modules())
            versions={name:value._version for name,value in model.state_dict().items()}
            with torch.inference_mode():
                for _ in range(5):
                    features=model(images,cam_label=cam,view_label=scene)
                torch.cuda.synchronize()
                assert features.shape==(8,5120 if variant=='demo' else 5632) and torch.isfinite(features).all()
                torch.cuda.reset_peak_memory_stats()
                allocated=torch.cuda.memory_allocated()
                device_ms,wall_ms=[],[]
                for _ in range(20):
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    started=time.perf_counter();begin.record()
                    features=model(images,cam_label=cam,view_label=scene)
                    end.record();torch.cuda.synchronize()
                    wall_ms.append(1000*(time.perf_counter()-started));device_ms.append(begin.elapsed_time(end))
                peak_allocated=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
                # Profile separately: instrumentation must not enter latency trials.
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,with_flops=True) as profile:
                    features=model(images,cam_label=cam,view_label=scene)
                    torch.cuda.synchronize()
            assert versions=={name:value._version for name,value in model.state_dict().items()}
            operators=[dict(operator=event.key,count=event.count,self_cpu_time_us=event.self_cpu_time_total,self_device_time_us=event.self_device_time_total,reported_flops=event.flops) for event in profile.key_averages()]
            lower_bound=sum(row['reported_flops'] for row in operators)
            assert lower_bound>0
            row=dict(dataset=dataset,variant=variant,batch_triplets=8,images_per_triplet=3,input_hw=list(cfg.INPUT.SIZE_TEST),
                parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),descriptor_dim=features.shape[1],
                precision='FP32 eval, no autocast',warmup_forwards=5,timed_forwards=20,profile_forwards=1,
                wall_batch_ms_median=statistics.median(wall_ms),wall_batch_ms_min=min(wall_ms),wall_batch_ms_max=max(wall_ms),
                cuda_event_batch_ms_median=statistics.median(device_ms),cuda_event_batch_ms_min=min(device_ms),cuda_event_batch_ms_max=max(device_ms),
                triplets_per_second=8000/statistics.median(wall_ms),images_per_second=24000/statistics.median(wall_ms),
                wall_batch_ms_samples=wall_ms,cuda_event_batch_ms_samples=device_ms,
                allocated_before_measurement_bytes=allocated,peak_allocated_bytes=peak_allocated,peak_reserved_bytes=peak_reserved,
                supported_operator_flops_per_triplet_lower_bound=lower_bound/8,complete_model_FLOPs=False,
                operators_without_reported_flops=[entry['operator'] for entry in operators if entry['reported_flops']==0],
                dataset_names=list(names),state_tensor_versions_unchanged=True,optimizer_updates=0,
                weights='public CLIP initialization and fresh ReID heads; architecture-cost comparison, not retrieval/checkpoint selection',
                limits='GPU0 only; other GPUs/host tasks can affect CPU dispatch. Data decoding and H2D excluded. Batch8 only, no latency percentile/stability claim beyond20 trials. Profiler estimates selected matmul/conv operations; FFT, normalization, activations and some fused attention uncounted, so no complete FLOPs claim.',
                profiler_documentation='https://docs.pytorch.org/docs/stable/profiler')
            write_json(output/(dataset+'_'+variant+'.json'),row)
            write_json(output/(dataset+'_'+variant+'_operators.json'),operators)
            rows.append(row)
            print('PROFILE_REID_CONDITION',json.dumps({key:row[key] for key in ('dataset','variant','wall_batch_ms_median','cuda_event_batch_ms_median','triplets_per_second','peak_allocated_bytes')}),flush=True)
            del model,features,profile
            gc.collect();torch.cuda.empty_cache()
    assert len(rows)==15
    for dataset in ('MSVR310','RGBNT201','RGBNT100'):
        enhanced=[row for row in rows if row['dataset']==dataset and row['variant']!='demo']
        assert len({(row['parameters'],row['trainable_parameters'],row['descriptor_dim']) for row in enhanced})==1
    write_json(output/'summary.json',dict(status='COMPLETE',observed_at=datetime.now().isoformat(timespec='seconds'),torch=torch.__version__,gpu=torch.cuda.get_device_name(),conditions=15,rows=rows,optimizer_updates=0,official_test_uses=0,complete_model_FLOPs=False))
    print('PROFILE_REID_ALL15_COMPLETE',flush=True)


if __name__=='__main__':
    main()
