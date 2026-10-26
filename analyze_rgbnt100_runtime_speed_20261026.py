"""Summarize the measured runtime screen; no NN and no method score selection."""
import csv,json
from pathlib import Path


def main():
    root=Path('results/reid2_runtime_speed_inference_R4_20261026/profile_output')
    result=json.loads((root/'controller_result.json').read_text())
    assert result['status']=='COMPLETE_INDEPENDENT_PROCESS_FULL_QG_RUNTIME_SCREEN'
    infer=result['arms']
    train=json.loads(Path('results/reid2_runtime_speed_20261026/profile_output/training_summary.json').read_text())['arms']
    assert len(infer)==8 and len(train)==5 and all(a['full_official_triplets']==10290 for a in infer)
    assert all(a['successful_updates']==27 and a['batch']==64 and a['P']==a['K']==8 for a in train)
    assert all(a['order']==train[0]['order'] for a in train)
    base_extract=(infer[0]['extract_seconds']+infer[-1]['extract_seconds'])/2
    base_total=(infer[0]['total_seconds']+infer[-1]['total_seconds'])/2
    base_train=(train[0]['timed_seconds']+train[-1]['timed_seconds'])/2
    inference=[];training=[]
    for arm in infer:
        inference.append(dict(arm=arm['arm'],threads=arm['threads'],workers=arm['workers'],batch=arm['batch'],
            extract_seconds=arm['extract_seconds'],total_seconds=arm['total_seconds'],triplets_per_second=arm['triplets_per_second'],
            extract_speedup_vs_mean_baselines=base_extract/arm['extract_seconds'],
            full_eval_speedup_vs_mean_baselines=base_total/arm['total_seconds'],
            peak_allocated_GiB=arm['peak_allocated_bytes']/2**30,peak_reserved_GiB=arm['peak_reserved_bytes']/2**30,
            max_observed_NVML_MiB=None,mean_GPU_util_percent=None,
            CPU_busy_percent=arm['resources']['host_CPU_busy_percent'],CPU_iowait_percent=arm['resources']['host_CPU_iowait_percent'],
            data_wait_seconds=arm['data_wait_seconds'],all_six_exact=arm['all_six_exact'],per_query_rank_exact=arm['per_query_rank_exact'],
            AP_changed_queries=arm['per_query_changes']['AP'],INP_changed_queries=arm['per_query_changes']['INP'],
            max_six_metric_difference_pp=max(map(abs,arm['metric_difference_from_original'].values()))))
    for index,arm in enumerate(train):
        samples=arm['resources']['samples']
        training.append(dict(arm=index,threads=arm['threads'],workers=arm['workers'],batch=64,
            timed_updates=24,timed_seconds=arm['timed_seconds'],main_triplets_per_second=arm['main_records_per_second'],
            speedup_vs_mean_baselines=base_train/arm['timed_seconds'],
            peak_allocated_GiB=arm['peak_allocated_bytes']/2**30,peak_reserved_GiB=arm['peak_reserved_bytes']/2**30,
            max_observed_NVML_MiB=max(r['memory_used_MiB'] for r in samples),
            mean_GPU_util_percent=sum(r['gpu_util_percent'] for r in samples)/len(samples),
            CPU_busy_percent=arm['resources']['host_CPU_busy_percent'],CPU_iowait_percent=arm['resources']['host_CPU_iowait_percent'],
            data_wait_seconds=arm['data_wait_seconds'],
            maximum_scalar_loss_difference_vs_first=max(abs(a-b) for a,b in zip(arm['scalar_losses'],train[0]['scalar_losses']))))
    eligible=[a for a in inference if a['all_six_exact'] and a['per_query_rank_exact']]
    selected=min(eligible,key=lambda a:a['total_seconds']) if eligible else None
    summary=dict(status='ACTUAL_RUNTIME_SPEED_SCREEN_CPU_SUMMARY',inference=inference,training=training,
        proposed_fastest_strict_equivalent_inference=selected,
        baseline_extract_drift_ratio=infer[-1]['extract_seconds']/infer[0]['extract_seconds'],
        baseline_train_drift_ratio=train[-1]['timed_seconds']/train[0]['timed_seconds'],
        preserved_formal_training_batch=64,checkpoint_writes=0,raw_array_writes=0,
        limits='Screen, not full50 convergence or isolated hardware scaling. Training arms share fixed identity/partial orders but worker counts can change augmentation assignment; timing wins do not certify learned result. No automatic formal sampler/batch/loss changes. Inference eligibility locked before measurement: all six exact and per-query ranks unchanged. CUDA event is metadataH2D+model+normalization, not pure model forward. Concurrent second-card workload and first/last baseline drift retained.',
        instrumentation_changelog=[dict(file='profile_rgbnt100_runtime_speed_20261026.py',change='created',purpose='Standalone actual-step and fullQG timing/resources/GT check; no production edits'),
            dict(file='profile_rgbnt100_runtime_speed_R3_20261026.py',change='created',purpose='Only narrow inference no_grad; actual NVML thread timeout+stall preserved, not complete8QG'),
            dict(file='profile_rgbnt100_runtime_speed_R4_20261026.py',change='created',purpose='One existing inference arm per process, no in-worker NVML thread after actual stall'),
            dict(file='launch_rgbnt100_runtime_speed_R4_20261026.py',change='created',purpose='Run eight isolated arms sequentially onGPU2, native exit records, no optimizer'),
            dict(file='launch_rgbnt100_runtime_speed_20261026.py',change='created',purpose='Original one profiler/card,5trainingarms complete;failedinference retained'),
            dict(file='analyze_rgbnt100_runtime_speed_20261026.py',change='created',purpose='CPU measured timing/VRAM comparison, no model access')])
    out=root.parent/'summary.json';assert not out.exists();out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n','utf-8')
    for name,data in (('inference',inference),('training',training)):
        with (root.parent/(name+'_speed.csv')).open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
