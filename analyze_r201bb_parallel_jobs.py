"""Reduce finite speed evidence only; no performance or neural experiment."""
import argparse,json,math
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root);data=json.loads((root/'controller_result.json').read_text())
    assert data['status']=='COMPLETE_BENCHMARK_NOT_METHOD_RESULT' and data['successful_updates']==228
    comparisons=[]
    for repeat in range(3):
        phases={p['arm']:p for p in data['phases'] if p['repeat']==repeat}
        serial,parallel=phases['serial'],phases['parallel']
        assert serial['paired_training_orders']==parallel['paired_training_orders']
        assert serial['successful_updates']==parallel['successful_updates']==38
        assert serial['query_gallery_forward_records']==parallel['query_gallery_forward_records']==20580
        comparisons.append(dict(repeat=repeat,serial_seconds=serial['seconds'],parallel_seconds=parallel['seconds'],
            speedup=serial['seconds']/parallel['seconds'],seconds_saved=serial['seconds']-parallel['seconds'],
            serial_peak_reserved_bytes=max(j['peak_reserved_bytes'] for j in serial['jobs']),
            parallel_sum_peak_reserved_bytes=sum(j['peak_reserved_bytes'] for j in parallel['jobs'])))
    serial_total=math.fsum(p['serial_seconds'] for p in comparisons);parallel_total=math.fsum(p['parallel_seconds'] for p in comparisons)
    result=dict(status='ACTUAL_BB_CPU_SPEED_READOUT_NOT_METHOD_RESULT',comparisons=comparisons,
        aggregate_speedup=serial_total/parallel_total,all_repeats_faster=all(p['speedup']>1 for p in comparisons),
        all_measured_seconds_retained=True,new_NN_calls=0,new_optimizer_updates=0,
        instrumentation_changelog=[dict(file='profile_r201bb_BA_job.py',change='created',purpose='Standalone existing BA step and full-QG workload timing; no production edits.'),
            dict(file='profile_r201bb_parallel_jobs.py',change='created',purpose='Serial/concurrent job orchestration, CPU and selected-card memory at phase nodes.'),
            dict(file='analyze_r201bb_parallel_jobs.py',change='created',purpose='CPU-only comparison from complete evidence.'),
            dict(file='results/preflight/r201bb_parallel_jobs_deploy_20261015.py',change='created',purpose='Deferred reviewed admission and explicit TRAIN-bank restore, no existing training edits.'),
            dict(file='results/preflight/r201bb_parallel_jobs_complete_intake_20261015.py',change='created',purpose='Collect exact finite evidence, CPU reduction and retire only dependency/package duplicate.')],
        limits='Process-reserved peaks are PyTorch allocator measurements; their sum is conservative concurrent allocation, not sampled NVIDIA peak. Full micro-job includes startup/GT evaluation but19 updates, not full50. No temperature/power/SM occupancy claim. Throughput evidence supports only this tested workload/card; benchmark scores do not become model results.')
    output=root/'CPU_speed_comparison.json';assert not output.exists();output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
