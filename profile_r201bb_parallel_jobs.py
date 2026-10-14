"""Matched two-job serial/parallel timing after BA consumers terminate; max2 NN."""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from run_experiment import write_json


def cpu():
    return list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))


def selected_memory(gpu):
    value=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    total,used,free=map(int,value.strip().split(','))
    return dict(total_MiB=total,used_MiB=used,free_MiB=free)


def main():
    parser=argparse.ArgumentParser()
    for key in ('data-root','pretrained','anchor-run-dir','closed-BA-root','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--gpu',type=int,choices=(2,3),required=True)
    args=parser.parse_args();closed=Path(args.closed_BA_root)
    owner=json.loads((closed/'remote_owner.json').read_text())
    assert not Path('/proc/'+str(owner['pid'])).exists()
    completed=json.loads((closed/'RGBNT100/controller_result.json').read_text())
    assert completed['status']=='COMPLETE' and completed['successful_updates']==13126 and completed['states_cases']==8
    for phase in ('native','training','audit','states'):
        for launch in (closed/'RGBNT100'/phase).glob('*_launch.json'):
            d=json.loads(launch.read_text());assert not Path('/proc/'+str(d['pid'])).exists()
            assert json.loads(launch.with_name(launch.name.replace('_launch.json','_exit.json')).read_text())['exit_code']==0
    apps=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not apps
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    write_json(root/'controller_launch.json',dict(pid=os.getpid(),started=time.time(),arguments=vars(args),
        selected_physical_gpu=args.gpu,max_NN=2,temperature_power_read=False,conditions='Current BA all consumers completed; selected GPU idle. Only future finite benchmark.'))
    phases=[]
    for repeat in range(3):
        arms=('serial','parallel') if repeat%2==0 else ('parallel','serial')
        for arm in arms:
            folder=root/('repeat_'+str(repeat)+'_'+arm);folder.mkdir()
            resources=selected_memory(args.gpu);before=cpu();started=time.perf_counter()
            def worker(index):
                output=folder/('job_'+str(index));env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                argv=[sys.executable,'-u','profile_r201bb_BA_job.py','--data-root',args.data_root,
                    '--pretrained',args.pretrained,'--anchor-run-dir',args.anchor_run_dir,'--output',str(output)]
                with (folder/('job_'+str(index)+'.log')).open('w') as log:
                    process=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                    write_json(folder/('job_'+str(index)+'_launch.json'),dict(pid=process.pid,started=time.time(),argv=argv,gpu=args.gpu))
                    status=process.wait()
                write_json(folder/('job_'+str(index)+'_exit.json'),dict(pid=process.pid,exit_code=status,finished=time.time()))
                assert status==0
                d=json.loads((output/'result.json').read_text())
                assert d['status']=='COMPLETE_BENCHMARK_NOT_METHOD_RESULT' and d['successful_updates']==19
                assert d['query_gallery_forward_records']==10290 and d['checkpoint_writes']==d['raw_distance_writes']==0
                assert d['parameters']==99585158 and d['trainable_parameters']==1887366 and d['descriptor_dim']==5120
                return d
            with ThreadPoolExecutor(max_workers=1 if arm=='serial' else 2) as pool:jobs=list(pool.map(worker,(0,1)))
            wall=time.perf_counter()-started;after=cpu();delta=[a-b for a,b in zip(after,before)];total=sum(delta)
            order=[]
            for index in (0,1):
                ds=[json.loads(line) for line in (folder/('job_'+str(index))/'training_details.jsonl').read_text().splitlines()]
                assert len(ds)==19 and all(d['optimizer_updated'] for d in ds)
                order.append([(d['names'],d['partial_set']) for d in ds])
            assert order[0]==order[1]
            phase=dict(repeat=repeat,arm=arm,seconds=wall,jobs=jobs,job_count=2,successful_updates=38,
                warm_updates=6,timed_updates=32,timed_main_records=2048,query_gallery_forward_records=20580,
                compute_apps_at_admission=apps,memory_before=resources,memory_after=selected_memory(args.gpu),
                CPU_busy_percent=100*(total-delta[3]-delta[4])/total,CPU_iowait_percent=100*delta[4]/total,
                paired_training_orders=order[0],checkpoint_writes=0,temperature_power_read=False)
            write_json(folder/'phase.json',phase);phases.append(phase)
    for repeat in range(3):
        pair=[d for d in phases if d['repeat']==repeat]
        assert pair[0]['paired_training_orders']==pair[1]['paired_training_orders']
    assert len(phases)==6 and sum(d['successful_updates'] for d in phases)==228
    write_json(root/'controller_result.json',dict(status='COMPLETE_BENCHMARK_NOT_METHOD_RESULT',phases=phases,
        successful_updates=228,warm_updates=36,timed_updates=192,query_gallery_forward_records=123480,
        top_level_training_full_partial_forward_records=29184,checkpoint_writes=0,
        max_NN=2,physical_gpu=args.gpu,temperature_power_read=False,
        limits='Three paired finite repeats; arms each run two identical short jobs, serial versus concurrent on one card. Includes full QG NN/GT ranking, not full50-epoch throughput. No scientific method result/selection, no training checkpoint writes. Orders and budgets matched; all timings retained.'))
    print('BB_FULL_WORKLOAD_PARALLEL_COMPARISON_COMPLETE',flush=True)


if __name__=='__main__':main()
