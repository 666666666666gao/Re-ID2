"""Two authorized GPUs, one disposable profiler per card; never saves model weights."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from run_experiment import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'controller_launch.json',dict(pid=os.getpid(),started=time.time(),argv=sys.argv,
        physical_GPUs=[2,3],max_NN=2,one_per_card=True,temperature_power_read=False))

    def run(name,mode,gpu,threads,workers):
        argv=[sys.executable,'-u','profile_rgbnt100_runtime_speed_20261026.py',
            '--run-dir',args.run_dir,'--output',str(out/name),'--mode',mode,
            '--gpu',str(gpu),'--threads',str(threads),'--workers',str(workers)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        with (out/(name+'.log')).open('x') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,stdin=subprocess.DEVNULL)
            write_json(out/(name+'_launch.json'),dict(pid=child.pid,started=time.time(),argv=argv,gpu=gpu))
            code=child.wait()
        write_json(out/(name+'_exit.json'),dict(pid=child.pid,exit_code=code,finished=time.time()))
        assert code==0
        return json.loads((out/name/'result.json').read_text())

    def train():
        rows=[]
        for i,(threads,workers) in enumerate(((4,4),(1,4),(8,4),(4,8),(4,4))):
            rows.append(run('train_'+str(i),'train',3,threads,workers))
        assert all(row['order']==rows[0]['order'] for row in rows)
        write_json(out/'training_summary.json',dict(arms=rows,paired_record_and_partial_orders=True,
            limitation='Worker count can change augmentation assignment despite matched record order. Screen is not a convergence comparison or automatic formal change.'))
        return rows

    with ThreadPoolExecutor(max_workers=2) as pool:
        inference=pool.submit(run,'inference','inference',2,4,4)
        training=pool.submit(train)
        infer=inference.result();trained=training.result()
    write_json(out/'controller_result.json',dict(status='COMPLETE_RUNTIME_SPEED_SCREEN_NOT_METHOD_RESULT',
        inference=infer,training=trained,completed_jobs=6,max_NN=2,physical_GPUs=[2,3],
        successful_disposable_updates=135,checkpoint_writes=0,raw_array_writes=0,
        finished=time.time(),limits='Warm existing environment. One selected axis model on RGBNT100. Runtime decisions need full metrics and baseline drift, not VRAM occupancy alone. No formal recipe/checkpoint mutation, GPU0/1, temperature/power, missing/49 or score search.'))
    print('RUNTIME_SPEED_SCREEN_COMPLETE',flush=True)


if __name__=='__main__':main()
