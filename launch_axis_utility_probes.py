"""Three idle cards: P0 frozen utility and split gradients, zero optimizer updates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from launch_axis_scaled import idle
from run_experiment import write_json


def execute(name,run,gpu,output):
    idle(gpu)
    for stage in ('smoke','utility','gradients'):
        leaf=output/(name+'_'+stage)
        argv=[sys.executable,'-u','probe_axis_task_gradients.py' if stage=='gradients' else 'probe_axis_retrieval_utility.py',
            '--run-dir',run,'--output',str(leaf.with_suffix('.json') if stage=='gradients' else leaf)]
        if stage=='smoke':argv.append('--smoke')
        with (output/(name+'_'+stage+'.log')).open('x') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            write_json(output/(name+'_'+stage+'_launch.json'),dict(pid=child.pid,gpu=gpu,command=argv,started=time.time()))
            code=child.wait()
        write_json(output/(name+'_'+stage+'_exit.json'),dict(exit_code=code,finished=time.time()))
        assert code==0,(name,stage,code)
        result_path=leaf.with_suffix('.json') if stage=='gradients' else leaf/'result.json'
        report=json.loads(result_path.read_text())
        assert report['status']=={'smoke':'PASS_FROZEN_UTILITY_SMOKE','utility':'PASS_FROZEN_RETRIEVAL_UTILITY',
            'gradients':'PASS_ACTUAL_SPLIT_TASK_GRADIENT_DIAGNOSTIC'}[stage]
        assert report['optimizer_updates']==0 and report['official_test_uses']==0
    return name


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--v5-root',required=True)
    parser.add_argument('--v6-run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(exist_ok=False)
    schedules={0:[('V5_MSVR310',str(Path(args.v5_root)/'MSVR310_axis_mass_fullref_s42'))],
        1:[('V5_RGBNT100',str(Path(args.v5_root)/'RGBNT100_axis_mass_fullref_s42')),('V6_MSVR310',args.v6_run)],
        3:[('V5_RGBNT201',str(Path(args.v5_root)/'RGBNT201_axis_mass_fullref_s42'))]}

    def worker(gpu,rows): return [execute(name,run,gpu,output) for name,run in rows]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(worker,gpu,rows) for gpu,rows in schedules.items()]
        completed=[name for future in futures for name in future.result()]
    write_json(output/'controller_result.json',dict(status='COMPLETE',cases=completed,optimizer_updates=0,official_test_uses=0))


if __name__=='__main__': main()
