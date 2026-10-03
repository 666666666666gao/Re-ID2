"""Two trained-checkpoint diagnostics on idle GPUs, zero optimizer updates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from launch_axis_scaled import idle
from run_experiment import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--v5-run',required=True)
    parser.add_argument('--v6-run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(exist_ok=False)
    jobs=[]
    for name,run,gpu in [('V5',args.v5_run,0),('V6',args.v6_run,1)]:
        idle(gpu)
        argv=[sys.executable,'-u','probe_contribution_gradients.py','--run-dir',run,'--output',str(output/(name+'.json'))]
        with (output/(name+'.log')).open('x') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        write_json(output/(name+'_launch.json'),dict(pid=child.pid,gpu=gpu,command=argv,started=time.time()))
        jobs.append((name,child))
    exits=[]
    for name,child in jobs:
        code=child.wait()
        write_json(output/(name+'_exit.json'),dict(exit_code=code,finished=time.time()))
        exits.append((name,code))
    assert all(code==0 for _,code in exits),exits
    results={}
    for name,_ in jobs:
        result=json.loads((output/(name+'.json')).read_text())
        assert result['status']=='PASS_ACTUAL_GRADIENT_DIAGNOSTIC' and result['optimizer_updates']==0
        assert not result['optimizer_constructed'] and not result['target_requires_grad']
        results[name]=result
    for key in ('names','labels','cameras','scenes','image_tensor_sha256'):
        assert results['V5'][key]==results['V6'][key],key
    write_json(output/'controller_result.json',dict(status='COMPLETE',cases=['V5','V6'],identical_seeded_training_batch=True,optimizer_updates=0,official_test_uses=0))


if __name__=='__main__':
    main()
