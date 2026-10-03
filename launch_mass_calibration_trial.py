"""One scalar-loss change: AMP smoke followed by fresh fifty-epoch training."""
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
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--pretrained',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(exist_ok=False)
    preflight=output/'preflight';development=output/'development'
    preflight.mkdir();development.mkdir()
    name='MSVR310_axis_mass_fullref_s42'
    common=['--dataset','MSVR310','--variant','axis_mass_fullref','--data-root',args.data_root,'--pretrained',args.pretrained,'--seed','42','--contribution-weight','1.0']
    jobs=[('smoke',preflight,name+'_smoke','smoke'),('development',development,name,'train')]
    rows=[]
    for stage,folder,job,mode in jobs:
        idle(1)
        argv=[sys.executable,'-u','run_mass_experiment.py',*common,'--mode',mode,'--output',str(folder/job)]
        with (folder/(job+'.log')).open('x') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            write_json(folder/(job+'_launch.json'),dict(pid=child.pid,gpu=1,command=argv,started=time.time()))
            code=child.wait()
        row=dict(stage=stage,name=job,exit_code=code,finished=time.time());write_json(folder/(job+'_exit.json'),row)
        assert code==0,(stage,job)
        result=json.loads((folder/job/('smoke.json' if stage=='smoke' else 'result.json')).read_text())
        assert result['arguments']['contribution_weight']==1.0
        if stage=='smoke':
            assert result['status']=='SMOKE_PASS' and result['steps']==3
            assert all(result['gradients'].values()) and result['strict_reload_equal']
            write_json(preflight/'controller_result.json',dict(status='PASS',checks=[row],engineering_updates=3,budget_epochs=50,fresh_public_CLIP=True))
        else:
            assert result['status']=='COMPLETE' and result['epochs']==50
        rows.append(row)
    write_json(output/'controller_result.json',dict(status='COMPLETE',runs=rows,budget_epochs=50,engineering_updates=3,official_test_uses=0,single_factor='V5 contribution loss weight .05 -> 1.0; original runner/model/input/sampler/optimizer and checkpoint serialization unchanged'))


if __name__=='__main__':
    main()
