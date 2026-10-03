"""Two real engineering gates before one fresh 50-epoch MSVR projection-tap trial."""
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
    dataset='MSVR310';variant='axis_mass_projected_fullref';name=dataset+'_'+variant+'_s42'
    common=['--dataset',dataset,'--data-root',args.data_root,'--pretrained',args.pretrained]
    jobs=[('tensor',preflight,'MSVR310_tensor',[sys.executable,'-u','verify_projected_axis_mass.py',*common,'--output',str(preflight/'MSVR310_tensor.json')]),
          ('smoke',preflight,name+'_smoke',[sys.executable,'-u','run_projected_mass_experiment.py',*common,'--variant',variant,'--seed','42','--contribution-weight','.05','--mode','smoke','--output',str(preflight/(name+'_smoke'))]),
          ('development',development,name,[sys.executable,'-u','run_projected_mass_experiment.py',*common,'--variant',variant,'--seed','42','--contribution-weight','.05','--mode','train','--output',str(development/name)])]
    rows=[]
    for stage,folder,job,argv in jobs:
        idle(1)
        with (folder/(job+'.log')).open('x') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            write_json(folder/(job+'_launch.json'),dict(pid=child.pid,gpu=1,command=argv,started=time.time()))
            code=child.wait()
        row=dict(stage=stage,name=job,exit_code=code,finished=time.time());write_json(folder/(job+'_exit.json'),row)
        assert code==0,(stage,job)
        if stage=='tensor':
            result=json.loads((folder/(job+'.json')).read_text())
            assert result['status']=='PASS' and result['optimizer_updates']==0
            assert all(value>0 for value in result['projected_auxiliary_CE_gradient_abs_sum_PM_PF'])
        elif stage=='smoke':
            result=json.loads((folder/job/'smoke.json').read_text())
            assert result['status']=='SMOKE_PASS' and result['steps']==3 and all(result['gradients'].values()) and result['strict_reload_equal']
            write_json(preflight/'controller_result.json',dict(status='PASS',checks=rows+[row],engineering_updates=3,budget_epochs=50,fresh_public_CLIP=True))
        else:
            result=json.loads((folder/job/'result.json').read_text())
            assert result['status']=='COMPLETE' and result['epochs']==50
        rows.append(row)
    write_json(output/'controller_result.json',dict(status='COMPLETE',runs=rows,budget_epochs=50,engineering_updates=3,official_test_uses=0,scope='Single-factor auxiliary supervision trial versus V5; no matched-V6 ordinary-control or mechanism claim yet'))


if __name__=='__main__':
    main()
