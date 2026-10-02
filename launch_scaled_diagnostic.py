"""Frozen V4 diagnostics after all eleven original development jobs finish."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from launch_runs import write_json
from launch_axis_scaled import SCHEDULE, idle


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    assert args.gpu in range(4)
    queue=Path('runs/axis_collaboration_v4_development')
    terminal=json.loads((queue/'controller_result.json').read_text())
    expected={dataset+'_'+variant+'_s42' for jobs in SCHEDULE.values() for dataset,variant in jobs}
    assert terminal['status']=='COMPLETE' and len(terminal['runs'])==11
    assert {row['name'] for row in terminal['runs']}==expected and all(row['exit_code']==0 for row in terminal['runs'])
    for name in expected:
        result=json.loads((queue/name/'result.json').read_text())
        assert result['status']=='COMPLETE' and result['epochs']==50
        assert json.loads((queue/(name+'_exit.json')).read_text())['exit_code']==0
    idle(args.gpu)
    output=Path('runs/axis_collaboration_v4_four_state_diagnostic')
    output.mkdir(exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    checks=[]
    for smoke in (True,False):
        for dataset in ('MSVR310','RGBNT201','RGBNT100'):
            name=dataset+('_smoke' if smoke else '_full')
            run=queue/(dataset+'_axis_scaled_fullref_s42')
            command=[sys.executable,'-u','diagnose_axis_collaboration.py','--run-dir',str(run),'--output',str(output/name)]
            if smoke:command.append('--smoke')
            with (output/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                write_json(output/(name+'_launch.json'),{'pid':child.pid,'gpu':args.gpu,'command':command})
                code=child.wait()
            row={'name':name,'exit_code':code}
            write_json(output/(name+'_exit.json'),row)
            checks.append(row)
            assert code==0,name
    write_json(output/'controller_result.json',{'status':'PASS','checks':checks,'optimizer_updates':0})


if __name__=='__main__':
    main()
