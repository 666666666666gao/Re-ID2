"""Fixed V4 tensor/smoke gates, then eleven same-budget development runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from launch_runs import write_json


SCHEDULE = {
    0: [('MSVR310','axis_scaled_fullref'), ('RGBNT100','frequency_scaled_fullref'), ('MSVR310','axis_raw_fullref')],
    1: [('RGBNT100','axis_scaled_fullref'), ('MSVR310','frequency_scaled_fullref')],
    2: [('RGBNT100','plain_scaled_fullref'), ('MSVR310','plain_scaled_fullref')],
    3: [('RGBNT201','axis_scaled_fullref'), ('RGBNT201','plain_scaled_fullref'),
        ('RGBNT201','frequency_scaled_fullref'), ('MSVR310','axis_scaled_base')],
}


def idle(gpu):
    while True:
        used = int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used < 500:
            return
        print('WAIT_V4_GPU',gpu,used,flush=True)
        time.sleep(240)


def execute(command, output, name, gpu):
    idle(gpu)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with (output/(name+'.log')).open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        write_json(output/(name+'_launch.json'),{'pid':child.pid,'gpu':gpu,'started':time.time(),'command':command})
        code=child.wait()
    row={'name':name,'exit_code':code,'finished':time.time()}
    write_json(output/(name+'_exit.json'),row)
    assert code==0,name
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--pretrained',required=True)
    args=parser.parse_args()
    predecessor=Path('runs/axis_collaboration_v3_development')
    terminal=json.loads((predecessor/'controller_result.json').read_text())
    assert len(terminal)==10 and all(row['exit_code']==0 for row in terminal)
    for row in terminal:
        result=json.loads((predecessor/row['run']/'result.json').read_text())
        assert result['status']=='COMPLETE' and result['epochs']==50
    diagnostic=json.loads(Path('runs/axis_collaboration_v3_four_state_diagnostic/controller_result.json').read_text())
    assert diagnostic['status']=='PASS' and len(diagnostic['checks'])==6 and all(row['exit_code']==0 for row in diagnostic['checks'])
    preflight=Path('runs/axis_collaboration_v4_preflight')
    preflight.mkdir(exist_ok=False)
    checks=[]
    for dataset in ('MSVR310','RGBNT201','RGBNT100'):
        command=[sys.executable,'-u','verify_axis_scaled.py','--dataset',dataset,'--data-root',args.data_root,
                 '--pretrained',args.pretrained,'--output',str(preflight/(dataset+'_tensor'))]
        checks.append(execute(command,preflight,dataset+'_tensor',1))
    for jobs in SCHEDULE.values():
        for dataset,variant in jobs:
            name=dataset+'_'+variant+'_smoke'
            command=[sys.executable,'-u','run_experiment.py','--dataset',dataset,'--variant',variant,'--seed','42',
                     '--mode','smoke','--contribution-weight','.05','--data-root',args.data_root,'--pretrained',args.pretrained,
                     '--output',str(preflight/name)]
            checks.append(execute(command,preflight,name,1))
    assert len(checks)==14
    write_json(preflight/'controller_result.json',{'status':'PASS','checks':checks})
    root=Path('runs/axis_collaboration_v4_development')
    root.mkdir(exist_ok=False)

    def slot(gpu):
        rows=[]
        for dataset,variant in SCHEDULE[gpu]:
            name=dataset+'_'+variant+'_s42'
            command=[sys.executable,'-u','run_experiment.py','--dataset',dataset,'--variant',variant,'--seed','42',
                     '--mode','train','--contribution-weight','.05','--data-root',args.data_root,'--pretrained',args.pretrained,
                     '--output',str(root/name)]
            rows.append(execute(command,root,name,gpu))
        return rows

    with ThreadPoolExecutor(max_workers=4) as workers:
        futures=[workers.submit(slot,gpu) for gpu in SCHEDULE]
        rows=[row for future in futures for row in future.result()]
    assert len(rows)==11
    write_json(root/'controller_result.json',{'status':'COMPLETE','runs':rows})


if __name__=='__main__':
    main()
