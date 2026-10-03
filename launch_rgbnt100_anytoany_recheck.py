"""One same-host RGBNT100 DeMo recheck after observed cross-host AP/INP drift."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
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
    parser.add_argument('--training-root',required=True)
    parser.add_argument('--variant',choices=['demo','axis_mass_fullref'],required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--pretrained',required=True)
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(exist_ok=False)
    jobs=[('RGBNT100',0)]

    def slot(job):
        dataset,gpu=job
        name=dataset+'_'+args.variant+'_s42';run=Path(args.training_root)/name
        exit_file=run/'exit.json' if args.variant=='demo' else run.parent/(name+'_exit.json')
        assert json.loads(exit_file.read_text())['exit_code']==0
        terminal=json.loads((run/'result.json').read_text())
        assert terminal['status']=='COMPLETE' and terminal['epochs']==50
        assert terminal['arguments']['dataset']==dataset and terminal['arguments']['variant']==args.variant
        inputs=[run/'best.pth',run/'best_dev_arrays.npz',run/'result.json',exit_file]
        proofs={str(path):dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in inputs}
        folder=output/name;folder.mkdir(exist_ok=False)
        write_json(folder/'frozen_inputs.json',proofs)
        stages=[]
        for smoke in (True,False):
            idle(gpu)
            stage='smoke' if smoke else 'full'
            argv=[sys.executable,'-u','missing_anytoany_development.py','--run-dir',str(run),'--output',str(folder/stage),
                '--data-root',args.data_root,'--pretrained',args.pretrained]
            if smoke:argv.append('--smoke')
            with (folder/(stage+'.log')).open('x') as log:
                child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                write_json(folder/(stage+'_launch.json'),dict(pid=child.pid,gpu=gpu,command=argv,started=time.time()))
                code=child.wait()
            row=dict(stage=stage,exit_code=code,finished=time.time())
            write_json(folder/(stage+'_exit.json'),row)
            assert code==0,(name,stage)
            result=json.loads((folder/stage/('smoke.json' if smoke else 'result.json')).read_text())
            assert result['status']==('PASS' if smoke else 'COMPLETE')
            assert result['optimizer_updates']==0 and result['state_tensor_versions_unchanged'] and result['normal_feature_max_error']==0
            if not smoke:assert len(result['measurements'])==49
            for path in inputs:
                assert path.stat().st_size==proofs[str(path)]['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==proofs[str(path)]['sha256']
            stages.append(row)
        record=dict(status='PASS',name=name,gpu=gpu,stages=stages,availability_pairs=49,optimizer_updates=0,official_test_uses=0)
        write_json(folder/'controller_result.json',record)
        return record

    with ThreadPoolExecutor(max_workers=1) as pool:runs=list(pool.map(slot,jobs))
    write_json(output/'controller_result.json',dict(status='COMPLETE',runs=runs,availability_pairs=49,
        optimizer_updates=0,official_test_uses=0,scope='Frozen seed42 development full7x7 availability evaluation, no missing-input retraining'))


if __name__=='__main__':main()
