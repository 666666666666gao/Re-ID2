"""One frozen V5 checkpoint per dataset, four states and thirteen missing conditions."""
import argparse
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
    parser.add_argument('--output',required=True)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--pretrained',required=True)
    args=parser.parse_args()
    training=Path(args.training_root);output=Path(args.output)
    output.mkdir(exist_ok=False)
    rows=[]
    for dataset in ('MSVR310','RGBNT201','RGBNT100'):
        name=dataset+'_axis_mass_fullref_s42';run=training/name
        exit_file=run.parent/(name+'_exit.json')
        while not exit_file.exists():
            print('WAIT_V5_CLEAN50',name,flush=True)
            time.sleep(240)
        assert json.loads(exit_file.read_text())['exit_code']==0,name
        terminal=json.loads((run/'result.json').read_text())
        assert terminal['status']=='COMPLETE' and terminal['epochs']==50
        assert terminal['arguments']['variant']=='axis_mass_fullref' and terminal['arguments']['dataset']==dataset
        inputs=(run/'best.pth',run/'best_dev_arrays.npz',run/'result.json',exit_file)
        proofs={str(path):dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in inputs}
        folder=output/name;folder.mkdir(exist_ok=False)
        write_json(folder/'frozen_inputs.json',proofs)
        stages=[]
        for script,kind in (('diagnose_mass_axis.py','four_state'),('missing_mass_development.py','missing')):
            for smoke in (True,False):
                idle(3)
                stage=kind+('_smoke' if smoke else '_full')
                argv=[sys.executable,'-u',script,'--run-dir',str(run),'--output',str(folder/stage),'--data-root',args.data_root,'--pretrained',args.pretrained]
                if smoke:argv.append('--smoke')
                with (folder/(stage+'.log')).open('x') as log:
                    child=subprocess.Popen(argv,env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
                    write_json(folder/(stage+'_launch.json'),dict(pid=child.pid,gpu=3,command=argv,started=time.time()))
                    code=child.wait()
                row=dict(stage=stage,exit_code=code,finished=time.time());write_json(folder/(stage+'_exit.json'),row)
                assert code==0,(name,stage)
                result_name='smoke.json' if smoke else ('diagnostic.json' if kind=='four_state' else 'result.json')
                result=json.loads((folder/stage/result_name).read_text())
                assert result['optimizer_updates']==0 and result['state_tensor_versions_unchanged']
                assert result['normal_inference_feature_max_error' if kind=='four_state' and not smoke else 'normal_feature_max_error']==0
                if smoke:assert result['status']=='PASS'
                elif kind=='four_state':assert len(result['metrics'])==4 and result['normal_distance_source'].startswith('saved best_dev_arrays.npz/distances')
                else:assert result['status']=='COMPLETE' and len(result['measurements'])==13
                for path in inputs:
                    assert path.stat().st_size==proofs[str(path)]['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==proofs[str(path)]['sha256']
                stages.append(row)
        record=dict(name=name,gpu=3,stages=stages,optimizer_updates=0,official_test_uses=0,normal_feature_max_error=0)
        write_json(folder/'controller_result.json',dict(status='PASS',**record));rows.append(record)
    write_json(output/'controller_result.json',dict(status='COMPLETE',runs=rows,missing_conditions=39,four_state_conditions=12,optimizer_updates=0,official_test_uses=0,scope='Frozen seed42 identity-heldout development diagnostics; no trained ablation or final test claim'))


if __name__=='__main__':
    main()
