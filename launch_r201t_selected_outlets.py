"""Two fixed T MS normal-outlet jobs on authorized2026 GPU0/1; live S RGB2/3 unchanged."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from evaluate_r201t_selected_outlets import PREFIX, STAGES
from gpu_01_diagnostic_execute import check_limits as check01
from launch_full_official_frozen_anchor_trial import save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--jobs',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    jobs=json.loads(Path(args.jobs).read_text())['jobs']
    assert len(jobs)==2 and {j['gpu'] for j in jobs}=={0,1}
    assert {(j['dataset'],j['variant']) for j in jobs}=={(d,v) for d in ('MSVR310',) for v in ('frequency_shared','axis_shared')}
    assert all(j['seed']==42 and j['gpu']=={'MSVR310':{'frequency_shared':0,'axis_shared':1}}[j['dataset']][j['variant']] for j in jobs)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);exchange=threading.Lock()
    save(out/'controller_launch.json',dict(pid=os.getpid(),started=time.time(),gpus=[0,1],optimizer_updates=0,temperature_power_control=False))
    def lane(job):
        name,gpu,run=job['name'],job['gpu'],Path(job['run'])
        selected=json.loads((run/'result.json').read_text())
        assert selected['best']['epoch']==job['selected_epoch'] and selected['full_metrics']==job['full_metrics']
        check=check01
        sample=check(gpu)
        while sample['memory_used_mib']>=500:
            with exchange:print('WAIT_SELECTED_GPU',gpu,sample['memory_used_mib'],flush=True)
            time.sleep(240);sample=check(gpu)
        argv=[sys.executable,'-u','evaluate_r201t_selected_outlets.py','--run-dir',str(run),'--normal-arrays',job['normal_restore'],'--output',str(out/name)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        with (out/(name+'.log')).open('x') as log:
            child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env,start_new_session=True)
            try:
                assert os.getpgid(child.pid)==child.pid
                save(out/(name+'_launch.json'),dict(pid=child.pid,process_group=child.pid,gpu=gpu,command=argv,started=time.time(),initial_resources=sample))
                ready,cleared,finished=False,False,False
                for line in child.stdout:
                    log.write(line);log.flush()
                    if not line.startswith(PREFIX):continue
                    message=json.loads(line[len(PREFIX):]);event=message['event']
                    if event=='RAW_READY':
                        assert message['condition']=='normal_outlets' and message['installed_gt_cases']==len(STAGES) and not ready;ready=True
                        with exchange:
                            print(PREFIX+json.dumps(dict(job=name,**message)),flush=True)
                            ack=json.loads(sys.stdin.readline())
                            assert ack==dict(job=name,event='ARCHIVED',condition=message['condition'],file=message['file'])
                            ack.pop('job');child.stdin.write(json.dumps(ack)+'\n');child.stdin.flush()
                    elif event=='RAW_CLEARED':
                        assert ready and not cleared and message['condition']=='normal_outlets';cleared=True
                        with exchange:print(PREFIX+json.dumps(dict(job=name,**message)),flush=True)
                    else:
                        assert event=='EVALUATION_COMPLETE' and not finished and message['conditions']==1 and message['cases']==len(STAGES) and message['optimizer_updates']==0 and ready and cleared;finished=True
                child.stdin.close();child.wait()
            finally:
                forced=child.poll() is None
                if forced:os.killpg(child.pid,signal.SIGKILL);child.wait()
                save(out/(name+'_exit.json'),dict(name=name,gpu=gpu,exit_code=child.returncode,finished=time.time(),forced_own_group_cleanup=forced))
        assert child.returncode==0 and finished
        result=json.loads((out/name/'result.json').read_text())
        assert result['status']=='COMPLETE' and result['stages']==list(STAGES) and result['normal_features_and_distance_exact'] and result['optimizer_updates']==result['new_weights']==0
        return dict(name=name,gpu=gpu,selected_epoch=job['selected_epoch'])
    with ThreadPoolExecutor(max_workers=2) as pool:completed=list(pool.map(lane,jobs))
    save(out/'controller_result.json',dict(status='COMPLETE',runs=completed,controls=2,normal_GT_stage_cases=2*len(STAGES),missing_GT_cases=0,optimizer_updates=0,new_weights=0))
    print(PREFIX+json.dumps(dict(event='CONTROLLER_COMPLETE',controls=2,conditions=2,cases=2*len(STAGES),optimizer_updates=0)),flush=True)


if __name__=='__main__':main()
