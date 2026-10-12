"""Two fixed Z MSVR adapting-modality normal-four-state jobs on0/1; no training/missing."""
import argparse,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from evaluate_r201z_normal4_states import PREFIX
from gpu_01_diagnostic_execute import check_limits
from launch_full_official_frozen_anchor_trial import save

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output);jobs=json.loads(Path(args.jobs).read_text())['jobs']
    assert len(jobs)==2 and {j['variant'] for j in jobs}=={'frequency_shared','axis_shared'} and len({j['name'] for j in jobs})==2
    assert all(j['dataset']=='MSVR310' and j['seed']==42 and j['gpu']=={'frequency_shared':0,'axis_shared':1}[j['variant']] and '_r201z_' in j['name'] for j in jobs)
    out.mkdir(parents=True,exist_ok=False);exchange=threading.Lock()
    save(out/'controller_launch.json',dict(pid=os.getpid(),started=time.time(),arguments=vars(args),gpus=[0,1],temperature_power_control=False,new_optimizer_updates=0))
    def lane(job):
        name,gpu,run=job['name'],job['gpu'],Path(job['run']);selected=json.loads((run/'result.json').read_text())
        assert selected['best']['epoch']==job['selected_epoch'] and selected['full_metrics']==job['full_metrics'] and selected['method_revision'].startswith('Z MS-only existing-modality-parameter gradient factor')
        assert selected['anchor']['final_full_CE_weight']==.05 and selected['anchor']['parent_contribution_regression_weight']==0.
        assert selected['anchor']['freeze_identity_encoder'] is False and selected['anchor']['partial_identity_adaptation']['adapted_parameters']==9958912
        restored=run/'best_official_arrays_restore_Z_normal4.npz';assert restored.is_file()
        sample=check_limits(gpu)
        while sample['memory_used_mib']>=500:
            with exchange:print('WAIT_SELECTED_GPU',gpu,sample['memory_used_mib'],flush=True)
            time.sleep(240);sample=check_limits(gpu)
        argv=[sys.executable,'-u','evaluate_r201z_normal4_states.py','--run-dir',str(run),'--normal-arrays',str(restored),'--output',str(out/name)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        with (out/(name+'.log')).open('x') as log:
            child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env,start_new_session=True)
            try:
                assert os.getpgid(child.pid)==child.pid
                save(out/(name+'_launch.json'),dict(pid=child.pid,process_group=child.pid,gpu=gpu,command=argv,started=time.time(),initial_resources=sample,temperature_power_control=False))
                ready,cleared,finished=False,False,False
                for line in child.stdout:
                    log.write(line);log.flush()
                    if not line.startswith(PREFIX):continue
                    message=json.loads(line[len(PREFIX):]);event=message['event']
                    if event=='RAW_READY':
                        assert message['condition']=='q_RNT_g_RNT' and message['installed_gt_cases']==4 and not ready;ready=True
                        with exchange:
                            print(PREFIX+json.dumps(dict(job=name,**message)),flush=True);ack=json.loads(sys.stdin.readline())
                            assert ack==dict(job=name,event='ARCHIVED',condition=message['condition'],file=message['file'])
                            ack.pop('job');child.stdin.write(json.dumps(ack)+'\n');child.stdin.flush()
                    elif event=='RAW_CLEARED':
                        assert ready and not cleared and message['condition']=='q_RNT_g_RNT';cleared=True
                        with exchange:print(PREFIX+json.dumps(dict(job=name,**message)),flush=True)
                    else:
                        assert event=='EVALUATION_COMPLETE' and not finished and message['conditions']==1 and message['cases']==4 and message['optimizer_updates']==0 and ready and cleared;finished=True
                child.stdin.close();child.wait()
            finally:
                forced=child.poll() is None
                if forced:os.killpg(child.pid,signal.SIGKILL);child.wait()
                exit_row=dict(name=name,gpu=gpu,exit_code=child.returncode,finished=time.time(),forced_own_group_cleanup=forced);save(out/(name+'_exit.json'),exit_row)
        assert exit_row['exit_code']==0 and finished
        result=json.loads((out/name/'result.json').read_text());assert result['status']=='COMPLETE' and result['conditions']==1 and result['state_cases']==4 and result['optimizer_updates']==result['new_weights']==0 and result['normal_only']
        return dict(name=name,gpu=gpu,selected_epoch=job['selected_epoch'],result=str(out/name/'result.json'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,job) for job in jobs];completed=[f.result() for f in futures]
    save(out/'controller_result.json',dict(status='COMPLETE',controls=2,runs=completed,normal_GT_state_cases=8,missing_GT_state_cases=0,all_raw_local_verified=True,optimizer_updates=0,new_weights=0))
    print(PREFIX+json.dumps(dict(event='CONTROLLER_COMPLETE',controls=2,conditions=2,cases=8,optimizer_updates=0)),flush=True)

if __name__=='__main__':main()
