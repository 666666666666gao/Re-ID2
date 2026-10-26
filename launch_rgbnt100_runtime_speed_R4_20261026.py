"""Run one inference arm per fresh process, without sampler threads in CUDA workers."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'controller_launch.json').write_text(json.dumps(dict(pid=os.getpid(),started=time.time(),argv=sys.argv,physical_GPU=2,max_NN=1))+'\n')
    arms=[]
    for index in range(8):
        folder=out/('arm_'+str(index));argv=[sys.executable,'-u','profile_rgbnt100_runtime_speed_R4_20261026.py',
            '--run-dir',args.run_dir,'--output',str(folder),'--mode','inference','--gpu','2','--arm',str(index)]
        with (out/('arm_'+str(index)+'.log')).open('x') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
            (out/('arm_'+str(index)+'_launch.json')).write_text(json.dumps(dict(pid=child.pid,started=time.time(),argv=argv))+'\n')
            code=child.wait()
        (out/('arm_'+str(index)+'_exit.json')).write_text(json.dumps(dict(pid=child.pid,exit_code=code,finished=time.time()))+'\n')
        assert code==0
        value=json.loads((folder/'result.json').read_text());assert len(value['arms'])==1 and value['total_full_QG_passes']==1
        arms.extend(value['arms'])
    assert len(arms)==8 and [r['arm'] for r in arms]==list(range(8))
    result=dict(status='COMPLETE_INDEPENDENT_PROCESS_FULL_QG_RUNTIME_SCREEN',arms=arms,total_full_QG_passes=8,finished=time.time(),max_NN=1,physical_GPU=2,optimizer_updates=0,checkpoint_writes=0,raw_array_writes=0,
        limits='One fresh model process per timing arm after actual R3 NVML thread timeout and inter-arm stall. No internal GPU-query thread or nvidia-smi in inference worker. Torch allocation peaks and CPU counters retained; no inference NVML utilization measurement. Timed extract includes full loader startup, not model construction. All source/precision/settings/GT unchanged. Prior135trainingupdates reused.')
    (out/'controller_result.json').write_text(json.dumps(result,indent=2)+'\n');print('INDEPENDENT_RUNTIME_SCREEN_COMPLETE',flush=True)


if __name__=='__main__':main()
