"""One native batch then all official Q/G; only physical GPU3, zero updates."""
import argparse,json,os,sys,time
from pathlib import Path
from gpu_thermal_execute import execute
from run_experiment import write_json


def main():
    parser=argparse.ArgumentParser()
    for key in ('run-dir','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    for phase in ('native','full'):(root/phase).mkdir()
    write_json(root/'controller_launch.json',dict(pid=os.getpid(),started=time.time(),physical_gpus=[3],maximum_NN=1,optimizer_updates=0))
    stages={}
    for phase in ('native','full'):
        out=root/phase/'AQ_axis_shared'
        exited=execute([sys.executable,'-u','probe_r201ar_AQ_spatial_attention.py','--run-dir',args.run_dir,'--output',str(out),'--phase',phase],root/phase,'AQ_axis_shared',3)
        result=json.loads((out/'result.json').read_text())
        assert result['status']==('SMOKE_PASS' if phase=='native' else 'COMPLETE')
        assert result['records']==(64 if phase=='native' else 10290) and result['attention_fields']==182
        assert result['optimizer_updates']==result['checkpoint_writes']==0 and result['output_features_exact_saved_selected_cache']
        stages[phase]=dict(exit=exited,result=str(out/'result.json'),records=result['records'])
        if phase=='native':write_json(root/'native_acceptance.json',dict(status='PASS',records=64,optimizer_updates=0))
    write_json(root/'controller_result.json',dict(status='COMPLETE',stages=stages,physical_gpus=[3],maximum_NN=1,
        optimizer_updates=0,checkpoint_writes=0,native_records=64,full_query_records=1715,full_gallery_records=8575,output_features_exact=True))
    print('AR_FULL_SPATIAL_ATTENTION_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
