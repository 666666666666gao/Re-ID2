"""Both native controls must pass before full Q/G; one process per GPU2/3."""
import argparse,json,os,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from gpu_thermal_execute import execute
from run_experiment import write_json


def main():
    parser=argparse.ArgumentParser()
    for key in ('as-root','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    for phase in ('native','full'):(root/phase).mkdir()
    controls=[('frequency_shared',2),('axis_shared',3)]
    write_json(root/'controller_launch.json',dict(pid=os.getpid(),started=time.time(),physical_gpus=[2,3],maximum_NN=2,optimizer_updates=0))
    stages={}
    for phase in ('native','full'):
        def run(control):
            variant,gpu=control;name='AS_'+variant;out=root/phase/name
            run_dir=Path(args.as_root)/'RGBNT100/training'/('RGBNT100_r201as_'+variant+'_s42')
            exited=execute([sys.executable,'-u','probe_r201at_AS_spatial_attention.py','--run-dir',str(run_dir),'--output',str(out),'--phase',phase],root/phase,name,gpu)
            result=json.loads((out/'result.json').read_text())
            assert result['status']==('SMOKE_PASS' if phase=='native' else 'COMPLETE')
            assert result['records']==(64 if phase=='native' else 10290) and result['attention_fields']==275
            assert result['optimizer_updates']==result['checkpoint_writes']==0 and result['output_features_exact_saved_selected_cache']
            return dict(model=name,gpu=gpu,exit=exited,result=str(out/'result.json'),records=result['records'])
        with ThreadPoolExecutor(max_workers=2) as pool:stages[phase]=list(pool.map(run,controls))
        if phase=='native':write_json(root/'native_acceptance.json',dict(status='PASS',controls=stages[phase],records_per_control=64,optimizer_updates=0))
    write_json(root/'controller_result.json',dict(status='COMPLETE',stages=stages,physical_gpus=[2,3],maximum_NN=2,
        optimizer_updates=0,checkpoint_writes=0,native_records_per_control=64,full_query_records_per_control=1715,full_gallery_records_per_control=8575,output_features_exact=True))
    print('AT_AS_BOTH_FULL_SPATIAL_ATTENTION_COMPLETE',flush=True)


if __name__=='__main__':main()
