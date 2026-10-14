"""Observe two selected AV RGB models, smoke first; physical GPU2/3 only."""
import argparse,hashlib,json,os,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from gpu_thermal_execute import execute
from run_experiment import write_json

CONTROLS=(('frequency_shared',2),('axis_shared',3))


def main():
    parser=argparse.ArgumentParser()
    for key in ('training-root','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    for phase in ('smoke','full'):(root/phase).mkdir()
    write_json(root/'controller_launch.json',dict(pid=os.getpid(),started=time.time(),physical_gpus=[2,3],maximum_NN=2,
        temperature_power_control=False,controls=CONTROLS,optimizer_updates=0,new_weights=0))
    def run(pair,mode):
        variant,gpu=pair;name='RGBNT100_r201av_'+variant+'_s42'
        source=Path(args.training_root)/name;out=root/mode/name
        closed=execute([sys.executable,'-u','evaluate_r201aw_AV_frequency_outlets.py','--run-dir',str(source),
            '--normal-arrays',str(source/'best_official_arrays.npz'),'--mode',mode,
            '--output',str(out)],root/mode,name,gpu)
        result=json.loads((out/'result.json').read_text())
        assert result['status']==('SMOKE_PASS' if mode=='smoke' else 'COMPLETE')
        assert result['optimizer_updates']==result['new_weights']==0
        assert result['state_tensor_versions_unchanged'] and len(result['stages'])==15
        assert result['attention_fields']==275
        if mode=='smoke':assert result['hook_vs_unhooked_same_batch_exact'] and result['output_features_exact_selected_cache']
        else:
            assert result['normal_features_and_distance_exact'] and result['query_records']==1715 and result['gallery_records']==8575 and result['spatial_records']==10290
            assert json.loads((out/'independent_cpu_audit.json').read_text())['status']=='PASS'
        return dict(name=name,variant=variant,gpu=gpu,exit=closed,output=str(out))
    with ThreadPoolExecutor(max_workers=2) as pool:smoke=list(pool.map(lambda pair:run(pair,'smoke'),CONTROLS))
    write_json(root/'native_acceptance.json',dict(status='PASS',controls=2,observed_records=128,plain_parity_records=128,
        actual_forward_record_executions=256,optimizer_updates=0,new_weights=0,records_per_control=64,exits=smoke))
    print('AW_NATIVE_PASS',flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:full=list(pool.map(lambda pair:run(pair,'full'),CONTROLS))
    raw={}
    for row in full:
        path=Path(row['output'])/'raw.npz'
        raw[row['name']]=dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write_json(root/'controller_result.json',dict(status='COMPLETE',runs=full,smoke=smoke,raw=raw,
        physical_gpus=[2,3],maximum_NN=2,cases=30,official_forward_record_executions=20580,optimizer_updates=0,new_weights=0,
        local_transfer_completed=False,scope='Full normal RGBNT100 selected-outlet diagnostic, no retraining,missing,49,seeds or201NN.'))
    print('AW_CONTROLLER_COMPLETE',flush=True)


if __name__=='__main__':main()
