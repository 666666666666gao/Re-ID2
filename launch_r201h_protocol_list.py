"""Four matched H201 controls: all native3 first, then full50 with local raw ACK."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

from gpu_thermal_execute import execute
from launch_full_official_frozen_anchor_trial import save
from launch_r201g_normal_priority import CONTROLS

REFERENCES=('label','camera')
PRINT_LOCK=threading.Lock()


def emit(value):
    with PRINT_LOCK:
        print('R201H_STREAM '+json.dumps(value),flush=True)


def main():
    parser=argparse.ArgumentParser()
    for key in ('data-root','pretrained','anchor-root','output'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--mode',choices=('native','full'),required=True)
    args=parser.parse_args()
    root,anchors=Path(args.output),Path(args.anchor_root)

    def name(reference,variant):
        return 'RGBNT201_r201h_'+reference+'_'+variant+'_s42'

    def argv(reference,variant,mode,output):
        return [sys.executable,'-u','run_r201h_protocol_list.py','--reference-mask',reference,
            '--dataset','RGBNT201','--variant',variant,'--freeze-identity-encoder','1','--seed','42',
            '--mode',mode,'--data-root',args.data_root,'--pretrained',args.pretrained,
            '--anchor-run-dir',str(anchors/'RGBNT201_demo_s42'),'--output',str(output)]

    if args.mode=='native':
        root.mkdir(parents=True,exist_ok=False)
        (root/'native').mkdir()
        save(root/'native_controller_launch.json',dict(pid=os.getpid(),started=time.time(),arguments=vars(args),
            physical_gpus=[2,3],maximum_NN=2,temperature_power_control=False))

        def native_lane(variant,gpu):
            rows=[]
            for reference in REFERENCES:
                job=name(reference,variant)+'_smoke'
                output=root/'native'/job
                execute(argv(reference,variant,'smoke',output),root/'native',job,gpu)
                data=json.loads((output/'smoke.json').read_text())
                assert data['status']=='SMOKE_PASS' and data['steps']==data['attempts']==3
                assert data['amp_skipped_steps']==0 and data['strict_reload_equal']
                assert data['descriptor_shape']==[8,5120] and data['descriptor_dim']==5120
                assert all(data['gradients'].values()) and not list(output.glob('*.pth'))
                for detail in data['details']:
                    assert detail['reference_mask']==reference and detail['primary_full_metric']=='protocol_smooth_AP_unit5120'
                    assert detail['AP_feature_gradient_finite'] and detail['all64_observations_in_classification']
                    assert detail['loss']==detail['full_loss'] and detail['partial_loss_effective']==0
                    assert detail['partial_CE_weight']==detail['partial_triplet_weight']==0
                    assert detail['fused_BN_calls']==2 and detail['full_unit_max_error']<1e-6
                    assert len(detail['names'])==64 and detail['temperature']==.01 and detail['metric_coefficient']==1.
                rows.append(dict(reference=reference,variant=variant,gpu=gpu,job=job,result=data))
            return rows

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(native_lane,variant,gpu) for variant,gpu in CONTROLS]
            rows=[row for future in futures for row in future.result()]
        assert len(rows)==4 and len({(r['result']['parameters'],r['result']['trainable_parameters']) for r in rows})==1
        orders=[[(d['names'],d['partial_set']) for d in row['result']['details']] for row in rows]
        assert all(order==orders[0] for order in orders)
        save(root/'native_acceptance.json',dict(status='PASS',controls=4,actual_updates=12,paired_sampling_exact=True,
            strict_reload_equal=True,all_active_gradients=True,partial_forward_BN_retained=True,
            no_positive_batch_zero_AP_only=True,added_weight_files=0,rows=rows))
        emit(dict(event='NATIVE_PASS',controls=4,actual_updates=12))
        save(root/'native_controller_result.json',dict(status='COMPLETE',controls=4,native_updates=12,new_formal_updates=0))
        emit(dict(event='CONTROLLER_COMPLETE',mode='native',controls=4,native_updates=12,new_formal_updates=0))
        return

    native=json.loads((root/'native_acceptance.json').read_text())
    assert native['status']=='PASS' and native['controls']==4 and native['actual_updates']==12
    assert json.loads((root/'native_controller_result.json').read_text())['status']=='COMPLETE'
    for phase in ('training','audit'):
        (root/phase).mkdir(exist_ok=False)
    save(root/'controller_launch.json',dict(pid=os.getpid(),started=time.time(),arguments=vars(args),
        physical_gpus=[2,3],maximum_NN=2,temperature_power_control=False,reference_order=list(REFERENCES)))

    def archive(run,job):
        path=run/'best_official_arrays.npz'
        file=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        emit(dict(event='NORMAL_READY',name=job,path=str(path),file=file))
        assert json.loads(sys.stdin.readline())==dict(event='NORMAL_ARCHIVED',name=job,file=file)
        assert path.resolve().is_relative_to(root.resolve())
        assert path.stat().st_size==file['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==file['sha256']
        path.unlink()
        save(run/'normal_local_archive.json',dict(status='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED',file=file))
        emit(dict(event='NORMAL_CLEARED',name=job))

    rows=[]
    with ThreadPoolExecutor(max_workers=1) as archives:
        for reference in REFERENCES:
            def train(variant,gpu):
                job=name(reference,variant)
                output=root/'training'/job
                trained=execute(argv(reference,variant,'train',output),root/'training',job,gpu)
                data=json.loads((output/'result.json').read_text())
                assert data['status']=='COMPLETE' and data['epochs']==50
                assert data['steps']==data['optimizer_steps']==2647 and data['amp_skipped_steps']==0
                assert (data['train_records'],data['query_records'],data['gallery_records'])==(3951,836,836)
                assert data['training_heldout_identities']==0 and data['descriptor_dim']==5120
                assert data['training_coverage']==dict(eligible=3951,visited=3951,unvisited=[])
                batches=[json.loads(line) for line in (output/'batch_orders.jsonl').read_text().splitlines()]
                assert len(batches)==2647 and all(d['reference_mask']==reference and d['optimizer_updated'] and d['loss']==d['full_loss']
                    and d['partial_loss_effective']==d['partial_CE_weight']==d['partial_triplet_weight']==0
                    and d['primary_full_metric']=='protocol_smooth_AP_unit5120' and d['all64_observations_in_classification']
                    and d['temperature']==.01 and d['metric_coefficient']==1. for d in batches)
                audited=execute([sys.executable,'-u','audit_full_official_normal.py','--run-dir',str(output)],root/'audit',job,gpu)
                assert json.loads((output/'normal_cpu_audit.json').read_text())['status']=='PASS'
                return dict(name=job,reference=reference,variant=variant,gpu=gpu,train=trained,audit=audited),archives.submit(archive,output,job)

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(train,variant,gpu) for variant,gpu in CONTROLS]
                wave=[future.result() for future in futures]
            for row,future in wave:
                future.result()
                rows.append(row)

    orders=[]
    for row in rows:
        batches=[json.loads(line) for line in (root/'training'/row['name']/'batch_orders.jsonl').read_text().splitlines()]
        orders.append([(d['epoch'],d['step'],d['names'],d['partial_set']) for d in batches])
    assert len(rows)==4 and all(order==orders[0] for order in orders)
    save(root/'controller_result.json',dict(status='COMPLETE',dataset='RGBNT201',controls=4,runs=rows,
        additional_epochs=200,successful_updates=10588,native_updates=12,paired_sampling_exact=True,
        normal_archives_local_verified=4,descriptor_dim=5120,
        limits='Known AP/reference-mask comparison, original50 plus new50, benchmark-selected seed42; no new loss novelty, guaranteed +2, three-dataset or missing success.'))
    emit(dict(event='CONTROLLER_COMPLETE',mode='full',controls=4,successful_updates=10588,additional_epochs=200))


if __name__=='__main__':
    main()
