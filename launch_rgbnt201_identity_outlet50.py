"""Two serial NN lanes; completed bundles archive while next model trains."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from gpu_thermal_execute import execute
from launch_full_official_frozen_anchor_trial import archive_frozen, emit, save


def main():
    parser=argparse.ArgumentParser()
    for key in ('data-root','pretrained','anchor-run-dir','native-root','output'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    native_root=Path(args.native_root)
    native=json.loads((native_root/'controller_result.json').read_text())
    assert native['status']=='COMPLETE_IDENTITY_OUTLET_NATIVE_ONLY' and native['actual_updates']==12
    assert native['new_weight_files']==native['formal50_started']==0
    native_args=json.loads((native_root/'controller_launch.json').read_text())['arguments']
    assert all(native_args[key]==vars(args)[key] for key in ('data_root','pretrained','anchor_run_dir'))
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    for phase in ('training','frozen49','audit'):
        (root/phase).mkdir()
    save(root/'controller_launch.json',dict(pid=os.getpid(),started=time.time(),arguments=vars(args),
        gpus=[2,3],temperature_power_control=False,models=4,epochs_per_model=50))
    def archive_bundle(folder,run,job):
        archive_frozen(root,folder,job)
        path=run/'best_official_arrays.npz'
        file=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        emit(dict(event='NORMAL_READY',job=job,path=str(path),file=file))
        ack=json.loads(sys.stdin.readline())
        assert ack==dict(event='NORMAL_ARCHIVED',job=job,file=file)
        assert path.stat().st_size==file['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==file['sha256']
        path.unlink()
        save(run/'normal_local_archive.json',dict(status='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED',file=file))
        emit(dict(event='NORMAL_CLEARED',job=job))
    with ThreadPoolExecutor(max_workers=1) as archives:
        def lane(variant,gpu):
            rows=[]
            for bypass in (0,1):
                job='RGBNT201_identity_'+variant+('_bypass' if bypass else '_narrow')+'_s42'
                run,folder=root/'training'/job,root/'frozen49'/job
                trained=execute([sys.executable,'-u','run_rgbnt201_identity_outlet.py',
                    '--dataset','RGBNT201','--variant',variant,'--bypass',str(bypass),
                    '--freeze-identity-encoder','1','--seed','42','--mode','train',
                    '--data-root',args.data_root,'--pretrained',args.pretrained,
                    '--anchor-run-dir',args.anchor_run_dir,'--output',str(run)],root/'training',job,gpu)
                result=json.loads((run/'result.json').read_text())
                assert result['status']=='COMPLETE' and result['epochs']==50
                assert result['steps']==result['optimizer_steps']==2647 and result['amp_skipped_steps']==0
                assert result['training_coverage']==dict(eligible=3951,visited=3951,unvisited=[])
                assert result['descriptor_dim']==5120 and result['training_heldout_identities']==0
                control=native['controls'][variant]['controls'][str(bypass)]
                assert (result['parameters'],result['trainable_parameters'])==(control['parameters'],control['trainable_parameters'])
                evaluated=execute([sys.executable,'-u','evaluate_rgbnt201_identity_states49.py',
                    '--run-dir',str(run),'--output',str(folder)],root/'frozen49',job,gpu)
                audited=execute([sys.executable,'-u','audit_rgbnt201_identity_states49.py',
                    '--run-dir',str(run),'--evaluation',str(folder)],root/'audit',job,gpu)
                audit=json.loads((folder/'independent_fourstate_cpu_audit.json').read_text())
                assert audit['status']=='PASS' and audit['state_cases']==196
                future=archives.submit(archive_bundle,folder,run,job)
                rows.append((dict(name=job,variant=variant,bypass=bypass,gpu=gpu,
                    train=trained,evaluation=evaluated,audit=audited),future))
            return rows
        with ThreadPoolExecutor(max_workers=2) as lanes:
            futures=[lanes.submit(lane,variant,gpu) for variant,gpu in (('frequency_shared',2),('axis_shared',3))]
            rows=[row for future in futures for row in future.result()]
        for row,future in rows:
            future.result()
    def orders(job):
        data=[json.loads(line) for line in (root/'training'/job/'batch_orders.jsonl').read_text().splitlines()]
        return [(r['epoch'],r['step'],r['names'],r['partial_set']) for r in data]
    reference=orders(rows[0][0]['name'])
    assert len(reference)==2647 and all(orders(row['name'])==reference for row,_ in rows)
    save(root/'controller_result.json',dict(status='COMPLETE',runs=[row for row,_ in rows],
        models=4,additional_epochs=200,successful_updates=10588,deployed_frozen_cases=196,
        closed_state_cases=784,paired_sampling_exact=True,all_raw_local_verified=True,
        limits='Original50 plus additional50 per model; seed42/benchmark-selected. Normal/overlapping interface '
               'test, not final any-to-any method. Full49 includes12 unsolved disjoint-source conditions.'))
    emit(dict(event='CONTROLLER_COMPLETE',runs=4,frozen_bundles=196,state_cases=784))


if __name__=='__main__':
    main()
