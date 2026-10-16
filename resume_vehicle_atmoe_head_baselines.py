"""Use completed native evidence under the original dynamic-AMP update contract."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from gpu_thermal_execute import execute
from launch_vehicle_atmoe_head_baselines import CASES, save


def main():
    parser = argparse.ArgumentParser()
    for name in ('previous-root','data-root','pretrained','output'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    previous = Path(args.previous_root)
    old = previous/'experiments'
    owner = json.loads((previous/'remote_owner.json').read_text())
    assert not Path('/proc/'+str(owner['pid'])).exists()
    assert not list((old/'training').iterdir())
    children = []
    for phase in ('parity','native'):
        for file in sorted((old/phase).glob('*_launch.json')):
            launch = json.loads(file.read_text())
            ended = json.loads(file.with_name(file.name.replace('_launch.json','_exit.json')).read_text())
            assert ended['exit_code'] == 0 and not Path('/proc/'+str(launch['pid'])).exists()
            children.append(dict(launch=launch,exit=ended))
    assert len(children) == 4
    evidence = []
    for case in CASES:
        parity = json.loads((old/'parity'/(case['dataset']+'_atmoe8_default_parity')/'result.json').read_text())
        assert parity['status'] == 'ACTUAL_VEHICLE_ATMOE_HEAD8_DEFAULT_CONFIG_STATE_RNG_FP32_AMP_PARITY_PASS'
        assert all(parity[key] for key in ('config_dump_exact','state_tensors_exact','CPU_CUDA_RNG_exact','fp32_output_exact','autocast_output_exact'))
        assert parity['parameters'] == case['original_parameters'] and parity['descriptor_dim'] == case['dimension']
        assert (parity['train_records'],parity['query_records'],parity['gallery_records']) == (case['train'],case['query'],case['gallery'])
        assert parity['forward_record_executions'] == 32 and parity['optimizer_updates'] == parity['checkpoint_writes'] == 0
        name = case['dataset']+'_atmoe'+str(case['heads'])+'_'+case['variant']+'_s42'
        native = json.loads((old/'native'/(name+'_smoke')/'smoke.json').read_text())
        details = native['details']
        assert native['status'] == 'SMOKE_PASS' and native['steps'] == 3
        assert all(isinstance(row['optimizer_updated'],bool) for row in details)
        assert len(details) == native['attempts'] and sum(row['optimizer_updated'] for row in details) == 3
        assert native['amp_skipped_steps'] == native['attempts']-native['steps']
        assert all(math.isfinite(row['loss']) and math.isfinite(row['amp_scale']) and row['amp_scale']>0 for row in details)
        assert native['strict_reload_equal'] and all(native['gradients'].values())
        assert native['descriptor_shape'] == [8,case['dimension']] and native['parameters'] == case['parameters']
        assert native['arguments']['atmoe_heads'] == case['heads']
        assert native['arguments']['dataset'] == case['dataset'] and native['arguments']['variant'] == case['variant']
        assert native['arguments']['seed'] == 42
        assert (native['train_records'],native['query_records'],native['gallery_records']) == (case['train'],case['query'],case['gallery'])
        assert native['training_heldout_identities'] == 0
        evidence.append(dict(dataset=case['dataset'],parity=parity,native=native))
    attempts = sum(row['native']['attempts'] for row in evidence)
    updates = sum(row['native']['steps'] for row in evidence)
    root = Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    for phase in ('training','audit'):
        (root/phase).mkdir()
    save(root/'native_acceptance.json',dict(status='PASS_ORIGINAL_NATIVE_DYNAMIC_AMP_ACCOUNTING',
        previous_root=str(previous),evidence_reused=True,new_native_updates=0,
        successful_optimizer_updates=updates,attempts=attempts,amp_skips=attempts-updates,
        parity_forward_record_executions=64,evidence=evidence,old_children=children,
        limits='Original GradScaler512 and original native SMOKE_PASS semantics retained. Final gradients and8query strict reload do not certify all steps or BatchNorm buffers; skipped forwards are not rolled back. Original zero-skip controller failure remains intact.'))
    save(root/'controller_launch.json',dict(pid=os.getpid(),started=time.time(),arguments=vars(args),
        cases=CASES,physical_gpus=[2,3],maximum_NN=2,
        scope='Explicit new full50 owner after diagnosed scheduler-accounting failure, not a native retry or model/AMP change. Two identity baselines; no RGBNT201, missing or seed search.'))

    def train(case):
        name = case['dataset']+'_atmoe'+str(case['heads'])+'_'+case['variant']+'_s42'
        out = root/'training'/name
        ended = execute([sys.executable,'-u','run_vehicle_atmoe_head_baseline.py',
            '--atmoe-heads',str(case['heads']),'--dataset',case['dataset'],'--variant',case['variant'],
            '--seed','42','--mode','train','--data-root',args.data_root,'--pretrained',args.pretrained,
            '--output',str(out)],root/'training',name,case['gpu'])
        data = json.loads((out/'result.json').read_text())
        assert data['status'] == 'COMPLETE' and data['epochs'] == 50 and data['training_heldout_identities'] == 0
        assert data['arguments']['atmoe_heads'] == case['heads'] and data['parameters'] == case['parameters']
        assert data['descriptor_dim'] == case['dimension']
        assert (data['train_records'],data['query_records'],data['gallery_records']) == (case['train'],case['query'],case['gallery'])
        assert 0 < data['optimizer_steps'] <= data['steps']
        assert data['amp_skipped_steps'] == data['steps']-data['optimizer_steps']
        assert data['training_coverage'] == dict(eligible=case['train'],visited=case['train'],unvisited=[])
        audit = execute([sys.executable,'-u','audit_vehicle_atmoe_head_normal.py','--run-dir',str(out)],
            root/'audit',name,case['gpu'])
        checked = json.loads((out/'normal_cpu_audit.json').read_text())
        assert checked['status'] == 'PASS'
        assert (checked['attempts'],checked['successful_optimizer_updates'],checked['amp_skips']) == (data['steps'],data['optimizer_steps'],data['amp_skipped_steps'])
        files = {}
        for key,filename in (('raw','best_official_arrays.npz'),('best','best.pth')):
            file = out/filename
            files[key] = dict(path=str(file),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        result = dict(name=name,dataset=case['dataset'],variant=case['variant'],heads=case['heads'],
            gpu=case['gpu'],train=ended,audit=audit,selected_epoch=data['best']['epoch'],
            attempts=data['steps'],successful_updates=data['optimizer_steps'],amp_skips=data['amp_skipped_steps'],
            full_metrics=data['full_metrics'],**files,
            paper_target=dict(mAP=case['target'][0],Rank_1=case['target'][1]),
            paper_both_plus1=data['full_metrics']['mAP']>=case['target'][0] and data['full_metrics']['Rank-1']>=case['target'][1],
            interpretation='Fresh identity baseline only, not new dual-axis contribution. Same50epochs; actual updates and skips are reported, no equality-of-updates claim.')
        save(root/(case['dataset']+'_component_complete.json'),result)
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        trained = list(pool.map(train,CASES))
    save(root/'controller_result.json',dict(status='COMPLETE',cases=2,runs=trained,
        native_successful_updates_reused=updates,native_attempts_reused=attempts,native_amp_skips_reused=attempts-updates,
        new_native_updates=0,formal_epochs=100,
        formal_attempts=sum(row['attempts'] for row in trained),
        formal_optimizer_updates=sum(row['successful_updates'] for row in trained),
        formal_amp_skips=sum(row['amp_skips'] for row in trained),
        full_training_coverage=True,normal_GT_audited=True,physical_gpus=[2,3],maximum_NN=2,
        missing_evaluated=False,raw_preserved_pending_local_archive=True,
        limits='Original model, GradScaler512, data and50epoch rules. Actual dynamic AMP skips are not updates. Historical head8 controls, benchmark-selected checkpoints; no equal-update or dual-axis gain claim.'))
    print('VEHICLE_ATMOE_DYNAMIC_AMP_ACCOUNTED_BASELINES_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
