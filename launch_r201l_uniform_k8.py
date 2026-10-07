"""Matched normal-priority controls: native3, full50, complete installed GT."""
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

CONTROLS = (('frequency_shared', 2), ('axis_shared', 3))
PRINT_LOCK = threading.Lock()


def emit(value):
    with PRINT_LOCK:
        sys.stdout.write('R201L_STREAM ' + json.dumps(value) + '\n')
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'anchor-root', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--dataset', choices=('RGBNT100','MSVR310'), required=True)
    args = parser.parse_args()
    root, anchors = Path(args.output), Path(args.anchor_root)
    root.mkdir(parents=True, exist_ok=False)
    for phase in ('native', 'training', 'audit'):
        (root / phase).mkdir()
    save(root / 'controller_launch.json', dict(pid=os.getpid(), started=time.time(), arguments=vars(args),
        gpus=[2,3], maximum_NN=2, temperature_power_control=False, experiment='L only uniform B64/P8K8; unchanged K graph/loss/params/5120/partial0;201 K8 reused'))
    def argv(variant, mode, output):
        return [sys.executable, '-u', 'run_r201l_uniform_k8.py', '--dataset', args.dataset, '--variant', variant,
            '--freeze-identity-encoder', '1', '--seed', '42', '--mode', mode, '--data-root', args.data_root,
            '--pretrained', args.pretrained, '--anchor-run-dir', str(anchors / (args.dataset+'_demo_s42')), '--output', str(output)]

    def native(variant, gpu):
        name = args.dataset+'_r201l_' + variant + '_s42_smoke'
        output = root / 'native' / name
        execute(argv(variant, 'smoke', output), root / 'native', name, gpu)
        data = json.loads((output / 'smoke.json').read_text())
        assert data['status'] == 'SMOKE_PASS' and data['steps'] == data['attempts'] == 3
        assert data['amp_skipped_steps'] == 0 and data['strict_reload_equal'] and '  NUM_INSTANCE: 8' in data['config'].splitlines()
        assert data['descriptor_shape'] == [8,5120] and data['descriptor_dim'] == 5120
        assert all(data['gradients'].values()) and not list(output.glob('*.pth'))
        assert all(d['loss']==d['full_loss'] and d['partial_loss_effective']==0 and d['partial_CE_weight']==d['partial_triplet_weight']==0
                   and d['fused_BN_calls']==2 and d['full_unit_max_error']<1e-6 and len(d['names'])==64
                   and d['sampling_identities']==8 and d['sampling_instances']==8 and d['sampling_batch']==64
                   and d['primary_margin']==.3 and d['primary_metric_coefficient']==1.
                   and d['primary_margin_definition_equal'] and d['primary_feature_gradient_finite']
                   and d['auxiliary_original_soft_triplet'] and d['PI_outlet']=='shared_relation_local'
                   and d['relation_PI_native']['pi_input_shape']==[64,7,1024]
                   and d['relation_PI_native']['pi_output_shape']==[64,7,512]
                   and d['relation_PI_native']['per_relation_outside_slot_errors']==[0.]*7
                   and d['relation_PI_native']['state_10_parent_equal']
                   and d['relation_PI_native']['state_01_parent_equal'] for d in data['details'])
        return data

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(native, variant, gpu) for variant, gpu in CONTROLS]
        native_rows = [future.result() for future in futures]
    assert len({(r['parameters'],r['trainable_parameters']) for r in native_rows}) == 1
    assert [(d['names'],d['partial_set']) for d in native_rows[0]['details']] == [(d['names'],d['partial_set']) for d in native_rows[1]['details']]
    save(root / 'native_acceptance.json', dict(status='PASS', controls=2, actual_updates=6, paired_sampling=True,
        primary_unit_triplet=True, primary_margin=.3, auxiliary_soft_unchanged=True, partial_forward_BN_retained=True, partial_loss_effective_zero=True,
        third_update_active_gradients=True, extra_weights=0, relation_PI_native_invariance=True, controlled10_01_parent_fuse_equal=True))
    emit(dict(event='NATIVE_PASS', controls=2, actual_updates=6, partial_loss_effective_zero=True, PI_outlet='shared_relation_local'))

    def archive(run, name):
        path = run / 'best_official_arrays.npz'
        file = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        emit(dict(event='NORMAL_READY', name=name, path=str(path), file=file))
        assert json.loads(sys.stdin.readline()) == dict(event='NORMAL_ARCHIVED', name=name, file=file)
        assert path.resolve().is_relative_to(root.resolve())
        assert path.stat().st_size == file['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == file['sha256']
        path.unlink()
        save(run / 'normal_local_archive.json', dict(status='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED', file=file))
        emit(dict(event='NORMAL_CLEARED', name=name))

    with ThreadPoolExecutor(max_workers=1) as archives:
        def train(variant, gpu):
            name = args.dataset+'_r201l_' + variant + '_s42'
            output = root / 'training' / name
            trained = execute(argv(variant, 'train', output), root / 'training', name, gpu)
            data = json.loads((output / 'result.json').read_text())
            assert data['status'] == 'COMPLETE' and data['epochs'] == 50 and '  NUM_INSTANCE: 8' in data['config'].splitlines()
            assert data['steps'] == data['optimizer_steps'] and data['amp_skipped_steps'] == 0
            assert all(r['loss']==r['full_loss'] and r['partial_loss_effective']==r['partial_CE_weight']==r['partial_triplet_weight']==0
                       and r['sampling_identities']==8 and r['sampling_instances']==8 and r['sampling_batch']==64
                       and r['primary_margin']==.3 and r['auxiliary_original_soft_triplet'] and r['PI_outlet']=='shared_relation_local'
                       for r in map(json.loads,(output/'batch_orders.jsonl').read_text().splitlines()))
            assert data['training_heldout_identities'] == 0 and data['training_coverage'] == dict(eligible=data['train_records'],visited=data['train_records'],unvisited=[])
            audited = execute([sys.executable,'-u','audit_full_official_normal.py','--run-dir',str(output)], root / 'audit', name, gpu)
            assert json.loads((output / 'normal_cpu_audit.json').read_text())['status'] == 'PASS'
            return dict(name=name,variant=variant,gpu=gpu,train=trained,audit=audited), archives.submit(archive,output,name)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(train,variant,gpu) for variant,gpu in CONTROLS]
            results = [future.result() for future in futures]
        for _, future in results:
            future.result()

    def orders(variant):
        file = root / 'training' / (args.dataset+'_r201l_' + variant + '_s42') / 'batch_orders.jsonl'
        return [(r['epoch'],r['step'],r['names'],r['partial_set']) for r in map(json.loads,file.read_text().splitlines())]

    assert orders('frequency_shared') == orders('axis_shared')
    updates=sum(json.loads((root/'training'/r['name']/'result.json').read_text())['optimizer_steps'] for r,_ in results)
    save(root / 'controller_result.json', dict(status='COMPLETE', dataset=args.dataset, controls=2, runs=[r for r,_ in results],
        additional_epochs=100, successful_updates=updates, native_updates=6, paired_sampling_exact=True,
        normal_archives_local_verified=2, missing_evaluation='Deferred until candidate normal comparison and three-dataset stage',
        limits='One seed, benchmark selected; original50 plus new50, fair paired frequency/axis, matched L batchsampling, partial BN/forward computation and zero partial gradient. Sampler length changes relative K; actual updates reported. No guaranteed normal/missing improvement.'))
    emit(dict(event='CONTROLLER_COMPLETE', controls=2, successful_updates=updates))


if __name__ == '__main__':
    main()
