"""Two matched RGBNT201 controls: CPU list fixture, native3, full50, GT."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from gpu_thermal_execute import execute
from launch_full_official_frozen_anchor_trial import save

CONTROLS = (('frequency_shared', 2), ('axis_shared', 3))
PRINT_LOCK = threading.Lock()


def emit(value):
    with PRINT_LOCK:
        sys.stdout.write('R201E_STREAM ' + json.dumps(value) + '\n')
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'anchor-root', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    root, anchors = Path(args.output), Path(args.anchor_root)
    root.mkdir(parents=True, exist_ok=False)
    for phase in ('native', 'training', 'audit'):
        (root / phase).mkdir()
    save(root / 'controller_launch.json', dict(pid=os.getpid(), started=time.time(), arguments=vars(args),
        gpus=[2,3], maximum_NN=2, temperature_power_control=False, experiment='R201E primary full-view list loss only'))
    fixture = subprocess.run([sys.executable, 'check_list_retrieval_objective.py'], check=True,
                             capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    checked = json.loads(fixture.stdout)
    assert checked['status'] == 'PASS_CPU_LIST_VALUE_GRADIENT_AND_PERMUTATION' and checked['optimizer_updates'] == 0
    save(root / 'cpu_fixture.json', checked)

    def argv(variant, mode, output):
        return [sys.executable, '-u', 'run_r201e_list_objective.py', '--dataset', 'RGBNT201', '--variant', variant,
            '--freeze-identity-encoder', '1', '--seed', '42', '--mode', mode, '--data-root', args.data_root,
            '--pretrained', args.pretrained, '--anchor-run-dir', str(anchors / 'RGBNT201_demo_s42'), '--output', str(output)]

    def native(variant, gpu):
        name = 'RGBNT201_r201e_' + variant + '_s42_smoke'
        output = root / 'native' / name
        execute(argv(variant, 'smoke', output), root / 'native', name, gpu)
        data = json.loads((output / 'smoke.json').read_text())
        assert data['status'] == 'SMOKE_PASS' and data['steps'] == data['attempts'] == 3
        assert data['amp_skipped_steps'] == 0 and data['strict_reload_equal']
        assert data['descriptor_shape'] == [8,5120] and data['descriptor_dim'] == 5120
        assert all(data['gradients'].values()) and not list(output.glob('*.pth'))
        assert all(0 <= d['smooth_ap_loss'] < 1 and d['AP_feature_gradient_l1'] > 0 and len(d['names']) == 64 for d in data['details'])
        return data

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(native, variant, gpu) for variant, gpu in CONTROLS]
        native_rows = [future.result() for future in futures]
    assert len({(r['parameters'],r['trainable_parameters']) for r in native_rows}) == 1
    assert [(d['names'],d['partial_set']) for d in native_rows[0]['details']] == [(d['names'],d['partial_set']) for d in native_rows[1]['details']]
    save(root / 'native_acceptance.json', dict(status='PASS', controls=2, actual_updates=6, paired_sampling=True,
        primary_list_loss_and_unit_retrieval=True, nonzero_primary_list_feature_gradients=True,
        third_update_active_gradients=True, extra_weights=0))
    emit(dict(event='NATIVE_PASS', controls=2, actual_updates=6, cpu_fixture='PASS'))

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
            name = 'RGBNT201_r201e_' + variant + '_s42'
            output = root / 'training' / name
            trained = execute(argv(variant, 'train', output), root / 'training', name, gpu)
            data = json.loads((output / 'result.json').read_text())
            assert data['status'] == 'COMPLETE' and data['epochs'] == 50
            assert data['steps'] == data['optimizer_steps'] == 2647 and data['amp_skipped_steps'] == 0
            assert data['training_heldout_identities'] == 0 and data['training_coverage'] == dict(eligible=3951,visited=3951,unvisited=[])
            audited = execute([sys.executable,'-u','audit_full_official_normal.py','--run-dir',str(output)], root / 'audit', name, gpu)
            assert json.loads((output / 'normal_cpu_audit.json').read_text())['status'] == 'PASS'
            return dict(name=name,variant=variant,gpu=gpu,train=trained,audit=audited), archives.submit(archive,output,name)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(train,variant,gpu) for variant,gpu in CONTROLS]
            results = [future.result() for future in futures]
        for _, future in results:
            future.result()

    def orders(variant):
        file = root / 'training' / ('RGBNT201_r201e_' + variant + '_s42') / 'batch_orders.jsonl'
        return [(r['epoch'],r['step'],r['names'],r['partial_set']) for r in map(json.loads,file.read_text().splitlines())]

    assert orders('frequency_shared') == orders('axis_shared')
    save(root / 'controller_result.json', dict(status='COMPLETE', controls=2, runs=[r for r,_ in results],
        additional_epochs=100, successful_updates=5294, native_updates=6, paired_sampling_exact=True,
        normal_archives_local_verified=2, missing_evaluation='Deferred until candidate normal comparison and three-dataset stage',
        limits='One seed, benchmark selected; original50 plus new50, fair paired frequency/axis. No guaranteed improvement.'))
    emit(dict(event='CONTROLLER_COMPLETE', controls=2, successful_updates=5294))


if __name__ == '__main__':
    main()
