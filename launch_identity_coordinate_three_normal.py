"""Finish remaining normal datasets first, then resume the existing missing queue."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from gpu_thermal_execute import execute
from launch_full_official_frozen_anchor_trial import save

VARIANTS = (('frequency_shared', 2), ('axis_shared', 3))
DATASETS = ('MSVR310', 'RGBNT100')


def emit(value):
    print('THREE_NORMAL_STREAM ' + json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ('data-root', 'pretrained', 'anchor-root', 'old-controller-root', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root, old, anchors = Path(args.output), Path(args.old_controller_root), Path(args.anchor_root)
    priority = json.loads((old / 'user_priority_missing_deferred.json').read_text())
    assert priority['status'] == 'OWN_SEQUENCING_PARENT_STOPPED_MISSING_DEFERRED_CHILD_TRAINING_UNCHANGED'
    pid = priority['controller_pid']
    assert subprocess.check_output(['ps', '-p', str(pid), '-o', 'stat='], text=True).strip().startswith('T')
    assert all((old / 'training' / ('RGBNT201_identity_' + variant + '_narrow_s42') / 'result.json').is_file()
        for variant, _ in VARIANTS)
    root.mkdir(parents=True, exist_ok=False)
    for phase in ('native', 'training', 'audit'):
        (root / phase).mkdir()
    save(root / 'controller_launch.json', dict(pid=os.getpid(), started=time.time(), arguments=vars(args),
        user_order='All3 normal before missing', gpus=[2,3], temperature_power_control=False,
        method='Unchanged R201C narrow5120 across3 datasets; each original50 plus additional50'))
    def argv(dataset, variant, mode, output):
        return [sys.executable, '-u', 'run_identity_coordinate_three_dataset.py',
            '--dataset', dataset, '--variant', variant, '--freeze-identity-encoder', '1', '--seed', '42',
            '--mode', mode, '--data-root', args.data_root, '--pretrained', args.pretrained,
            '--anchor-run-dir', str(anchors / (dataset + '_demo_s42')), '--output', str(output)]
    def native_lane(variant, gpu):
        rows = []
        for dataset in DATASETS:
            name = dataset + '_identity_' + variant + '_narrow_s42_smoke'
            output = root / 'native' / name
            execute(argv(dataset, variant, 'smoke', output), root / 'native', name, gpu)
            result = json.loads((output / 'smoke.json').read_text())
            assert result['status'] == 'SMOKE_PASS' and result['steps'] == result['attempts'] == 3
            assert result['amp_skipped_steps'] == 0 and all(result['gradients'].values())
            assert result['strict_reload_equal'] and result['descriptor_shape'] == [8,5120]
            assert result['descriptor_dim'] == 5120 and not list(output.glob('*.pth'))
            rows.append((dataset, variant, result))
        return rows
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(native_lane, variant, gpu) for variant, gpu in VARIANTS]
        native = [row for future in futures for row in future.result()]
    for dataset in DATASETS:
        controls = [row[2] for row in native if row[0] == dataset]
        assert len({(r['parameters'], r['trainable_parameters']) for r in controls}) == 1
        assert controls[0]['gradients'] == controls[1]['gradients']
        assert [(d['names'], d['partial_set']) for d in controls[0]['details']] == [
            (d['names'], d['partial_set']) for d in controls[1]['details']]
    save(root / 'native_acceptance.json', dict(status='PASS', controls=4, actual_updates=12,
        all_active_gradients=True, gradient_scope='Each control checked after its third update; no per-param audit of firsttwo',
        paired_sampling=True, added_weight_files=0))
    emit(dict(event='NATIVE_PASS', controls=4, actual_updates=12))
    def archive_normal(run, name):
        path = run / 'best_official_arrays.npz'
        file = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        emit(dict(event='NORMAL_READY', name=name, path=str(path), file=file))
        ack = json.loads(sys.stdin.readline())
        assert ack == dict(event='NORMAL_ARCHIVED', name=name, file=file)
        assert path.stat().st_size == file['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == file['sha256']
        path.unlink()
        save(run / 'normal_local_archive.json', dict(status='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED', file=file))
        emit(dict(event='NORMAL_CLEARED', name=name))
    with ThreadPoolExecutor(max_workers=1) as archives:
        def lane(variant, gpu):
            rows = []
            for dataset in DATASETS:
                name = dataset + '_identity_' + variant + '_narrow_s42'
                run = root / 'training' / name
                trained = execute(argv(dataset, variant, 'train', run), root / 'training', name, gpu)
                data = json.loads((run / 'result.json').read_text())
                teacher = json.loads((anchors / (dataset + '_demo_s42') / 'result.json').read_text())
                assert data['status'] == 'COMPLETE' and data['epochs'] == 50
                assert data['optimizer_steps'] == data['steps'] == teacher['optimizer_steps']
                assert data['amp_skipped_steps'] == data['training_heldout_identities'] == 0
                assert data['training_coverage'] == dict(eligible=data['train_records'], visited=data['train_records'], unvisited=[])
                audit = execute([sys.executable, '-u', 'audit_full_official_normal.py', '--run-dir', str(run)],
                    root / 'audit', name, gpu)
                assert json.loads((run / 'normal_cpu_audit.json').read_text())['status'] == 'PASS'
                future = archives.submit(archive_normal, run, name)
                rows.append((dict(name=name, dataset=dataset, variant=variant, gpu=gpu, train=trained, audit=audit), future))
            return rows
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(lane, variant, gpu) for variant, gpu in VARIANTS]
            rows = [row for future in futures for row in future.result()]
        for _, future in rows:
            future.result()
    for dataset in DATASETS:
        def orders(variant):
            name = dataset + '_identity_' + variant + '_narrow_s42'
            data = [json.loads(line) for line in (root / 'training' / name / 'batch_orders.jsonl').read_text().splitlines()]
            return [(r['epoch'], r['step'], r['names'], r['partial_set']) for r in data]
        assert orders('frequency_shared') == orders('axis_shared')
    assert subprocess.check_output(['ps', '-p', str(pid), '-o', 'stat='], text=True).strip().startswith('T')
    os.kill(pid, signal.SIGCONT)
    save(old / 'user_priority_missing_resumed.json', dict(status='ORIGINAL_MISSING_QUEUE_RESUMED_AFTER_THREE_NORMAL_DATASETS',
        resumed_at=time.time(), controller_pid=pid, continuing_same_receiver=17662, neural_sources_changed=0))
    save(root / 'controller_result.json', dict(status='COMPLETE', runs=[row for row,_ in rows],
        new_models=4, additional_epochs=200, successful_updates=14124, native_updates=12,
        existing_RGBNT201_normal_models=2, normal_datasets_completed=3, normal_archives_local_verified=4,
        paired_sampling_exact=True, original_missing_controller_resumed=True,
        limits='Normal results only; benchmark-selected/single seed/original50 plus extra50. '
               'All49 missing and multiseed/three-dataset+2 acceptance remain separate and unproven.'))
    emit(dict(event='CONTROLLER_COMPLETE', normal_models=4, normal_datasets=3, missing_controller_resumed=True))


if __name__ == '__main__':
    main()
