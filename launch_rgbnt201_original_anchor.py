"""Two original-anchor controls: native gates first, then full50 plus installed-GT49."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
import time

from gpu_thermal_execute import execute
from launch_full_official_frozen_anchor_trial import archive_frozen, emit, save


def name(freeze):
    return 'RGBNT201_original_shared_' + ('fixed' if freeze else 'adaptive') + '_s42'


def wait_m8_lane(root, gpu):
    variant = 'axis_shared' if gpu == 2 else 'twins_shared'
    old = 'MSVR310_anchor_' + variant + '_fixed_s42'
    terminal = root / 'diagnosis' / (old + '_exit.json')
    while not terminal.exists():
        print('WAIT_EXISTING_M8_FINAL_NEURAL_PHASE', gpu, old, flush=True)
        time.sleep(240)
    assert json.loads(terminal.read_text())['exit_code'] == 0
    launch = json.loads((root / 'diagnosis' / (old + '_launch.json')).read_text())
    assert not Path('/proc/' + str(launch['pid'])).exists()
    result = json.loads((root / 'diagnosis' / old / 'result.json').read_text())
    assert result['status'] == 'COMPLETE'


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'output', 'anchor-run-dir', 'm8-root'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--mode', choices=('preflight', 'train'), required=True)
    parser.add_argument('--preflight-root')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    save(root / 'controller_launch.json', dict(pid=os.getpid(), started=time.time(), arguments=vars(args),
        gpus=[2, 3], temperature_power_control=False, original_M8_unchanged=True))
    common = ['--data-root', args.data_root, '--pretrained', args.pretrained, '--anchor-run-dir', args.anchor_run_dir]

    def training_command(freeze):
        return [sys.executable, '-u', 'run_rgbnt201_original_anchor.py', '--dataset', 'RGBNT201',
            '--variant', 'demo_shared', '--freeze-identity-encoder', str(freeze), '--seed', '42', *common]

    if args.mode == 'preflight':
        for phase in ('contract', 'smoke'):
            (root / phase).mkdir()
        wait_m8_lane(Path(args.m8_root), 2)
        execute([sys.executable, '-u', 'verify_rgbnt201_original_anchor.py', *common,
            '--output', str(root / 'contract' / 'paired')], root / 'contract', 'paired', 2)
        native = json.loads((root / 'contract/paired/result.json').read_text())
        assert native['status'] == 'PASS_RGBNT201_ORIGINAL_ANCHOR_TWO_NATIVE_UPDATES'
        assert native['matched_initial_state_tensors_exact'] and native['actual_optimizer_updates'] == 2
        assert native['amp_skipped_steps'] == 0 and native['new_weight_files'] == 0
        assert (native['batch_size'], native['P'], native['K']) == (64, 8, 8)
        rows = []
        for freeze in (1, 0):
            run_name = name(freeze)
            execute(training_command(freeze) + ['--mode', 'smoke', '--output', str(root / 'smoke' / run_name)],
                root / 'smoke', run_name, 2)
            result = json.loads((root / 'smoke' / run_name / 'smoke.json').read_text())
            contract = native['controls'][str(freeze)]
            assert result['status'] == 'SMOKE_PASS' and result['steps'] == result['attempts'] == 3
            assert result['amp_skipped_steps'] == 0 and result['strict_reload_equal']
            assert result['training_heldout_identities'] == 0 and all(result['gradients'].values())
            assert (result['train_records'], result['query_records'], result['gallery_records']) == (3951, 836, 836)
            assert (result['parameters'], result['trainable_parameters']) == (contract['parameters'], contract['trainable_parameters'])
            assert result['pretraining'].startswith('Full-official original RGBNT201 DeMo50')
            assert result['anchor']['shared_projection_frozen'] is False
            rows.append(dict(freeze=freeze, name=run_name, parameters=result['parameters'],
                trainable_parameters=result['trainable_parameters'], active_names=sorted(result['gradients'])))
        assert rows[0]['parameters'] == rows[1]['parameters']
        assert not list(root.rglob('*.pth'))
        save(root / 'preflight_result.json', dict(status='PASS_RGBNT201_ORIGINAL_ANCHOR_NATIVE2_SMOKE6',
            native=native, controls=rows, actual_native_updates=2, actual_smoke_updates=6,
            amp_skipped_steps=0, full50_started=0, new_weight_files=0, gpus=[2]))
        print('RGBNT201_ORIGINAL_ANCHOR_PREFLIGHT_COMPLETE', flush=True)
        return

    preflight_root = Path(args.preflight_root)
    preflight = json.loads((preflight_root / 'preflight_result.json').read_text())
    assert preflight['status'] == 'PASS_RGBNT201_ORIGINAL_ANCHOR_NATIVE2_SMOKE6'
    assert preflight['actual_native_updates'] == 2 and preflight['actual_smoke_updates'] == 6
    assert preflight['amp_skipped_steps'] == 0 and preflight['full50_started'] == 0
    previous = json.loads((preflight_root / 'controller_launch.json').read_text())
    assert all(previous['arguments'][key] == vars(args)[key] for key in ('data_root', 'pretrained', 'anchor_run_dir', 'm8_root'))
    for phase in ('training', 'frozen49', 'audit'):
        (root / phase).mkdir()
    lock = threading.Lock()

    def job(gpu, freeze):
        wait_m8_lane(Path(args.m8_root), gpu)
        run_name = name(freeze)
        run, frozen = root / 'training' / run_name, root / 'frozen49' / run_name
        trained = execute(training_command(freeze) + ['--mode', 'train', '--output', str(run)], root / 'training', run_name, gpu)
        result = json.loads((run / 'result.json').read_text())
        assert result['status'] == 'COMPLETE' and result['epochs'] == 50
        assert result['steps'] == result['optimizer_steps'] == 2647 and result['amp_skipped_steps'] == 0
        assert result['training_coverage'] == dict(eligible=3951, visited=3951, unvisited=[])
        assert (result['train_records'], result['query_records'], result['gallery_records']) == (3951, 836, 836)
        assert result['training_heldout_identities'] == 0 and result['descriptor_dim'] == 5632
        evaluated = execute([sys.executable, '-u', 'evaluate_rgbnt201_original_anchor49.py',
            '--run-dir', str(run), '--output', str(frozen)], root / 'frozen49', run_name, gpu)
        audited = execute([sys.executable, '-u', 'audit_full_official49.py', '--run-dir', str(run),
            '--evaluation', str(frozen)], root / 'audit', run_name, gpu)
        checked = json.loads((frozen / 'independent_cpu_audit.json').read_text())
        assert checked['status'] == 'PASS' and checked['cases'] == 49
        with lock:
            archive_frozen(root, frozen, run_name)
            emit(dict(event='RUN_COMPLETE', job=run_name, completed_epochs=50, frozen_conditions=49))
        row = dict(freeze=freeze, gpu=gpu, name=run_name, train=trained, evaluation=evaluated, audit=audited)
        save(root / ('gpu_' + str(gpu) + '_completed.json'), row)
        return row

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(job, gpu, freeze) for gpu, freeze in ((2, 1), (3, 0))]
        rows = [future.result() for future in futures]
    orders = []
    for row in rows:
        batches = [json.loads(line) for line in (root / 'training' / row['name'] / 'batch_orders.jsonl').read_text().splitlines()]
        orders.append([(entry['epoch'], entry['step'], entry['names'], entry['partial_set']) for entry in batches])
    assert len(orders[0]) == 2647 and orders[0] == orders[1]
    save(root / 'controller_result.json', dict(status='COMPLETE', runs=rows, frozen_metric_cases=98,
        all_frozen_cpu_audits_passed=True, paired_identity_and_partial_sampling_exact=True,
        training_heldout_identities=0, all98_raw_local_verified=True,
        limits='Original DeMo trained50 then additional50/new optimizer per control. Different trainable counts by intended freezing factor; no dual-axis method, expert-contribution, multi-seed or three-dataset success claim.'))
    emit(dict(event='CONTROLLER_COMPLETE', runs=2, frozen_cases=98))


if __name__ == '__main__':
    main()
