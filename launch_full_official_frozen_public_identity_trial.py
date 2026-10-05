"""Three fixed M8 experts; reuse M7 inference and per-condition archive protocol."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
import time

from gpu_thermal_execute import check_limits, execute
from launch_full_official_frozen_anchor_trial import archive_frozen, emit, name, save, stream_diagnosis
from launch_frozen_public_identity_preflight import check_public_identity


SCHEDULE = {2: ('axis_shared',), 3: ('frequency_shared', 'twins_shared')}


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'output', 'anchor-run-dir', 'preflight-root', 'm7-root'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    preflight_root, m7 = Path(args.preflight_root), Path(args.m7_root)
    preflight = json.loads((preflight_root / 'preflight_result.json').read_text())
    assert preflight['status'] == 'PASS_THREE_M8_FIXED_NATIVE_AND_PUBLIC_CE_SMOKES'
    assert preflight['actual_native_updates'] == 3 and preflight['actual_smoke_updates'] == 9
    assert preflight['amp_skipped_steps'] == 0 and preflight['formal50_started'] == 0
    previous = json.loads((m7 / 'controller_result.json').read_text())
    assert previous['status'] == 'COMPLETE' and len(previous['runs']) == 5
    assert previous['frozen_metric_cases'] == 245 and previous['enhanced_state_metric_cases'] == 1176
    assert previous['all_frozen_and_state_cpu_audits_passed'] and previous['paired_identity_and_partial_sampling_exact']
    assert previous['anchor_run_dir'] == args.anchor_run_dir
    launched = json.loads((preflight_root / 'controller_launch.json').read_text())
    assert all(launched['arguments'][k] == vars(args)[k] for k in ('data_root', 'pretrained', 'anchor_run_dir', 'm7_root'))
    for gpu in SCHEDULE:
        sample = check_limits(gpu)
        assert sample['memory_used_mib'] < 500
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    for phase in ('training', 'frozen49', 'audit', 'diagnosis'):
        (root / phase).mkdir()
    save(root / 'controller_train_launch.json', dict(pid=os.getpid(), started=time.time(), arguments=vars(args),
        gpus=[2, 3], original_M7_complete=True, temperature_power_control=False))
    print('M8_FULL50_CONTROLLER_STARTED', os.getpid(), 'GPU2/3', flush=True)
    lock = threading.Lock()

    def jobs(gpu):
        rows = []
        for variant in SCHEDULE[gpu]:
            run_name = name(variant, 1)
            contract = json.loads((preflight_root / 'contract' / run_name / 'result.json').read_text())
            smoke = json.loads((preflight_root / 'preflight' / run_name / 'smoke.json').read_text())
            assert contract['actual_optimizer_updates'] == 1 and contract['amp_skipped_steps'] == 0
            assert smoke['steps'] == smoke['attempts'] == 3 and smoke['amp_skipped_steps'] == 0
            check_public_identity(contract['detail'])
            for detail in smoke['details']:
                check_public_identity(detail)
            run, frozen = root / 'training' / run_name, root / 'frozen49' / run_name
            trained = execute([sys.executable, '-u', 'run_full_official_frozen_public_identity_experiment.py',
                '--dataset', 'MSVR310', '--variant', variant, '--freeze-identity-encoder', '1', '--seed', '42',
                '--data-root', args.data_root, '--pretrained', args.pretrained, '--anchor-run-dir', args.anchor_run_dir,
                '--mode', 'train', '--output', str(run)], root / 'training', run_name, gpu)
            result = json.loads((run / 'result.json').read_text())
            assert result['status'] == 'COMPLETE' and result['epochs'] == 50
            assert result['steps'] == result['optimizer_steps'] == 705 and result['amp_skipped_steps'] == 0
            assert result['training_heldout_identities'] == 0
            assert (result['train_records'], result['query_records'], result['gallery_records']) == (1032, 591, 1055)
            assert result['training_coverage'] == dict(eligible=1032, visited=1032, unvisited=[])
            assert result['arguments']['public_identity_weight'] == .1 and result['descriptor_dim'] == 5632
            assert (result['parameters'], result['trainable_parameters']) == (contract['parameters'], contract['trainable_parameters'])
            with lock:
                evaluated = execute([sys.executable, '-u', 'evaluate_full_official_frozen_anchor49.py',
                    '--run-dir', str(run), '--output', str(frozen)], root / 'frozen49', run_name, gpu)
                audited = execute([sys.executable, '-u', 'audit_full_official49.py', '--run-dir', str(run),
                    '--evaluation', str(frozen)], root / 'audit', run_name, gpu)
                checked = json.loads((frozen / 'independent_cpu_audit.json').read_text())
                assert checked['status'] == 'PASS' and checked['cases'] == 49
                diagnosis = root / 'diagnosis' / run_name
                diagnosed = stream_diagnosis([sys.executable, '-u', 'diagnose_full_official_frozen_anchor_stream.py',
                    '--run-dir', str(run), '--previous-frozen', str(frozen), '--output', str(diagnosis)],
                    root / 'diagnosis', run_name, gpu)
                checked = json.loads((diagnosis / 'independent_cpu_audit.json').read_text())
                assert checked['status'] == 'PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT' and checked['cases'] == 294
                archive_frozen(root, frozen, run_name)
            rows.append(dict(variant=variant, freeze=1, name=run_name, gpu=gpu,
                train=trained, evaluation=evaluated, audit=audited, diagnosis=diagnosed))
            save(root / ('gpu_' + str(gpu) + '_completed.json'), dict(status='IN_PROGRESS', runs=rows))
            emit(dict(event='RUN_COMPLETE', job=run_name, completed_epochs=50, frozen_conditions=49, state_cases=294))
        return rows

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(jobs, gpu) for gpu in SCHEDULE]
        rows = [row for future in futures for row in future.result()]
    orders = []
    for row in rows:
        records = [json.loads(line) for line in (root / 'training' / row['name'] / 'batch_orders.jsonl').read_text().splitlines()]
        old = [json.loads(line) for line in (m7 / 'training' / row['name'] / 'batch_orders.jsonl').read_text().splitlines()]
        order = [(r['epoch'], r['step'], r['names'], r['partial_set']) for r in records]
        assert len(order) == 705 and order == [(r['epoch'], r['step'], r['names'], r['partial_set']) for r in old]
        orders.append(order)
    assert len(rows) == 3 and all(order == orders[0] for order in orders)
    save(root / 'controller_result.json', dict(status='COMPLETE', runs=rows, frozen_metric_cases=147,
        enhanced_state_metric_cases=882, all_frozen_and_state_cpu_audits_passed=True,
        paired_identity_and_partial_sampling_exact=True, M7_corresponding_sampling_exact=True,
        training_heldout_identities=0, temperature_power_control=False, anchor_run_dir=args.anchor_run_dir,
        archive_sensitive_phases_serial=True, all147_enhanced_raw_and147_frozen_raw_local_verified=True,
        limits='Common benchmark-selected50-epoch anchor plus additional50 fixed-expert updates; existing publicCE adds two BN/head calls. Single-seed MSVR, not unified three-dataset superiority.'))
    emit(dict(event='CONTROLLER_COMPLETE', runs=3, frozen_cases=147, state_cases=882))


if __name__ == '__main__':
    main()
