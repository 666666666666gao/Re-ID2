"""Five common-anchor controls, native gates, two GPU lanes and serial raw archive phases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from gpu_thermal_execute import execute, check_limits
from diagnose_full_official_streaming_states import STREAM_PREFIX


SCHEDULE = {2: (('demo_shared', 0), ('axis_shared', 1), ('axis_shared', 0)),
            3: (('frequency_shared', 1), ('twins_shared', 1))}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def emit(value):
    print(STREAM_PREFIX + json.dumps(value), flush=True)


def name(variant, freeze):
    return 'MSVR310_anchor_' + variant + ('_fixed' if freeze else '_adaptive') + '_s42'


def stream_diagnosis(command, output, run_name, gpu):
    sample = check_limits(gpu)
    assert sample['memory_used_mib'] < 500
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    with (output / (run_name + '.log')).open('x') as log:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                 text=True, env=env, start_new_session=True)
        try:
            assert os.getpgid(child.pid) == child.pid
            save(output / (run_name + '_launch.json'), dict(pid=child.pid, process_group=child.pid, gpu=gpu,
                started=time.time(), command=command, initial_resources=sample, temperature_power_control=False))
            ready, cleared, complete = set(), set(), False
            for line in child.stdout:
                log.write(line); log.flush()
                if not line.startswith(STREAM_PREFIX):
                    continue
                message = json.loads(line[len(STREAM_PREFIX):])
                event = message['event']
                if event == 'RAW_READY':
                    assert message['condition'] not in ready
                    ready.add(message['condition'])
                    emit(dict(job=run_name, **message))
                    acknowledgement = json.loads(sys.stdin.readline())
                    assert acknowledgement == dict(job=run_name, event='ARCHIVED', condition=message['condition'], file=message['file'])
                    acknowledgement.pop('job')
                    child.stdin.write(json.dumps(acknowledgement) + '\n'); child.stdin.flush()
                elif event == 'RAW_CLEARED':
                    assert message['condition'] in ready and message['condition'] not in cleared
                    cleared.add(message['condition'])
                    emit(dict(job=run_name, **message))
                else:
                    assert event == 'DIAGNOSIS_COMPLETE' and message['cases'] == 294 and message['conditions'] == 49
                    assert not complete and len(ready) == len(cleared) == 49
                    complete = True
            child.stdin.close()
            child.wait()
        finally:
            forced = child.poll() is None
            if forced:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
            row = dict(name=run_name, exit_code=child.returncode, finished=time.time(), gpu=gpu,
                       temperature_power_control=False, forced_own_group_cleanup=forced)
            save(output / (run_name + '_exit.json'), row)
    assert row['exit_code'] == 0 and complete
    return row


def archive_frozen(root, folder, run_name):
    paths = sorted(folder.glob('q_*_g_*.npz'))
    assert len(paths) == 49 and folder.resolve().is_relative_to(root.resolve())
    files = {str(path): dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths}
    emit(dict(event='FROZEN_READY', job=run_name, files=files))
    acknowledgement = json.loads(sys.stdin.readline())
    assert acknowledgement == dict(event='FROZEN_ARCHIVED', job=run_name, files=files)
    for path in paths:
        assert path.stat().st_size == files[str(path)]['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == files[str(path)]['sha256']
        path.unlink()
    assert not list(folder.glob('q_*_g_*.npz'))
    save(folder / 'local_archive.json', dict(status='ALL49_LOCAL_SIZE_SHA_VERIFIED_SERVER_RAW_CLEARED', files=files))
    emit(dict(event='FROZEN_CLEARED', job=run_name))


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'output', 'anchor-run-dir'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--mode', choices=('preflight', 'train'), required=True)
    args = parser.parse_args()
    root = Path(args.output)

    def command(variant, freeze):
        return [sys.executable, '-u', 'run_full_official_frozen_anchor_experiment.py', '--dataset', 'MSVR310',
                '--variant', variant, '--freeze-identity-encoder', str(freeze), '--seed', '42',
                '--data-root', args.data_root, '--pretrained', args.pretrained, '--anchor-run-dir', args.anchor_run_dir]

    if args.mode == 'preflight':
        root.mkdir(exist_ok=False)
        for phase in ('contract', 'preflight', 'training', 'frozen49', 'audit', 'diagnosis'):
            (root / phase).mkdir()
        save(root / 'controller_preflight_launch.json', dict(pid=os.getpid(), started=time.time(), arguments=vars(args), gpus=[2, 3]))

        def check(gpu):
            rows = []
            for variant, freeze in SCHEDULE[gpu]:
                run_name = name(variant, freeze)
                contract = execute([sys.executable, '-u', 'verify_full_official_frozen_anchor.py', '--dataset', 'MSVR310',
                    '--variant', variant, '--freeze-identity-encoder', str(freeze), '--seed', '42', '--data-root', args.data_root,
                    '--pretrained', args.pretrained, '--anchor-run-dir', args.anchor_run_dir,
                    '--output', str(root / 'contract' / run_name)], root / 'contract', run_name, gpu)
                checked = json.loads((root / 'contract' / run_name / 'result.json').read_text())
                assert checked['status'] == 'PASS_FULL_OFFICIAL_ANCHOR_NATIVE_UPDATE_AND_IDENTITY_REFERENCE'
                assert checked['actual_optimizer_updates'] == 1 and checked['amp_skipped_steps'] == 0
                assert checked['initial_all7_base_descriptors_exact'] == {s: True for s in ('RNT', 'R', 'N', 'T', 'RN', 'RT', 'NT')}
                assert all(checked['active_gradients'].values()) and checked['strict_reload_equal']
                assert checked['fixed_encoder_unchanged_after_update_and_reload'] == bool(freeze)
                assert checked['full_split_counts'] == dict(train=1032, query=591, gallery=1055)
                smoke = execute(command(variant, freeze) + ['--mode', 'smoke', '--output', str(root / 'preflight' / run_name)],
                                root / 'preflight', run_name, gpu)
                result = json.loads((root / 'preflight' / run_name / 'smoke.json').read_text())
                assert result['status'] == 'SMOKE_PASS' and result['steps'] == 3 and result['strict_reload_equal']
                assert result['training_heldout_identities'] == 0 and all(result['gradients'].values())
                assert (result['train_records'], result['query_records'], result['gallery_records']) == (1032, 591, 1055)
                assert result['parameters'] == checked['parameters'] and result['trainable_parameters'] == checked['trainable_parameters']
                rows.append(dict(variant=variant, freeze=freeze, gpu=gpu, name=run_name, contract=contract, smoke=smoke,
                    parameters=checked['parameters'], trainable_parameters=checked['trainable_parameters'],
                    active_names=sorted(checked['active_gradients']), anchor=checked['anchor']))
            return rows

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(check, gpu) for gpu in SCHEDULE]
            rows = [row for future in futures for row in future.result()]
        frozen = [row for row in rows if row['freeze']]
        assert len(rows) == 5 and len(frozen) == 3
        assert all((row['parameters'], row['trainable_parameters'], row['active_names']) ==
                   (frozen[0]['parameters'], frozen[0]['trainable_parameters'], frozen[0]['active_names']) for row in frozen)
        assert all(row['anchor']['run_dir'] == args.anchor_run_dir for row in rows)
        save(root / 'preflight_result.json', dict(status='PASS', runs=rows, arguments=vars(args),
            contract_actual_optimizer_updates=5, smoke_actual_optimizer_updates=15,
            full_split_counts=dict(train=1032, query=591, gallery=1055), training_heldout_identities=0,
            same_fixed_expert_capacity_and_active_gradient_names=True, new_full50_training_runs=0))
        emit(dict(event='PREFLIGHT_COMPLETE', contracts=5, smoke_updates=15, full50_runs=0))
        return

    preflight = json.loads((root / 'preflight_result.json').read_text())
    assert preflight['status'] == 'PASS' and len(preflight['runs']) == 5
    assert all(preflight['arguments'][k] == vars(args)[k] for k in ('data_root', 'pretrained', 'output', 'anchor_run_dir'))
    gate = json.loads((root / 'neural_stream_gate.json').read_text())
    assert gate['status'] == 'PASS_FULL49_NEURAL_STREAM_EQUALS_COMPLETED_M6_GT_AUDIT'
    assert gate['conditions'] == 49 and gate['cases'] == 294 and gate['optimizer_updates'] == 0
    save(root / 'controller_train_launch.json', dict(pid=os.getpid(), started=time.time(), arguments=vars(args), gpus=[2, 3]))
    lock = threading.Lock()

    def jobs(gpu):
        rows = []
        for variant, freeze in SCHEDULE[gpu]:
            run_name = name(variant, freeze)
            run, frozen = root / 'training' / run_name, root / 'frozen49' / run_name
            trained = execute(command(variant, freeze) + ['--mode', 'train', '--output', str(run)], root / 'training', run_name, gpu)
            result = json.loads((run / 'result.json').read_text())
            assert result['status'] == 'COMPLETE' and result['epochs'] == 50
            assert result['training_heldout_identities'] == 0 and result['training_coverage'] == dict(eligible=1032, visited=1032, unvisited=[])
            with lock:
                evaluated = execute([sys.executable, '-u', 'evaluate_full_official_frozen_anchor49.py', '--run-dir', str(run),
                                     '--output', str(frozen)], root / 'frozen49', run_name, gpu)
                audited = execute([sys.executable, '-u', 'audit_full_official49.py', '--run-dir', str(run),
                                   '--evaluation', str(frozen)], root / 'audit', run_name, gpu)
                assert json.loads((frozen / 'independent_cpu_audit.json').read_text())['cases'] == 49
                row = dict(variant=variant, freeze=freeze, name=run_name, gpu=gpu, train=trained, evaluation=evaluated, audit=audited)
                if variant != 'demo_shared':
                    diagnosis = root / 'diagnosis' / run_name
                    row['diagnosis'] = stream_diagnosis([sys.executable, '-u', 'diagnose_full_official_frozen_anchor_stream.py',
                        '--run-dir', str(run), '--previous-frozen', str(frozen), '--output', str(diagnosis)], root / 'diagnosis', run_name, gpu)
                    checked = json.loads((diagnosis / 'independent_cpu_audit.json').read_text())
                    assert checked['status'] == 'PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT' and checked['cases'] == 294
                archive_frozen(root, frozen, run_name)
            rows.append(row)
            save(root / ('gpu_' + str(gpu) + '_completed.json'), dict(status='IN_PROGRESS', runs=rows))
            emit(dict(event='RUN_COMPLETE', job=run_name, completed_epochs=50, frozen_conditions=49,
                      state_cases=294 if variant != 'demo_shared' else 0))
        return rows

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(jobs, gpu) for gpu in SCHEDULE]
        rows = [row for future in futures for row in future.result()]
    orders = []
    for row in rows:
        records = [json.loads(line) for line in (root / 'training' / row['name'] / 'batch_orders.jsonl').read_text().splitlines()]
        orders.append([(r['epoch'], r['step'], r['names'], r['partial_set']) for r in records])
        assert len(records) == 705
    assert len(rows) == 5 and all(order == orders[0] for order in orders)
    save(root / 'controller_result.json', dict(status='COMPLETE', runs=rows, frozen_metric_cases=245,
        enhanced_state_metric_cases=1176, all_frozen_and_state_cpu_audits_passed=True,
        paired_identity_and_partial_sampling_exact=True, training_heldout_identities=0,
        temperature_power_control=False, anchor_run_dir=args.anchor_run_dir,
        archive_sensitive_phases_serial=True, all196_enhanced_raw_and245_frozen_raw_local_verified=True,
        limits='Common benchmark-selected50-epoch anchor plus additional50 for each control. Single-seed MSVR development, not unified three-dataset superiority.'))
    emit(dict(event='CONTROLLER_COMPLETE', runs=5, frozen_cases=245, state_cases=1176))


if __name__ == '__main__':
    main()
