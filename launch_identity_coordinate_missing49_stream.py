"""Run native inference gates, then four fixed-best missing evaluations on GPU2/3."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from evaluate_identity_coordinate_missing49_stream import PREFIX
from gpu_thermal_execute import check_limits
from launch_full_official_frozen_anchor_trial import save

LANES = (('frequency_shared', 2), ('axis_shared', 3))
DATASETS = ('MSVR310', 'RGBNT100')


def main():
    parser = argparse.ArgumentParser()
    for key in ('normal-root', 'old-root', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--mode', choices=('native', 'full'), required=True)
    args = parser.parse_args()
    normal, old, out = map(Path, (args.normal_root, args.old_root, args.output))
    prior = json.loads((normal / 'controller_result.json').read_text())
    original = json.loads((old / 'controller_result.json').read_text())
    assert prior['status'] == original['status'] == 'COMPLETE'
    assert prior['new_models'] == 4 and prior['normal_datasets_completed'] == 3
    assert prior['normal_archives_local_verified'] == 4 and prior['paired_sampling_exact']
    assert original['models'] == 4 and original['closed_state_cases'] == 784 and original['all_raw_local_verified']
    if args.mode == 'native':
        out.mkdir(parents=True, exist_ok=False)
        (out / 'native').mkdir()
    else:
        accepted = json.loads((out / 'controller_native_result.json').read_text())
        assert accepted['status'] == 'PASS' and accepted['controls'] == 4 and accepted['normal_GT_state_cases'] == 16
        assert accepted['all7_small_batch_checks_per_control'] and accepted['all_raw_local_verified']
        (out / 'full').mkdir()
    save(out / ('controller_' + args.mode + '_launch.json'), dict(pid=os.getpid(), started=time.time(),
        arguments=vars(args), gpus=[2, 3], temperature_power_control=False, new_optimizer_updates=0))
    exchange = threading.Lock()

    def lane(variant, gpu):
        completed = []
        for dataset in DATASETS:
            name = dataset + '_identity_' + variant + '_narrow_s42'
            run, output = normal / 'training' / name, out / args.mode / name
            normal_arrays = run / 'best_official_arrays_restore_missing49.npz'
            assert normal_arrays.is_file()
            sample = check_limits(gpu)
            while sample['memory_used_mib'] >= 500:
                with exchange:
                    print('WAIT_SELECTED_GPU', gpu, sample['memory_used_mib'], flush=True)
                time.sleep(240)
                sample = check_limits(gpu)
            argv = [sys.executable, '-u', 'evaluate_identity_coordinate_missing49_stream.py', '--run-dir', str(run),
                '--normal-arrays', str(normal_arrays), '--output', str(output), '--mode', args.mode]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            with (out / args.mode / (name + '.log')).open('x') as log:
                child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                    text=True, env=env, start_new_session=True)
                try:
                    assert os.getpgid(child.pid) == child.pid
                    save(out / args.mode / (name + '_launch.json'), dict(pid=child.pid, process_group=child.pid, gpu=gpu,
                        command=argv, started=time.time(), initial_resources=sample, temperature_power_control=False))
                    ready, cleared, finished = set(), set(), False
                    for line in child.stdout:
                        log.write(line)
                        log.flush()
                        if not line.startswith(PREFIX):
                            continue
                        message = json.loads(line[len(PREFIX):])
                        event = message['event']
                        if event == 'RAW_READY':
                            assert message['installed_gt_cases'] == 4 and message['condition'] not in ready
                            ready.add(message['condition'])
                            with exchange:
                                print(PREFIX + json.dumps(dict(job=name, **message)), flush=True)
                                ack = json.loads(sys.stdin.readline())
                                assert ack == dict(job=name, event='ARCHIVED', condition=message['condition'], file=message['file'])
                                ack.pop('job')
                                child.stdin.write(json.dumps(ack) + '\n')
                                child.stdin.flush()
                        elif event == 'RAW_CLEARED':
                            assert message['condition'] in ready and message['condition'] not in cleared
                            cleared.add(message['condition'])
                            with exchange:
                                print(PREFIX + json.dumps(dict(job=name, **message)), flush=True)
                        else:
                            assert event == 'EVALUATION_COMPLETE' and not finished
                            expected = 1 if args.mode == 'native' else 49
                            assert message['conditions'] == expected and message['cases'] == 4 * expected
                            assert len(ready) == len(cleared) == expected
                            finished = True
                    child.stdin.close()
                    child.wait()
                finally:
                    forced = child.poll() is None
                    if forced:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    exit_row = dict(name=name, gpu=gpu, exit_code=child.returncode, finished=time.time(), forced_own_group_cleanup=forced)
                    save(out / args.mode / (name + '_exit.json'), exit_row)
            assert exit_row['exit_code'] == 0 and finished
            result = json.loads((output / 'result.json').read_text())
            assert result['status'] == 'COMPLETE' and result['optimizer_updates'] == result['new_weights'] == 0
            if args.mode == 'native':
                assert set(result['native_small_batch_checks']) == {'R', 'N', 'T', 'RN', 'RT', 'NT', 'RNT'}
            completed.append(dict(name=name, dataset=dataset, variant=variant, gpu=gpu, result=str(output / 'result.json')))
        return completed

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(lane, variant, gpu) for variant, gpu in LANES]
        completed = [row for future in futures for row in future.result()]
    expected = 4 if args.mode == 'native' else 196
    result = dict(status='PASS' if args.mode == 'native' else 'COMPLETE', controls=4, runs=completed,
        normal_GT_state_cases=16 if args.mode == 'native' else 0,
        missing_GT_state_cases=784 if args.mode == 'full' else 0, raw_conditions=expected,
        all7_small_batch_checks_per_control=args.mode == 'native', all_raw_local_verified=True,
        optimizer_updates=0, new_weights=0, normal_phase_preceded_missing=True,
        limits='Frozen selected checkpoints; source/runtime acceptance does not establish +2 improvements or multiseed success.')
    save(out / ('controller_' + args.mode + '_result.json'), result)
    print(PREFIX + json.dumps(dict(event='CONTROLLER_COMPLETE', mode=args.mode, controls=4,
        conditions=expected, cases=4 * expected, optimizer_updates=0)), flush=True)


if __name__ == '__main__':
    main()
