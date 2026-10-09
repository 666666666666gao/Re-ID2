"""Fixed N42 paper-six on weakdatasets; retained K201 six results already archived."""
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

from evaluate_r201n_paper_missing6_stream import PREFIX
from gpu_thermal_execute import check_limits
from launch_full_official_frozen_anchor_trial import save

LANES = (('frequency_shared', 2), ('axis_shared', 3))


def main():
    parser = argparse.ArgumentParser()
    for key in ('jobs', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--mode', choices=('native', 'full'), required=True)
    args = parser.parse_args()
    out=Path(args.output)
    jobs=json.loads(Path(args.jobs).read_text())['jobs']
    controls=len(jobs)
    assert controls==4 and len({j['name'] for j in jobs})==controls
    for dataset in ('MSVR310','RGBNT100'):
        assert {j['variant'] for j in jobs if j['dataset']==dataset and j['seed']==42}=={'frequency_shared','axis_shared'}
    assert all(j['gpu']==dict(LANES)[j['variant']] for j in jobs)
    assert all(j['seed']==42 and j['forward_graph']=='relation_local_shared_PI' and j['dataset'] in ('MSVR310','RGBNT100') and '_r201n_' in j['name'] for j in jobs)
    if args.mode == 'native':
        out.mkdir(parents=True, exist_ok=False)
        (out / 'native').mkdir()
    else:
        accepted = json.loads((out / 'controller_native_result.json').read_text())
        assert accepted['status'] == 'PASS' and accepted['controls'] == controls and accepted['normal_GT_state_cases'] == 4*controls
        assert accepted['all7_small_batch_checks_per_control'] and accepted['all_raw_local_verified']
        (out / 'full').mkdir()
    save(out / ('controller_' + args.mode + '_launch.json'), dict(pid=os.getpid(), started=time.time(),
        arguments=vars(args), gpus=[2, 3], temperature_power_control=False, new_optimizer_updates=0))
    exchange = threading.Lock()

    def lane(variant, gpu):
        completed = []
        for job in [j for j in jobs if j['variant']==variant]:
            name,dataset=job['name'],job['dataset']
            run,output=Path(job['run']),out/args.mode/name
            selected=json.loads((run/'result.json').read_text())
            assert selected['arguments']['seed']==job['seed'] and selected['best']['epoch']==job['selected_epoch']
            assert selected['full_metrics']==job['full_metrics']
            assert selected['method_revision'].startswith('R201N one-factor final unit5120 shift bound')
            assert 'Only MSVR310/RGBNT100' in selected['method_revision']
            assert selected['anchor']['method']==selected['method_revision']
            assert selected['anchor']['final_unit_descriptor_bound']['epsilon']==.10
            assert selected['anchor']['final_unit_descriptor_bound']['parameter_change']==0
            normal_arrays = run / 'best_official_arrays_restore_N_paper6.npz'
            assert normal_arrays.is_file()
            sample = check_limits(gpu)
            while sample['memory_used_mib'] >= 500:
                with exchange:
                    print('WAIT_SELECTED_GPU', gpu, sample['memory_used_mib'], flush=True)
                time.sleep(240)
                sample = check_limits(gpu)
            argv = [sys.executable, '-u', 'evaluate_r201n_paper_missing6_stream.py', '--run-dir', str(run),
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
                            expected = 1 if args.mode == 'native' else 6
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
            assert result['final_unit_shift_epsilon']==.10
            assert all(b['epsilon']==.10 and b['all_samples_budget_pass'] and max(b['maximum_shift'].values())<=.10+1e-6 for b in result['final_descriptor_budgets'].values())
            if args.mode == 'native':
                assert set(result['native_small_batch_checks']) == {'R', 'N', 'T', 'RN', 'RT', 'NT', 'RNT'}
            completed.append(dict(name=name,dataset=dataset,variant=variant,seed=job['seed'],selection_scope=job['selection_scope'],gpu=gpu,result=str(output/'result.json')))
        return completed

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(lane, variant, gpu) for variant, gpu in LANES]
        completed = [row for future in futures for row in future.result()]
    expected = controls if args.mode == 'native' else 6*controls
    result = dict(status='PASS' if args.mode == 'native' else 'COMPLETE', controls=controls, runs=completed,
        normal_GT_state_cases=4*controls if args.mode == 'native' else 0,
        missing_GT_state_cases=24*controls if args.mode == 'full' else 0, raw_conditions=expected,
        all7_small_batch_checks_per_control=args.mode == 'native', all_raw_local_verified=True,
        optimizer_updates=0, new_weights=0, normal_phase_preceded_missing=True, final_unit_shift_epsilon=.10,controlled_final_shift_bound=True,
        limits='Only six symmetric paper masks, no new49. Frozen normal-mAP selected checkpoints; no missing-specific selection. MSVR is paper-protocol extension; preflight/closure does not establish +1 or multiseed gains.')
    save(out / ('controller_' + args.mode + '_result.json'), result)
    print(PREFIX + json.dumps(dict(event='CONTROLLER_COMPLETE', mode=args.mode, controls=controls,
        conditions=expected, cases=4 * expected, optimizer_updates=0)), flush=True)


if __name__ == '__main__':
    main()
