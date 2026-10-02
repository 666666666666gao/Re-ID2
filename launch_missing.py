"""Per-card missing-modality evaluations after the existing normal and stress tests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from launch_runs import SCHEDULE as FIRST, write_json
from launch_repeats import SCHEDULE as REPEATS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', choices=list(FIRST), required=True)
    parser.add_argument('--sanity-run')
    args = parser.parse_args()
    expected_host = sum(len(FIRST[args.host][g]) + len(REPEATS[args.host][g][1]) for g in FIRST[args.host])
    # Both existing controllers can have later jobs on an apparently free card.
    # Do not enter their queues after only an individual run's exit receipt.
    for filename in ('full_evaluation_controller_result.json', 'stress_controller_result.json'):
        predecessor = Path('runs') / filename
        while True:
            if predecessor.exists():
                completed = json.loads(predecessor.read_text())
                assert all(row['exit_code'] == 0 for row in completed), str(predecessor)
                if len(completed) == expected_host:
                    break
            time.sleep(240)

    def slot(gpu):
        jobs = [(Path('runs/dynamic_amp_comparison') / f'{d}_{v}_s42') for d, v in FIRST[args.host][gpu]]
        jobs += [(Path('runs/three_seed_extension') / f'{d}_{v}_s{s}') for d, v, s in REPEATS[args.host][gpu][1]]
        if args.sanity_run:
            jobs = [run for run in jobs if run.name == args.sanity_run]
        results = []
        for run in jobs:
            receipt = run / 'missing_exit.json'
            if receipt.exists():
                assert json.loads(receipt.read_text())['exit_code'] == 0, str(run)
                assert json.loads((run / 'missing_evaluation/result.json').read_text())['status'] == 'COMPLETE'
                results.append({'run': str(run), 'exit_code': 0, 'existing_completed': True})
                continue
            for name in ('evaluation_exit.json', 'stress_exit.json'):
                predecessor = run / name
                while not predecessor.exists():
                    time.sleep(240)
                assert json.loads(predecessor.read_text())['exit_code'] == 0, str(predecessor)
            while True:
                used = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                                                '--format=csv,noheader,nounits'], text=True)
                if int(used.strip()) < 500:
                    break
                print(f'WAIT_MISSING_GPU gpu={gpu} memory_used_mib={used.strip()}', flush=True)
                time.sleep(240)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            command = [sys.executable, '-u', 'missing_evaluation.py', '--run-dir', str(run)]
            with (run / 'missing_stdout.log').open('x') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                write_json(run / 'missing_launch.json', {'gpu': gpu, 'pid': process.pid, 'started': time.time(), 'command': command})
                code = process.wait()
            write_json(run / 'missing_exit.json', {'exit_code': code, 'finished': time.time()})
            results.append({'run': str(run), 'exit_code': code})
            assert code == 0, str(run)
        return results

    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(slot, gpu) for gpu in FIRST[args.host]]
        results = [row for future in futures for row in future.result()]
    expected = 1 if args.sanity_run else expected_host
    assert len(results) == expected and all(row['exit_code'] == 0 for row in results)
    write_json(Path('runs/missing_sanity_controller_result.json' if args.sanity_run else 'runs/missing_controller_result.json'), results)


if __name__ == '__main__':
    main()
