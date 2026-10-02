"""Evaluate all fixed best checkpoints after each GPU's three-seed training ends."""
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
    parser.add_argument('--host', choices=list(REPEATS), required=True)
    args = parser.parse_args()

    def slot(gpu):
        _, repeat_jobs = REPEATS[args.host][gpu]
        dataset, variant, seed = repeat_jobs[-1]
        predecessor = Path('runs/three_seed_extension') / f'{dataset}_{variant}_s{seed}' / 'exit.json'
        while not predecessor.exists():
            time.sleep(240)
        assert json.loads(predecessor.read_text())['exit_code'] == 0, str(predecessor)
        used = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                                        '--format=csv,noheader,nounits'], text=True)
        assert int(used.strip()) < 500, (gpu, used)
        first = [('dynamic_amp_comparison', d, v, 42) for d, v in FIRST[args.host][gpu]]
        repeat = [('three_seed_extension', d, v, s) for d, v, s in repeat_jobs]
        results = []
        for campaign, dataset, variant, seed in first + repeat:
            run = Path('runs') / campaign / f'{dataset}_{variant}_s{seed}'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            command = [sys.executable, '-u', 'full_evaluation.py', '--run-dir', str(run)]
            with (run / 'evaluation_stdout.log').open('x') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                write_json(run / 'evaluation_launch.json', {'gpu': gpu, 'pid': process.pid,
                                                           'command': command, 'started': time.time()})
                code = process.wait()
            write_json(run / 'evaluation_exit.json', {'exit_code': code, 'finished': time.time()})
            results.append({'campaign': campaign, 'run': run.name, 'exit_code': code})
            if code != 0:
                return results
        return results

    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(slot, gpu) for gpu in REPEATS[args.host]]
        results = [row for future in futures for row in future.result()]
    write_json(Path('runs/full_evaluation_controller_result.json'), results)
    expected = sum(len(FIRST[args.host][g]) + len(REPEATS[args.host][g][1]) for g in REPEATS[args.host])
    assert len(results) == expected and all(row['exit_code'] == 0 for row in results), results


if __name__ == '__main__':
    main()
