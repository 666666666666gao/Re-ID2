"""Three-seed continuation on each GPU after its existing assigned work ends."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from launch_runs import write_json

SCHEDULE = {
    '2025': {
        0: ('RGBNT201_demo_s42', [('RGBNT201', 'demo', 43), ('RGBNT201', 'demo', 44)]),
        1: ('RGBNT201_ordinary_s42', [('RGBNT201', 'ordinary', 43), ('RGBNT201', 'ordinary', 44)]),
        2: ('RGBNT201_dual_s42', [('RGBNT201', 'dual', 43), ('RGBNT201', 'dual', 44)]),
        3: ('MSVR310_demo_s42', [('MSVR310', v, 43) for v in ('demo', 'ordinary', 'dual')]),
    },
    '2026': {
        0: ('RGBNT100_demo_s42', [('RGBNT100', 'demo', 43), ('RGBNT100', 'demo', 44)]),
        1: ('RGBNT100_ordinary_s42', [('RGBNT100', 'ordinary', 43), ('RGBNT100', 'ordinary', 44)]),
        2: ('RGBNT100_dual_s42', [('RGBNT100', 'dual', 43), ('RGBNT100', 'dual', 44)]),
        3: ('MSVR310_dual_s42', [('MSVR310', v, 44) for v in ('demo', 'ordinary', 'dual')]),
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', choices=list(SCHEDULE), required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    args = parser.parse_args()
    root = Path('runs/three_seed_extension')
    root.mkdir(parents=True, exist_ok=False)

    def slot(gpu, predecessor, jobs):
        terminal = Path('runs/dynamic_amp_comparison') / predecessor / 'exit.json'
        while not terminal.exists():
            time.sleep(240)
        assert json.loads(terminal.read_text())['exit_code'] == 0, predecessor
        used = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                                        '--format=csv,noheader,nounits'], text=True)
        assert int(used.strip()) < 500, (gpu, used)
        results = []
        for dataset, variant, seed in jobs:
            name = f'{dataset}_{variant}_s{seed}'
            output = root / name
            output.mkdir()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            command = [sys.executable, '-u', 'run_experiment.py', '--mode', 'train',
                       '--dataset', dataset, '--variant', variant, '--seed', str(seed),
                       '--data-root', args.data_root, '--pretrained', args.pretrained, '--output', str(output)]
            with (output / 'stdout.log').open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                write_json(output / 'launch.json', {'host': args.host, 'gpu': gpu, 'pid': process.pid,
                                                    'command': command, 'started': time.time()})
                code = process.wait()
            write_json(output / 'exit.json', {'exit_code': code, 'finished': time.time()})
            results.append({'run': name, 'exit_code': code})
            if code != 0:
                return results
        return results

    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(slot, gpu, predecessor, jobs)
                   for gpu, (predecessor, jobs) in SCHEDULE[args.host].items()]
        results = [row for future in futures for row in future.result()]
    write_json(root / 'controller_result.json', results)
    assert len(results) == sum(len(jobs) for _, jobs in SCHEDULE[args.host].values())
    assert all(row['exit_code'] == 0 for row in results), results


if __name__ == '__main__':
    main()
