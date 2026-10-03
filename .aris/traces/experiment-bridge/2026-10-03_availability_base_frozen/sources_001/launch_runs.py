"""Fixed single-GPU schedules on the two currently idle servers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCHEDULE = {
    '2025': {0: [('RGBNT201', 'demo')], 1: [('RGBNT201', 'ordinary')], 2: [('RGBNT201', 'dual')], 3: [('MSVR310', 'demo')]},
    '2026': {0: [('RGBNT100', 'demo')], 1: [('RGBNT100', 'ordinary')], 2: [('RGBNT100', 'dual')], 3: [('MSVR310', 'ordinary'), ('MSVR310', 'dual')]},
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', choices=list(SCHEDULE), required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--pretrained', required=True)
    args = p.parse_args()
    root = Path('runs/dynamic_amp_comparison')
    root.mkdir(parents=True, exist_ok=False)
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    assert all(int(row.split(',')[1]) < 500 for row in gpu_rows.strip().splitlines()), gpu_rows

    def slot(gpu, jobs):
        results = []
        for dataset, variant in jobs:
            name = f'{dataset}_{variant}_s42'
            output = root / name
            output.mkdir()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            command = [sys.executable, '-u', 'run_experiment.py', '--mode', 'train', '--dataset', dataset, '--variant', variant,
                       '--data-root', args.data_root, '--pretrained', args.pretrained, '--output', str(output)]
            with (output / 'stdout.log').open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                write_json(output / 'launch.json', {'host': args.host, 'gpu': gpu, 'pid': process.pid, 'command': command, 'started': time.time()})
                code = process.wait()
            write_json(output / 'exit.json', {'exit_code': code, 'finished': time.time()})
            results.append({'run': name, 'exit_code': code})
            if code != 0:
                return results
        return results

    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(slot, gpu, jobs) for gpu, jobs in SCHEDULE[args.host].items()]
        results = [result for future in futures for result in future.result()]
    write_json(root / 'controller_result.json', results)
    assert len(results) == sum(len(j) for j in SCHEDULE[args.host].values()) and all(r['exit_code'] == 0 for r in results), results


if __name__ == '__main__':
    main()
