"""First fixed V2 seed42 development tests after the host's missing evaluations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from launch_runs import write_json

SCHEDULE = {0: ('RGBNT201', 'dual_residual'), 1: ('RGBNT100', 'dual_residual'),
            2: ('MSVR310', 'dual_residual'), 3: ('MSVR310', 'ordinary_residual')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    args = parser.parse_args()
    predecessor = Path('runs/missing_controller_result.json')
    while not predecessor.exists():
        time.sleep(240)
    terminal = json.loads(predecessor.read_text())
    assert len(terminal) == 13 and all(row['exit_code'] == 0 for row in terminal)
    root = Path('runs/residual_development_v2')
    root.mkdir(exist_ok=False)

    def slot(gpu, dataset, variant):
        while True:
            used = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                                            '--format=csv,noheader,nounits'], text=True)
            if int(used.strip()) < 500:
                break
            print(f'WAIT_RESIDUAL_GPU gpu={gpu} memory_used_mib={used.strip()}', flush=True)
            time.sleep(240)
        output = root/f'{dataset}_{variant}_s42'
        output.mkdir()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        command = [sys.executable, '-u', 'run_experiment.py', '--mode', 'train', '--dataset', dataset,
                   '--variant', variant, '--seed', '42', '--data-root', args.data_root,
                   '--pretrained', args.pretrained, '--output', str(output)]
        with (output/'stdout.log').open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            write_json(output/'launch.json', {'gpu': gpu, 'pid': process.pid, 'started': time.time(), 'command': command})
            code = process.wait()
        row = {'run': output.name, 'exit_code': code, 'finished': time.time()}
        write_json(output/'exit.json', row)
        return row

    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(slot, gpu, dataset, variant) for gpu, (dataset, variant) in SCHEDULE.items()]
        results = [future.result() for future in futures]
    write_json(root/'controller_result.json', results)
    assert all(row['exit_code'] == 0 for row in results)


if __name__ == '__main__':
    main()
