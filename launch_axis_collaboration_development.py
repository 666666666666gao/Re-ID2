"""Fixed V3 development budget after actual V2 exits; no official-test selection."""
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
    0: [('MSVR310', 'axis_collaboration', .05), ('RGBNT100', 'ordinary_frequency', .05)],
    1: [('RGBNT100', 'axis_collaboration', .05), ('MSVR310', 'ordinary_frequency', .05)],
    2: [('RGBNT100', 'plain_twins', .05), ('MSVR310', 'plain_twins', .05)],
    3: [('RGBNT201', 'axis_collaboration', .05), ('RGBNT201', 'plain_twins', .05),
        ('RGBNT201', 'ordinary_frequency', .05), ('MSVR310', 'axis_collaboration', 0.)],
}
PREDECESSORS = {
    0: ['RGBNT201_dual_residual_s42'],
    1: ['RGBNT100_dual_residual_s42'],
    2: ['MSVR310_dual_residual_s42', 'RGBNT201_ordinary_residual_s42'],
    3: ['MSVR310_ordinary_residual_s42', 'RGBNT100_ordinary_residual_s42'],
}


def run_name(dataset, variant, contribution):
    suffix = '_no_contribution' if contribution == 0 else ''
    return f'{dataset}_{variant}{suffix}_s42'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    args = parser.parse_args()
    preflight = Path('runs/axis_collaboration_v3_preflight_fp32/controller_result.json')
    while not preflight.exists():
        time.sleep(240)
    checks = json.loads(preflight.read_text())
    assert checks['status'] == 'PASS' and len(checks['checks']) == 13 and all(row['exit_code'] == 0 for row in checks['checks'])
    root = Path('runs/axis_collaboration_v3_development')
    root.mkdir(exist_ok=False)

    def slot(gpu):
        for predecessor in PREDECESSORS[gpu]:
            run = Path('runs/residual_development_v2')/predecessor
            while not (run/'exit.json').exists():
                time.sleep(240)
            assert json.loads((run/'exit.json').read_text())['exit_code'] == 0
            result = json.loads((run/'result.json').read_text())
            assert result['status'] == 'COMPLETE' and result['epochs'] == 50
        results = []
        for dataset, variant, contribution in SCHEDULE[gpu]:
            while True:
                used = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
                if int(used.strip()) < 500:
                    break
                print('WAIT_V3_GPU', gpu, used.strip(), flush=True)
                time.sleep(240)
            name = run_name(dataset, variant, contribution)
            output = root/name
            output.mkdir()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            command = [sys.executable, '-u', 'run_experiment.py', '--dataset', dataset, '--variant', variant,
                       '--seed', '42', '--mode', 'train', '--contribution-weight', str(contribution),
                       '--data-root', args.data_root, '--pretrained', args.pretrained, '--output', str(output)]
            with (output/'stdout.log').open('x') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                write_json(output/'launch.json', {'pid': child.pid, 'gpu': gpu, 'started': time.time(), 'command': command})
                code = child.wait()
            row = {'run': name, 'exit_code': code, 'finished': time.time()}
            write_json(output/'exit.json', row)
            results.append(row)
            assert code == 0, name
        return results

    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(slot, gpu) for gpu in SCHEDULE]
        results = [row for future in futures for row in future.result()]
    assert len(results) == 10
    write_json(root/'controller_result.json', results)


if __name__ == '__main__':
    main()
