"""Three real config anchors and six fresh three-update training/restoration checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_experiment import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    args = parser.parse_args()
    root = Path('runs/residual_v2_preflight')
    root.mkdir(exist_ok=False)

    def slot(gpu, dataset):
        used = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
                                        '--format=csv,noheader,nounits'], text=True)
        assert int(used.strip()) < 500, (gpu, used)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        base = ['--dataset', dataset, '--data-root', args.data_root, '--pretrained', args.pretrained]
        commands = [(dataset + '_anchor', [sys.executable, '-u', 'verify_residual.py', *base,
                                          '--output', str(root/(dataset + '_anchor'))])]
        commands += [(dataset + '_' + variant, [sys.executable, '-u', 'run_experiment.py', *base,
                        '--mode', 'smoke', '--variant', variant, '--seed', '42', '--output', str(root/(dataset + '_' + variant))])
                     for variant in ('ordinary_residual', 'dual_residual')]
        results = []
        for name, command in commands:
            with (root/(name + '.log')).open('x') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                write_json(root/(name + '_launch.json'), {'gpu': gpu, 'pid': process.pid, 'command': command, 'started': time.time()})
                code = process.wait()
            row = {'name': name, 'exit_code': code, 'finished': time.time()}
            write_json(root/(name + '_exit.json'), row)
            results.append(row)
            assert code == 0, name
        return results

    with ThreadPoolExecutor(max_workers=3) as workers:
        futures = [workers.submit(slot, gpu, dataset) for gpu, dataset in ((1, 'RGBNT201'), (2, 'RGBNT100'), (3, 'MSVR310'))]
        results = [row for future in futures for row in future.result()]
    assert len(results) == 9 and all(row['exit_code'] == 0 for row in results)
    write_json(root/'controller_result.json', {'status': 'PASS', 'checks': results, 'optimizer_updates': 18})


if __name__ == '__main__':
    main()
