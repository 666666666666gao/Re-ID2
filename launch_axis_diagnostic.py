"""Run frozen V3 diagnostics after the selected GPU's final scheduled V3 job."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from launch_runs import write_json
from launch_axis_collaboration_development import SCHEDULE, run_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    assert args.gpu in range(4)
    queue = Path('runs/axis_collaboration_v3_development')
    predecessor = queue / run_name(*SCHEDULE[args.gpu][-1])
    while not (predecessor / 'exit.json').exists():
        print('WAIT_V3_SLOT_FINAL', args.gpu, predecessor.name, flush=True)
        time.sleep(240)
    assert json.loads((predecessor / 'exit.json').read_text())['exit_code'] == 0
    terminal = json.loads((predecessor / 'result.json').read_text())
    assert terminal['status'] == 'COMPLETE' and terminal['epochs'] == 50
    for dataset in ('MSVR310', 'RGBNT201', 'RGBNT100'):
        run = queue / (dataset + '_axis_collaboration_s42')
        assert json.loads((run / 'exit.json').read_text())['exit_code'] == 0
        terminal = json.loads((run / 'result.json').read_text())
        assert terminal['status'] == 'COMPLETE' and terminal['epochs'] == 50
    while True:
        used = int(subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        if used < 500:
            break
        print('WAIT_DIAGNOSTIC_GPU', args.gpu, used, flush=True)
        time.sleep(240)
    output = Path('runs/axis_collaboration_v3_four_state_diagnostic')
    output.mkdir(exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    results = []
    for smoke in (True, False):
        for dataset in ('MSVR310', 'RGBNT201', 'RGBNT100'):
            name = dataset + ('_smoke' if smoke else '_full')
            run = queue / (dataset + '_axis_collaboration_s42')
            destination = output / name
            command = [sys.executable, '-u', 'diagnose_axis_collaboration.py', '--run-dir', str(run), '--output', str(destination)]
            if smoke:
                command.append('--smoke')
            with (output / (name + '.log')).open('x') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                write_json(output / (name + '_launch.json'), {'pid': child.pid, 'gpu': args.gpu, 'started': time.time(), 'command': command})
                code = child.wait()
            result = {'name': name, 'exit_code': code, 'finished': time.time()}
            write_json(output / (name + '_exit.json'), result)
            results.append(result)
            assert code == 0, name
    write_json(output / 'controller_result.json', {'status': 'PASS', 'checks': results, 'optimizer_updates': 0})


if __name__ == '__main__':
    main()
