"""Use the freed GPU3 for every completed checkpoint's fixed-weight diagnostics."""
import argparse
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
    last_msvr = 'MSVR310_dual_s43' if args.host == '2025' else 'MSVR310_dual_s44'
    terminal = Path('runs/three_seed_extension') / last_msvr / 'evaluation_exit.json'
    assert json.loads(terminal.read_text())['exit_code'] == 0
    pending = []
    for gpu in FIRST[args.host]:
        pending.extend(Path('runs/dynamic_amp_comparison') / f'{d}_{v}_s42' for d, v in FIRST[args.host][gpu])
        pending.extend(Path('runs/three_seed_extension') / f'{d}_{v}_s{s}' for d, v, s in REPEATS[args.host][gpu][1])
    results = []
    if args.sanity_run:
        pending = [run for run in pending if run.name == args.sanity_run]
        assert len(pending) == 1
    for run in list(pending):
        receipt = run / 'stress_exit.json'
        if receipt.exists():
            assert json.loads(receipt.read_text())['exit_code'] == 0, str(run)
            assert json.loads((run / 'stress_evaluation/result.json').read_text())['status'] == 'COMPLETE'
            results.append({'run': str(run), 'exit_code': 0, 'existing_completed': True})
            pending.remove(run)
    while pending:
        ready = []
        for run in pending:
            exit_path = run / 'exit.json'
            if exit_path.exists():
                assert json.loads(exit_path.read_text())['exit_code'] == 0, str(run)
                assert json.loads((run / 'result.json').read_text())['status'] == 'COMPLETE'
                ready.append(run)
        if not ready:
            time.sleep(240)
            continue
        run = ready[0]
        used = subprocess.check_output(['nvidia-smi', '-i', '3', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
        assert int(used.strip()) < 500, used
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='3', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        command = [sys.executable, '-u', 'stress_evaluation.py', '--run-dir', str(run)]
        with (run / 'stress_stdout.log').open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            write_json(run / 'stress_launch.json', {'gpu': 3, 'pid': process.pid, 'started': time.time(), 'command': command})
            code = process.wait()
        write_json(run / 'stress_exit.json', {'exit_code': code, 'finished': time.time()})
        results.append({'run': str(run), 'exit_code': code})
        write_json(Path('runs/stress_sanity_result.json' if args.sanity_run else 'runs/stress_controller_result.json'), results)
        assert code == 0, str(run)
        pending.remove(run)


if __name__ == '__main__':
    main()
