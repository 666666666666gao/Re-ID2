"""Sequential true-data V3 verification and native-AMP smokes on released GPU0."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from launch_runs import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    args = parser.parse_args()
    predecessor = Path('runs/residual_development_v2/RGBNT201_dual_residual_s42')
    assert json.loads((predecessor/'exit.json').read_text())['exit_code'] == 0
    result = json.loads((predecessor/'result.json').read_text())
    assert result['status'] == 'COMPLETE' and result['epochs'] == 50
    while True:
        used = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
        if int(used.strip()) < 500:
            break
        print('WAIT_V3_PREFLIGHT_GPU0', used.strip(), flush=True)
        time.sleep(240)
    root = Path('runs/axis_collaboration_v3_preflight_fp32')
    root.mkdir(exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    checks = []
    for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
        commands = [('access', [sys.executable, '-u', 'verify_axis_collaboration.py', '--dataset', dataset])]
        commands += [(variant, [sys.executable, '-u', 'run_experiment.py', '--dataset', dataset,
                                '--variant', variant, '--seed', '42', '--mode', 'smoke'])
                     for variant in ('axis_collaboration', 'plain_twins', 'ordinary_frequency')]
        if dataset == 'MSVR310':
            commands.append(('axis_collaboration_no_contribution', [sys.executable, '-u', 'run_experiment.py',
                             '--dataset', dataset, '--variant', 'axis_collaboration', '--seed', '42',
                             '--mode', 'smoke', '--contribution-weight', '0']))
        for label, command in commands:
            name = dataset + '_' + label
            command += ['--data-root', args.data_root, '--pretrained', args.pretrained, '--output', str(root/name)]
            with (root/(name+'.log')).open('x') as output:
                process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
                code = process.wait()
            row = {'name': name, 'exit_code': code, 'finished': time.time()}
            checks.append(row)
            write_json(root/(name+'_exit.json'), row)
            assert code == 0, name
    assert len(checks) == 13
    write_json(root/'controller_result.json', {'status': 'PASS', 'checks': checks})


if __name__ == '__main__':
    main()
