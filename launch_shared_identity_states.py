"""Four frozen V11 jobs, each seven-bank smoke before full controlled evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

from launch_axis_scaled import execute
from run_experiment import write_json


VARIANTS = ('axis_shared', 'frequency_shared', 'twins_shared', 'demo_shared')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    args = parser.parse_args()
    campaign, output = Path(args.campaign), Path(args.output)
    assert json.loads((campaign / 'controller_result.json').read_text())['status'] == 'COMPLETE'
    output.mkdir(exist_ok=False)

    def slot(gpu, variant):
        name = 'MSVR310_' + variant + '_s42'
        root = output / name
        root.mkdir()
        command = [sys.executable, '-u', 'diagnose_shared_identity_states.py', '--run-dir', str(campaign / 'development' / name),
                   '--previous-frozen', str(campaign / 'frozen49' / name), '--data-root', args.data_root, '--pretrained', args.pretrained]
        smoke = execute(command + ['--output', str(root / 'smoke'), '--smoke'], root, 'smoke', gpu)
        assert json.loads((root / 'smoke/smoke.json').read_text())['status'] == 'PASS'
        full = execute(command + ['--output', str(root / 'full')], root, 'full', gpu)
        result = json.loads((root / 'full/result.json').read_text())
        assert result['status'] == 'COMPLETE' and result['metric_count'] == 294
        assert result['normal_feature_max_error'] == 0 and result['previous_all49_full11_sixmetric_and_perquery_exact']
        write_json(root / 'controller_result.json', dict(status='COMPLETE', smoke=smoke, full=full))
        return dict(variant=variant, smoke=smoke, full=full)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(slot, index, variant) for index, variant in enumerate(VARIANTS)]
        rows = [future.result() for future in futures]
    write_json(output / 'controller_result.json', dict(status='COMPLETE', runs=rows))


if __name__ == '__main__':
    main()
