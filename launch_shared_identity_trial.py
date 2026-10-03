"""One reviewed four-GPU wave: contract, all smokes, matched fresh50, frozen49."""
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
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    preflight = output / 'preflight'
    preflight.mkdir()
    tensor = execute([sys.executable, '-u', 'verify_shared_identity_axis.py', '--data-root', args.data_root,
                      '--pretrained', args.pretrained, '--output', str(preflight / 'tensor')], preflight, 'tensor', 0)
    assert json.loads((preflight / 'tensor/result.json').read_text())['status'] == 'PASS_SHARED_IDENTITY_TENSOR_CONTRACT'

    def command(variant):
        return [sys.executable, '-u', 'run_shared_identity_experiment.py', '--dataset', 'MSVR310',
                '--variant', variant, '--seed', '42', '--data-root', args.data_root, '--pretrained', args.pretrained]

    def smoke(index, variant):
        row = execute(command(variant) + ['--mode', 'smoke', '--output', str(preflight / variant)], preflight, variant, index)
        result = json.loads((preflight / variant / 'smoke.json').read_text())
        assert result['status'] == 'SMOKE_PASS' and result['steps'] == 3 and result['strict_reload_equal']
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(smoke, index, variant) for index, variant in enumerate(VARIANTS)]
        smokes = [future.result() for future in futures]
    write_json(output / 'preflight_result.json', {'status': 'PASS', 'tensor': tensor, 'smokes': smokes})
    development = output / 'development'
    development.mkdir()
    frozen = output / 'frozen49'
    frozen.mkdir()

    def slot(index, variant):
        name = 'MSVR310_' + variant + '_s42'
        run = development / name
        train = execute(command(variant) + ['--mode', 'train', '--output', str(run)], development, name, index)
        result = json.loads((run / 'result.json').read_text())
        assert result['status'] == 'COMPLETE' and result['epochs'] == 50
        evaluation = execute([sys.executable, '-u', 'missing_shared_identity_development.py', '--run-dir', str(run),
                              '--output', str(frozen / name), '--data-root', args.data_root, '--pretrained', args.pretrained],
                             frozen, name, index)
        evaluated = json.loads((frozen / name / 'result.json').read_text())
        assert evaluated['status'] == 'COMPLETE' and len(evaluated['measurements']) == 49
        assert evaluated['normal_feature_max_error'] == 0 and evaluated['state_tensor_versions_unchanged']
        return {'variant': variant, 'train': train, 'frozen49': evaluation}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(slot, index, variant) for index, variant in enumerate(VARIANTS)]
        rows = [future.result() for future in futures]
    write_json(output / 'controller_result.json', {'status': 'COMPLETE', 'runs': rows})


if __name__ == '__main__':
    main()
