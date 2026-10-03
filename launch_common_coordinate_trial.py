"""Four matched V12 fresh50 jobs on the user's two selected GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

from launch_axis_scaled import execute
from run_experiment import write_json


VARIANTS = ('axis_shared', 'frequency_shared', 'twins_shared', 'demo_shared')
GPUS = (2, 3)


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
    tensor = execute([sys.executable, '-u', 'verify_common_coordinate_axis.py', '--data-root', args.data_root,
                      '--pretrained', args.pretrained, '--output', str(preflight / 'tensor')], preflight, 'tensor', GPUS[0])
    contract = json.loads((preflight / 'tensor/result.json').read_text())
    assert contract['status'] == 'PASS_SHARED_IDENTITY_TENSOR_CONTRACT'
    assert contract['common_increments_only'] and contract['ordinary_frequency_legacy_parity']

    def command(variant):
        return [sys.executable, '-u', 'run_common_coordinate_experiment.py', '--dataset', 'MSVR310',
                '--variant', variant, '--seed', '42', '--data-root', args.data_root, '--pretrained', args.pretrained]

    def smoke(index, variant):
        row = execute(command(variant) + ['--mode', 'smoke', '--output', str(preflight / variant)], preflight, variant, index)
        result = json.loads((preflight / variant / 'smoke.json').read_text())
        assert result['status'] == 'SMOKE_PASS' and result['steps'] == 3 and result['strict_reload_equal']
        return row

    def smoke_queue(index, gpu):
        return [smoke(gpu, variant) for variant in VARIANTS[index::len(GPUS)]]

    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = [pool.submit(smoke_queue, index, gpu) for index, gpu in enumerate(GPUS)]
        smokes = [row for future in futures for row in future.result()]
    write_json(output / 'preflight_result.json', {'status': 'PASS', 'tensor': tensor, 'smokes': smokes})
    development, frozen, controlled = (output / name for name in ('development', 'frozen49', 'controlled_states'))
    for folder in (development, frozen, controlled):
        folder.mkdir()

    def slot(index, variant):
        name = 'MSVR310_' + variant + '_s42'
        run = development / name
        train = execute(command(variant) + ['--mode', 'train', '--output', str(run)], development, name, index)
        result = json.loads((run / 'result.json').read_text())
        assert result['status'] == 'COMPLETE' and result['epochs'] == 50
        evaluation = execute([sys.executable, '-u', 'missing_common_coordinate_development.py', '--run-dir', str(run),
                              '--output', str(frozen / name), '--data-root', args.data_root, '--pretrained', args.pretrained],
                             frozen, name, index)
        evaluated = json.loads((frozen / name / 'result.json').read_text())
        assert evaluated['status'] == 'COMPLETE' and len(evaluated['measurements']) == 49
        assert evaluated['normal_feature_max_error'] == 0 and evaluated['state_tensor_versions_unchanged']
        state_root = controlled / name
        state_root.mkdir()
        state_command = [sys.executable, '-u', 'diagnose_common_coordinate_states.py', '--run-dir', str(run),
                         '--previous-frozen', str(frozen / name), '--data-root', args.data_root, '--pretrained', args.pretrained]
        state_smoke = execute(state_command + ['--output', str(state_root / 'smoke'), '--smoke'], state_root, 'smoke', index)
        assert json.loads((state_root / 'smoke/smoke.json').read_text())['status'] == 'PASS'
        state_full = execute(state_command + ['--output', str(state_root / 'full')], state_root, 'full', index)
        states = json.loads((state_root / 'full/result.json').read_text())
        assert states['status'] == 'COMPLETE' and states['metric_count'] == 294
        assert states['normal_feature_max_error'] == 0 and states['previous_all49_full11_sixmetric_and_perquery_exact']
        return dict(variant=variant, train=train, frozen49=evaluation, state_smoke=state_smoke, state_full=state_full)

    def train_queue(index, gpu):
        return [slot(gpu, variant) for variant in VARIANTS[index::len(GPUS)]]

    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = [pool.submit(train_queue, index, gpu) for index, gpu in enumerate(GPUS)]
        rows = [row for future in futures for row in future.result()]
    write_json(controlled / 'controller_result.json', dict(status='COMPLETE', runs=rows))
    write_json(output / 'controller_result.json', dict(status='COMPLETE', runs=rows))


if __name__ == '__main__':
    main()
