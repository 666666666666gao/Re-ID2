"""Two serial GPU queues: matched pooling fresh50, full49 and294 controlled-state cases each."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

from gpu_thermal_execute import check_limits, execute


SCHEDULE = {
    2: [('original_mean', 'axis_shared'), ('eligible_mean', 'axis_shared'), ('seed_query', 'axis_shared'),
        ('original_mean', 'twins_shared'), ('eligible_mean', 'twins_shared')],
    3: [('original_mean', 'frequency_shared'), ('eligible_mean', 'frequency_shared'), ('seed_query', 'frequency_shared'),
        ('seed_query', 'twins_shared'), ('original_mean', 'demo_shared')],
}


def save(path, record):
    path.write_text(json.dumps(record, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    for gpu in SCHEDULE:
        check_limits(gpu)
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    preflight = output / 'preflight'
    preflight.mkdir()
    tensor = execute([sys.executable, '-u', 'verify_common_outlet_axis.py', '--data-root', args.data_root,
                      '--pretrained', args.pretrained, '--output', str(preflight / 'tensor')], preflight, 'tensor', 2)
    assert json.loads((preflight / 'tensor/result.json').read_text())['status'] == 'PASS_COMMON_OUTLET_TENSOR_CONTRACT'

    def command(pooling, variant):
        return [sys.executable, '-u', 'run_common_outlet_experiment.py', '--dataset', 'MSVR310', '--pooling', pooling,
                '--variant', variant, '--seed', '42', '--data-root', args.data_root, '--pretrained', args.pretrained]

    def smokes(gpu):
        rows = []
        for pooling, variant in SCHEDULE[gpu]:
            name = pooling + '_' + variant
            row = execute(command(pooling, variant) + ['--mode', 'smoke', '--output', str(preflight / name)], preflight, name, gpu)
            result = json.loads((preflight / name / 'smoke.json').read_text())
            assert result['status'] == 'SMOKE_PASS' and result['steps'] == 3 and result['strict_reload_equal']
            assert result['arguments']['pooling'] == pooling
            rows.append(row)
        return rows

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(smokes, gpu) for gpu in SCHEDULE]
        rows = [row for future in futures for row in future.result()]
    assert len(rows) == 10
    save(output / 'preflight_result.json', dict(status='PASS', tensor=tensor, smokes=rows))
    for pooling in ('original_mean', 'eligible_mean', 'seed_query'):
        root = output / pooling
        root.mkdir()
        for name in ('development', 'frozen49', 'controlled_states'):
            (root / name).mkdir()

    def jobs(gpu):
        rows = []
        for pooling, variant in SCHEDULE[gpu]:
            root = output / pooling
            name = 'MSVR310_' + variant + '_s42'
            run = root / 'development' / name
            train = execute(command(pooling, variant) + ['--mode', 'train', '--output', str(run)], root / 'development', name, gpu)
            result = json.loads((run / 'result.json').read_text())
            assert result['status'] == 'COMPLETE' and result['epochs'] == 50 and result['arguments']['pooling'] == pooling
            frozen = root / 'frozen49' / name
            evaluation = execute([sys.executable, '-u', 'missing_common_outlet_development.py', '--run-dir', str(run),
                                  '--output', str(frozen), '--data-root', args.data_root, '--pretrained', args.pretrained],
                                 root / 'frozen49', name, gpu)
            measured = json.loads((frozen / 'result.json').read_text())
            assert measured['status'] == 'COMPLETE' and len(measured['measurements']) == 49
            assert measured['normal_feature_max_error'] == 0 and measured['state_tensor_versions_unchanged']
            state = root / 'controlled_states' / name
            state.mkdir()
            command_states = [sys.executable, '-u', 'diagnose_common_outlet_states.py', '--run-dir', str(run), '--previous-frozen',
                              str(frozen), '--data-root', args.data_root, '--pretrained', args.pretrained]
            smoke = execute(command_states + ['--output', str(state / 'smoke'), '--smoke'], state, 'smoke', gpu)
            assert json.loads((state / 'smoke/smoke.json').read_text())['status'] == 'PASS'
            full = execute(command_states + ['--output', str(state / 'full')], state, 'full', gpu)
            terminal = json.loads((state / 'full/result.json').read_text())
            assert terminal['status'] == 'COMPLETE' and terminal['metric_count'] == 294
            assert terminal['normal_feature_max_error'] == 0 and terminal['previous_all49_full11_sixmetric_and_perquery_exact']
            rows.append(dict(pooling=pooling, variant=variant, gpu=gpu, train=train, frozen49=evaluation, state_smoke=smoke, state_full=full))
        return rows

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(jobs, gpu) for gpu in SCHEDULE]
        rows = [row for future in futures for row in future.result()]
    assert len(rows) == 10
    for pooling in ('original_mean', 'eligible_mean', 'seed_query'):
        selected = [row for row in rows if row['pooling'] == pooling]
        save(output / pooling / 'controller_result.json', dict(status='COMPLETE', runs=selected))
        save(output / pooling / 'controlled_states/controller_result.json', dict(status='COMPLETE', runs=selected))
    save(output / 'controller_result.json', dict(status='COMPLETE', runs=rows, power_limit_w=250, temperature_ceiling_c=75))


if __name__ == '__main__':
    main()
