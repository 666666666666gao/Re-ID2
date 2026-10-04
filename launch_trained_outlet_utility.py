"""Frozen all-three-pooling axis/frequency diagnostic, with a six-smoke barrier."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

from gpu_thermal_execute import execute


POOLS = ('original_mean', 'eligible_mean', 'seed_query')
QUEUES = {2: 'axis_shared', 3: 'frequency_shared'}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    for name in ('campaign', 'output', 'data-root', 'pretrained'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    campaign, output = Path(args.campaign), Path(args.output)
    terminal = json.loads((campaign / 'controller_result.json').read_text())
    assert terminal['status'] == 'COMPLETE' and len(terminal['runs']) == 10
    output.mkdir(exist_ok=False)
    for pooling in POOLS:
        for variant in QUEUES.values():
            (output / pooling / ('MSVR310_' + variant + '_s42')).mkdir(parents=True)

    def run(gpu, variant, smoke):
        rows = []
        for pooling in POOLS:
            name = 'MSVR310_' + variant + '_s42'
            folder = output / pooling / name
            stage = 'smoke' if smoke else 'full'
            command = [sys.executable, '-u', 'diagnose_trained_outlet_utility.py',
                       '--run-dir', str(campaign / pooling / 'development' / name),
                       '--previous-frozen', str(campaign / pooling / 'frozen49' / name),
                       '--output', str(folder / stage), '--data-root', args.data_root,
                       '--pretrained', args.pretrained]
            if smoke:
                command.append('--smoke')
            row = execute(command, folder, stage, gpu)
            result = json.loads((folder / stage / ('smoke.json' if smoke else 'result.json')).read_text())
            assert result['status'] == ('PASS' if smoke else 'COMPLETE')
            assert result['pooling'] == pooling and result['normal_feature_max_error'] == 0
            assert result['original_fuse_reconstructed_exact'] and result['state_tensor_versions_unchanged']
            if not smoke:
                assert result['metric_count'] == 392 and result['previous_all49_deployed_exact']
                assert result['optimizer_updates'] == result['official_test_uses'] == 0
            rows.append(dict(pooling=pooling, variant=variant, **row))
        return rows

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, gpu, variant, True) for gpu, variant in QUEUES.items()]
        smokes = [row for future in futures for row in future.result()]
    assert len(smokes) == 6
    save(output / 'smoke_result.json', dict(status='PASS', runs=smokes))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, gpu, variant, False) for gpu, variant in QUEUES.items()]
        full = [row for future in futures for row in future.result()]
    assert len(full) == 6
    save(output / 'controller_result.json', dict(status='COMPLETE', runs=full, metric_count=2352,
        optimizer_updates=0, new_weights=0, selected_gpus=[2, 3], temperature_power_control=False))


if __name__ == '__main__':
    main()
