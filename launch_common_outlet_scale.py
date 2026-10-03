"""Read-only three-control outlet measurement on physical GPUs2/3."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

from launch_axis_scaled import execute
from run_experiment import write_json


QUEUES = {2: ('axis_shared', 'twins_shared'), 3: ('frequency_shared',)}


def main():
    parser = argparse.ArgumentParser()
    for name in ('campaign', 'output', 'data-root', 'pretrained'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    campaign, output = Path(args.campaign), Path(args.output)
    assert json.loads((campaign / 'controller_result.json').read_text())['status'] == 'COMPLETE'
    output.mkdir(exist_ok=False)
    for variants in QUEUES.values():
        for variant in variants:
            (output / ('MSVR310_' + variant + '_s42')).mkdir()

    def run(gpu, variant, smoke):
        name = 'MSVR310_' + variant + '_s42'
        folder = output / name
        stage = 'smoke' if smoke else 'full'
        command = [sys.executable, '-u', 'diagnose_common_outlet_scale.py',
                   '--run-dir', str(campaign / 'development' / name),
                   '--previous-frozen', str(campaign / 'frozen49' / name),
                   '--output', str(folder / stage), '--data-root', args.data_root, '--pretrained', args.pretrained]
        if smoke:
            command.append('--smoke')
        row = execute(command, folder, stage, gpu)
        result = json.loads((folder / stage / ('smoke.json' if smoke else 'result.json')).read_text())
        assert result['status'] == ('PASS' if smoke else 'COMPLETE')
        assert result['normal_feature_max_error'] == 0 and result['original_fuse_reconstructed_exact']
        if not smoke:
            assert result['metric_count'] == 294 and result['previous_all49_deployed_exact']
        return dict(variant=variant, gpu=gpu, **row)

    def queue(gpu, variants, smoke):
        return [run(gpu, variant, smoke) for variant in variants]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(queue, gpu, variants, True) for gpu, variants in QUEUES.items()]
        smokes = [row for future in futures for row in future.result()]
    write_json(output / 'smoke_result.json', dict(status='PASS', runs=smokes))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(queue, gpu, variants, False) for gpu, variants in QUEUES.items()]
        full = [row for future in futures for row in future.result()]
    write_json(output / 'controller_result.json', dict(status='COMPLETE', runs=full, optimizer_updates=0,
        selected_gpus=[2, 3], metric_count=882, new_weights=0))


if __name__ == '__main__':
    main()
