"""Three frozen M2 best models, all-smoke barrier, physical GPU2/3 only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

from gpu_thermal_execute import execute


QUEUES = {2: ('axis_shared', 'twins_shared'), 3: ('frequency_shared',)}


def main():
    parser = argparse.ArgumentParser()
    for name in ('campaign', 'output', 'data-root', 'pretrained'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    campaign, output = Path(args.campaign), Path(args.output)
    terminal = json.loads((campaign / 'controller_result.json').read_text())
    audit = json.loads((campaign / 'independent_cpu_audit.json').read_text())
    assert terminal['status'] == 'COMPLETE' and len(terminal['runs']) == 3
    assert audit['status'] == 'PASS' and audit['cases'] == 2058
    output.mkdir(exist_ok=False)
    for variant in ('axis_shared', 'frequency_shared', 'twins_shared'):
        (output / variant).mkdir()

    def queue(gpu, variants, smoke):
        rows = []
        for variant in variants:
            name = 'MSVR310_' + variant + '_s42'
            folder = output / variant
            stage = 'smoke' if smoke else 'full'
            command = [sys.executable, '-u', 'diagnose_cross_identity_coordinates.py',
                '--run-dir', str(campaign / 'original_mean/development' / name),
                '--previous-utility', str(campaign / 'original_mean/utility' / name / 'full'),
                '--output', str(folder / stage), '--data-root', args.data_root,
                '--pretrained', args.pretrained]
            if smoke:
                command.append('--smoke')
            row = execute(command, folder, stage, gpu)
            result = json.loads((folder / stage / ('smoke.json' if smoke else 'result.json')).read_text())
            assert result['status'] == ('PASS' if smoke else 'COMPLETE')
            assert result['variant'] == variant and result['normal_feature_max_error'] == 0
            assert result['original_fuse_reconstructed_exact'] and result['state_tensor_versions_unchanged']
            assert result['optimizer_updates'] == result['new_weights'] == 0
            if not smoke:
                assert result['metric_count'] == 343 and result['previous_same_coordinate_metrics_perquery_exact']
            rows.append(dict(variant=variant, **row))
        return rows

    for smoke in (True, False):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(queue, gpu, variants, smoke) for gpu, variants in QUEUES.items()]
            rows = [row for future in futures for row in future.result()]
        assert len(rows) == 3
        (output / ('smoke_result.json' if smoke else 'controller_result.json')).write_text(
            json.dumps(dict(status='PASS' if smoke else 'COMPLETE', runs=rows,
                metric_count=0 if smoke else 1029, optimizer_updates=0, new_weights=0,
                selected_gpus=[2, 3], temperature_power_control=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
