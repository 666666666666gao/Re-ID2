"""AN sanity then train-only selected-weight probes on physical GPU2/3."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

from gpu_thermal_execute import execute
from run_experiment import write_json

CONTROLS = (('frequency_shared', 2), ('axis_shared', 3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    for phase in ('native', 'full'):
        (root / phase).mkdir()
    write_json(root/'controller_launch.json', dict(pid=os.getpid(), started=time.time(),
        physical_gpus=[2, 3], maximum_NN=2, optimizer_updates=0, temperature_power_control=False))

    def run(pair, phase):
        variant, gpu = pair
        name = 'RGBNT100_r201al_' + variant + '_s42'
        output = root / phase / (name + '.json')
        source = Path(args.training_root) / name
        exited = execute([sys.executable, '-u', 'probe_r201an_AL_selected_task_gradients.py',
            '--run-dir', str(source), '--output', str(output), '--batches', '1' if phase == 'native' else '4'],
            root / phase, name, gpu)
        result = json.loads(output.read_text())
        assert result['status'] == ('SMOKE_PASS' if phase == 'native' else 'COMPLETE')
        assert result['optimizer_updates'] == result['checkpoint_writes'] == result['official_query_gallery_neural_uses'] == 0
        assert result['parameters_unchanged'] and result['frozen_identity_unchanged']
        return dict(name=name, gpu=gpu, exited=exited, result=str(output), batches=result['batches'])

    with ThreadPoolExecutor(max_workers=2) as pool:
        native = list(pool.map(lambda pair: run(pair, 'native'), CONTROLS))
    write_json(root/'native_acceptance.json', dict(status='PASS', batches=2, optimizer_updates=0, runs=native))
    print('AN_NATIVE_PASS', flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        full = list(pool.map(lambda pair: run(pair, 'full'), CONTROLS))
    a, b = [json.loads(Path(r['result']).read_text()) for r in full]
    assert [(r['names'], r['training_labels']) for r in a['rows']] == [(r['names'], r['training_labels']) for r in b['rows']]
    write_json(root/'controller_result.json', dict(status='COMPLETE', native=native, full=full,
        physical_gpus=[2, 3], maximum_NN=2, neural_forwards=10, native_forwards=2, full_forwards=8,
        optimizer_updates=0, checkpoint_writes=0, official_query_gallery_neural_uses=0, paired_names_labels=True))
    print('AN_CONTROLLER_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
