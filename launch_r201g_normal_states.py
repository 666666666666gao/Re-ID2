"""After closed R201G, two selected checkpoints, full normal four-state utility."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
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
    source, out = Path(args.training_root), Path(args.output)
    controller = json.loads((source / 'controller_result.json').read_text())
    assert controller['status'] == 'COMPLETE' and controller['successful_updates'] == 5294
    assert controller['normal_archives_local_verified'] == 2 and controller['paired_sampling_exact']
    launch = json.loads((source / 'controller_launch.json').read_text())
    for pid in [launch['pid']] + [json.loads((source / 'training' /
            ('RGBNT201_r201g_' + variant + '_s42_launch.json')).read_text())['pid'] for variant, _ in CONTROLS]:
        assert subprocess.run(['ps', '-p', str(pid)], stdout=subprocess.DEVNULL).returncode == 1
    out.mkdir(exist_ok=False)
    write_json(out / 'controller_launch.json', dict(pid=os.getpid(), started=time.time(),
        source_root=str(source), gpus=[2, 3], maximum_NN=2, temperature_power_control=False))

    def evaluate(variant, gpu):
        name = 'RGBNT201_r201g_' + variant + '_s42'
        run = source / 'training' / name
        weights = run / 'best.pth'
        before = hashlib.sha256(weights.read_bytes()).hexdigest()
        execute([sys.executable, '-u', 'evaluate_identity_coordinate_normal_states.py',
                 '--run-dir', str(run), '--output', str(out / name)], out, name, gpu)
        assert hashlib.sha256(weights.read_bytes()).hexdigest() == before
        result = json.loads((out / name / 'result.json').read_text())
        assert result['status'] == 'COMPLETE' and result['state_cases'] == 4
        assert result['optimizer_updates'] == result['checkpoint_writes'] == 0
        return dict(name=name, variant=variant, gpu=gpu, checkpoint_unchanged=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(evaluate, variant, gpu) for variant, gpu in CONTROLS]
        controls = [future.result() for future in futures]
    # ACK one original raw archive at a time before clearing its remote copy.
    for control in controls:
        raw = out / control['name'] / 'normal_states.npz'
        info = dict(bytes=raw.stat().st_size, sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
        sys.stdout.write('R201G_STATES ' + json.dumps(dict(event='RAW_READY', name=control['name'],
            path=str(raw), file=info)) + '\n'); sys.stdout.flush()
        assert json.loads(sys.stdin.readline()) == dict(event='RAW_ARCHIVED', name=control['name'], file=info)
        assert raw.resolve().is_relative_to(out.resolve())
        assert raw.stat().st_size == info['bytes'] and hashlib.sha256(raw.read_bytes()).hexdigest() == info['sha256']
        raw.unlink()
        write_json(raw.parent / 'local_archive.json', dict(status='LOCAL_SIZE_SHA_ACK_REMOTE_CLEARED', file=info))
        sys.stdout.write('R201G_STATES ' + json.dumps(dict(event='RAW_CLEARED', name=control['name'])) + '\n'); sys.stdout.flush()
    write_json(out / 'controller_result.json', dict(status='COMPLETE', controls=controls,
        normal_conditions=1, state_cases=8, optimizer_updates=0, checkpoint_writes=0,
        original_raw_local_verified=2, missing_evaluation=False))
    print('R201G_NORMAL_TWO_CHECKPOINTS_FOUR_STATES_CLOSED', flush=True)


if __name__ == '__main__':
    main()
