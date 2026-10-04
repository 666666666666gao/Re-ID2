"""Two existing best weights, real smoke then frozen full588-readout audits per GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

from gpu_thermal_execute import execute


SCHEDULE = {2: 'measurement_only', 3: 'independent_control'}
STAGES = ('deployed', 'base_common', 'M_pre', 'M_post', 'F_pre', 'F_post', 'M_aux', 'F_aux')


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, root = map(Path, (args.source_root, args.output))
    terminal = json.loads((source / 'controller_result.json').read_text())
    assert terminal['status'] == 'COMPLETE' and len(terminal['runs']) == 6
    assert terminal['all_frozen_state_cpu_audits_passed'] and terminal['frozen_state_metric_cases'] == 1764
    assert terminal['paired_identity_and_partial_sampling_exact']
    assert os.statvfs(source).f_bavail * os.statvfs(source).f_frsize > 3500 * 1024 * 1024
    root.mkdir(exist_ok=False)
    for phase in ('preflight', 'frozen', 'audit'):
        (root / phase).mkdir()

    def paths(mode):
        name = 'MSVR310_' + mode + '_axis_shared_s42'
        return source / 'training' / name, source / 'frozen49' / name

    def command(mode, output):
        run, previous = paths(mode)
        return [sys.executable, '-u', 'diagnose_full_official_control_readout.py',
            '--run-dir', str(run), '--previous-frozen', str(previous), '--output', str(output)]

    def smoke(gpu):
        mode = SCHEDULE[gpu]
        checked = execute(command(mode, root / 'preflight' / mode) + ['--smoke'], root / 'preflight', mode, gpu)
        result = json.loads((root / 'preflight' / mode / 'smoke.json').read_text())
        assert result['status'] == 'PASS' and result['records'] == 64 and result['availability_sets'] == 7
        assert tuple(result['stages']) == STAGES and result['normal_feature_max_error'] == 0
        assert result['original_fuse_reconstructed_exact'] and result['state_tensor_versions_unchanged']
        assert result['optimizer_updates'] == result['new_weights'] == 0
        return dict(mode=mode, gpu=gpu, execution=checked, result=result)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(smoke, gpu) for gpu in SCHEDULE]
        checked = [future.result() for future in futures]
    save(root / 'preflight_result.json', dict(status='PASS', runs=checked, new_training=0))

    def frozen(gpu):
        mode = SCHEDULE[gpu]
        run, previous = paths(mode)
        output = root / 'frozen' / mode
        evaluated = execute(command(mode, output), root / 'frozen', mode, gpu)
        audited = execute([sys.executable, '-u', 'audit_full_official_control_readout.py',
            '--run-dir', str(run), '--previous-frozen', str(previous), '--diagnosis', str(output)], root / 'audit', mode, gpu)
        result = json.loads((output / 'independent_cpu_audit.json').read_text())
        assert result['status'] == 'PASS_FULL_OFFICIAL_READOUT_INSTALLED_GT' and result['cases'] == 588
        assert result['repeated_condition_query_rows'] == 347508 and result['optimizer_updates'] == 0
        return dict(mode=mode, gpu=gpu, evaluation=evaluated, audit=audited)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(frozen, gpu) for gpu in SCHEDULE]
        rows = [future.result() for future in futures]
    save(root / 'controller_result.json', dict(status='COMPLETE', runs=rows, metric_cases=1176,
        repeated_condition_query_rows=695016, train_records=1032, query_records=591, gallery_records=1055,
        training_heldout_identities=0, optimizer_updates=0, new_weights=0, temperature_power_control=False,
        limits='Frozen two-axis-mode readout utility diagnostics only; no successful method or three-dataset result.'))


if __name__ == '__main__':
    main()
