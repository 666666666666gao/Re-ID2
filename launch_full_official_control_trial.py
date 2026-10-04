"""Full-data M3a/M3b and fair ordinary controls, after all six baseline audits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

from gpu_thermal_execute import execute


VARIANTS = ('axis_shared', 'frequency_shared', 'twins_shared')
SCHEDULE = {2: 'measurement_only', 3: 'independent_control'}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'output', 'baseline-campaign'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    prior = Path(args.baseline_campaign)
    completed = json.loads((prior / 'controller_result.json').read_text())
    assert completed['status'] == 'COMPLETE' and len(completed['runs']) == 6
    assert completed['paired_identity_sampling_exact'] and completed['training_heldout_identities'] == 0
    for row in completed['runs']:
        name = row['dataset'] + '_' + row['variant'] + '_s42'
        train = json.loads((prior / 'training' / name / 'result.json').read_text())
        audit = json.loads((prior / 'frozen49' / name / 'independent_cpu_audit.json').read_text())
        assert train['status'] == 'COMPLETE' and train['epochs'] == 50 and train['training_coverage']['unvisited'] == []
        assert audit['status'] == 'PASS' and audit['cases'] == 49
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    for phase in ('contract', 'preflight', 'training', 'frozen49', 'audit', 'diagnosis', 'diagnosis_audit'):
        (root / phase).mkdir()
    contract = execute([sys.executable, '-u', 'verify_full_official_control_axis.py',
        '--data-root', args.data_root, '--pretrained', args.pretrained, '--output', str(root / 'contract' / 'gradient')],
        root / 'contract', 'gradient', 2)
    checked = json.loads((root / 'contract' / 'gradient' / 'result.json').read_text())
    assert checked['status'] == 'PASS_FULL_OFFICIAL_CONTROL_GRADIENT_CONTRACT'

    def name(mode, variant):
        return 'MSVR310_' + mode + '_' + variant + '_s42'

    def command(mode, variant):
        return [sys.executable, '-u', 'run_full_official_control_experiment.py', '--dataset', 'MSVR310',
            '--variant', variant, '--gate-gradient-mode', mode, '--seed', '42',
            '--data-root', args.data_root, '--pretrained', args.pretrained]

    def smokes(gpu):
        mode, rows = SCHEDULE[gpu], []
        for variant in VARIANTS:
            run_name = name(mode, variant)
            row = execute(command(mode, variant) + ['--mode', 'smoke', '--output', str(root / 'preflight' / run_name)],
                root / 'preflight', run_name, gpu)
            smoke = json.loads((root / 'preflight' / run_name / 'smoke.json').read_text())
            assert smoke['status'] == 'SMOKE_PASS' and smoke['steps'] == 3 and smoke['strict_reload_equal']
            assert smoke['training_heldout_identities'] == 0 and all(smoke['gradients'].values())
            rows.append(row)
        return rows

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(smokes, gpu) for gpu in SCHEDULE]
        smokes_checked = [row for future in futures for row in future.result()]
    assert len(smokes_checked) == 6
    for mode in SCHEDULE.values():
        counts = []
        for variant in VARIANTS:
            smoke = json.loads((root / 'preflight' / name(mode, variant) / 'smoke.json').read_text())
            counts.append((smoke['parameters'], smoke['trainable_parameters'], smoke['descriptor_dim']))
        assert len(set(counts)) == 1 and counts[0][-1] == 5632
    save(root / 'preflight_result.json', dict(status='PASS', contract=contract, smokes=smokes_checked))

    def jobs(gpu):
        mode, rows = SCHEDULE[gpu], []
        for variant in VARIANTS:
            run_name = name(mode, variant)
            run, frozen = root / 'training' / run_name, root / 'frozen49' / run_name
            trained = execute(command(mode, variant) + ['--mode', 'train', '--output', str(run)], root / 'training', run_name, gpu)
            result = json.loads((run / 'result.json').read_text())
            assert result['status'] == 'COMPLETE' and result['epochs'] == 50 and result['training_heldout_identities'] == 0
            evaluated = execute([sys.executable, '-u', 'evaluate_full_official_control49.py',
                '--run-dir', str(run), '--output', str(frozen)], root / 'frozen49', run_name, gpu)
            audited = execute([sys.executable, '-u', 'audit_full_official49.py',
                '--run-dir', str(run), '--evaluation', str(frozen)], root / 'audit', run_name, gpu)
            audit = json.loads((frozen / 'independent_cpu_audit.json').read_text())
            assert audit['status'] == 'PASS' and audit['cases'] == 49 and audit['training_coverage']['unvisited'] == []
            diagnosis = root / 'diagnosis' / run_name
            diagnosed = execute([sys.executable, '-u', 'diagnose_full_official_control_states.py',
                '--run-dir', str(run), '--previous-frozen', str(frozen), '--output', str(diagnosis)],
                root / 'diagnosis', run_name, gpu)
            diagnosis_audited = execute([sys.executable, '-u', 'audit_full_official_control_states.py',
                '--run-dir', str(run), '--previous-frozen', str(frozen), '--diagnosis', str(diagnosis)],
                root / 'diagnosis_audit', run_name, gpu)
            states = json.loads((diagnosis / 'independent_cpu_audit.json').read_text())
            assert states['status'] == 'PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT' and states['cases'] == 294
            rows.append(dict(dataset='MSVR310', mode=mode, variant=variant, gpu=gpu,
                train=trained, evaluation=evaluated, audit=audited, diagnosis=diagnosed, diagnosis_audit=diagnosis_audited))
            save(root / ('gpu_' + str(gpu) + '_completed.json'), dict(status='IN_PROGRESS', runs=rows))
        return rows

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(jobs, gpu) for gpu in SCHEDULE]
        rows = [row for future in futures for row in future.result()]
    assert len(rows) == 6
    paired = []
    for mode in SCHEDULE.values():
        for variant in VARIANTS:
            with (root / 'training' / name(mode, variant) / 'batch_orders.jsonl').open() as handle:
                paired.append([(r['epoch'], r['step'], r['names'], r['partial_set']) for r in map(json.loads, handle)])
    assert all(order == paired[0] for order in paired[1:])
    save(root / 'controller_result.json', dict(status='COMPLETE', runs=rows, training_heldout_identities=0,
        frozen_state_metric_cases=1764, frozen_state_repeated_query_rows=1764 * 591,
        contribution_rows=6 * 49 * 591, all_frozen_state_cpu_audits_passed=True,
        paired_identity_and_partial_sampling_exact=True, temperature_power_control=False,
        limits='Single-seed full-data MSVR mechanism trial only; not a final three-dataset or multiseed method.'))


if __name__ == '__main__':
    main()
