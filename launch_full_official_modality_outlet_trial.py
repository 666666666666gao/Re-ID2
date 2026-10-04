"""Three matched M4 fresh50 trials, serialized per2026 GPU2/3, then full49/state audits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

from gpu_thermal_execute import execute


SCHEDULE = {2: ('axis_shared',), 3: ('frequency_shared', 'twins_shared')}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'output', 'previous-campaign'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    prior, root = Path(args.previous_campaign), Path(args.output)
    previous = json.loads((prior / 'controller_result.json').read_text())
    assert previous['status'] == 'COMPLETE' and previous['all_frozen_state_cpu_audits_passed']
    assert previous['frozen_state_metric_cases'] == 1764 and previous['paired_identity_and_partial_sampling_exact']
    root.mkdir(exist_ok=False)
    for phase in ('contract', 'preflight', 'training', 'frozen49', 'audit', 'diagnosis', 'diagnosis_audit'):
        (root / phase).mkdir()

    def name(variant):
        return 'MSVR310_measurement_only_' + variant + '_s42'

    def command(variant):
        return [sys.executable, '-u', 'run_full_official_modality_outlet_experiment.py', '--dataset', 'MSVR310',
            '--variant', variant, '--seed', '42', '--data-root', args.data_root, '--pretrained', args.pretrained]

    def preflight(gpu):
        rows = []
        for variant in SCHEDULE[gpu]:
            run_name = name(variant)
            contract = execute([sys.executable, '-u', 'verify_full_official_modality_outlet_axis.py',
                '--data-root', args.data_root, '--pretrained', args.pretrained, '--variant', variant,
                '--output', str(root / 'contract' / run_name)], root / 'contract', run_name, gpu)
            checked = json.loads((root / 'contract' / run_name / 'result.json').read_text())
            assert checked['status'] == 'PASS_FULL_OFFICIAL_PM_OUTLET_CONTRACT' and checked['added_parameters'] == 0
            smoke = execute(command(variant) + ['--mode', 'smoke', '--output', str(root / 'preflight' / run_name)],
                root / 'preflight', run_name, gpu)
            result = json.loads((root / 'preflight' / run_name / 'smoke.json').read_text())
            assert result['status'] == 'SMOKE_PASS' and result['steps'] == 3 and result['strict_reload_equal']
            assert result['training_heldout_identities'] == 0 and all(result['gradients'].values())
            assert all(row['modality_alignment_raw'] > 0 and row['modality_alignment_weight'] == .1 for row in result['details'])
            old = json.loads((prior / 'training' / run_name / 'result.json').read_text())
            keys = ('parameters', 'trainable_parameters', 'descriptor_dim', 'train_records', 'query_records', 'gallery_records')
            assert all(result[key] == old[key] for key in keys)
            rows.append(dict(variant=variant, gpu=gpu, contract=contract, smoke=smoke,
                counts={key: result[key] for key in keys}))
        return rows

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(preflight, gpu) for gpu in SCHEDULE]
        checked = [row for future in futures for row in future.result()]
    assert len(checked) == 3 and len({tuple(row['counts'].values()) for row in checked}) == 1
    save(root / 'preflight_result.json', dict(status='PASS', runs=checked, actual_optimizer_updates=9,
        full_data_counts=dict(train=1032, query=591, gallery=1055), training_heldout_identities=0))

    def jobs(gpu):
        rows = []
        for variant in SCHEDULE[gpu]:
            run_name = name(variant)
            run, frozen = root / 'training' / run_name, root / 'frozen49' / run_name
            trained = execute(command(variant) + ['--mode', 'train', '--output', str(run)], root / 'training', run_name, gpu)
            result = json.loads((run / 'result.json').read_text())
            assert result['status'] == 'COMPLETE' and result['epochs'] == 50 and result['training_heldout_identities'] == 0
            assert result['training_coverage'] == dict(eligible=1032, visited=1032, unvisited=[])
            evaluated = execute([sys.executable, '-u', 'evaluate_full_official_modality_outlet49.py',
                '--run-dir', str(run), '--output', str(frozen)], root / 'frozen49', run_name, gpu)
            audited = execute([sys.executable, '-u', 'audit_full_official49.py',
                '--run-dir', str(run), '--evaluation', str(frozen)], root / 'audit', run_name, gpu)
            assert json.loads((frozen / 'independent_cpu_audit.json').read_text())['cases'] == 49
            diagnosis = root / 'diagnosis' / run_name
            diagnosed = execute([sys.executable, '-u', 'diagnose_full_official_modality_outlet_states.py',
                '--run-dir', str(run), '--previous-frozen', str(frozen), '--output', str(diagnosis)],
                root / 'diagnosis', run_name, gpu)
            diagnosis_audited = execute([sys.executable, '-u', 'audit_full_official_control_states.py',
                '--run-dir', str(run), '--previous-frozen', str(frozen), '--diagnosis', str(diagnosis)],
                root / 'diagnosis_audit', run_name, gpu)
            audit = json.loads((diagnosis / 'independent_cpu_audit.json').read_text())
            assert audit['status'] == 'PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT' and audit['cases'] == 294
            rows.append(dict(mode='measurement_only', variant=variant, dataset='MSVR310', gpu=gpu,
                train=trained, evaluation=evaluated, audit=audited, diagnosis=diagnosed, diagnosis_audit=diagnosis_audited))
            save(root / ('gpu_' + str(gpu) + '_completed.json'), dict(status='IN_PROGRESS', runs=rows))
        return rows

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(jobs, gpu) for gpu in SCHEDULE]
        rows = [row for future in futures for row in future.result()]
    orders = []
    for row in rows:
        for campaign in (root, prior):
            with (campaign / 'training' / name(row['variant']) / 'batch_orders.jsonl').open() as handle:
                orders.append([(r['epoch'], r['step'], r['names'], r['partial_set']) for r in map(json.loads, handle)])
    assert len(rows) == 3 and all(order == orders[0] for order in orders)
    save(root / 'controller_result.json', dict(status='COMPLETE', runs=rows,
        frozen_state_metric_cases=882, frozen_state_repeated_query_rows=882 * 591,
        all_frozen_state_cpu_audits_passed=True, paired_identity_and_partial_sampling_exact=True,
        training_heldout_identities=0, temperature_power_control=False, added_model_parameters=0,
        limits='Three single-seed MSVR full-data M4 versus same-variant priorM3a controls; benchmark-selected best. '
            'No successful unified three-dataset/multiseed method is implied.'))


if __name__ == '__main__':
    main()
