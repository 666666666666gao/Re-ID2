"""Four matched public-identity trials, two serial lanes, full frozen49 and enhanced six states."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

from gpu_thermal_execute import execute

SCHEDULE = {2: ('demo_shared', 'axis_shared'), 3: ('frequency_shared', 'twins_shared')}
SETS = ('R', 'N', 'T', 'RN', 'RT', 'NT', 'RNT')


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'pretrained', 'output', 'previous-campaign', 'baseline-campaign', 'closed-graph-campaign'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    prior, baseline, root = Path(args.previous_campaign), Path(args.baseline_campaign), Path(args.output)
    previous = json.loads((prior / 'controller_result.json').read_text())
    assert previous['status'] == 'COMPLETE' and previous['frozen_state_metric_cases'] == 882
    assert previous['all_frozen_state_cpu_audits_passed'] and previous['paired_identity_and_partial_sampling_exact']
    graph = json.loads((Path(args.closed_graph_campaign) / 'controller_result.json').read_text())
    assert graph['status'] == 'COMPLETE' and graph['frozen_metric_cases'] == 196
    assert graph['enhanced_state_metric_cases'] == 882 and graph['all_frozen_and_state_cpu_audits_passed']
    assert graph['paired_identity_and_partial_sampling_exact']
    root.mkdir(exist_ok=False)
    phases = ('contract', 'preflight', 'training', 'frozen49', 'audit', 'diagnosis', 'diagnosis_audit')
    for phase in phases:
        (root / phase).mkdir()

    def name(variant):
        return 'MSVR310_public_identity_' + variant + '_s42'

    def old_run(variant):
        if variant == 'demo_shared':
            return baseline / 'training' / 'MSVR310_demo_shared_s42'
        return prior / 'training' / ('MSVR310_measurement_only_' + variant + '_s42')

    def command(variant):
        return [sys.executable, '-u', 'run_full_official_public_outlet_identity_experiment.py',
            '--dataset', 'MSVR310', '--variant', variant, '--seed', '42',
            '--data-root', args.data_root, '--pretrained', args.pretrained]

    def preflight(gpu):
        rows = []
        for variant in SCHEDULE[gpu]:
            run_name = name(variant)
            contract = execute([sys.executable, '-u', 'verify_full_official_public_outlet_identity.py',
                '--data-root', args.data_root, '--pretrained', args.pretrained, '--variant', variant,
                '--output', str(root / 'contract' / run_name)], root / 'contract', run_name, gpu)
            checked = json.loads((root / 'contract' / run_name / 'result.json').read_text())
            assert checked['status'] == 'PASS_FULL_OFFICIAL_PUBLIC_IDENTITY_PARENT_AMP_STEP'
            assert checked['parent_step_loss_gradient_equivalence'] and checked['optimizer_update_consistency']
            assert checked['added_parameters'] == 0
            smoke = execute(command(variant) + ['--mode', 'smoke', '--output', str(root / 'preflight' / run_name)],
                root / 'preflight', run_name, gpu)
            result = json.loads((root / 'preflight' / run_name / 'smoke.json').read_text())
            assert result['status'] == 'SMOKE_PASS' and result['steps'] == 3 and result['strict_reload_equal']
            assert result['training_heldout_identities'] == 0 and all(result['gradients'].values())
            assert all(r['public_identity_weight'] == .1 and r['public_identity_extra_parameters'] == 0
                and r['public_identity_extra_backbone_passes'] == 0 and r['public_identity_extra_shared_neck_calls'] == 2
                and r['public_full_ce'] > 0 and r['public_partial_ce'] > 0 for r in result['details'])
            old = json.loads((old_run(variant) / 'result.json').read_text())
            keys = ('parameters', 'trainable_parameters', 'descriptor_dim', 'train_records', 'query_records', 'gallery_records')
            assert all(result[key] == old[key] for key in keys)
            rows.append(dict(variant=variant, gpu=gpu, contract=contract, smoke=smoke,
                counts={key: result[key] for key in keys}))
        return rows

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(preflight, gpu) for gpu in SCHEDULE]
        checked = [row for future in futures for row in future.result()]
    enhanced = [row for row in checked if row['variant'] != 'demo_shared']
    assert len(checked) == 4 and len({tuple(row['counts'].values()) for row in enhanced}) == 1
    save(root / 'preflight_result.json', dict(status='PASS', runs=checked,
        smoke_actual_optimizer_updates=12, contract_actual_optimizer_updates=16,
        full_data_counts=dict(train=1032, query=591, gallery=1055), training_heldout_identities=0))

    def jobs(gpu):
        rows = []
        for variant in SCHEDULE[gpu]:
            run_name = name(variant)
            run, frozen = root / 'training' / run_name, root / 'frozen49' / run_name
            trained = execute(command(variant) + ['--mode', 'train', '--output', str(run)],
                root / 'training', run_name, gpu)
            result = json.loads((run / 'result.json').read_text())
            assert result['status'] == 'COMPLETE' and result['epochs'] == 50 and result['training_heldout_identities'] == 0
            assert result['training_coverage'] == dict(eligible=1032, visited=1032, unvisited=[])
            evaluated = execute([sys.executable, '-u', 'evaluate_full_official_public_outlet_identity49.py',
                '--run-dir', str(run), '--output', str(frozen)], root / 'frozen49', run_name, gpu)
            audited = execute([sys.executable, '-u', 'audit_full_official49.py',
                '--run-dir', str(run), '--evaluation', str(frozen)], root / 'audit', run_name, gpu)
            audit = json.loads((frozen / 'independent_cpu_audit.json').read_text())
            assert audit['cases'] == 49
            row = dict(variant=variant, dataset='MSVR310', gpu=gpu, train=trained, evaluation=evaluated, audit=audited)
            if variant != 'demo_shared':
                diagnosis = root / 'diagnosis' / run_name
                row['diagnosis'] = execute([sys.executable, '-u', 'diagnose_full_official_public_outlet_identity_states.py',
                    '--run-dir', str(run), '--previous-frozen', str(frozen), '--output', str(diagnosis)],
                    root / 'diagnosis', run_name, gpu)
                row['diagnosis_audit'] = execute([sys.executable, '-u', 'audit_full_official_control_states.py',
                    '--run-dir', str(run), '--previous-frozen', str(frozen), '--diagnosis', str(diagnosis)],
                    root / 'diagnosis_audit', run_name, gpu)
                audited_states = json.loads((diagnosis / 'independent_cpu_audit.json').read_text())
                assert audited_states['status'] == 'PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT'
                assert audited_states['cases'] == 294
            rows.append(row)
            save(root / ('gpu_' + str(gpu) + '_completed.json'), dict(status='IN_PROGRESS', runs=rows))
        return rows

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(jobs, gpu) for gpu in SCHEDULE]
        rows = [row for future in futures for row in future.result()]
    orders, coverage = [], {}
    for row in rows:
        variant = row['variant']
        records = [json.loads(line) for line in (root / 'training' / name(variant) / 'batch_orders.jsonl').read_text().splitlines()]
        order = [(r['epoch'], r['step'], r['names'], r['partial_set']) for r in records]
        parent_order = [(r['epoch'], r['step'], r['names'], r['partial_set'])
                        for r in map(json.loads, (old_run(variant) / 'batch_orders.jsonl').read_text().splitlines())]
        assert order == parent_order
        orders.append(order)
        counts = {a: 0 for a in SETS}
        for r in records:
            assert r['public_identity_weight'] == .1 and r['public_identity_extra_parameters'] == 0
            assert r['public_identity_extra_backbone_passes'] == 0 and r['public_identity_extra_shared_neck_calls'] == 2
            assert r['public_full_ce'] > 0 and r['public_partial_ce'] > 0 and not r['reference_requires_grad']
            counts['RNT'] += 1
            counts[r['partial_set']] += 1
        assert len(counts) == 7 and all(value > 0 for value in counts.values())
        coverage[variant] = counts
    assert len(rows) == 4 and all(order == orders[0] for order in orders)
    save(root / 'controller_result.json', dict(status='COMPLETE', runs=rows,
        frozen_metric_cases=196, enhanced_state_metric_cases=882,
        enhanced_state_repeated_query_rows=882 * 591, all_frozen_and_state_cpu_audits_passed=True,
        paired_identity_and_partial_sampling_exact=True, public_identity_input_training_coverage=coverage,
        training_heldout_identities=0, temperature_power_control=False, added_model_parameters=0,
        limits='Single-seed full MSVR public-identity CE development trial relative to M4 parents, not a M5-to-M6 single-factor ablation. Benchmark-selected checkpoint; no unified three-dataset/multiseed superiority implied.'))


if __name__ == '__main__':
    main()
