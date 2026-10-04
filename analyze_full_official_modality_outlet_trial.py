"""Compare completed M4 against matched M3a, ordinary controls and full-data DeMo."""
import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

from analyze_full_official_baseline_pairs import METRICS, SETS, query_rows, summarize
from analyze_full_official_control_trial import VARIANTS, STATES, compare, comparison_groups


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def analyze(trial, prior, baseline):
    terminal, intake = read(trial / 'controller_result.json'), read(trial / 'intake_receipt.json')
    assert terminal['status'] == 'COMPLETE' and len(terminal['runs']) == 3
    assert terminal['training_heldout_identities'] == 0 and terminal['paired_identity_and_partial_sampling_exact']
    assert terminal['all_frozen_state_cpu_audits_passed'] and terminal['frozen_state_metric_cases'] == 882
    assert terminal['frozen_state_repeated_query_rows'] == 521262 and terminal['added_model_parameters'] == 0
    assert intake['status'] == 'THREE_FULL_OFFICIAL_M4_RUNS_AND882_GT_STATE_CASES_TEXT_COLLECTED'
    assert intake['state_metric_cases'] == 882 and intake['repeated_condition_query_rows'] == 521262
    assert {row['variant'] for row in terminal['runs']} == set(VARIANTS)
    baseline_report, prior_report = read(baseline / 'summary.json'), read(prior / 'analysis.json')
    assert baseline_report['status'] == 'SIX_FULL_OFFICIAL_BASELINES_COMPLETE_ALL294_CPU_AUDITED'
    assert prior_report['status'] == 'SIX_FULL_OFFICIAL_CONTROL_RUNS_AND1764_STATE_CASES_ANALYZED'
    conditions = sorted('q_' + q + '_g_' + g for q in SETS for g in SETS)
    banks, references = {}, {}
    for label, root, name in [
        *[(variant, baseline, 'MSVR310_' + variant + '_s42') for variant in ('demo', 'demo_shared')],
        *[('M3a/' + variant, prior, 'MSVR310_measurement_only_' + variant + '_s42') for variant in VARIANTS]]:
        audit = read(root / 'frozen49' / name / 'independent_cpu_audit.json')
        assert audit['status'] == 'PASS' and audit['cases'] == 49 and set(audit['conditions']) == set(conditions)
        banks[label] = {}
        for condition in conditions:
            rows = query_rows(root / 'frozen49' / name / (condition + '.csv'), 591)
            assert all(abs(summarize(rows)[metric] - audit['conditions'][condition][metric]) < 1e-8 for metric in METRICS)
            banks[label][condition] = rows
        references[label] = dict(selected_epoch=audit['selected_epoch'], normal=audit['normal'], conditions=audit['conditions'])
    models, all_orders = {}, []
    for variant in VARIANTS:
        name = 'MSVR310_measurement_only_' + variant + '_s42'
        run, frozen, diagnosis = (trial / phase / name for phase in ('training', 'frozen49', 'diagnosis'))
        trained = read(run / 'result.json')
        old = read(prior / 'training' / name / 'result.json')
        audit = read(frozen / 'independent_cpu_audit.json')
        states = read(diagnosis / 'independent_cpu_audit.json')
        frozen_result = read(frozen / 'result.json')
        arguments = trained['arguments']
        assert trained['status'] == old['status'] == frozen_result['status'] == 'COMPLETE' and trained['epochs'] == old['epochs'] == 50
        assert arguments['modality_alignment_weight'] == .1
        assert {key: value for key, value in arguments.items() if key not in ('modality_alignment_weight', 'output')} == {
            key: value for key, value in old['arguments'].items() if key != 'output'}
        assert (arguments['dataset'], arguments['variant'], arguments['gate_gradient_mode'], arguments['seed']) == ('MSVR310', variant, 'measurement_only', 42)
        for key in ('parameters', 'trainable_parameters', 'descriptor_dim', 'train_records', 'query_records', 'gallery_records'):
            assert trained[key] == old[key]
        assert trained['descriptor_dim'] == 5632 and trained['training_heldout_identities'] == 0
        assert (trained['train_records'], trained['query_records'], trained['gallery_records']) == (1032, 591, 1055)
        assert trained['training_coverage'] == dict(eligible=1032, visited=1032, unvisited=[])
        assert audit['status'] == 'PASS' and audit['cases'] == 49 and audit['perquery_count'] == 28959
        assert states['status'] == 'PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT' and states['cases'] == 294
        assert states['repeated_condition_query_rows'] == 173754 and states['calibration_rows'] == 28959
        assert states['model_arguments'] == audit['model_arguments'] == frozen_result['model_arguments'] == arguments
        assert trained['best']['epoch'] == audit['selected_epoch'] == frozen_result['selected_epoch']
        assert set(states['conditions']) == set(audit['conditions']) == set(conditions)
        phase_wall_seconds = {}
        for phase in ('contract', 'preflight', 'training', 'frozen49', 'audit', 'diagnosis', 'diagnosis_audit'):
            finished = read(trial / phase / (name + '_exit.json'))
            started = read(trial / phase / (name + '_launch.json'))
            assert finished['exit_code'] == 0
            phase_wall_seconds[phase] = finished['finished'] - started['started']
            assert phase_wall_seconds[phase] > 0
        with (run / 'batch_orders.jsonl').open(encoding='utf-8') as handle:
            orders = list(map(json.loads, handle))
        assert len(orders) == trained['steps'] and sum(row['optimizer_updated'] for row in orders) == trained['optimizer_steps']
        assert trained['steps'] - trained['optimizer_steps'] == trained['amp_skipped_steps']
        assert all(row['modality_alignment_raw'] > 0 and row['modality_alignment_weight'] == .1
            and row['modality_alignment_temperature'] == .07
            and not row['identity_alignment_reference_requires_grad'] for row in orders)
        current = [(row['epoch'], row['step'], row['names'], row['partial_set']) for row in orders]
        with (prior / 'training' / name / 'batch_orders.jsonl').open(encoding='utf-8') as handle:
            original = [(row['epoch'], row['step'], row['names'], row['partial_set']) for row in map(json.loads, handle)]
        assert current == original
        all_orders.append(current)
        banks['M4/' + variant] = {}
        effects = {key: {} for key in ('full11_minus00', 'full11_minus10', 'full11_minus01')}
        for condition in conditions:
            rows = {state: query_rows(diagnosis / condition / ('state_' + state + '.csv'), 591) for state in STATES}
            for state in STATES:
                measured = summarize(rows[state])
                assert all(abs(measured[metric] - states['conditions'][condition]['metrics'][state][metric]) < 1e-8 for metric in METRICS)
            prior_rows = query_rows(frozen / (condition + '.csv'), 591)
            assert rows['11'] == prior_rows
            assert all(abs(summarize(rows['11'])[metric] - audit['conditions'][condition][metric]) < 1e-8 for metric in METRICS)
            banks['M4/' + variant][condition] = rows['11']
            for key, before in (('full11_minus00', '00'), ('full11_minus10', '10'), ('full11_minus01', '01')):
                measured = compare(rows[before], rows['11'])
                saved = states['conditions'][condition][key]
                assert all(abs(measured['delta_pp'][metric] - saved['delta_pp'][metric]) < 1e-8 for metric in METRICS)
                for field in ('Rank1_harm_queries', 'Rank1_rescue_queries', 'AP_improved_queries', 'AP_worsened_queries', 'AP_unchanged_queries'):
                    assert measured[field] == saved[field]
                effects[key][condition] = measured
        assert all(abs(states['normal']['metrics']['11'][metric] - trained['full_metrics'][metric]) < 1e-8 for metric in METRICS)
        calibration = [states['conditions'][condition]['calibration'] for condition in conditions]
        models[variant] = dict(selected_epoch=trained['best']['epoch'], normal=states['normal'], conditions=states['conditions'],
            parameters=trained['parameters'], trainable_parameters=trained['trainable_parameters'], descriptor_dim=trained['descriptor_dim'],
            optimizer_steps=trained['optimizer_steps'], amp_skipped_steps=trained['amp_skipped_steps'], training_coverage=trained['training_coverage'],
            runtime=trained['runtime'], phase_wall_seconds=phase_wall_seconds,
            peak_training_memory_bytes=trained['peak_memory'], frozen_availability_runtime=frozen_result['runtime'],
            alignment_mean_raw=fmean(row['modality_alignment_raw'] for row in orders), heads=states['heads'],
            inference_effects={key: dict(conditions=values, groups=comparison_groups(values), normal=values['q_RNT_g_RNT'],
                all49_equal_condition_mean_delta_pp={metric: fmean(row['delta_pp'][metric] for row in values.values()) for metric in METRICS})
                for key, values in effects.items()},
            calibration_equal_condition_mean={key: [fmean(row[key][i] for row in calibration) for i in range(3)]
                for key in ('target_mean', 'target_std', 'MAE', 'zero_prediction_MAE')},
            calibration_MAE_better_than_zero_conditions=[sum(row['MAE'][i] < row['zero_prediction_MAE'][i] for row in calibration) for i in range(3)])
    assert all(order == all_orders[0] for order in all_orders)
    assert len({(value['parameters'], value['trainable_parameters'], value['descriptor_dim']) for value in models.values()}) == 1
    pairs = [(before, 'M4/axis_shared') for before in ('demo', 'demo_shared', 'M4/frequency_shared', 'M4/twins_shared')]
    pairs += [('M3a/' + variant, 'M4/' + variant) for variant in VARIANTS]
    comparisons = {}
    for before, after in pairs:
        values = {condition: compare(banks[before][condition], banks[after][condition]) for condition in conditions}
        comparisons[before + ' -> ' + after] = dict(before=before, after=after, normal=values['q_RNT_g_RNT'], conditions=values,
            groups=comparison_groups(values), both_metrics_plus2_conditions=sum(row['both_metrics_plus2'] for row in values.values()),
            all49_equal_condition_mean_delta_pp={metric: fmean(row['delta_pp'][metric] for row in values.values()) for metric in METRICS})
    return dict(status='THREE_FULL_OFFICIAL_M4_RUNS_AND882_STATE_CASES_ANALYZED', dataset='MSVR310', seed=42,
        full_split_counts=dict(train=1032, query=591, gallery=1055), training_heldout_identities=0,
        state_metric_cases=882, repeated_condition_query_rows=521262, contribution_rows=86877,
        models=models, references=references, comparisons=comparisons, comparison_condition_rows=343,
        original_DeMo_plus2_thresholds={metric: references['demo']['normal'][metric] + 2 for metric in ('mAP', 'Rank-1')},
        paired_identity_and_partial_sampling_exact=True, added_model_parameters=0, goal_complete=False,
        limits='Single-seed full-data MSVR M4 is one additional PM outlet training objective, not the goal multiseed milestone. '
            'Matched M3a parents retain identical architecture and all other saved arguments, active parameters, descriptors and PK/subset orders. '
            'Checkpoints use full official benchmark mAP; no untouched test or artificial holdout. Conditions/groups repeat591 queries. '
            'All signs, six metrics and frozen11−00/10/01 harm/rescue retained; a frozen00 is not a retrained DeMo or final candidate. '
            'This stdlib text analysis verifies saved CSV and CPU-audit outputs, not independent neural regeneration or a new raw-distance audit. '
            'Runtime from the training result is selected-checkpoint full-input extraction including data loading, not total training time; phase_wall_seconds comes from actual process launch/exit timestamps. '
            'Three-dataset/+2/+2, multiseed, fair-control and mechanism requirements remain unproven until each has actual evidence.')


def main():
    parser = argparse.ArgumentParser()
    for key in ('trial-root', 'prior-root', 'baseline-root'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    trial = Path(args.trial_root); output = trial / 'analysis.json'
    assert not output.exists()
    report = analyze(trial, Path(args.prior_root), Path(args.baseline_root))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fields = ['comparison', 'condition', *METRICS, 'Rank1_harm_queries', 'Rank1_rescue_queries',
        'AP_improved_queries', 'AP_worsened_queries', 'AP_unchanged_queries', 'both_metrics_plus2']
    with (trial / 'comparison_condition_deltas.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for label, comparison in report['comparisons'].items():
            for condition, values in comparison['conditions'].items():
                writer.writerow(dict(comparison=label, condition=condition, **values['delta_pp'], **{key: values[key] for key in fields[8:]}))
    print('FULL_OFFICIAL_M4_ALL882_STATE_CASES_ANALYZED', flush=True)


if __name__ == '__main__':
    main()
