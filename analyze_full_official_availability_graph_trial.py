"""All signed graph-trial comparisons and frozen expert utility from GT-audited CSV."""
import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

from analyze_full_official_baseline_pairs import METRICS, SETS, query_rows, summarize
from analyze_full_official_control_trial import VARIANTS, STATES, compare, comparison_groups

GRAPH_VARIANTS = ('demo_shared', *VARIANTS)
REFERENCE_CYCLE = ('R', 'N', 'T', 'RN', 'RT', 'NT', 'RNT')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_bank(frozen, audit, conditions):
    assert audit['status'] == 'PASS' and audit['cases'] == 49 and audit['perquery_count'] == 28959
    assert set(audit['conditions']) == set(conditions)
    bank = {}
    for condition in conditions:
        rows = query_rows(frozen / (condition + '.csv'), 591)
        assert all(abs(summarize(rows)[m] - audit['conditions'][condition][m]) < 1e-8 for m in METRICS)
        bank[condition] = rows
    return bank


def analyze(trial, prior, baseline):
    done, intake = read(trial / 'controller_result.json'), read(trial / 'intake_receipt.json')
    assert done['status'] == 'COMPLETE' and len(done['runs']) == 4
    assert done['all_frozen_and_state_cpu_audits_passed'] and done['paired_identity_and_partial_sampling_exact']
    assert done['frozen_metric_cases'] == 196 and done['enhanced_state_metric_cases'] == 882
    assert done['enhanced_state_repeated_query_rows'] == 521262 and done['training_heldout_identities'] == 0
    assert done['added_model_parameters'] == 0
    assert intake['status'] == 'FOUR_FULL_OFFICIAL_GRAPH_RUNS196_FROZEN_AND882_STATE_GT_CASES_TEXT_COLLECTED'
    assert intake['frozen_metric_cases'] == 196 and intake['enhanced_state_metric_cases'] == 882
    assert {r['variant'] for r in done['runs']} == set(GRAPH_VARIANTS)
    assert read(prior / 'analysis.json')['status'] == 'THREE_FULL_OFFICIAL_M4_RUNS_AND882_STATE_CASES_ANALYZED'
    assert read(baseline / 'summary.json')['status'] == 'SIX_FULL_OFFICIAL_BASELINES_COMPLETE_ALL294_CPU_AUDITED'
    conditions = sorted('q_' + q + '_g_' + g for q in SETS for g in SETS)
    banks, references, parents = {}, {}, {}
    reference_runs = [(v, baseline, 'MSVR310_' + v + '_s42') for v in ('demo', 'demo_shared')]
    reference_runs += [('M4/' + v, prior, 'MSVR310_measurement_only_' + v + '_s42') for v in VARIANTS]
    for label, root, name in reference_runs:
        trained = read(root / 'training' / name / 'result.json')
        audit = read(root / 'frozen49' / name / 'independent_cpu_audit.json')
        assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50
        assert trained['training_heldout_identities'] == 0
        assert trained['training_coverage'] == dict(eligible=1032, visited=1032, unvisited=[])
        assert all(abs(trained['full_metrics'][m] - audit['normal'][m]) < 1e-8 for m in METRICS)
        banks[label] = load_bank(root / 'frozen49' / name, audit, conditions)
        references[label] = dict(selected_epoch=audit['selected_epoch'], normal=audit['normal'], conditions=audit['conditions'])
        parents[label] = (root / 'training' / name, trained)
    models, all_orders = {}, []
    for variant in GRAPH_VARIANTS:
        name = 'MSVR310_graph_' + variant + '_s42'
        run, frozen = (trial / phase / name for phase in ('training', 'frozen49'))
        trained, result, audit = map(read, (run / 'result.json', frozen / 'result.json', frozen / 'independent_cpu_audit.json'))
        args = trained['arguments']
        old_label = 'demo_shared' if variant == 'demo_shared' else 'M4/' + variant
        old_run, old = parents[old_label]
        assert trained['status'] == result['status'] == 'COMPLETE' and trained['epochs'] == 50
        assert args['dataset'] == 'MSVR310' and args['variant'] == variant and args['seed'] == 42
        assert args['graph_weight'] == .1 and args['graph_temperature'] == .07
        assert args['graph_reference_cycle'] == list(REFERENCE_CYCLE)
        ignored = {'output', 'graph_weight', 'graph_temperature', 'graph_reference_cycle'}
        assert {k: v for k, v in args.items() if k not in ignored} == {k: v for k, v in old['arguments'].items() if k not in ignored}
        for key in ('parameters', 'trainable_parameters', 'descriptor_dim', 'train_records', 'query_records', 'gallery_records'):
            assert trained[key] == old[key]
        assert (trained['train_records'], trained['query_records'], trained['gallery_records']) == (1032, 591, 1055)
        assert trained['descriptor_dim'] == 5632 and trained['training_heldout_identities'] == 0
        assert trained['training_coverage'] == dict(eligible=1032, visited=1032, unvisited=[])
        assert result['model_arguments'] == audit['model_arguments'] == args
        assert audit['selected_epoch'] == result['selected_epoch'] == trained['best']['epoch']
        assert all(abs(trained['full_metrics'][m] - audit['normal'][m]) < 1e-8 for m in METRICS)
        banks['M5/' + variant] = load_bank(frozen, audit, conditions)
        orders = [json.loads(line) for line in (run / 'batch_orders.jsonl').read_text().splitlines()]
        assert len(orders) == trained['steps']
        assert sum(r['optimizer_updated'] for r in orders) == trained['optimizer_steps']
        assert trained['steps'] - trained['optimizer_steps'] == trained['amp_skipped_steps']
        current_order = [(r['epoch'], r['step'], r['names'], r['partial_set']) for r in orders]
        parent_order = [(r['epoch'], r['step'], r['names'], r['partial_set'])
            for r in map(json.loads, (old_run / 'batch_orders.jsonl').read_text().splitlines())]
        assert current_order == parent_order
        all_orders.append(current_order)
        coverage = {q + '_' + g: 0 for q in REFERENCE_CYCLE for g in REFERENCE_CYCLE}
        for index, row in enumerate(orders):
            assert row['graph_step'] == index and row['graph_reference_set'] == REFERENCE_CYCLE[index % 7]
            assert row['graph_query_sets'] == ['RNT', row['partial_set']]
            assert row['graph_weight'] == .1 and row['graph_temperature'] == .07
            assert not row['graph_reference_requires_grad'] and row['graph_extra_backbone_passes'] == 0
            for key in ('graph_full_audit', 'graph_partial_audit'):
                assert row[key]['identity_alignment_valid_anchors'] > 0
                assert row[key]['identity_alignment_positive_observations_distinct']
                assert not row[key]['identity_alignment_reference_requires_grad']
            for q in row['graph_query_sets']:
                coverage[q + '_' + row['graph_reference_set']] += 1
        assert all(v > 0 for v in coverage.values()) and coverage == done['graph_pair_training_coverage'][variant]
        phases = ['contract', 'preflight', 'training', 'frozen49', 'audit']
        model = dict(selected_epoch=audit['selected_epoch'], normal=audit['normal'], conditions=audit['conditions'],
            parameters=trained['parameters'], trainable_parameters=trained['trainable_parameters'], descriptor_dim=trained['descriptor_dim'],
            optimizer_steps=trained['optimizer_steps'], amp_skipped_steps=trained['amp_skipped_steps'],
            training_coverage=trained['training_coverage'], graph_pair_training_coverage=coverage,
            graph_full_mean_raw=fmean(r['graph_full_raw'] for r in orders),
            graph_partial_mean_raw=fmean(r['graph_partial_raw'] for r in orders),
            peak_training_memory_bytes=trained['peak_memory'], runtime=trained['runtime'], frozen_availability_runtime=result['runtime'])
        if variant != 'demo_shared':
            diagnosis = trial / 'diagnosis' / name
            states = read(diagnosis / 'independent_cpu_audit.json')
            assert states['status'] == 'PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT' and states['cases'] == 294
            assert states['repeated_condition_query_rows'] == 173754 and states['calibration_rows'] == 28959
            assert states['model_arguments'] == args and set(states['conditions']) == set(conditions)
            effects = {k: {} for k in ('full11_minus00', 'full11_minus10', 'full11_minus01')}
            for condition in conditions:
                rows = {state: query_rows(diagnosis / condition / ('state_' + state + '.csv'), 591) for state in STATES}
                for state in STATES:
                    assert all(abs(summarize(rows[state])[m] - states['conditions'][condition]['metrics'][state][m]) < 1e-8 for m in METRICS)
                assert rows['11'] == banks['M5/' + variant][condition]
                for key, before in (('full11_minus00', '00'), ('full11_minus10', '10'), ('full11_minus01', '01')):
                    measured, saved = compare(rows[before], rows['11']), states['conditions'][condition][key]
                    assert all(abs(measured['delta_pp'][m] - saved['delta_pp'][m]) < 1e-8 for m in METRICS)
                    for field in ('Rank1_harm_queries', 'Rank1_rescue_queries', 'AP_improved_queries', 'AP_worsened_queries', 'AP_unchanged_queries'):
                        assert measured[field] == saved[field]
                    effects[key][condition] = measured
            calibration = [states['conditions'][c]['calibration'] for c in conditions]
            model.update(diagnosis_normal=states['normal'], diagnosis_conditions=states['conditions'], heads=states['heads'],
                inference_effects={key: dict(conditions=values, normal=values['q_RNT_g_RNT'], groups=comparison_groups(values),
                    all49_equal_condition_mean_delta_pp={m: fmean(r['delta_pp'][m] for r in values.values()) for m in METRICS})
                    for key, values in effects.items()},
                calibration_equal_condition_mean={k: [fmean(r[k][i] for r in calibration) for i in range(3)]
                    for k in ('target_mean', 'target_std', 'MAE', 'zero_prediction_MAE')},
                calibration_MAE_better_than_zero_conditions=[sum(r['MAE'][i] < r['zero_prediction_MAE'][i] for r in calibration) for i in range(3)])
            phases += ['diagnosis', 'diagnosis_audit']
        model['phase_wall_seconds'] = {}
        for phase in phases:
            launched = read(trial / phase / (name + '_launch.json'))
            finished = read(trial / phase / (name + '_exit.json'))
            assert finished['exit_code'] == 0
            seconds = finished['finished'] - launched['started']; assert seconds > 0
            model['phase_wall_seconds'][phase] = seconds
        models[variant] = model
    assert all(order == all_orders[0] for order in all_orders)
    assert len({(models[v]['parameters'], models[v]['trainable_parameters'], models[v]['descriptor_dim']) for v in VARIANTS}) == 1
    pairs = [(b, 'M5/axis_shared') for b in ('demo', 'M5/demo_shared', 'M5/frequency_shared', 'M5/twins_shared')]
    pairs += [('demo_shared', 'M5/demo_shared')] + [('M4/' + v, 'M5/' + v) for v in VARIANTS]
    comparisons = {}
    for before, after in pairs:
        values = {c: compare(banks[before][c], banks[after][c]) for c in conditions}
        comparisons[before + ' -> ' + after] = dict(before=before, after=after, normal=values['q_RNT_g_RNT'],
            conditions=values, groups=comparison_groups(values),
            both_metrics_plus2_conditions=sum(r['both_metrics_plus2'] for r in values.values()),
            all49_equal_condition_mean_delta_pp={m: fmean(r['delta_pp'][m] for r in values.values()) for m in METRICS})
    return dict(status='FOUR_FULL_OFFICIAL_GRAPH_RUNS196_FROZEN_AND882_STATE_CASES_ANALYZED', dataset='MSVR310', seed=42,
        full_split_counts=dict(train=1032, query=591, gallery=1055), training_heldout_identities=0,
        frozen_metric_cases=196, enhanced_state_metric_cases=882, repeated_condition_query_rows=521262, contribution_rows=86877,
        models=models, references=references, comparisons=comparisons, comparison_condition_rows=392,
        original_DeMo_plus2_thresholds={m: references['demo']['normal'][m] + 2 for m in ('mAP', 'Rank-1')},
        paired_identity_and_partial_sampling_exact=True, added_model_parameters=0, goal_complete=False,
        limits='Single-seed full MSVR trial with benchmark-selected checkpoints; no untouched-test or three-dataset/multiseed success. '
            '49 conditions repeat the same591 queries; equal-condition means and dependent query sums are descriptive, not official unified mAP or independent observations. '
            'Frozen00 is this new-method-trained base, not an independently retrained DeMo. All six metrics and signed harm/rescue changes retained. '
            'Text-only reconstruction from installed-GT CSV/CPU audits; no new neural inference or raw-distance audit. '
            'Selected-checkpoint extraction runtime is not training time; process launch/exit wall time reported separately.')


def main():
    parser = argparse.ArgumentParser()
    for key in ('trial-root', 'prior-root', 'baseline-root'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args(); trial = Path(args.trial_root)
    output = trial / 'analysis.json'; assert not output.exists()
    report = analyze(trial, Path(args.prior_root), Path(args.baseline_root))
    output.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    fields = ['comparison', 'condition', *METRICS, 'Rank1_harm_queries', 'Rank1_rescue_queries',
        'AP_improved_queries', 'AP_worsened_queries', 'AP_unchanged_queries', 'both_metrics_plus2']
    with (trial / 'comparison_condition_deltas.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for label, comparison in report['comparisons'].items():
            for condition, values in comparison['conditions'].items():
                writer.writerow(dict(comparison=label, condition=condition, **values['delta_pp'], **{k: values[k] for k in fields[8:]}))
    print('FULL_OFFICIAL_GRAPH_COMPLETE_ANALYSIS', flush=True)


if __name__ == '__main__':
    main()
