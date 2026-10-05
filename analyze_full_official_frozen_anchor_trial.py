"""Complete signed common-anchor comparisons and all49 inference-utility diagnostics."""
import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

from analyze_full_official_baseline_pairs import METRICS, SETS, query_rows, summarize
from analyze_full_official_control_trial import STATES, compare, comparison_groups
from analyze_full_official_public_outlet_identity_trial import load_bank


MODELS = {'demo_continued': ('demo_shared', 0), 'axis_fixed': ('axis_shared', 1),
          'frequency_fixed': ('frequency_shared', 1), 'twins_fixed': ('twins_shared', 1),
          'axis_adaptive': ('axis_shared', 0)}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def analyze(trial, baseline):
    done, intake = read(trial / 'controller_result.json'), read(trial / 'intake_receipt.json')
    assert done['status'] == 'COMPLETE' and len(done['runs']) == 5
    assert done['frozen_metric_cases'] == 245 and done['enhanced_state_metric_cases'] == 1176
    assert done['all_frozen_and_state_cpu_audits_passed'] and done['paired_identity_and_partial_sampling_exact']
    assert done['archive_sensitive_phases_serial'] and done['all196_enhanced_raw_and245_frozen_raw_local_verified']
    assert intake['status'] == 'FIVE_FULL_OFFICIAL_ANCHOR_RUNS245_FROZEN_AND1176_STATE_CASES_TEXT_COLLECTED'
    assert read(baseline / 'summary.json')['status'] == 'SIX_FULL_OFFICIAL_BASELINES_COMPLETE_ALL294_CPU_AUDITED'
    conditions = sorted('q_' + q + '_g_' + g for q in SETS for g in SETS)
    banks, references = {}, {}
    for variant in ('demo', 'demo_shared'):
        root = baseline / 'frozen49' / ('MSVR310_' + variant + '_s42')
        audit = read(root / 'independent_cpu_audit.json')
        banks[variant] = load_bank(root, audit, conditions)
        references[variant] = dict(normal=audit['normal'], selected_epoch=audit['selected_epoch'], conditions=audit['conditions'])
    anchor = read(baseline / 'training/MSVR310_demo_shared_s42/result.json')
    anchor_launch = read(baseline / 'training/MSVR310_demo_shared_s42_launch.json')
    anchor_exit = read(baseline / 'training/MSVR310_demo_shared_s42_exit.json')
    assert anchor_exit['exit_code'] == 0
    anchor_training_seconds = anchor_exit['finished'] - anchor_launch['started']
    assert anchor_training_seconds > 0
    parent_orders = [(r['epoch'], r['step'], r['names'], r['partial_set']) for r in
                     map(json.loads, (baseline / 'training/MSVR310_demo_shared_s42/batch_orders.jsonl').read_text().splitlines())]
    models, orders = {}, []
    for label, (variant, freeze) in MODELS.items():
        name = 'MSVR310_anchor_' + variant + ('_fixed' if freeze else '_adaptive') + '_s42'
        run, frozen = (trial / phase / name for phase in ('training', 'frozen49'))
        trained, result, audit = map(read, (run / 'result.json', frozen / 'result.json', frozen / 'independent_cpu_audit.json'))
        args = trained['arguments']
        assert trained['status'] == result['status'] == 'COMPLETE' and trained['epochs'] == 50
        assert args['dataset'] == 'MSVR310' and args['variant'] == variant and args['seed'] == 42
        assert args['freeze_identity_encoder'] == freeze and args['anchor_run_dir'] == done['anchor_run_dir']
        assert trained['training_heldout_identities'] == 0 and trained['training_coverage'] == dict(eligible=1032, visited=1032, unvisited=[])
        assert (trained['train_records'], trained['query_records'], trained['gallery_records']) == (1032, 591, 1055)
        assert trained['descriptor_dim'] == 5632
        assert result['model_arguments'] == audit['model_arguments'] == args
        assert audit['selected_epoch'] == result['selected_epoch'] == trained['best']['epoch']
        assert all(abs(trained['full_metrics'][m] - audit['normal'][m]) < 1e-8 for m in METRICS)
        banks[label] = load_bank(frozen, audit, conditions)
        batches = list(map(json.loads, (run / 'batch_orders.jsonl').read_text().splitlines()))
        assert len(batches) == trained['steps'] == 705
        assert sum(r['optimizer_updated'] for r in batches) == trained['optimizer_steps']
        assert trained['steps'] - trained['optimizer_steps'] == trained['amp_skipped_steps']
        order = [(r['epoch'], r['step'], r['names'], r['partial_set']) for r in batches]
        assert order == parent_orders
        orders.append(order)
        assert all(not row['reference_requires_grad'] and row['freeze_identity_encoder'] == bool(freeze) for row in batches)
        model = dict(variant=variant, freeze_identity_encoder=bool(freeze), selected_epoch=audit['selected_epoch'],
                     normal=audit['normal'], conditions=audit['conditions'], parameters=trained['parameters'],
                     trainable_parameters=trained['trainable_parameters'], descriptor_dim=5632,
                     optimizer_steps=trained['optimizer_steps'], amp_skipped_steps=trained['amp_skipped_steps'],
                     peak_training_memory_bytes=trained['peak_memory'], runtime=trained['runtime'],
                     frozen_availability_runtime=result['runtime'], anchor_pretraining_epochs=50, additional_epochs=50)
        phases = ['contract', 'preflight', 'training', 'frozen49', 'audit']
        if variant != 'demo_shared':
            diagnosis = trial / 'diagnosis' / name
            states = read(diagnosis / 'independent_cpu_audit.json')
            assert states['status'] == 'PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT' and states['cases'] == 294
            assert states['model_arguments'] == args and set(states['conditions']) == set(conditions)
            effects = {key: {} for key in ('full11_minus00', 'full11_minus10', 'full11_minus01')}
            base_comparisons = {}
            for condition in conditions:
                rows = {state: query_rows(diagnosis / condition / ('state_' + state + '.csv'), 591) for state in STATES}
                assert all(abs(summarize(rows[state])[m] - states['conditions'][condition]['metrics'][state][m]) < 1e-8 for state in STATES for m in METRICS)
                assert rows['11'] == banks[label][condition]
                base_comparisons[condition] = compare(banks['demo_shared'][condition], rows['00'])
                for key, before in (('full11_minus00', '00'), ('full11_minus10', '10'), ('full11_minus01', '01')):
                    measured, saved = compare(rows[before], rows['11']), states['conditions'][condition][key]
                    assert all(abs(measured['delta_pp'][m] - saved['delta_pp'][m]) < 1e-8 for m in METRICS)
                    assert all(measured[field] == saved[field] for field in
                               ('Rank1_harm_queries', 'Rank1_rescue_queries', 'AP_improved_queries', 'AP_worsened_queries', 'AP_unchanged_queries'))
                    effects[key][condition] = measured
            calibration = [states['conditions'][c]['calibration'] for c in conditions]
            model.update(diagnosis_normal=states['normal'], diagnosis_conditions=states['conditions'], heads=states['heads'],
                         inference_effects={key: dict(normal=values['q_RNT_g_RNT'], conditions=values, groups=comparison_groups(values),
                             all49_equal_condition_mean_delta_pp={m: fmean(v['delta_pp'][m] for v in values.values()) for m in METRICS}) for key, values in effects.items()},
                         anchor_to_trained00=dict(normal=base_comparisons['q_RNT_g_RNT'], conditions=base_comparisons, groups=comparison_groups(base_comparisons)),
                         calibration_equal_condition_mean={key: [fmean(v[key][i] for v in calibration) for i in range(3)] for key in
                             ('target_mean', 'target_std', 'MAE', 'zero_prediction_MAE')},
                         calibration_MAE_better_than_zero_conditions=[sum(v['MAE'][i] < v['zero_prediction_MAE'][i] for v in calibration) for i in range(3)])
            phases.append('diagnosis')
        model['phase_wall_seconds'] = {}
        for phase in phases:
            launched = read(trial / phase / (name + '_launch.json'))
            finished = read(trial / phase / (name + '_exit.json'))
            assert finished['exit_code'] == 0 and finished['finished'] > launched['started']
            model['phase_wall_seconds'][phase] = finished['finished'] - launched['started']
        models[label] = model
    assert all(order == orders[0] for order in orders)
    assert len({(models[v]['parameters'], models[v]['trainable_parameters'], models[v]['descriptor_dim']) for v in ('axis_fixed', 'frequency_fixed', 'twins_fixed')}) == 1
    pairs = [(before, 'axis_fixed') for before in ('demo', 'demo_shared', 'demo_continued', 'frequency_fixed', 'twins_fixed', 'axis_adaptive')]
    pairs += [('demo_shared', 'demo_continued'), ('demo_shared', 'axis_adaptive')]
    comparisons = {}
    for before, after in pairs:
        values = {c: compare(banks[before][c], banks[after][c]) for c in conditions}
        comparisons[before + ' -> ' + after] = dict(before=before, after=after, normal=values['q_RNT_g_RNT'], conditions=values,
            groups=comparison_groups(values), both_metrics_plus2_conditions=sum(v['both_metrics_plus2'] for v in values.values()),
            all49_equal_condition_mean_delta_pp={m: fmean(v['delta_pp'][m] for v in values.values()) for m in METRICS})
    return dict(status='FIVE_FULL_OFFICIAL_ANCHOR_RUNS245_FROZEN_AND1176_STATE_CASES_ANALYZED', dataset='MSVR310', seed=42,
                full_split_counts=dict(train=1032, query=591, gallery=1055), training_heldout_identities=0,
                frozen_metric_cases=245, enhanced_state_metric_cases=1176, repeated_condition_query_rows=1176 * 591,
                contribution_rows=196 * 591, models=models, references=references, comparisons=comparisons,
                comparison_condition_rows=len(pairs) * 49, paired_identity_and_partial_sampling_exact=True,
                anchor_pretraining_cost=dict(epochs=50, selected_epoch=anchor['best']['epoch'],
                    training_wall_seconds=anchor_training_seconds, selected_checkpoint_extraction_runtime=anchor['runtime'],
                    optimizer_steps=anchor['optimizer_steps']),
                original_DeMo_plus2_thresholds={m: references['demo']['normal'][m] + 2 for m in ('mAP', 'Rank-1')}, goal_complete=False,
                limits='One common completed50 anchor bestE37 plus additional50/new optimizer for each control; not a fresh-CLIP50 single-factor claim. '
                       'Single-seed MSVR benchmark checkpoint selection, not untouched test or unified three-dataset/multiseed acceptance. '
                       'All six metrics/signed harm-rescue/49 groups retained; repeated conditions are dependent, equal-condition means descriptive. '
                       'Frozen00 is compared explicitly with the original anchor; adaptive00 is trained, not an independently trained baseline. '
                       'Text-only verified GT CSV reconstruction, not new NN or raw-distance recount. Stream diagnosis wall time includes audit/archive waiting.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('trial-root', 'baseline-root'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    trial = Path(args.trial_root)
    assert not (trial / 'analysis.json').exists()
    report = analyze(trial, Path(args.baseline_root))
    (trial / 'analysis.json').write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    fields = ['comparison', 'condition', *METRICS, 'Rank1_harm_queries', 'Rank1_rescue_queries',
              'AP_improved_queries', 'AP_worsened_queries', 'AP_unchanged_queries', 'both_metrics_plus2']
    with (trial / 'comparison_condition_deltas.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for label, comparison in report['comparisons'].items():
            for condition, values in comparison['conditions'].items():
                writer.writerow(dict(comparison=label, condition=condition, **values['delta_pp'], **{key: values[key] for key in fields[8:]}))
    print('FULL_OFFICIAL_ANCHOR_COMPLETE_ANALYSIS', flush=True)
