"""Report audited M3a comparisons; no model execution or checkpoint selection."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
METRICS = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
VARIANTS = ('axis_shared', 'frequency_shared', 'twins_shared')
NORMAL = 'q_RNT_g_RNT'


def groups(conditions):
    predicates = dict(
        normal=lambda q, g: q == g == 'RNT',
        all49=lambda q, g: True,
        same_availability=lambda q, g: q == g,
        overlap_mismatch=lambda q, g: q != g and bool(set(q) & set(g)),
        source_disjoint=lambda q, g: not bool(set(q) & set(g)),
        partial_query_full_gallery=lambda q, g: q != 'RNT' and g == 'RNT',
        both_partial=lambda q, g: q != 'RNT' and g != 'RNT',
    )
    result = {name: [c for c in conditions if predicate(c.split('_')[1], c.split('_')[3])]
              for name, predicate in predicates.items()}
    assert [len(result[k]) for k in predicates] == [1, 49, 7, 30, 12, 6, 36]
    return result


def summarize(candidate, reference, condition_groups):
    delta = {c: {m: candidate[c][m] - reference[c][m] for m in METRICS}
             for c in candidate}
    grouped = {}
    for name, conditions in condition_groups.items():
        grouped[name] = dict(
            conditions=len(conditions),
            candidate_mean={m: statistics.mean(candidate[c][m] for c in conditions) for m in METRICS},
            reference_mean={m: statistics.mean(reference[c][m] for c in conditions) for m in METRICS},
            delta_pp={m: statistics.mean(delta[c][m] for c in conditions) for m in METRICS},
            both_plus2_conditions=[c for c in conditions if delta[c]['mAP'] >= 2 and delta[c]['Rank-1'] >= 2],
            mAP_degraded_conditions=[c for c in conditions if delta[c]['mAP'] < 0],
            Rank1_degraded_conditions=[c for c in conditions if delta[c]['Rank-1'] < 0],
            either_primary_degraded_conditions=[c for c in conditions if delta[c]['mAP'] < 0 or delta[c]['Rank-1'] < 0],
            both_primary_improved_conditions=[c for c in conditions if delta[c]['mAP'] > 0 and delta[c]['Rank-1'] > 0],
        )
    return dict(groups=grouped, per_condition_delta_pp=delta)


def complementarity(root, variant, condition_groups):
    """Pair exported, GT-audited query rows; these counts are not extra trials."""
    name = 'MSVR310_' + variant + '_s42'
    state_root = root / 'original_mean/controlled_states' / name / 'full'
    utility_root = root / 'original_mean/utility' / name / 'full'
    measured = {}
    for condition in condition_groups['all49']:
        files = {s: state_root / condition / ('state_' + s + '.csv') for s in ('00', '10', '01', '11')}
        files.update({s: utility_root / condition / (s + '.csv') for s in ('F_pre', 'F_post')})
        records = {}
        for stage, path in files.items():
            with path.open(encoding='utf-8', newline='') as f:
                records[stage] = list(csv.DictReader(f))
        queries = [(r['query_index'], r['name'], r['identity'], r['camera'], r['scene']) for r in records['00']]
        assert len(queries) == 210
        assert all([(r['query_index'], r['name'], r['identity'], r['camera'], r['scene']) for r in data] == queries for data in records.values())
        sets = dict(F_post_rescues_base=[], F_post_rescues_realized_by_joint=[], F_post_rescues_missed_by_joint=[],
                    F_pre_rescues_retained_by_PF=[], F_pre_rescues_lost_by_PF=[],
                    joint_harm=[], joint_rescue=[], M_only_correct=[], F_only_correct=[], both_correct=[], both_wrong=[])
        for i, query in enumerate(queries):
            correct = {s: int(data[i]['first_match']) == 1 for s, data in records.items()}
            if not correct['00'] and correct['F_post']:
                sets['F_post_rescues_base'].append(query[0])
                sets['F_post_rescues_realized_by_joint' if correct['11'] else 'F_post_rescues_missed_by_joint'].append(query[0])
            if not correct['00'] and correct['F_pre']:
                sets['F_pre_rescues_retained_by_PF' if correct['F_post'] else 'F_pre_rescues_lost_by_PF'].append(query[0])
            if correct['00'] and not correct['11']:
                sets['joint_harm'].append(query[0])
            if not correct['00'] and correct['11']:
                sets['joint_rescue'].append(query[0])
            if correct['10'] and correct['01']:
                sets['both_correct'].append(query[0])
            elif correct['10']:
                sets['M_only_correct'].append(query[0])
            elif correct['01']:
                sets['F_only_correct'].append(query[0])
            else:
                sets['both_wrong'].append(query[0])
        measured[condition] = sets
    grouped = {group: dict(condition_query_occurrences=210 * len(conditions),
                           counts={k: sum(len(measured[c][k]) for c in conditions) for k in measured[conditions[0]]})
               for group, conditions in condition_groups.items()}
    return dict(groups=grouped, per_condition_query_indices=measured,
                limits='Standalone PF successes indicate potential complementarity, not a guarantee that fusion can recover them. Four-state correctness is a frozen intervention on the same trained model. Repeated condition-query occurrences are not independent samples.')


def contribution_calibration(root, variant, condition_groups):
    folder = root / 'original_mean/controlled_states' / ('MSVR310_' + variant + '_s42') / 'full'
    reported = json.loads((folder / 'result.json').read_text(encoding='utf-8'))['contributions']
    keys = (('M', 'delta_M_given_F'), ('F', 'delta_F_given_M'), ('I', 'empirical_interaction'))
    records = {}
    for condition in condition_groups['all49']:
        with (folder / condition / 'contributions.csv').open(encoding='utf-8', newline='') as f:
            records[condition] = list(csv.DictReader(f))
        assert len(records[condition]) == 210
        for i, (name, target) in enumerate(keys):
            actual = statistics.mean(abs(float(r['prediction_' + name]) - float(r[target])) for r in records[condition])
            # Actual M2 exports show float32 mean rounding up to 1.43e-7.
            assert abs(actual - reported[condition]['MAE'][i]) < 1e-6
    result = {}
    for group, conditions in condition_groups.items():
        rows = [r for c in conditions for r in records[c]]
        result[group] = {}
        for name, key in keys:
            targets = [float(r[key]) for r in rows]
            predictions = [float(r['prediction_' + name]) for r in rows]
            gates = [1 / (1 + math.exp(-20 * value)) for value in predictions]
            target_mean = statistics.mean(targets)
            result[group][name] = dict(
                condition_query_occurrences=len(rows), target_mean=target_mean,
                target_std=statistics.pstdev(targets), prediction_mean=statistics.mean(predictions),
                prediction_std=statistics.pstdev(predictions),
                MAE=statistics.mean(abs(p - t) for p, t in zip(predictions, targets)),
                RMSE=statistics.mean((p - t) ** 2 for p, t in zip(predictions, targets)) ** .5,
                zero_prediction_MAE=statistics.mean(abs(t) for t in targets),
                evaluation_target_mean_constant_MAE=statistics.mean(abs(t - target_mean) for t in targets),
                implied_gate_mean=statistics.mean(gates), implied_gate_std=statistics.pstdev(gates),
                implied_gate_min=min(gates), implied_gate_max=max(gates),
            )
    return result


def build_comparisons():
    root = PROJECT / 'results/measurement_gate_m3a_20261004'
    paths = dict(
        M3a=root / 'independent_cpu_audit.json',
        M2b=PROJECT / 'results/identity_alignment_m2b_20261004/independent_cpu_audit.json',
        augmented_DeMo=PROJECT / 'results/common_outlet_m1_v3_complete/independent_cpu_audit.json',
        original_DeMo=PROJECT / 'results/axis_collaboration_v4_missing_development27/MSVR310_demo_s42/full/clean.json',
    )
    loaded = {name: json.loads(path.read_text(encoding='utf-8')) for name, path in paths.items()}
    assert all(loaded[k]['status'] == 'PASS' for k in ('M3a', 'M2b', 'augmented_DeMo'))
    assert loaded['M3a']['cases'] == 3087 and loaded['M3a']['perquery_count'] == 648270
    assert loaded['M2b']['cases'] == 3087 and loaded['M2b']['perquery_count'] == 648270
    assert loaded['M3a']['M3a_gate_gradient_contract_verified']
    deployed = {v: {c: r['metrics']['11'] for c, r in loaded['M3a']['runs'][v]['states']['conditions'].items()}
                for v in VARIANTS}
    conditions = list(deployed['axis_shared'])
    condition_groups = groups(conditions)
    demo = {c: r['metrics']['11'] for c, r in loaded['augmented_DeMo']['runs']['original_mean/demo_shared']['conditions'].items()}
    assert set(demo) == set(conditions)
    result = dict(
        status='COMPLETE_AUDITED_COMPARISONS_NOT_FINAL_METHOD',
        evidence={k: dict(path=p.relative_to(PROJECT).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k, p in paths.items()},
        limits='MSVR310 fit/dev seed42 only; equal-condition means and condition counts are diagnostics, not independent samples or statistical significance. Original DeMo is used for the normal +2 gate; availability-screened augmented DeMo is used for missing conditions. Augmented DeMo has no frequency losses and is not an equal-parameter expert control.',
        original_DeMo_normal={m: loaded['original_DeMo'][m] for m in METRICS},
        augmented_DeMo_normal=demo[NORMAL],
        comparisons={},
        normal_original_DeMo_gate={},
        cross_compatibility={},
        query_complementarity={},
        contribution_calibration={},
        goal_complete=False,
    )
    rows = []
    references = {}
    for variant in VARIANTS:
        reference = {c: r['metrics']['11'] for c, r in loaded['M2b']['runs'][variant]['states']['conditions'].items()}
        references['M3a_minus_M2b_' + variant] = (variant, reference)
    references.update({
        'M3a_axis_minus_frequency_shared': ('axis_shared', deployed['frequency_shared']),
        'M3a_axis_minus_twins_shared': ('axis_shared', deployed['twins_shared']),
        'M3a_axis_minus_augmented_DeMo': ('axis_shared', demo),
    })
    for comparison, (variant, reference) in references.items():
        assert set(reference) == set(conditions)
        measured = summarize(deployed[variant], reference, condition_groups)
        result['comparisons'][comparison] = measured
        for condition, delta in measured['per_condition_delta_pp'].items():
            rows.append(dict(comparison=comparison, model=variant, condition=condition,
                             both_plus2=delta['mAP'] >= 2 and delta['Rank-1'] >= 2, **delta))
    for variant in VARIANTS:
        delta = {m: deployed[variant][NORMAL][m] - loaded['original_DeMo'][m] for m in METRICS}
        result['normal_original_DeMo_gate'][variant] = dict(
            candidate=deployed[variant][NORMAL], delta_pp=delta,
            both_plus2=delta['mAP'] >= 2 and delta['Rank-1'] >= 2,
        )
        current = loaded['M3a']['runs'][variant]['cross_coordinates']['conditions']
        prior = loaded['M2b']['runs'][variant]['cross_coordinates']['conditions']
        assert set(current) == set(prior) == set(conditions)
        cross = {}
        for pair in ('F_post_common', 'common_F_post'):
            candidate = {c: r['metrics'][pair] for c, r in current.items()}
            reference = {c: r['metrics'][pair] for c, r in prior.items()}
            cross[pair] = summarize(candidate, reference, condition_groups)
            cross[pair]['below_own_common_mAP_conditions'] = [c for c in conditions if candidate[c]['mAP'] < current[c]['metrics']['common_common']['mAP']]
            cross[pair]['below_own_F_post_mAP_conditions'] = [c for c in conditions if candidate[c]['mAP'] < current[c]['metrics']['F_post_F_post']['mAP']]
        cross['standalone_F_post'] = summarize(
            {c: r['metrics']['F_post_F_post'] for c, r in current.items()},
            {c: r['metrics']['F_post_F_post'] for c, r in prior.items()}, condition_groups)
        result['cross_compatibility'][variant] = cross
        result['query_complementarity'][variant] = complementarity(root, variant, condition_groups)
        result['contribution_calibration'][variant] = contribution_calibration(root, variant, condition_groups)
    assert len(rows) == 294
    with (root / 'all49_baseline_and_expert_comparisons.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['comparison', 'model', 'condition', 'both_plus2', *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    with (root / 'comparison_summary.json').open('x', encoding='utf-8') as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    entry = root / 'comparison_entrypoint.py'
    assert not entry.exists()
    entry.write_bytes(Path(__file__).read_bytes())
    return result


if __name__ == '__main__':
    summary = build_comparisons()
    print(json.dumps(dict(status=summary['status'], normal=summary['normal_original_DeMo_gate'],
                          comparisons={k: v['groups']['all49'] for k, v in summary['comparisons'].items()}), ensure_ascii=False))
