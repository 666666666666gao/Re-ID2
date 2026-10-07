"""RGBNT201 fixed normal Best-of-five winners; existing text/GT evidence, CPU only."""
from datetime import datetime
import math
from pathlib import Path

from analyze_identity_coordinate_missing49 import SETS, STATES, availability_groups, load, verified_rows
from analyze_identity_coordinate_three_normal import METRICS, table

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT / 'results/r201k_best5_ordinary_missing49_20261007'


def main():
    proof = load(PROJECT / 'results/preflight/r201k_best5_ordinary_missing49_20261007_actual_session.json')
    assert proof['status'] == 'ACTUAL_K_BEST5_ORDINARY45_FIXED49_GT_RAW_COMPLETE_CPU_COMPARISON_PENDING'
    assert proof['phases']['full']['conditions'] == 49 and proof['phases']['full']['state_cases'] == 196
    assert proof['native_duplicate_raw_retired'] == 1 and proof['restored_normal_copies_cleared']
    assert proof['new_optimizer_updates'] == 0 and proof['unique_raw_lost'] == 0
    best = {r['model']: r for r in load(PROJECT / 'results/r201k_expert_N5_20261007/normal_seed_analysis/result.json')['best_of_five']}
    assert best['axis_shared']['seed'] == 42 and best['axis_shared']['selected_epoch'] == 30
    assert best['frequency_shared']['seed'] == 45 and best['frequency_shared']['selected_epoch'] == 33
    jobs = load(ROOT / 'jobs.json')['jobs']
    controller = load(ROOT / 'controller_full_result.json')
    assert len(jobs) == 1 and controller['status'] == 'COMPLETE' and controller['controls'] == 1
    assert controller['raw_conditions'] == 49 and controller['missing_GT_state_cases'] == 196
    assert controller['all_raw_local_verified'] and controller['optimizer_updates'] == controller['new_weights'] == 0
    folders = {
        'Best5_axis_s42': PROJECT / 'results/r201k_missing49_20261007/full/RGBNT201_r201k_axis_shared_s42',
        'Best5_ordinary_s45': ROOT / 'full/RGBNT201_r201k_frequency_shared_s45',
    }
    models, geometry = {}, []
    for label, folder in folders.items():
        result, audit = load(folder / 'result.json'), load(folder / 'independent_cpu_audit.json')
        variant = 'axis_shared' if label == 'Best5_axis_s42' else 'frequency_shared'
        assert result['status'] == 'COMPLETE' and result['conditions'] == 49 and result['state_cases'] == 196
        assert audit['status'] == 'PASS' and audit['conditions'] == 49 and audit['state_cases'] == 196
        assert result['optimizer_updates'] == result['new_weights'] == 0 and result['normal_features_and_distance_exact']
        assert result['selected_epoch'] == best[variant]['selected_epoch'] and result['model_arguments']['seed'] == best[variant]['seed']
        assert all(result['measurements']['q_RNT_g_RNT']['11'][m] == best[variant][m] for m in METRICS)
        for state in STATES:
            models[label + '/' + state] = ({c: v[state] for c, v in result['measurements'].items()}, folder, state)
        for condition, item in audit['measured'].items():
            if 'disjoint_private_distance' in item:
                assert item['cross_source_identity_comparison_not_established']
                geometry.extend(dict(model=label, condition=condition, state=s, **v) for s, v in item['disjoint_private_distance'].items())
    original = PROJECT / 'results/full_official_baselines_20261004/frozen49/RGBNT201_demo_s42'
    assert load(original / 'independent_cpu_audit.json')['status'] == 'PASS'
    models['original_DeMo42'] = (load(original / 'result.json')['measurements'], original, 'baseline')
    axis, ordinary = 'Best5_axis_s42', 'Best5_ordinary_s45'
    comparisons = [(axis + '/11', r) for r in ('original_DeMo42', axis + '/00', axis + '/10', axis + '/01', ordinary + '/11')]
    comparisons += [(ordinary + '/11', r) for r in ('original_DeMo42', ordinary + '/00', ordinary + '/10', ordinary + '/01')]
    metrics, groups, deltas, pairs, grouped = [], [], [], [], []
    checked = 0
    for qs in SETS:
        for gs in SETS:
            condition = 'q_' + qs + '_g_' + gs
            rows = {}
            for label, (values, folder, state) in models.items():
                file = folder / (condition + '.csv') if state == 'baseline' else folder / condition / ('state_' + state + '.csv')
                current = verified_rows(file, values[condition], 836, 836)
                rows[label] = current
                checked += len(current)
                metrics.append(dict(model=label, condition=condition, **{m: values[condition][m] for m in METRICS}))
                groups.extend(dict(model=label, condition=condition, grouping=k, **v) for k, entries in values[condition]['groups'].items() for v in entries)
            assert rows[axis + '/00'] == rows[ordinary + '/00']
            for improved, reference in comparisons:
                left, right = rows[improved], rows[reference]
                assert all(all(a[k] == b[k] for k in ('query_index', 'name', 'identity', 'camera', 'scene', 'valid', 'relevant_gallery', 'kept_gallery')) for a, b in zip(left, right))
                av, bv = models[improved][0][condition], models[reference][0][condition]
                diff = {m: av[m] - bv[m] for m in METRICS}
                rescued = sum(int(a['Rank-1']) == 1 and int(b['Rank-1']) == 0 for a, b in zip(left, right))
                harmed = sum(int(a['Rank-1']) == 0 and int(b['Rank-1']) == 1 for a, b in zip(left, right))
                assert abs(diff['Rank-1'] - 100 * (rescued - harmed) / 836) < 1e-8
                deltas.append(dict(condition=condition, improved=improved, reference=reference, **diff, rescued=rescued, harmed=harmed,
                    AP_improved=sum(float(a['AP']) > float(b['AP']) for a, b in zip(left, right)),
                    AP_worsened=sum(float(a['AP']) < float(b['AP']) for a, b in zip(left, right)),
                    both_plus2=diff['mAP'] >= 2 and diff['Rank-1'] >= 2))
                if improved == axis + '/11' and reference in (axis + '/00', ordinary + '/11'):
                    pairs.extend(dict(condition=condition, improved=improved, reference=reference,
                        **{k: a[k] for k in ('query_index', 'name', 'identity', 'camera', 'scene')},
                        delta_AP_pp=100 * (float(a['AP']) - float(b['AP'])),
                        delta_INP_pp=100 * (float(a['INP']) - float(b['INP'])),
                        delta_rank1=int(a['Rank-1']) - int(b['Rank-1'])) for a, b in zip(left, right))
    for improved, reference in comparisons:
        for group, size in dict(all49=49, same_availability=7, overlap_mismatch=30, source_disjoint=12,
                               partial_query_full_gallery=6, full_query_partial_gallery=6, both_partial=36).items():
            chosen = [r for r in deltas if r['improved'] == improved and r['reference'] == reference
                      and group in availability_groups(*r['condition'][2:].split('_g_'))]
            assert len(chosen) == size
            grouped.append(dict(improved=improved, reference=reference, group=group, conditions=size,
                **{m: math.fsum(r[m] for r in chosen) / size for m in METRICS},
                rescued=sum(r['rescued'] for r in chosen), harmed=sum(r['harmed'] for r in chosen),
                both_plus2_conditions=sum(r['both_plus2'] for r in chosen)))
    assert len(metrics) == 441 and len(deltas) == 441 and checked == 368676 and len(pairs) == 81928
    out = ROOT / 'missing_best5_analysis'
    out.mkdir(exist_ok=False)
    for name, values in (('six_metrics', metrics), ('group_metrics', groups), ('comparisons', deltas),
                         ('availability_group_deltas', grouped), ('disjoint_distance_geometry', geometry), ('paired_query_changes', pairs)):
        table(out / (name + '.csv'), values)
    result = dict(status='ACTUAL_RGBNT201_NORMAL_BEST5_FIXED49_CPU_COMPLETE', completed_at=datetime.now().isoformat(timespec='seconds'),
        normal_best_of_five=best, new_missing_control=1, reused_missing_control=1, conditions_per_control=49,
        states_per_control=196, six_metric_rows=len(metrics), comparisons=len(deltas), checked_condition_query_rows=checked,
        unique_queries=836, paired_query_rows=len(pairs), all_six_CMC50_perquery_identity_camera_scene=True,
        availability_group_deltas=grouped, new_neural_calls=0, new_optimizer_updates=0,
        limits='Normal benchmark-selected Best-of-five axis42 versus ordinary45 is different-seed best comparison, not paired seed variance. '
               'Conditional expert repeat on fixed DeMo42, not five pipelines; original50 versus additional50 budget disclosed. '
               'Same normal-selected weights across49, no percondition seed/epoch selection. Repeated condition-query rows not independent. '
               'Twelve disjoint private-coordinate comparisons do not establish cross-source identity. Negatives and harms preserved; '
               'execution completion does not establish three-dataset +2, calibration/causality or retrained module necessity.')
    (out / 'result.json').write_text(__import__('json').dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
