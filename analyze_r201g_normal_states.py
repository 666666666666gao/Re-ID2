"""CPU reduce every normal query from closed selected four-state readouts."""
import csv
from datetime import datetime
import json
from pathlib import Path

from analyze_identity_coordinate_three_normal import METRICS, load, query_rows, six, table

PROJECT = Path(__file__).resolve().parent


def main():
    proof = load(PROJECT / 'results/preflight/r201g_normal_states_actual_20261006.json')
    assert proof['status'] == 'ACTUAL_R201G_TWO_NORMAL_FOURSTATE_GT_RAW_COMPLETE' and proof['state_cases'] == 8
    root = PROJECT / 'results/r201g_normal_states_20261006'
    reference, original = query_rows(PROJECT / 'results/full_official_baselines_20261004/training/RGBNT201_demo_s42', 'RGBNT201')
    reads, metrics, groups, queries = {}, [], [], []
    diagnostics = {}
    for variant in ('frequency_shared', 'axis_shared'):
        name = 'RGBNT201_r201g_' + variant + '_s42'
        folder = root / name
        result = load(folder / 'result.json')
        archive = proof['archives'][name]
        assert archive['full11_distance_equal_selected_normal'] and archive['base00_distance_equal_original_DeMo']
        assert archive['eight_GT_metadata_arrays_equal_selected_normal']
        assert result['status'] == 'COMPLETE' and result['state_cases'] == 4 and result['query_records'] == 836
        assert result['optimizer_updates'] == result['checkpoint_writes'] == 0 and result['state_tensor_versions_unchanged']
        for state in ('00', '10', '01', '11'):
            with (folder / ('state_' + state + '.csv')).open(encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == 836 and all(row['valid'] == 'True' for row in rows)
            assert all(tuple(row[key] for key in ('query_index', 'name', 'identity', 'camera', 'scene')) ==
                       tuple(other[key] for key in ('query_index', 'name', 'identity', 'camera', 'scene'))
                       for row, other in zip(rows, reference))
            summary = six(rows)
            recorded = result['measurements'][state]
            assert all(abs(summary[key] - recorded[key]) < 1e-8 for key in METRICS)
            assert result['installed_GT_audit'][state]['max_metric_error'] < 1e-8
            if state == '00':
                assert all(abs(summary[key] - original[key]) < 1e-8 for key in METRICS)
            reads[(variant, state)] = rows, summary
            metrics.append(dict(variant=variant, state=state, selected_epoch=result['selected_epoch'], **summary))
            for grouping, values in recorded['groups'].items():
                groups.extend(dict(variant=variant, state=state, grouping=grouping, **value) for value in values)
            queries.extend(dict(variant=variant, state=state, **row) for row in rows)
        base = reads[(variant, '00')][0]
        m = reads[(variant, '10')][0]
        f = reads[(variant, '01')][0]
        full = reads[(variant, '11')][0]
        diagnostics[variant] = dict(
            base_M_correct_F_wrong=sum(int(a['Rank-1']) == 1 and int(b['Rank-1']) == 0 for a, b in zip(m, f)),
            base_F_correct_M_wrong=sum(int(a['Rank-1']) == 1 and int(b['Rank-1']) == 0 for a, b in zip(f, m)),
            both_single_states_wrong_full_correct=sum(int(a['Rank-1']) == int(b['Rank-1']) == 0 and int(c['Rank-1']) == 1 for a, b, c in zip(m, f, full)),
            either_single_state_correct_full_wrong=sum((int(a['Rank-1']) == 1 or int(b['Rank-1']) == 1) and int(c['Rank-1']) == 0 for a, b, c in zip(m, f, full)),
            base_wrong_full_correct=sum(int(a['Rank-1']) == 0 and int(b['Rank-1']) == 1 for a, b in zip(base, full)),
            base_correct_full_wrong=sum(int(a['Rank-1']) == 1 and int(b['Rank-1']) == 0 for a, b in zip(base, full)))
    comparisons, paired = [], []
    cases = [(variant, '11', variant, state) for variant in ('frequency_shared', 'axis_shared') for state in ('00', '10', '01')]
    cases.append(('axis_shared', '11', 'frequency_shared', '11'))
    for av, ast, bv, bst in cases:
        a, am = reads[(av, ast)]
        b, bm = reads[(bv, bst)]
        label = dict(improved_variant=av, improved_state=ast, reference_variant=bv, reference_state=bst)
        delta = {key: am[key] - bm[key] for key in METRICS}
        comparisons.append(dict(**label, **delta, both_plus2=delta['mAP'] >= 2 and delta['Rank-1'] >= 2,
            rank1_rescued=sum(int(x['Rank-1']) == 1 and int(y['Rank-1']) == 0 for x, y in zip(a, b)),
            rank1_harmed=sum(int(x['Rank-1']) == 0 and int(y['Rank-1']) == 1 for x, y in zip(a, b)),
            AP_improved=sum(float(x['AP']) > float(y['AP']) for x, y in zip(a, b)),
            AP_worsened=sum(float(x['AP']) < float(y['AP']) for x, y in zip(a, b))))
        paired.extend(dict(**label, **{key: x[key] for key in ('query_index', 'name', 'identity', 'camera', 'scene')},
            delta_AP_pp=100 * (float(x['AP']) - float(y['AP'])),
            delta_INP_pp=100 * (float(x['INP']) - float(y['INP'])),
            delta_Rank1=int(x['Rank-1']) - int(y['Rank-1'])) for x, y in zip(a, b))
    assert len(metrics) == 8 and len(comparisons) == 7 and len(paired) == 5852 and len(queries) == 6688
    output = root / 'normal_state_analysis'
    output.mkdir(exist_ok=False)
    for name, rows in (('six_metrics', metrics), ('group_metrics', groups), ('all_state_query_rows', queries),
                       ('comparisons', comparisons), ('paired_query_changes', paired)):
        table(output / (name + '.csv'), rows)
    result = dict(status='ACTUAL_R201G_SELECTED_NORMAL_FOURSTATE_CPU_READOUT_COMPLETE',
        completed_at=datetime.now().isoformat(timespec='seconds'), comparisons=comparisons,
        single_state_query_diagnostics=diagnostics, original_queries=836, repeated_state_query_rows=6688,
        paired_query_rows=5852, six_metrics_CMC50_identity_camera_scene=True,
        new_neural_calls=0, new_optimizer_updates=0,
        limits='Same selected weights, normal controlled inference; base+M/base+F are not standalone expert descriptors. '
               'No retraining ablation, independent/psi routing necessity, calibrated benefit, missing-modality, three-dataset or multi-seed claim. '
               'The four repeated states do not create independent samples. Ground truth acceptance is the actual preceding installed-filename audit.')
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], comparisons=comparisons), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
