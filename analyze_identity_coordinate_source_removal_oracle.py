"""Read sealed unit-control exports; GT-guided source removal is diagnostic only."""
from datetime import datetime
import json
from pathlib import Path

from analyze_identity_coordinate_missing49 import verified_rows
from analyze_identity_coordinate_three_normal import COUNTS, METRICS, table
from analyze_full_official_baseline_pairs import summarize

PROJECT = Path(__file__).resolve().parent
SETS = ('R', 'N', 'T', 'RN', 'RT', 'NT')
KEYS = ('query_index', 'name', 'identity', 'camera', 'scene', 'valid', 'relevant_gallery', 'kept_gallery')


def main():
    stream = PROJECT / 'results/identity_coordinate_missing49_stream_full_completed_20261005'
    r201 = PROJECT / 'results/rgbnt201_identity_outlet_r201c_20261005'
    closed = json.loads((stream / 'three_dataset_missing_analysis/result.json').read_text(encoding='utf-8'))
    assert closed['status'] == 'ACTUAL_THREE_DATASET_FIXED_BEST_ALL49_FOURSTATE_CPU_READOUT'
    assert json.loads((stream / 'controller_full_result.json').read_text())['status'] == 'COMPLETE'
    output = PROJECT / 'results/identity_coordinate_source_removal_oracle_20261006'
    assert not output.exists()
    comparisons, query_details, summaries = [], [], []
    checked = 0
    for dataset, (_, count, gallery, _) in COUNTS.items():
        for variant in ('frequency_shared', 'axis_shared'):
            name = dataset + '_identity_' + variant + '_narrow_s42'
            folder = r201 / 'frozen49' / name if dataset == 'RGBNT201' else stream / 'full' / name
            data = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
            audit = json.loads((folder / ('independent_fourstate_cpu_audit.json' if dataset == 'RGBNT201' else 'independent_cpu_audit.json')).read_text(encoding='utf-8'))
            assert data['status'] == 'COMPLETE' and data['state_cases'] == audit['state_cases'] == 196 and audit['status'] == 'PASS'
            measurements = data['state_measurements'] if dataset == 'RGBNT201' else data['measurements']
            for state in ('00', '11'):
                def rows(q, g):
                    nonlocal checked
                    condition = 'q_' + q + '_g_' + g
                    path = folder / (condition + ('' if state == '11' else '_state' + state) + '.csv') if dataset == 'RGBNT201' else folder / condition / ('state_' + state + '.csv')
                    values = verified_rows(path, measurements[condition][state], count, gallery)
                    checked += len(values)
                    return values
                full = rows('RNT', 'RNT')
                normal = summarize(full)
                for protocol in ('query_missing', 'both_missing_same_set'):
                    candidates = {subset: rows(subset, 'RNT' if protocol == 'query_missing' else subset) for subset in SETS}
                    for subset, partial in candidates.items():
                        assert all(all(a[key] == b[key] for key in KEYS) for a, b in zip(full, partial))
                        values = summarize(partial)
                        comparisons.append(dict(dataset=dataset, variant=variant, state=state, protocol=protocol, subset=subset,
                            **{key + '_delta_pp': values[key] - normal[key] for key in METRICS},
                            rescued=sum(int(a['first_match']) != 1 and int(b['first_match']) == 1 for a, b in zip(full, partial)),
                            harmed=sum(int(a['first_match']) == 1 and int(b['first_match']) != 1 for a, b in zip(full, partial))))
                    records = []
                    for index, base in enumerate(full):
                        correct_sets = [subset for subset, partial in candidates.items() if int(partial[index]['first_match']) == 1]
                        better_AP_sets = [subset for subset, partial in candidates.items() if float(partial[index]['AP']) > float(base['AP'])]
                        harm_sets = [subset for subset, partial in candidates.items() if int(partial[index]['first_match']) != 1]
                        record = dict(dataset=dataset, variant=variant, state=state, protocol=protocol,
                            **{key: base[key] for key in ('query_index', 'name', 'identity', 'camera', 'scene')},
                            full_AP=float(base['AP']), full_Rank1=int(base['first_match']) == 1,
                            any_partial_Rank1=bool(correct_sets), correct_partial_sets='|'.join(correct_sets),
                            better_AP_sets='|'.join(better_AP_sets), incorrect_partial_sets='|'.join(harm_sets),
                            oracle_rescue=int(base['first_match']) != 1 and bool(correct_sets),
                            any_removal_harms=int(base['first_match']) == 1 and bool(harm_sets))
                        records.append(record)
                    query_details.extend(records)
                    rescued = sum(row['oracle_rescue'] for row in records)
                    wrong = sum(not row['full_Rank1'] for row in records)
                    summaries.append(dict(dataset=dataset, variant=variant, state=state, protocol=protocol,
                        selected_epoch=data['selected_epoch'], queries=count, full_Rank1=normal['Rank-1'], full_errors=wrong,
                        GT_oracle_rescuable_errors=rescued, GT_oracle_Rank1=normal['Rank-1'] + 100 * rescued / count,
                        full_correct_with_at_least_one_harmful_removal=sum(row['any_removal_harms'] for row in records)))
    assert len(comparisons) == 144 and len(summaries) == 24
    assert checked == 13 * 4 * sum(counts[1] for counts in COUNTS.values())
    assert len(query_details) == 8 * sum(counts[1] for counts in COUNTS.values())
    output.mkdir()
    for name, values in (('fixed_removal_comparisons', comparisons), ('query_potential', query_details), ('GT_oracle_summary', summaries)):
        table(output / (name + '.csv'), values)
    result = dict(status='ACTUAL_SEALED_SOURCE_REMOVAL_GT_ORACLE_CPU_DIAGNOSTIC', completed_at=datetime.now().isoformat(timespec='seconds'),
        checked_condition_query_rows=checked, query_potential_rows=len(query_details), summary_rows=24, fixed_comparisons=144,
        new_neural_calls=0, new_optimizer_updates=0, new_model_selection=0,
        limits='Uses only previously sealed unit-control checkpoints and installed-GT CSVs. Oracle uses query GT to choose among six removals plus original; not deployable performance, a learned route, a held-out test, or a causal modality-quality finding. Query-only and symmetric query/gallery removal change different matching states. No new training, source change, missing evaluation or candidate promotion.')
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))
    for row in summaries:
        if row['dataset'] == 'RGBNT201' and row['state'] == '11':
            print(json.dumps(row))


if __name__ == '__main__':
    main()
