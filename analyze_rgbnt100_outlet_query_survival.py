"""CPU query overlap of already audited AF outlets; no new inference or selector."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / 'results/r201af_AD_selected_outlets_20261012/RGBNT100/full'
OUTPUT = PROJECT / 'results/rgbnt100_AF_outlet_query_survival_CPU_20261012'
METRICS = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
IDENTITY = ('query_index', 'name', 'identity', 'camera', 'scene')


def main():
    assert not OUTPUT.exists()
    manifest = json.loads((PROJECT / 'results/r201ad_RGB_precision_matched_20261012/RGBNT100/training/'
                           'RGBNT100_r201ad_hard_axis_shared_s42/official_split_manifest.json').read_text())
    summaries, transitions, flags, inputs = [], [], [], {}
    for variant in ('frequency_shared', 'axis_shared'):
        folder = SOURCE / ('RGBNT100_r201ad_hard_' + variant + '_s42')
        result = json.loads((folder / 'result.json').read_text())
        assert result['status'] == 'COMPLETE' and result['normal_features_and_distance_exact']
        assert result['optimizer_updates'] == result['new_weights'] == 0
        tables, all_metrics = {}, {}
        for stage in result['stages']:
            path = folder / (stage + '.csv')
            with path.open(encoding='utf-8', newline='') as file:
                rows = list(csv.DictReader(file))
            assert len(rows) == len(manifest['query']) == 1715
            assert all((row['name'], int(row['identity']), int(row['camera']), int(row['scene'])) ==
                       tuple(official[k] for k in ('name', 'identity', 'camera', 'scene'))
                       for row, official in zip(rows, manifest['query']))
            assert all(row['valid'] == 'True' for row in rows)
            measures = {'mAP': 100 * sum(float(r['AP']) for r in rows) / len(rows),
                        'mINP': 100 * sum(float(r['INP']) for r in rows) / len(rows)}
            measures.update({key: 100 * sum(int(r[key]) for r in rows) / len(rows)
                             for key in METRICS[2:]})
            assert all(abs(measures[key] - result['measurements'][stage][key]) < 1e-8
                       for key in METRICS)
            tables[stage] = rows
            all_metrics[stage] = measures
            inputs[path.relative_to(PROJECT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        base = tables['base00']
        bad = {i for i, row in enumerate(base) if row['Rank-1'] == '0'}
        assert len(bad) == 58
        correct = {stage: {i for i in bad if rows[i]['Rank-1'] == '1'}
                   for stage, rows in tables.items()}
        for stage, rows in tables.items():
            summaries.append(dict(variant=variant, stage=stage, dimension=result['dimensions'][stage],
                original_errors=58, original_error_correct=len(correct[stage]),
                original_correct_harmed=sum(rows[i]['Rank-1'] == '0' for i in range(1715) if i not in bad),
                corrected_cameras=dict(Counter(base[i]['camera'] for i in correct[stage])),
                corrected_identities=dict(Counter(base[i]['identity'] for i in correct[stage])),
                **all_metrics[stage]))
        for expert in ('M', 'F'):
            names = [expert + suffix for suffix in ('_projection_input', '_projection_output',
                     '_unit_direction', '_projected_auxiliary')] + ['normal11']
            for left, right in zip(names, names[1:]):
                transitions.append(dict(variant=variant, expert=expert, left=left, right=right,
                    left_dimension=result['dimensions'][left], right_dimension=result['dimensions'][right],
                    left_correct=len(correct[left]), right_correct=len(correct[right]),
                    retained=len(correct[left] & correct[right]), lost=len(correct[left] - correct[right]),
                    added=len(correct[right] - correct[left])))
        for i in sorted(bad):
            flag = dict(variant=variant, **{key: base[i][key] for key in IDENTITY})
            for stage, rows in tables.items():
                flag[stage + '_correct'] = int(rows[i]['Rank-1'])
                flag[stage + '_first_positive_rank'] = int(rows[i]['first_match'])
                flag[stage + '_AP'] = float(rows[i]['AP'])
            flags.append(flag)
        inputs[(folder / 'result.json').relative_to(PROJECT).as_posix()] = hashlib.sha256(
            (folder / 'result.json').read_bytes()).hexdigest()
    OUTPUT.mkdir()
    for name, rows in (('stage_counts.csv', summaries), ('transitions.csv', transitions),
                       ('original_error_flags.csv', flags)):
        with (OUTPUT / name).open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = dict(status='ACTUAL_AF_RGBNT100_OUTLET_QUERY_SURVIVAL_CPU_COMPLETE',
        models=2, outlets=30, query_rows_checked=3430, export_csv_rows_checked=51450,
        original_error_rows=116,
        new_neural_calls=0, new_optimizer_updates=0, new_weights=0, sources_sha256=inputs,
        summaries=summaries, transitions=transitions,
        limits='Reuse previously installed-GT audited fixed AF CSVs, not new raw-distance or NN recomputation. '
               'Individual outlet retrievals differ in dimensions/normalization/role; overlap is not a causal '
               'information-loss proof or a deployable GT selector. All58 errors kept, all1715 metadata/six '
               'recounts checked. Neither this diagnosis nor interim AG counts prove final +1 performance.')
    (OUTPUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'models', 'outlets', 'query_rows_checked',
          'original_error_rows', 'new_neural_calls', 'new_optimizer_updates')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
