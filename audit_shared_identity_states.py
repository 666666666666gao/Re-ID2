"""CPU-only independent GT recount of frozen V11 distances and query changes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


VARIANTS = ('axis_shared', 'frequency_shared', 'twins_shared', 'demo_shared')
STATES = ('00', '10', '01', '11', 'base_private', 'base_shared')
METRICS = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')


def installed_gt(data_root, splits):
    dev_ids = set(json.loads(splits.read_text())['MSVR310']['dev_ids'])
    records = []
    for path in (data_root / 'MSVR310/bounding_box_train').glob('*/vis/*'):
        name = path.name
        pid, camera, scene = int(name[:4]), int(name[11]), int(name[6:9])
        assert 0 <= camera <= 7
        assert (path.parent.parent / 'ni' / name).is_file()
        assert (path.parent.parent / 'th' / name).is_file()
        if pid in dev_ids:
            records.append((str(path), pid, camera, scene, name))
    records.sort(key=lambda row: row[0])
    assert records
    ids, cameras, scenes, names = (np.asarray([r[i] for r in records]) for i in range(1, 5))
    query = np.asarray([i for i in range(len(records)) if ((ids == ids[i]) & (scenes != scenes[i])).any()])
    return dict(ids=ids, cameras=cameras, scenes=scenes, names=names, query_indices=query)


def recount(distances, gt):
    ids, scenes, query = (gt[key] for key in ('ids', 'scenes', 'query_indices'))
    assert distances.shape == (len(query), len(ids)) and np.isfinite(distances).all()
    rows = []
    gallery = np.arange(len(ids))
    for ordinal, row in enumerate(query):
        # Independent lexicographic order: original gallery row breaks distance ties.
        order = np.lexsort((gallery, distances[ordinal]))
        keep = order[~((ids[order] == ids[row]) & (scenes[order] == scenes[row]))]
        ranks = np.nonzero(ids[keep] == ids[row])[0] + 1
        assert len(ranks)
        ap = float(np.mean(np.arange(1, len(ranks) + 1) / ranks))
        rows.append(dict(query_index=ordinal, name=str(gt['names'][row]), identity=int(ids[row]),
                         camera=int(gt['cameras'][row]), scene=int(scenes[row]), valid=True,
                         relevant_gallery=len(ranks), kept_gallery=len(keep), AP=ap,
                         INP=float(len(ranks) / ranks[-1]), first_match=int(ranks[0]), last_match=int(ranks[-1]),
                         **{f'Rank-{k}': int(ranks[0] <= k) for k in (1, 5, 10, 20)}))
    return rows


def summarize(rows):
    return dict(mAP=100 * float(np.mean([r['AP'] for r in rows])),
                mINP=100 * float(np.mean([r['INP'] for r in rows])),
                **{f'Rank-{k}': 100 * float(np.mean([r[f'Rank-{k}'] for r in rows])) for k in (1, 5, 10, 20)})


def changes(before, after):
    assert len(before) == len(after)
    delta = np.asarray([b['AP'] - a['AP'] for a, b in zip(before, after)])
    harm = sum(a['Rank-1'] == 1 and b['Rank-1'] == 0 for a, b in zip(before, after))
    rescue = sum(a['Rank-1'] == 0 and b['Rank-1'] == 1 for a, b in zip(before, after))
    return dict(delta_pp={key: summarize(after)[key] - summarize(before)[key] for key in METRICS},
                Rank1_harm_queries=harm, Rank1_rescue_queries=rescue,
                AP_improved_queries=int((delta > 0).sum()), AP_worsened_queries=int((delta < 0).sum()),
                AP_unchanged_queries=int((delta == 0).sum()), queries=len(before))


def audit_variant(root, variant, gt):
    full = root / ('MSVR310_' + variant + '_s42') / 'full'
    result = json.loads((full / 'result.json').read_text())
    assert result['status'] == 'COMPLETE' and result['metric_count'] == 294
    assert len(result['measurements']) == 49
    assert result['optimizer_updates'] == result['official_test_uses'] == 0
    assert result['variant'] == variant and result['dataset'] == 'MSVR310'
    assert result['normal_feature_max_error'] == 0 and result['previous_all49_full11_sixmetric_and_perquery_exact']
    with np.load(full / 'raw_distances.npz') as raw:
        assert len(raw.files) == 294 + len(gt)
        assert all(np.array_equal(raw[key], values) for key, values in gt.items())
        conditions = {}
        max_error = 0.
        query = gt['query_indices']
        for condition, reported in result['measurements'].items():
            reconstructed = {}
            for state in STATES:
                distances = raw[condition + '_' + state]
                rows = recount(distances, gt)
                values = summarize(rows)
                reconstructed[state] = rows
                old = reported[state]
                error = max(abs(values[key] - old[key]) for key in METRICS)
                max_error = max(max_error, error)
                assert error < 1e-8, (variant, condition, state, error)
                cmc = [100 * float(np.mean([r['first_match'] <= k for r in rows])) for k in range(1, 51)]
                assert np.max(np.abs(np.asarray(cmc) - old['CMC_1_to_50'])) < 1e-8
                assert old['query_count'] == old['valid_queries'] == len(rows)
                assert old['invalid_queries'] == 0 and old['gallery_count'] == len(gt['ids'])
                for key in ('camera', 'scene', 'identity'):
                    groups = old['groups'][key]
                    assert [item['value'] for item in groups] == sorted({r[key] for r in rows})
                    for group in groups:
                        members = [r for r in rows if r[key] == group['value']]
                        assert len(members) == group['queries']
                        assert all(abs(summarize(members)[m] - group[m]) < 1e-8 for m in METRICS)
                with (full / condition / ('state_' + state + '.csv')).open() as handle:
                    exported = list(csv.DictReader(handle))
                assert len(exported) == len(rows)
                for actual, expected in zip(exported, rows):
                    assert set(actual) == set(expected)
                    for key, value in expected.items():
                        if key in ('AP', 'INP'):
                            assert abs(float(actual[key]) - value) < 1e-12
                        else:
                            assert actual[key] == str(value), (variant, condition, state, key)
                _, qset, _, gset = condition.split('_')
                if state == 'base_private' and set(qset).isdisjoint(set(gset)):
                    assert np.all(distances == 2), (variant, condition, 'private coordinates must have zero cross-source dot product')
            with (full / condition / 'contributions.csv').open() as handle:
                contribution_rows = list(csv.DictReader(handle))
            assert len(contribution_rows) == len(query)
            targets = []
            for qrow, record in zip(query, contribution_rows):
                assert int(record['query_index']) == qrow
                pos, neg = (int(record[key]) for key in ('gallery_positive_index', 'gallery_negative_index'))
                assert gt['ids'][pos] == gt['ids'][qrow] and gt['scenes'][pos] != gt['scenes'][qrow]
                assert gt['ids'][neg] != gt['ids'][qrow]
                scores = {key: float(record['R' + key]) for key in ('00', '10', '01', '11')}
                expected = (scores['11'] - scores['01'], scores['11'] - scores['10'],
                            scores['11'] - scores['10'] - scores['01'] + scores['00'])
                actual = [float(record[key]) for key in ('delta_M_given_F', 'delta_F_given_M', 'empirical_interaction')]
                # Stored targets use float32 subtraction; this checks serialized arithmetic.
                assert max(abs(a - b) for a, b in zip(actual, expected)) < 1e-7
                targets.append(actual)
            assert np.max(np.abs(np.mean(targets, 0) - result['contributions'][condition]['target_mean'])) < 1e-7
            assert np.max(np.abs(np.std(targets, axis=0) - result['contributions'][condition]['target_std'])) < 1e-7
            conditions[condition] = dict(metrics={state: summarize(rows) for state, rows in reconstructed.items()},
                full_vs_base=changes(reconstructed['00'], reconstructed['11']),
                M_vs_base=changes(reconstructed['00'], reconstructed['10']),
                F_vs_base=changes(reconstructed['00'], reconstructed['01']),
                full_vs_M=changes(reconstructed['10'], reconstructed['11']),
                full_vs_F=changes(reconstructed['01'], reconstructed['11']))
    return dict(selected_epoch=result['selected_epoch'], conditions=conditions,
                max_sixmetric_recount_error_pp=max_error, cases=294,
                raw_archive_bytes=(full / 'raw_distances.npz').stat().st_size,
                residual_scale=result['residual_scale'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--splits', default='splits.json')
    args = parser.parse_args()
    root = Path(args.root)
    assert json.loads((root / 'controller_result.json').read_text())['status'] == 'COMPLETE'
    output = root / 'independent_cpu_audit.json'
    assert not output.exists()
    gt = installed_gt(Path(args.data_root), Path(args.splits))
    runs = {variant: audit_variant(root, variant, gt) for variant in VARIANTS}
    report = dict(status='PASS', source='installed training filenames and fixed dev IDs; independent lexicographic ranking of exact stored distances',
                  same_identity_same_scene_excluded=True, cases=1176,
                  perquery_count=1176 * len(gt['query_indices']), queries=len(gt['query_indices']), gallery=len(gt['ids']),
                  sixmetrics_CMC50_groups_perquery_recount=True, contribution_GT_and_arithmetic=True,
                  optimizer_updates=0, official_test_uses=0, runs=runs,
                  limits='single dataset and seed; frozen intervention evidence; shared/private blocks are diagnostic; no claim of causal or information-theoretic synergy')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('INDEPENDENT_RAW_CPU_AUDIT_PASS', json.dumps(dict(cases=1176,queries=report['queries'],gallery=report['gallery'])), flush=True)


if __name__ == '__main__':
    main()
