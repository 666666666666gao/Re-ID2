"""CPU installed-GT recount and independent cross-stage matrix wiring check."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_shared_identity_states import installed_gt, recount, summarize, changes, METRICS


PAIRS = {
    'common_common': ('base_common', 'base_common'),
    'F_pre_F_pre': ('F_pre', 'F_pre'),
    'F_post_F_post': ('F_post', 'F_post'),
    'F_pre_common': ('F_pre', 'base_common'),
    'common_F_pre': ('base_common', 'F_pre'),
    'F_post_common': ('F_post', 'base_common'),
    'common_F_post': ('base_common', 'F_post'),
}
SETS = ('RNT', 'R', 'N', 'T', 'RN', 'RT', 'NT')


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'data-root'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--splits', default='splits.json')
    args = parser.parse_args()
    root = Path(args.root)
    terminal = json.loads((root / 'controller_result.json').read_text())
    assert terminal['status'] == 'COMPLETE' and len(terminal['runs']) == 3
    target = root / 'independent_cpu_audit.json'
    assert not target.exists()
    gt = installed_gt(Path(args.data_root), Path(args.splits))
    expected_conditions = {'q_' + q + '_g_' + g for q in SETS for g in SETS}
    runs = {}
    for row in terminal['runs']:
        variant = row['variant']
        assert variant in ('axis_shared', 'frequency_shared', 'twins_shared') and variant not in runs
        assert row['exit_code'] == 0 and row['gpu'] in (2, 3)
        folder = root / variant / 'full'
        result = json.loads((folder / 'result.json').read_text())
        assert result['status'] == 'COMPLETE' and result['metric_count'] == 343
        assert result['variant'] == variant and result['pairs'] == {k: list(v) for k, v in PAIRS.items()}
        assert result['optimizer_updates'] == result['new_weights'] == result['official_test_uses'] == 0
        assert result['normal_feature_max_error'] == 0 and result['original_fuse_reconstructed_exact']
        assert result['state_tensor_versions_unchanged'] and result['previous_same_coordinate_metrics_perquery_exact']
        assert set(result['measurements']) == expected_conditions
        conditions, matrix_error, metric_error = {}, 0., 0.
        with np.load(folder / 'raw_distances.npz') as raw, np.load(folder / 'normalized_features.npz') as features:
            assert set(raw.files) == {c + '_' + p for c in expected_conditions for p in PAIRS} | set(gt)
            assert set(features.files) == {a + '_' + s for a in SETS for s in ('base_common', 'F_pre', 'F_post')} | set(gt)
            assert all(np.array_equal(archive[name], value) for archive in (raw, features) for name, value in gt.items())
            for availability in SETS:
                for stage in ('base_common', 'F_pre', 'F_post'):
                    values = features[availability + '_' + stage]
                    assert values.shape == (len(gt['ids']), 512) and np.isfinite(values).all()
                    assert np.allclose(np.linalg.norm(values, axis=1), 1., atol=1e-6, rtol=0)
            for condition, reported in result['measurements'].items():
                assert set(reported) == set(PAIRS)
                qset, gset = condition.split('_')[1], condition.split('_')[3]
                rows = {}
                for pair, (qstage, gstage) in PAIRS.items():
                    q = features[qset + '_' + qstage][gt['query_indices']]
                    g = features[gset + '_' + gstage]
                    independently_rebuilt = (q * q).sum(1)[:, None] + (g * g).sum(1)[None] - 2 * (q @ g.T)
                    distances = raw[condition + '_' + pair]
                    assert distances.shape == independently_rebuilt.shape and np.isfinite(distances).all()
                    error = float(np.abs(independently_rebuilt - distances).max())
                    matrix_error = max(matrix_error, error)
                    assert error < 2e-6
                    rows[pair] = recount(distances, gt)
                    calculated = summarize(rows[pair])
                    metric_error = max(metric_error, max(abs(calculated[m] - reported[pair][m]) for m in METRICS))
                    assert all(abs(calculated[m] - reported[pair][m]) < 1e-8 for m in METRICS)
                    cmc = [100 * np.mean([r['first_match'] <= k for r in rows[pair]]) for k in range(1, 51)]
                    assert np.max(np.abs(np.asarray(cmc) - reported[pair]['CMC_1_to_50'])) < 1e-8
                    assert reported[pair]['query_count'] == reported[pair]['valid_queries'] == len(rows[pair])
                    assert reported[pair]['invalid_queries'] == 0 and reported[pair]['gallery_count'] == len(gt['ids'])
                    for key in ('identity', 'camera', 'scene'):
                        groups = reported[pair]['groups'][key]
                        assert [g['value'] for g in groups] == sorted({r[key] for r in rows[pair]})
                        for group in groups:
                            subset = [r for r in rows[pair] if r[key] == group['value']]
                            assert len(subset) == group['queries']
                            assert all(abs(summarize(subset)[m] - group[m]) < 1e-8 for m in METRICS)
                    with (folder / condition / (pair + '.csv')).open() as f:
                        exported = list(csv.DictReader(f))
                    assert len(exported) == len(rows[pair])
                    for actual, expected in zip(exported, rows[pair]):
                        assert set(actual) == set(expected)
                        for name, value in expected.items():
                            assert abs(float(actual[name]) - value) < 1e-12 if name in ('AP', 'INP') else actual[name] == str(value)
                conditions[condition] = dict(metrics={p: summarize(r) for p, r in rows.items()},
                    F_post_query_vs_common_query=changes(rows['common_common'], rows['F_post_common']),
                    F_post_gallery_vs_common_gallery=changes(rows['common_common'], rows['common_F_post']))
        runs[variant] = dict(selected_epoch=result['selected_epoch'], conditions=conditions,
            max_cross_distance_reconstruction_error=matrix_error, max_sixmetric_recount_error_pp=metric_error)
    assert len(runs) == 3
    target.write_text(json.dumps(dict(status='PASS', cases=1029, perquery_count=1029 * len(gt['query_indices']),
        runs=runs, independent_installed_GT_recount=True, matrix_reconstruction_tolerance=2e-6,
        optimizer_updates=0, new_weights=0, official_test_uses=0,
        limits='Numerical matrix agreement verifies stage wiring; ranking is independently recounted from exact exported distances. Cross-stage compatibility is diagnostic, not a deployed method.'), indent=2) + '\n')
    print('CROSS_IDENTITY_COORDINATES_CPU_PASS', flush=True)


if __name__ == '__main__':
    main()
