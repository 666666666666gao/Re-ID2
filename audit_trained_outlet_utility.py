"""CPU independent installed-GT recount of all six frozen M1 utility runs."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from audit_shared_identity_states import installed_gt, recount, summarize, changes, METRICS


STAGES = ('deployed', 'base_common', 'M_pre', 'M_post', 'F_pre', 'F_post', 'M_aux', 'F_aux')
RELATIONS = ('R', 'N', 'T', 'RN', 'RT', 'NT', 'RNT')


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'data-root'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--splits', default='splits.json')
    args = parser.parse_args()
    root = Path(args.root)
    terminal = json.loads((root / 'controller_result.json').read_text())
    assert terminal['status'] == 'COMPLETE' and len(terminal['runs']) == 6
    target = root / 'independent_cpu_audit.json'
    assert not target.exists()
    gt = installed_gt(Path(args.data_root), Path(args.splits))
    runs = {}
    for row in terminal['runs']:
        key = row['pooling'] + '/' + row['variant']
        assert key not in runs
        folder = root / row['pooling'] / ('MSVR310_' + row['variant'] + '_s42') / 'full'
        result = json.loads((folder / 'result.json').read_text())
        assert result['status'] == 'COMPLETE' and result['metric_count'] == 392
        assert result['pooling'] == row['pooling'] and result['variant'] == row['variant']
        assert result['dataset'] == 'MSVR310' and result['stages'] == list(STAGES)
        assert result['optimizer_updates'] == result['official_test_uses'] == 0
        assert result['normal_feature_max_error'] == 0 and result['original_fuse_reconstructed_exact']
        assert result['previous_all49_deployed_exact'] and result['state_tensor_versions_unchanged']
        conditions, error = {}, 0.
        with np.load(folder / 'raw_distances.npz') as raw:
            assert len(raw.files) == 392 + len(gt)
            assert all(np.array_equal(raw[name], value) for name, value in gt.items())
            assert len(result['measurements']) == 49
            for condition, reported in result['measurements'].items():
                assert set(reported) == set(STAGES)
                rows = {}
                for stage in STAGES:
                    rows[stage] = recount(raw[condition + '_' + stage], gt)
                    calculated = summarize(rows[stage])
                    maximum = max(abs(calculated[m] - reported[stage][m]) for m in METRICS)
                    error = max(error, maximum)
                    assert maximum < 1e-8
                    cmc = [100 * np.mean([r['first_match'] <= k for r in rows[stage]]) for k in range(1, 51)]
                    assert np.max(np.abs(np.asarray(cmc) - reported[stage]['CMC_1_to_50'])) < 1e-8
                    assert reported[stage]['query_count'] == reported[stage]['valid_queries'] == len(rows[stage])
                    assert reported[stage]['invalid_queries'] == 0 and reported[stage]['gallery_count'] == len(gt['ids'])
                    for group_key in ('identity', 'camera', 'scene'):
                        groups = reported[stage]['groups'][group_key]
                        assert [g['value'] for g in groups] == sorted({r[group_key] for r in rows[stage]})
                        for group in groups:
                            subset = [r for r in rows[stage] if r[group_key] == group['value']]
                            assert len(subset) == group['queries']
                            assert all(abs(summarize(subset)[m] - group[m]) < 1e-8 for m in METRICS)
                    with (folder / condition / (stage + '.csv')).open() as f:
                        exported = list(csv.DictReader(f))
                    assert len(exported) == len(rows[stage])
                    for actual, expected in zip(exported, rows[stage]):
                        assert set(actual) == set(expected)
                        for name, value in expected.items():
                            assert abs(float(actual[name]) - value) < 1e-12 if name in ('AP', 'INP') else actual[name] == str(value)
                conditions[condition] = dict(metrics={s:summarize(r) for s,r in rows.items()},
                    M_projection=changes(rows['M_pre'], rows['M_post']), F_projection=changes(rows['F_pre'], rows['F_post']),
                    M_aux_to_route=changes(rows['M_aux'], rows['M_pre']), F_aux_to_route=changes(rows['F_aux'], rows['F_pre']))
        for availability in ('RNT', 'R', 'N', 'T', 'RN', 'RT', 'NT'):
            with (folder / ('scale_' + availability + '.csv')).open() as f:
                scales = list(csv.DictReader(f))
            assert len(scales) == len(gt['ids'])
            legal_count = 2 ** len(availability) - 1
            eligible = [float(set(relation) <= set(availability)) for relation in RELATIONS]
            for index, values in enumerate(scales):
                assert int(values['row']) == index and values['names'] == gt['names'][index]
                assert values['availability'] == availability and float(values['legal_relations']) == legal_count
                coefficients = [float(values['outlet_coefficient_' + str(i)]) for i in range(7)]
                anchors = [float(values['anchor_' + str(i)]) for i in range(7)]
                masses = [float(values['relation_mass_' + str(i)]) for i in range(7)]
                assert all(math.isfinite(v) and v >= 0 for v in coefficients + anchors + masses)
                assert abs(sum(masses) - 1) < 1e-6
                expected_sum = legal_count / 7 if row['pooling'] == 'original_mean' else 1
                assert abs(sum(coefficients) - expected_sum) < 1e-6
                for i in range(7):
                    assert float(values['eligible_' + str(i)]) == eligible[i]
                    if eligible[i] == 0:
                        assert anchors[i] == masses[i] == coefficients[i] == 0
                    if row['pooling'] in ('original_mean', 'eligible_mean'):
                        denominator = 7 if row['pooling'] == 'original_mean' else legal_count
                        assert abs(coefficients[i] - eligible[i] / denominator) < 1e-6
                weighted = sum(a * m * legal_count for a, m in zip(anchors, masses))
                readout = sum(a * m * legal_count * c for a, m, c in zip(anchors, masses, coefficients))
                assert np.isclose(float(values['weighted_anchor']), weighted, rtol=1e-6, atol=1e-6)
                assert np.isclose(float(values['weighted_anchor_readout']), readout, rtol=1e-6, atol=1e-6)
                assert np.isclose(float(values['weighted_anchor_relative_to_shared']),
                    readout / float(values['shared_norm']), rtol=1e-6, atol=1e-6)
        runs[key] = dict(selected_epoch=result['selected_epoch'], cases=392,
            max_sixmetric_error_pp=error, conditions=conditions)
    assert set(runs) == {p+'/'+v for p in ('original_mean','eligible_mean','seed_query') for v in ('axis_shared','frequency_shared')}
    target.write_text(json.dumps(dict(status='PASS', cases=2352, perquery_count=2352*len(gt['query_indices']),
        runs=runs, independent_installed_GT_recount=True, optimizer_updates=0, official_test_uses=0,
        limits='Frozen single-split/seed diagnostics, not retrained superiority or causal evidence.'),indent=2)+'\n')
    print('TRAINED_OUTLET_UTILITY_CPU_PASS',json.dumps(dict(cases=2352,perquery_count=2352*len(gt['query_indices']))),flush=True)


if __name__ == '__main__':
    main()
