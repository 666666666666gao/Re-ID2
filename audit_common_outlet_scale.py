"""CPU independent installed-GT recount and alignment of frozen outlet stages."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_shared_identity_states import installed_gt, recount, summarize, changes, METRICS


VARIANTS = ('axis_shared', 'frequency_shared', 'twins_shared')
STAGES = ('deployed', 'base_common', 'M_pre', 'M_post', 'F_pre', 'F_post')


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'data-root'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--splits', default='splits.json')
    args = parser.parse_args()
    root = Path(args.root)
    assert json.loads((root / 'controller_result.json').read_text())['status'] == 'COMPLETE'
    output = root / 'independent_cpu_audit.json'
    assert not output.exists()
    gt = installed_gt(Path(args.data_root), Path(args.splits))
    runs = {}
    for variant in VARIANTS:
        folder = root / ('MSVR310_' + variant + '_s42') / 'full'
        result = json.loads((folder / 'result.json').read_text())
        assert result['status'] == 'COMPLETE' and result['metric_count'] == 294
        assert result['optimizer_updates'] == result['official_test_uses'] == 0
        conditions, error = {}, 0.
        with np.load(folder / 'raw_distances.npz') as raw:
            assert len(raw.files) == 294 + len(gt)
            assert all(np.array_equal(raw[key], values) for key, values in gt.items())
            for condition, reported in result['measurements'].items():
                rows = {}
                for stage in STAGES:
                    rows[stage] = recount(raw[condition + '_' + stage], gt)
                    calculated = summarize(rows[stage])
                    maximum = max(abs(calculated[key] - reported[stage][key]) for key in METRICS)
                    error = max(error, maximum)
                    assert maximum < 1e-8
                    expected_cmc = [100 * np.mean([r['first_match'] <= k for r in rows[stage]]) for k in range(1, 51)]
                    assert np.max(np.abs(np.asarray(expected_cmc) - reported[stage]['CMC_1_to_50'])) < 1e-8
                    for key in ('identity', 'camera', 'scene'):
                        for group in reported[stage]['groups'][key]:
                            subset = [r for r in rows[stage] if r[key] == group['value']]
                            assert len(subset) == group['queries']
                            assert all(abs(summarize(subset)[m] - group[m]) < 1e-8 for m in METRICS)
                    with (folder / condition / (stage + '.csv')).open() as handle:
                        exported = list(csv.DictReader(handle))
                    assert len(exported) == len(rows[stage])
                    for actual, expected in zip(exported, rows[stage]):
                        assert set(actual) == set(expected)
                        for key, value in expected.items():
                            if key in ('AP', 'INP'):
                                assert abs(float(actual[key]) - value) < 1e-12
                            else:
                                assert actual[key] == str(value)
                conditions[condition] = dict(metrics={stage: summarize(value) for stage, value in rows.items()},
                    M_projection=changes(rows['M_pre'], rows['M_post']),
                    F_projection=changes(rows['F_pre'], rows['F_post']))
        for availability in ('RNT', 'R', 'N', 'T', 'RN', 'RT', 'NT'):
            with (folder / ('scale_' + availability + '.csv')).open() as handle:
                scales = list(csv.DictReader(handle))
            assert len(scales) == len(gt['ids'])
            legal_count = 2 ** len(availability) - 1
            for index, row in enumerate(scales):
                assert int(row['row']) == index and row['names'] == gt['names'][index]
                assert row['availability'] == availability and float(row['legal_relations']) == legal_count
                assert abs(float(row['legal_mean_amplitude_factor']) - 7 / legal_count) < 1e-6
                assert sum(float(row['eligible_' + str(i)]) for i in range(7)) == legal_count
                for i in range(7):
                    if float(row['eligible_' + str(i)]) == 0:
                        assert float(row['anchor_' + str(i)]) == float(row['relation_mass_' + str(i)]) == 0
        runs[variant] = dict(selected_epoch=result['selected_epoch'], cases=294, max_sixmetric_error_pp=error,
            conditions=conditions, raw_archive_bytes=(folder / 'raw_distances.npz').stat().st_size)
    output.write_text(json.dumps(dict(status='PASS', cases=882, perquery_count=882 * len(gt['query_indices']),
        queries=len(gt['query_indices']), gallery=len(gt['ids']), runs=runs,
        independent_installed_GT_raw_distance_recount=True, scale_order_and_legal_count_exact=True,
        limits='Frozen same-checkpoint diagnostics, not retrained pooling superiority or causal attribution.'), indent=2) + '\n')
    print('INDEPENDENT_OUTLET_CPU_PASS', json.dumps(dict(cases=882, queries=len(gt['query_indices']))), flush=True)


if __name__ == '__main__':
    main()
