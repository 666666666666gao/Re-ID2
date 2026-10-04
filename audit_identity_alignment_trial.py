"""Reuse installed-GT state/utility/cross audits and verify actual M2b sampling."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

import audit_cross_identity_coordinates as cross_audit
import audit_frequency_relation_trial as parent_audit


def main():
    parser = argparse.ArgumentParser()
    for name in ('root','reference-root','data-root'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    root, reference = Path(args.root), Path(args.reference_root)
    output = root/'independent_cpu_audit.json'
    parent_output = root/'states_utility_cpu_audit.json'
    assert not output.exists() and not parent_output.exists()
    alignment = {}
    for variant in ('axis_shared','frequency_shared','twins_shared'):
        run = root/'original_mean/development'/('MSVR310_'+variant+'_s42')
        trained = json.loads((run/'result.json').read_text())
        assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50
        assert trained['arguments']['alignment_weight'] == .1 and trained['arguments']['alignment_temperature'] == .07
        prior = json.loads((reference/'original_mean/development'/run.name/'result.json').read_text())
        for key in ('parameters','trainable_parameters','descriptor_dim','fit_records','dev_records','dev_queries'):
            assert trained[key] == prior[key]
        orders = [json.loads(line) for line in (run/'batch_orders.jsonl').read_text().splitlines()]
        for row in orders:
            names = row['names']
            positive_counts = [sum(q[:4] == g[:4] and q != g for g in names) for q in names]
            valid = sum(n > 0 for n in positive_counts)
            assert row['identity_alignment_valid_anchors'] == valid
            assert row['identity_alignment_excluded_anchors'] == len(names)-valid
            assert row['identity_alignment_positive_pairs'] == sum(positive_counts)
            assert row['identity_alignment_positive_observations_distinct'] is True
            assert row['identity_alignment_reference_requires_grad'] is False
            assert row['identity_alignment_weight'] == .1 and row['identity_alignment_temperature'] == .07
            assert valid > 0 and np.isfinite(row['identity_alignment_raw'])
        alignment[variant] = dict(raw_first=orders[0]['identity_alignment_raw'],raw_last=orders[-1]['identity_alignment_raw'],
            valid_anchor_occurrences=sum(r['identity_alignment_valid_anchors'] for r in orders),
            excluded_anchor_occurrences=sum(r['identity_alignment_excluded_anchors'] for r in orders),
            sampling_and_active_parameters_match_original_M2=True)
    parent_audit.main()
    output.rename(parent_output)
    parent = json.loads(parent_output.read_text())
    assert parent['status'] == 'PASS' and parent['cases'] == 2058 and parent['perquery_count'] == 432180
    sys.argv = [sys.argv[0], '--root', str(root/'cross_coordinates'), '--data-root', args.data_root]
    cross_audit.main()
    cross = json.loads((root/'cross_coordinates/independent_cpu_audit.json').read_text())
    assert cross['status'] == 'PASS' and cross['cases'] == 1029 and cross['perquery_count'] == 216090
    for variant,run in parent['runs'].items():
        run['alignment_training'] = alignment[variant]
        run['cross_coordinates'] = cross['runs'][variant]
    output.write_text(json.dumps(dict(status='PASS',cases=3087,perquery_count=648270,runs=parent['runs'],
        states_utility_cases=2058,cross_coordinate_cases=1029,independent_installed_GT_recount=True,
        optimizer_updates=0,official_test_uses=0,
        limits='One MSVR310 development split/seed; repeated queries and identity alignment losses do not prove final method superiority.'),indent=2)+'\n')
    print('IDENTITY_ALIGNMENT_ALL3087_CPU_PASS',flush=True)


if __name__ == '__main__':
    main()
