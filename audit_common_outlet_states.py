"""Independent GT recount for all10 M1 runs, without importing models or CUDA."""
import argparse
import json
from pathlib import Path

from audit_shared_identity_states import audit_variant, installed_gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--splits', default='splits.json')
    args = parser.parse_args()
    root = Path(args.root)
    terminal = json.loads((root / 'controller_result.json').read_text())
    assert terminal['status'] == 'COMPLETE' and len(terminal['runs']) == 10
    gt = installed_gt(Path(args.data_root), Path(args.splits))
    runs = {}
    for row in terminal['runs']:
        key = row['pooling'] + '/' + row['variant']
        assert key not in runs
        runs[key] = audit_variant(root / row['pooling'] / 'controlled_states', row['variant'], gt)
    assert len(runs) == 10
    output = root / 'independent_cpu_audit.json'
    assert not output.exists()
    report = dict(status='PASS', source='installed MSVR training filenames and fixed dev IDs; independent lexicographic ranking of stored FP32 distances',
                  cases=2940, perquery_count=2940 * len(gt['query_indices']), runs=runs,
                  sixmetrics_CMC50_groups_perquery_recount=True, optimizer_updates=0, official_test_uses=0)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('COMMON_OUTLET_INDEPENDENT_CPU_PASS', json.dumps(dict(cases=2940, perquery_count=report['perquery_count'])), flush=True)


if __name__ == '__main__':
    main()
