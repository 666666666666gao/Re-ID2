"""Verify completed epoch selection, actual update accounting and paired sampling."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METRICS = ('mAP', 'Rank-1', 'Rank-5', 'Rank-10')


def analyze():
    base = ROOT / 'results/first_comparison'
    results, orders, rows = {}, {}, []
    for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
        for variant in ('demo', 'ordinary', 'dual'):
            name = f'{dataset}_{variant}_s42'
            folder = base / name
            result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
            exit_record = json.loads((folder / 'exit.json').read_text(encoding='utf-8'))
            assert result['status'] == 'COMPLETE' and exit_record['exit_code'] == 0
            with (folder / 'epochs.csv').open(encoding='utf-8', newline='') as source:
                epochs = list(csv.DictReader(source))
            assert [int(e['epoch']) for e in epochs] == list(range(1, 51))
            assert all(math.isfinite(float(e[k])) for e in epochs for k in (*METRICS, 'loss'))
            selected = max(epochs, key=lambda e: float(e['mAP']))  # first row on ties
            assert int(selected['epoch']) == result['best']['epoch']
            for metric in METRICS:
                assert abs(float(selected[metric]) - result['best'][metric]) < 1e-8
                assert abs(result['strict_reload'][metric] - result['best'][metric]) < 1e-8
            with (folder / 'batch_orders.jsonl').open(encoding='utf-8') as source:
                batches = [json.loads(line) for line in source]
            assert len(batches) == result['steps']
            assert [b['step'] for b in batches] == list(range(1, len(batches) + 1))
            assert all(len(b['names']) == 64 for b in batches)
            updates = sum(b['optimizer_updated'] for b in batches)
            skips = len(batches) - updates
            assert updates == result['optimizer_steps'] and skips == result['amp_skipped_steps']
            cumulative, cumulative_updates = 0, 0
            for epoch in epochs:
                current = [b for b in batches if b['epoch'] == int(epoch['epoch'])]
                assert current
                cumulative += len(current)
                cumulative_updates += sum(b['optimizer_updated'] for b in current)
                assert int(epoch['steps']) == cumulative
                assert int(epoch['optimizer_steps']) == cumulative_updates
                assert int(epoch['amp_skipped_steps']) == cumulative - cumulative_updates
            orders[(dataset, variant)] = [(b['epoch'], b['step'], b['names']) for b in batches]
            results[(dataset, variant)] = result
            rows.append({'dataset': dataset, 'variant': variant, 'parameters': result['parameters'],
                         'descriptor_dim': result['strict_reload']['descriptor_dim'], 'best_epoch': result['best']['epoch'],
                         **{m: result['best'][m] for m in METRICS}, 'batch_attempts': len(batches),
                         'optimizer_updates': updates, 'amp_skips': skips})
    comparisons = []
    for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
        demo, ordinary, dual = [results[(dataset, v)] for v in ('demo', 'ordinary', 'dual')]
        assert orders[(dataset, 'demo')] == orders[(dataset, 'ordinary')] == orders[(dataset, 'dual')]
        assert ordinary['parameters'] == dual['parameters']
        assert ordinary['trainable_parameters'] == dual['trainable_parameters']
        assert ordinary['strict_reload']['descriptor_dim'] == dual['strict_reload']['descriptor_dim'] == 5632
        proof = json.loads((ROOT / f'results/preflight/checks/{dataset}_pair/pair.json').read_text())
        assert proof['all_initial_state_tensors_equal'] and proof['parameters_each'] == dual['parameters']
        comparisons.append({'dataset': dataset,
                            'dual_minus_ordinary': {m: dual['best'][m] - ordinary['best'][m] for m in METRICS},
                            'ordinary_minus_demo': {m: ordinary['best'][m] - demo['best'][m] for m in METRICS},
                            'dual_minus_demo': {m: dual['best'][m] - demo['best'][m] for m in METRICS},
                            'same_full_batch_sequence': True, 'matched_parameters_and_dimension': True,
                            'optimizer_updates_ordinary': ordinary['optimizer_steps'], 'optimizer_updates_dual': dual['optimizer_steps']})
    report = {'verification': 'PASS', 'scope': 'Nine complete seed42 identity-heldout development experiments; no official test',
              'rows': rows, 'comparisons': comparisons,
              'total_epochs': 450, 'batch_attempts': sum(r['batch_attempts'] for r in rows),
              'optimizer_updates': sum(r['optimizer_updates'] for r in rows), 'amp_skips': sum(r['amp_skips'] for r in rows),
              'claims_boundary': 'Not a multi-seed stability claim, causal attribution to psi, contribution-supervision result or SOTA.'}
    (base / 'verified_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (base / 'verified_metrics.csv').open('w', encoding='utf-8', newline='') as table:
        writer = csv.DictWriter(table, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    handoff = ROOT / 'docs/实验交接.md'
    original = handoff.read_text(encoding='utf-8')
    original += '\n### 完整结果确定性核验\n\n'
    original += f"九端450epoch全部完成且exit0。每端best是50行CSV中开发mAP最高、并列最早的epoch，四项指标与strict reload一致。三个数据集各三模型的完整batch文件名及顺序完全一致，ordinary/dual参数与5632D描述子匹配。总batch尝试{report['batch_attempts']}，成功optimizer更新{report['optimizer_updates']}，AMP跳过{report['amp_skips']}。逐端明细见results/first_comparison/verified_metrics.csv；核验见verified_comparison.json。\n\n"
    for r in rows:
        original += f"- {r['dataset']} {r['variant']}：best{r['best_epoch']}，mAP/Rank-1/5/10={r['mAP']:.4f}/{r['Rank-1']:.4f}/{r['Rank-5']:.4f}/{r['Rank-10']:.4f}，optimizer {r['optimizer_updates']}/{r['batch_attempts']}，AMP跳过{r['amp_skips']}。\n"
    handoff.write_bytes(original.encode('utf-8'))
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    analyze()
