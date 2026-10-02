"""Collect all 27 runs, verify budgets, publish complete three-seed evaluation."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import time

from collect_results import PROJECT, HOSTS, command, copy_file, remote_python, sync_handoff
from launch_runs import SCHEDULE as FIRST
from launch_repeats import SCHEDULE as REPEATS

BASE = PROJECT / 'results/full_suite'
METRICS = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
JOBS = []
for host in FIRST:
    for gpu in FIRST[host]:
        JOBS.extend((host, gpu, 'dynamic_amp_comparison', d, v, 42) for d, v in FIRST[host][gpu])
        JOBS.extend((host, gpu, 'three_seed_extension', d, v, s) for d, v, s in REPEATS[host][gpu][1])


def read_host(host):
    root, _ = HOSTS[host]
    code = f'''import json
from pathlib import Path
root=Path({root!r})
records={{}}
for campaign in ['dynamic_amp_comparison','three_seed_extension']:
    for folder in sorted((root/'runs'/campaign).glob('*_s*')):
        record={{}}
        for key in ['launch','run','status','best','exit','result','evaluation_launch','evaluation_exit']:
            p=folder/(key+'.json')
            if p.exists(): record[key]=json.loads(p.read_text())
        p=folder/'full_evaluation/metrics.json'
        if p.exists(): record['full_metrics']=json.loads(p.read_text())
        record['training_text_files']=[f for f in ['epochs.csv','batch_orders.jsonl'] if (folder/f).exists()]
        records[folder.name]=record
print(json.dumps(records))
'''
    return json.loads(remote_python(host, code))


def collect():
    BASE.mkdir(parents=True, exist_ok=True)
    snapshots = {host: read_host(host) for host in HOSTS}
    now = datetime.now().astimezone().isoformat(timespec='seconds')
    (BASE / 'snapshot.json').write_text(json.dumps({'time': now, 'hosts': snapshots}, indent=2), encoding='utf-8')
    rows, failures = [], []
    for host, gpu, campaign, dataset, variant, seed in JOBS:
        name = f'{dataset}_{variant}_s{seed}'
        record = snapshots[host].get(name, {})
        train_exit = record.get('exit', {}).get('exit_code')
        eval_exit = record.get('evaluation_exit', {}).get('exit_code')
        done = record.get('result', {}).get('status') == 'COMPLETE' and record.get('full_metrics') and eval_exit == 0 and train_exit == 0
        failed = train_exit not in (None, 0) or eval_exit not in (None, 0)
        phase = 'FAILED' if failed else ('COMPLETE' if done else ('EVALUATING' if 'evaluation_launch' in record else
                ('WAIT_EVALUATION' if train_exit == 0 else ('TRAINING' if 'launch' in record else 'QUEUED'))))
        row = {'dataset': dataset, 'variant': variant, 'seed': seed, 'host': host, 'gpu': gpu, 'status': phase,
               'epoch': record.get('status', {}).get('epochs', record.get('status', {}).get('epoch', 0))}
        metrics = record.get('full_metrics', {})
        row.update({f'{scope}_{m}': metrics.get(scope, {}).get(m) for scope in ('dev', 'official_test') for m in METRICS})
        rows.append(row)
        if failed:
            failures.append(name)
        destination = BASE / name
        remote = f'{HOSTS[host][0]}/runs/{campaign}/{name}'
        if train_exit is not None and not (destination / 'train_intake.json').exists():
            for key in ('launch', 'run', 'status', 'best', 'exit', 'result'):
                if key in record:
                    copy_file(host, f'{remote}/{key}.json', destination / f'{key}.json')
            copy_file(host, f'{remote}/stdout.log', destination / 'stdout.log')
            for filename in record.get('training_text_files', []):
                copy_file(host, f'{remote}/{filename}', destination / filename)
            (destination / 'train_intake.json').write_text(json.dumps({'time': now, 'exit_code': train_exit}), encoding='utf-8')
        if eval_exit is not None and not (destination / 'eval_intake.json').exists():
            for key in ('evaluation_launch', 'evaluation_exit'):
                copy_file(host, f'{remote}/{key}.json', destination / f'{key}.json')
            copy_file(host, f'{remote}/evaluation_stdout.log', destination / 'evaluation_stdout.log')
            if done:
                for filename in ('metrics.json', 'dev_per_query.json', 'dev_per_query.csv', 'test_per_query.json', 'test_per_query.csv'):
                    copy_file(host, f'{remote}/full_evaluation/{filename}', destination / 'full_evaluation' / filename)
            (destination / 'eval_intake.json').write_text(json.dumps({'time': now, 'exit_code': eval_exit}), encoding='utf-8')
    with (BASE / 'status.csv').open('w', encoding='utf-8', newline='') as table:
        writer = csv.DictWriter(table, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    text = f'\n## 三种子完整评测进度\n\n实际采集：{now}。完整训练及开发/官方测试评测完成{sum(r["status"] == "COMPLETE" for r in rows)}/27，失败{len(failures)}。等待状态不是已完成指标。\n\n'
    text += '|数据集|模型|seed|GPU|epoch|状态|开发mAP|官方测试mAP|\n|---|---|---|---|---|---|---|---|\n'
    for row in rows:
        display = lambda key: '—' if row[key] is None else f'{row[key]:.4f}'
        text += f'|{row["dataset"]}|{row["variant"]}|{row["seed"]}|{row["host"]}:{row["gpu"]}|{row["epoch"]}/50|{row["status"]}|{display("dev_mAP")}|{display("official_test_mAP")}|\n'
    update_handoff(text)
    digest = sync_handoff()
    print('FULL_SUITE_COLLECTED', now, 'complete', sum(r['status'] == 'COMPLETE' for r in rows), 'failed', failures, 'sha256', digest, flush=True)
    return rows, failures


def update_handoff(text):
    document = PROJECT / 'docs/实验交接.md'
    original = document.read_text(encoding='utf-8')
    start, end = '<!-- FULL_SUITE_START -->', '<!-- FULL_SUITE_END -->'
    if start in original:
        before, rest = original.split(start, 1)
        _, after = rest.split(end, 1)
    else:
        before, after = original, ''
    document.write_bytes((before + start + text + end + after).encode('utf-8'))


def audit_and_aggregate():
    results, orders, metrics = {}, {}, {}
    for _, _, _, dataset, variant, seed in JOBS:
        name = f'{dataset}_{variant}_s{seed}'
        folder = BASE / name
        result = json.loads((folder / 'result.json').read_text())
        assert result['status'] == 'COMPLETE' and result['epochs'] == 50
        assert json.loads((folder / 'exit.json').read_text())['exit_code'] == 0
        assert json.loads((folder / 'evaluation_exit.json').read_text())['exit_code'] == 0
        with (folder / 'epochs.csv').open(encoding='utf-8', newline='') as table:
            epochs = list(csv.DictReader(table))
        assert [int(r['epoch']) for r in epochs] == list(range(1, 51))
        selected = max(epochs, key=lambda row: float(row['mAP']))
        assert int(selected['epoch']) == result['best']['epoch']
        assert all(abs(float(selected[m]) - result['best'][m]) < 1e-8 for m in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10'))
        batches = [json.loads(line) for line in (folder / 'batch_orders.jsonl').read_text().splitlines()]
        assert len(batches) == result['steps'] and [b['step'] for b in batches] == list(range(1, len(batches) + 1))
        assert all(len(b['names']) == 64 for b in batches)
        updates = sum(b['optimizer_updated'] for b in batches)
        assert updates == result['optimizer_steps'] and len(batches) - updates == result['amp_skipped_steps']
        cumulative, cumulative_updates = 0, 0
        for epoch in epochs:
            current = [b for b in batches if b['epoch'] == int(epoch['epoch'])]
            assert current
            cumulative += len(current)
            cumulative_updates += sum(b['optimizer_updated'] for b in current)
            assert int(epoch['steps']) == cumulative and int(epoch['optimizer_steps']) == cumulative_updates
            assert int(epoch['amp_skipped_steps']) == cumulative - cumulative_updates
        key = (dataset, variant, seed)
        results[key] = result
        orders[key] = [(b['epoch'], b['step'], b['names']) for b in batches]
        metrics[key] = json.loads((folder / 'full_evaluation/metrics.json').read_text())
        assert metrics[key]['selected_epoch'] == result['best']['epoch']
        assert all(abs(metrics[key]['dev'][m] - result['strict_reload'][m]) < 1e-8 for m in ('mAP', 'Rank-1', 'Rank-5', 'Rank-10'))
    for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
        for seed in (42, 43, 44):
            assert orders[(dataset, 'demo', seed)] == orders[(dataset, 'ordinary', seed)] == orders[(dataset, 'dual', seed)]
            ordinary, dual = results[(dataset, 'ordinary', seed)], results[(dataset, 'dual', seed)]
            assert ordinary['parameters'] == dual['parameters'] and ordinary['trainable_parameters'] == dual['trainable_parameters']
            assert metrics[(dataset, 'ordinary', seed)]['descriptor_dim'] == metrics[(dataset, 'dual', seed)]['descriptor_dim'] == 5632
    aggregates, comparisons = [], []
    for scope in ('dev', 'official_test'):
        for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
            for variant in ('demo', 'ordinary', 'dual'):
                row = {'scope': scope, 'dataset': dataset, 'variant': variant, 'seeds': [42, 43, 44]}
                for metric in METRICS:
                    values = [metrics[(dataset, variant, seed)][scope][metric] for seed in (42, 43, 44)]
                    row[metric] = {'values': values, 'mean': statistics.mean(values), 'sample_std': statistics.stdev(values)}
                aggregates.append(row)
            for seed in (42, 43, 44):
                folder = 'test' if scope == 'official_test' else 'dev'
                query_rows = {}
                for variant in ('ordinary', 'dual'):
                    with (BASE / f'{dataset}_{variant}_s{seed}' / 'full_evaluation' / f'{folder}_per_query.csv').open(encoding='utf-8', newline='') as source:
                        query_rows[variant] = list(csv.DictReader(source))
                ordinary, dual = query_rows['ordinary'], query_rows['dual']
                identity = lambda row: tuple(row[k] for k in ('query_index', 'name', 'identity', 'camera', 'scene', 'valid'))
                assert [identity(r) for r in ordinary] == [identity(r) for r in dual]
                paired = [(a, b) for a, b in zip(ordinary, dual) if a['valid'] == 'True']
                harms = sum(a['Rank-1'] == '1' and b['Rank-1'] == '0' for a, b in paired)
                rescues = sum(a['Rank-1'] == '0' and b['Rank-1'] == '1' for a, b in paired)
                comparison = {'scope': scope, 'dataset': dataset, 'seed': seed, 'valid_queries': len(paired),
                              'dual_minus_ordinary': {m: metrics[(dataset, 'dual', seed)][scope][m] - metrics[(dataset, 'ordinary', seed)][scope][m] for m in METRICS},
                              'rank1_harm_count': harms, 'rank1_rescue_count': rescues,
                              'harm_percent_of_all_valid': 100 * harms / len(paired), 'rescue_percent_of_all_valid': 100 * rescues / len(paired),
                              'ordinary_correct': sum(a['Rank-1'] == '1' for a, _ in paired),
                              'ordinary_wrong': sum(a['Rank-1'] == '0' for a, _ in paired)}
                comparisons.append(comparison)
    report = {'verification': 'PASS', 'runs': 27, 'epochs': 1350, 'seeds': [42, 43, 44],
              'aggregates': aggregates, 'paired_comparisons': comparisons,
              'batch_attempts': sum(r['steps'] for r in results.values()),
              'optimizer_updates': sum(r['optimizer_steps'] for r in results.values()),
              'amp_skips': sum(r['amp_skipped_steps'] for r in results.values()),
              'limits': 'Fixed fit subset; not full official-train reproduction. Three seeds do not establish broad stability or SOTA. No contribution supervision. Structural+router combination, not psi-only attribution.'}
    (BASE / 'verified_aggregate.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    flat_rows = []
    for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
        for variant in ('demo', 'ordinary', 'dual'):
            for seed in (42, 43, 44):
                result, metric = results[(dataset, variant, seed)], metrics[(dataset, variant, seed)]
                row = {'dataset': dataset, 'variant': variant, 'seed': seed, 'best_epoch': result['best']['epoch'],
                       'parameters': result['parameters'], 'descriptor_dim': metric['descriptor_dim'],
                       'batch_attempts': result['steps'], 'optimizer_updates': result['optimizer_steps'],
                       'amp_skips': result['amp_skipped_steps'], 'training_seconds': result['finished'] - result['started'],
                       'train_peak_memory_bytes': result['peak_memory'], 'eval_peak_memory_bytes': metric['eval_peak_memory_bytes'],
                       'query_seconds': metric['query_runtime']['end_to_end_seconds'],
                       'gallery_seconds': metric['gallery_runtime']['end_to_end_seconds']}
                row.update({f'{scope}_{m}': metric[scope][m] for scope in ('dev', 'official_test') for m in METRICS})
                flat_rows.append(row)
    with (BASE / 'metrics_per_run.csv').open('w', encoding='utf-8', newline='') as table:
        writer = csv.DictWriter(table, fieldnames=list(flat_rows[0])); writer.writeheader(); writer.writerows(flat_rows)
    text = '\n## 三种子完整评测终态\n\n27次训练共1350epoch及全部开发/官方测试评测exit0。审计50轮完整性、开发best选择、严格重载、逐batch样本序列、真实更新/AMP跳过、参数/维度匹配均通过。训练仍为固定fit子集。下表为全部三个种子的均值±样本标准差，指标均为百分数；原始逐次、逐查询、分组及成本记录见results/full_suite。\n\n'
    for scope in ('dev', 'official_test'):
        text += f'### {scope}\n\n|数据集|模型|mAP|mINP|Rank-1|Rank-5|Rank-10|Rank-20|\n|---|---|---|---|---|---|---|---|\n'
        for row in aggregates:
            if row['scope'] == scope:
                values = '|'.join(f'{row[m]["mean"]:.4f}±{row[m]["sample_std"]:.4f}' for m in METRICS)
                text += f'|{row["dataset"]}|{row["variant"]}|{values}|\n'
        text += '\n双轴相对ordinary的逐种子mAP差值（seed42/43/44）：\n\n'
        for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
            values = [r['dual_minus_ordinary']['mAP'] for r in comparisons if r['scope'] == scope and r['dataset'] == dataset]
            text += f'- {dataset}：' + '/'.join(f'{v:+.4f}' for v in values) + f'；均值{statistics.mean(values):+.4f}，样本标准差{statistics.stdev(values):.4f}。\n'
        text += '\n全部逐次分数：\n\n|数据集|模型|seed|开发best epoch|mAP|mINP|Rank-1|Rank-5|Rank-10|Rank-20|\n|---|---|---|---|---|---|---|---|---|---|\n'
        for row in flat_rows:
            values = '|'.join(f'{row[f"{scope}_{m}"]:.4f}' for m in METRICS)
            text += f'|{row["dataset"]}|{row["variant"]}|{row["seed"]}|{row["best_epoch"]}|{values}|\n'
    text += f'\n总batch尝试{report["batch_attempts"]}，真实optimizer更新{report["optimizer_updates"]}，AMP跳过{report["amp_skips"]}。推理增加的时间与显存、FLOPs覆盖限制按前述实测和各run metrics.json原样报告。三个种子的全部结果及退步/恢复数均保留，不能只选某数据集或某指标建立结论。\n'
    update_handoff(text)
    print('FULL_SUITE_AUDIT_PASS', json.dumps({'runs': 27, 'epochs': 1350, 'batch_attempts': report['batch_attempts']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--primary-receipt', type=Path)
    args = parser.parse_args()
    if args.watch:
        assert args.primary_receipt is not None
        first = PROJECT / 'results/first_comparison/collection.json'
        # The running primary collector truncates collection.json, then publishes Git.
        # Its terminal receipt follows both writes and publication; wait before reading.
        while 'TERMINAL_PUBLISHED' not in args.primary_receipt.read_text(encoding='utf-8'):
            time.sleep(240)
        primary = json.loads(first.read_text())
        assert len(primary['completed']) == 9 and not primary['failed'], 'inspect first campaign failure'
        from analyze_comparison import analyze
        analyze()
    while True:
        rows, failures = collect()
        if failures:
            print('FAILURE_REQUIRES_LOG_DIAGNOSIS', failures, flush=True)
            break
        if all(row['status'] == 'COMPLETE' for row in rows):
            audit_and_aggregate()
            print('FINAL_HANDOFF_SHA256', sync_handoff(), flush=True)
            break
        if not args.watch:
            return
        time.sleep(240)
    if args.watch:
        command(['git', 'add', 'docs/实验交接.md', 'results/full_suite', 'results/first_comparison'], cwd=PROJECT)
        command(['git', 'commit', '-m', 'Publish complete three-seed evaluation and recorded failures'], cwd=PROJECT)
        command(['git', 'push', 'origin', 'main'], cwd=PROJECT)
        print('FULL_SUITE_TERMINAL_PUBLISHED', flush=True)


if __name__ == '__main__':
    main()
