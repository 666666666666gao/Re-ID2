"""Collect every missing condition; publish after the original suite's writer ends."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import time

from collect_results import PROJECT, HOSTS, command, copy_file, remote_python, sync_handoff
from collect_full_suite import JOBS

BASE = PROJECT / 'results/missing_modalities'
METRICS = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
CONDITIONS = ['clean'] + [scope + '_missing_' + code for scope in ('both', 'query') for code in ('r', 'n', 't', 'rn', 'rt', 'nt')]


def read_host(host):
    root, _ = HOSTS[host]
    code = f'''import json
from pathlib import Path
root=Path({root!r})
records={{}}
for campaign in ['dynamic_amp_comparison','three_seed_extension']:
    for run in sorted((root/'runs'/campaign).glob('*_s*')):
        record={{}}
        for key in ['missing_launch','missing_exit']:
            path=run/(key+'.json')
            if path.exists(): record[key]=json.loads(path.read_text())
        path=run/'missing_evaluation/result.json'
        if path.exists(): record['result']=json.loads(path.read_text())
        record['files']=[p.name for p in sorted((run/'missing_evaluation').glob('*')) if p.suffix in ['.json','.csv']]
        records[run.name]=record
print(json.dumps(records))
'''
    return json.loads(remote_python(host, code))


def collect():
    BASE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        snapshots = dict(zip(HOSTS, pool.map(read_host, HOSTS)))
    now = datetime.now().astimezone().isoformat(timespec='seconds')
    rows = []
    for host, gpu, campaign, dataset, variant, seed in JOBS:
        name = f'{dataset}_{variant}_s{seed}'
        record = snapshots[host].get(name, {})
        code = record.get('missing_exit', {}).get('exit_code')
        complete = code == 0 and record.get('result', {}).get('status') == 'COMPLETE'
        phase = 'FAILED' if code not in (None, 0) else ('COMPLETE' if complete else ('RUNNING_OR_WAITING' if 'missing_launch' in record else 'QUEUED'))
        rows.append({'run': name, 'host': host, 'gpu': gpu, 'status': phase})
        destination = BASE / name
        if code is not None and not (destination / 'intake.json').exists():
            source = f'{HOSTS[host][0]}/runs/{campaign}/{name}'
            transfers = [(f'{source}/{key}.json', destination/f'{key}.json') for key in ('missing_launch', 'missing_exit') if key in record]
            transfers.append((f'{source}/missing_stdout.log', destination/'missing_stdout.log'))
            transfers += [(f'{source}/missing_evaluation/{filename}', destination/'missing_evaluation'/filename) for filename in record['files']]
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(copy_file, host, remote, local) for remote, local in transfers]
                for future in futures:
                    future.result()
            (destination/'intake.json').write_text(json.dumps({'time': now, 'exit_code': code}), encoding='utf-8')
    summary = {'time': now, 'rows': rows, 'complete': sum(r['status'] == 'COMPLETE' for r in rows),
               'failed': [r['run'] for r in rows if r['status'] == 'FAILED']}
    (BASE/'status.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('MISSING_COLLECTED', now, summary['complete'], 'failed', summary['failed'], flush=True)
    return summary


def audit_and_publish():
    measurements, rows = {}, []
    for _, _, _, dataset, variant, seed in JOBS:
        name = f'{dataset}_{variant}_s{seed}'
        folder = BASE/name
        result = json.loads((folder/'missing_evaluation/result.json').read_text())
        assert result['status'] == 'COMPLETE' and result['optimizer_updates'] == 0
        assert json.loads((folder/'missing_exit.json').read_text())['exit_code'] == 0
        assert set(result['measurements']) == set(CONDITIONS)
        parent = PROJECT/'results/full_suite'/name
        train = json.loads((parent/'result.json').read_text())
        normal = json.loads((parent/'full_evaluation/metrics.json').read_text())['official_test']
        assert result['selected_epoch'] == train['best']['epoch']
        assert all(abs(result['measurements']['clean']['metrics'][m] - normal[m]) < 1e-8 for m in METRICS)
        if variant == 'demo':
            assert result['original_mask_check']['original_forward_bitwise_equal_masks'] == ['r', 'n', 't', 'rn', 'rt', 'nt']
        for condition in CONDITIONS:
            metric = result['measurements'][condition]['metrics']
            assert all(metric[k] == normal[k] for k in ('query_count', 'valid_queries', 'invalid_queries', 'gallery_count'))
            measurements[(dataset, variant, seed, condition)] = metric
            rows.append({'dataset': dataset, 'variant': variant, 'seed': seed, 'condition': condition,
                         **{m: metric[m] for m in METRICS}})
    assert len(rows) == 351
    aggregates, comparisons = [], []
    for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
        for condition in CONDITIONS:
            for variant in ('demo', 'ordinary', 'dual'):
                group = [measurements[(dataset, variant, s, condition)] for s in (42, 43, 44)]
                aggregate = {'dataset': dataset, 'condition': condition, 'variant': variant}
                aggregate.update({m: {'mean': statistics.mean(g[m] for g in group), 'sample_std': statistics.stdev(g[m] for g in group)} for m in METRICS})
                aggregates.append(aggregate)
            for seed in (42, 43, 44):
                query_rows = {}
                for variant in ('demo', 'ordinary', 'dual'):
                    with (BASE/f'{dataset}_{variant}_s{seed}'/'missing_evaluation'/f'{condition}.csv').open(encoding='utf-8', newline='') as source:
                        query_rows[variant] = list(csv.DictReader(source))
                identity = lambda r: tuple(r[k] for k in ('query_index', 'name', 'identity', 'camera', 'scene', 'valid'))
                assert [identity(r) for r in query_rows['demo']] == [identity(r) for r in query_rows['ordinary']] == [identity(r) for r in query_rows['dual']]
                paired = [(a, b) for a, b in zip(query_rows['demo'], query_rows['dual']) if a['valid'] == 'True']
                ordinary_pairs = [(a, b) for a, b in zip(query_rows['ordinary'], query_rows['dual']) if a['valid'] == 'True']
                harms = sum(a['Rank-1'] == '1' and b['Rank-1'] == '0' for a, b in paired)
                rescues = sum(a['Rank-1'] == '0' and b['Rank-1'] == '1' for a, b in paired)
                comparisons.append({'dataset': dataset, 'condition': condition, 'seed': seed, 'valid_queries': len(paired),
                    'dual_minus_demo': {m: measurements[(dataset, 'dual', seed, condition)][m] - measurements[(dataset, 'demo', seed, condition)][m] for m in METRICS},
                    'dual_minus_ordinary': {m: measurements[(dataset, 'dual', seed, condition)][m] - measurements[(dataset, 'ordinary', seed, condition)][m] for m in METRICS},
                    'rank1_harms_vs_demo': harms, 'rank1_rescues_vs_demo': rescues,
                    'rank1_harms_vs_ordinary': sum(a['Rank-1'] == '1' and b['Rank-1'] == '0' for a, b in ordinary_pairs),
                    'rank1_rescues_vs_ordinary': sum(a['Rank-1'] == '0' and b['Rank-1'] == '1' for a, b in ordinary_pairs)})
    mean_comparisons = []
    for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
        for condition in CONDITIONS:
            group = [r for r in comparisons if (r['dataset'], r['condition']) == (dataset, condition)]
            row = {'dataset': dataset, 'condition': condition,
                   'dual_minus_demo': {m: statistics.mean(r['dual_minus_demo'][m] for r in group) for m in METRICS},
                   'dual_minus_ordinary': {m: statistics.mean(r['dual_minus_ordinary'][m] for r in group) for m in METRICS}}
            row['mAP_Rank1_at_least_2pp'] = all(row['dual_minus_demo'][m] >= 2 for m in ('mAP', 'Rank-1'))
            row['all_six_metrics_at_least_2pp'] = all(row['dual_minus_demo'][m] >= 2 for m in METRICS)
            mean_comparisons.append(row)
    macro_aggregates = []
    for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
        for scope in ('both', 'query'):
            conditions = [c for c in CONDITIONS if c.startswith(scope + '_')]
            assert len(conditions) == 6
            for variant in ('demo', 'ordinary', 'dual'):
                macro = {'dataset': dataset, 'scope': scope, 'variant': variant}
                for m in METRICS:
                    values = [statistics.mean(measurements[(dataset, variant, s, c)][m] for c in conditions) for s in (42, 43, 44)]
                    macro[m] = {'per_seed_equal_six_condition_means': values, 'mean': statistics.mean(values), 'sample_std': statistics.stdev(values)}
                macro_aggregates.append(macro)
    report = {'verification': 'PASS', 'runs': 27, 'conditions': 351, 'seeds': [42, 43, 44],
              'aggregates': aggregates, 'per_seed_comparisons': comparisons, 'mean_comparisons': mean_comparisons,
              'equal_six_condition_macro_aggregates': macro_aggregates,
              'mAP_Rank1_2pp_every_dataset_and_condition': all(r['mAP_Rank1_at_least_2pp'] for r in mean_comparisons),
              'all_six_2pp_every_dataset_and_condition': all(r['all_six_metrics_at_least_2pp'] for r in mean_comparisons),
              'limits': 'Diagnostic of previously fixed checkpoints trained with full modalities on fit subset. No test-time updates or learned availability masking. Both/query-only protocols reported separately; these results cannot be used to select checkpoint or hide harmful conditions.'}
    (BASE/'verified_aggregate.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (BASE/'metrics_per_run.csv').open('w', encoding='utf-8', newline='') as table:
        writer = csv.DictWriter(table, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    text = '\n\n## 完整缺失模态评测终态\n\n27个固定开发best、351个条件全部exit0；九个DeMo checkpoint分别在前八个真实query样本上核验六种mask：原始缺失分支与外部置零前向逐bit一致；这是八样本实现核验，不是全测试张量逐bit认证。所有clean六项指标与正常官方测试相同。无训练更新。缺失均在归一化输入张量上置零，r/n/t分别指缺失RGB/NIR/TIR，rn/rt/nt指同时缺失两个模态。both表示查询与图库都缺失同一集合，query表示仅查询缺失、图库完整。各协议独立列出，所有种子完整保留。指标是百分数，差值单位为百分点。\n\n'
    text += f'当前双轴相对DeMo，全部数据集/条件的mAP与Rank-1均≥+2点：{report["mAP_Rank1_2pp_every_dataset_and_condition"]}；全部六项指标均≥+2点：{report["all_six_2pp_every_dataset_and_condition"]}。这是实际验收结果，未达到不能宣称已达标。\n'
    text += '\n### 六种缺失设置的等权平均\n\n每个seed先对六种缺失条件等权平均，再对三个seed计算均值±样本标准差。该平均不替代逐条件验收。\n\n|数据集|模型|协议|mAP|mINP|Rank-1|Rank-5|Rank-10|Rank-20|\n|---|---|---|---|---|---|---|---|---|\n'
    for row in macro_aggregates:
        values = '|'.join(f'{row[m]["mean"]:.4f}±{row[m]["sample_std"]:.4f}' for m in METRICS)
        text += f'|{row["dataset"]}|{row["variant"]}|{row["scope"]}|{values}|\n'
    for scope in ('clean', 'both', 'query'):
        text += f'\n### 缺失协议：{scope}\n\n|数据集|模型|条件|mAP|mINP|Rank-1|Rank-5|Rank-10|Rank-20|\n|---|---|---|---|---|---|---|---|---|\n'
        for row in aggregates:
            selected = row['condition'] == 'clean' if scope == 'clean' else row['condition'].startswith(scope + '_')
            if selected:
                values = '|'.join(f'{row[m]["mean"]:.4f}±{row[m]["sample_std"]:.4f}' for m in METRICS)
                text += f'|{row["dataset"]}|{row["variant"]}|{row["condition"]}|{values}|\n'
    text += '\n全部351条原始六项分数在results/missing_modalities/metrics_per_run.csv；每query AP/INP、首/末匹配rank、相机/场景/身份分组、路由统计及原始日志分别在各run的missing_evaluation。与DeMo及普通频域分支的逐种子差值、改错/恢复数量保存在verified_aggregate.json。固定权重置零实验不等于缺失模态训练或可用性mask学习。\n'
    document = PROJECT/'docs/实验交接.md'
    original = document.read_text(encoding='utf-8')
    assert '## 完整缺失模态评测终态' not in original
    document.write_bytes((original + text).encode('utf-8'))
    print('MISSING_AUDIT_PASS', '27 runs / 351 conditions', 'HANDOFF_SHA256', sync_handoff(), flush=True)
    command(['git', 'add', 'docs/实验交接.md', 'results/missing_modalities'], cwd=PROJECT)
    command(['git', 'commit', '-m', 'Publish all six missing modality masks and both evaluation protocols'], cwd=PROJECT)
    command(['git', 'push', 'origin', 'main'], cwd=PROJECT)
    print('MISSING_TERMINAL_PUBLISHED', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--suite-receipt', type=Path)
    args = parser.parse_args()
    if args.watch:
        assert args.suite_receipt is not None
    while True:
        status = collect()
        if status['failed']:
            print('MISSING_FAILURE_REQUIRES_LOG_DIAGNOSIS', status['failed'], flush=True)
            return
        if status['complete'] == 27:
            if args.watch:
                while 'FULL_SUITE_TERMINAL_PUBLISHED' not in args.suite_receipt.read_text(encoding='utf-8'):
                    time.sleep(240)
            else:
                return
            audit_and_publish()
            return
        if not args.watch:
            return
        time.sleep(240)


if __name__ == '__main__':
    main()
