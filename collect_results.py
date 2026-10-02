"""Read the existing controllers; update the single handoff and its exact mirrors."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

PROJECT = Path(__file__).resolve().parent
DESKTOP = Path('C:/Users/gb/Desktop/document/DeMo双轴实验交接.md')
HOSTS = {'2025': ('/data2/gb/Re-ID/DeMo-DualAxis', '/data2/gb/Re-ID/conda-envs/tri_reid/bin/python'),
         '2026': ('/data/gaob/Re-ID/DeMo-DualAxis', '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python')}
RUNS = [('RGBNT201', 'demo', '2025', '0'), ('RGBNT201', 'ordinary', '2025', '1'), ('RGBNT201', 'dual', '2025', '2'),
        ('MSVR310', 'demo', '2025', '3'), ('RGBNT100', 'demo', '2026', '0'), ('RGBNT100', 'ordinary', '2026', '1'),
        ('RGBNT100', 'dual', '2026', '2'), ('MSVR310', 'ordinary', '2026', '3'), ('MSVR310', 'dual', '2026', '3')]
OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', '-o', 'ConnectTimeout=15']
CAMPAIGN = 'dynamic_amp_comparison'


def command(argv, **kwargs):
    return subprocess.run(argv, check=True, capture_output=True, text=True, encoding='utf-8', **kwargs).stdout


def remote_python(host, code):
    _, python = HOSTS[host]
    return command(['ssh', *OPTIONS, host, shlex.quote(python) + ' -c ' + shlex.quote(code)])


def read_host(host):
    root, _ = HOSTS[host]
    code = f'''import json
from pathlib import Path
root=Path({root!r})/'runs/{CAMPAIGN}'
result={{}}
for folder in sorted(root.glob('*_s42')):
    record={{}}
    for key in ['launch','run','status','best','exit','result']:
        path=folder/(key+'.json')
        if path.exists():
            record[key]=json.loads(path.read_text())
    result[folder.name]=record
print(json.dumps(result))
'''
    return json.loads(remote_python(host, code))


def copy_file(host, source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    command(['scp', *OPTIONS, host + ':' + source, str(destination)])


def sync_handoff():
    document = PROJECT / 'docs/实验交接.md'
    content = document.read_bytes()
    DESKTOP.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    for host, (root, _) in HOSTS.items():
        command(['scp', *OPTIONS, str(document), host + ':' + root + '/docs/实验交接.md'])
        code = 'import hashlib;from pathlib import Path;print(hashlib.sha256(Path(' + repr(root + '/docs/实验交接.md') + ').read_bytes()).hexdigest())'
        assert remote_python(host, code).strip() == digest, host + ' handoff differs'
    assert DESKTOP.read_bytes() == content
    return digest


def collect_once():
    snapshots = {host: read_host(host) for host in HOSTS}
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    output = PROJECT / 'results/first_comparison'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'latest_snapshot.json').write_text(json.dumps({'observed_at': now, 'hosts': snapshots}, indent=2), encoding='utf-8')
    rows, completed, failed = [], [], []
    for dataset, variant, host, gpu in RUNS:
        name = f'{dataset}_{variant}_s42'
        record = snapshots[host].get(name, {})
        result, status = record.get('result', {}), record.get('status', {})
        exit_code = record.get('exit', {}).get('exit_code')
        phase = 'FAILED' if exit_code is not None and exit_code != 0 else ('COMPLETE' if result.get('status') == 'COMPLETE' and exit_code == 0 else ('RUNNING' if 'launch' in record else 'PENDING'))
        epoch = result.get('epochs', status.get('epoch', 0))
        best = result.get('best', record.get('best', {}))
        rows.append({'dataset': dataset, 'variant': variant, 'host': host, 'gpu': gpu, 'status': phase, 'epoch': epoch,
                     'best_epoch': best.get('epoch'), 'mAP': best.get('mAP'), 'Rank-1': best.get('Rank-1'), 'Rank-5': best.get('Rank-5'), 'Rank-10': best.get('Rank-10')})
        # Fetch complete text evidence once; no checkpoint or training replay.
        if phase in ('COMPLETE', 'FAILED'):
            destination = output / name
            if not (destination / 'intake.json').exists():
                root, _ = HOSTS[host]
                files = list(record.keys()) + ['stdout']
                for key in files:
                    suffix = '.log' if key == 'stdout' else '.json'
                    copy_file(host, f'{root}/runs/{CAMPAIGN}/{name}/{key}{suffix}', destination / (key + suffix))
                if epoch:
                    copy_file(host, f'{root}/runs/{CAMPAIGN}/{name}/epochs.csv', destination / 'epochs.csv')
                (destination / 'intake.json').write_text(json.dumps({'collected_at': now, 'status': phase}), encoding='utf-8')
        if phase == 'COMPLETE':
            completed.append(name)
        if phase == 'FAILED':
            failed.append(name)
    with (output / 'summary.csv').open('w', encoding='utf-8', newline='') as table:
        writer = csv.DictWriter(table, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    handoff = PROJECT / 'docs/实验交接.md'
    original = handoff.read_text(encoding='utf-8')
    before, rest = original.split('## 运行计划和当前状态', 1)
    _, after = rest.split('## 证据和下一步', 1)
    text = f'## 运行计划和当前状态\n\n实际观察：{now}。完成{len(completed)}/9，失败{len(failed)}/9；未完成模型的最佳分数是中间开发指标，不能当作50轮终态结果。官方测试评估次数0。\n\n'
    text += '当前批次为dynamic_amp_comparison。MSVR310 ordinary→dual在2026:3串行，其余同时运行。所有端50轮、seed42、B64；实际成功optimizer更新和AMP跳过次数单独记录。两端的loss、优化器、初始化与开发协议固定。\n\n'
    text += '| 数据集 | 模型 | GPU | 已完成epoch | 状态 | 当前best epoch | mAP | Rank-1/5/10 |\n|---|---|---|---|---|---|---|---|\n'
    for row in rows:
        display = lambda k: '—' if row[k] is None else f'{row[k]:.4f}'
        ranks = '/'.join(display(k) for k in ('Rank-1', 'Rank-5', 'Rank-10'))
        text += f"| {row['dataset']} | {row['variant']} | {row['host']}:{row['gpu']} | {row['epoch']}/50 | {row['status']} | {row['best_epoch'] or '—'} | {display('mAP')} | {ranks} |\n"
    if len(completed) == 9:
        text += '\n三数据集对照差值（dual减ordinary，开发best mAP）：\n\n'
        for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310'):
            ordinary = next(r for r in rows if r['dataset'] == dataset and r['variant'] == 'ordinary')
            dual = next(r for r in rows if r['dataset'] == dataset and r['variant'] == 'dual')
            text += f"- {dataset}：{dual['mAP'] - ordinary['mAP']:+.4f} mAP，{dual['Rank-1'] - ordinary['Rank-1']:+.4f} Rank-1。\n"
        text += '\n这是单seed开发比较，尚不构成稳定提升或最终测试结论。不根据此结果自动改变频带、学习率、种子或重启训练。\n'
    if failed:
        text += '\n实际失败端：' + ', '.join(failed) + '。原日志和失败状态保留；控制器不自动重试，后续处理须基于真实错误。\n'
    handoff.write_text(before + text + '\n## 证据和下一步' + after, encoding='utf-8')
    digest = sync_handoff()
    result = {'observed_at': now, 'completed': completed, 'failed': failed, 'handoff_sha256': digest}
    (output / 'collection.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('COLLECTED', json.dumps(result), flush=True)
    return snapshots, completed, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        snapshots, complete, failed = collect_once()
        if not args.watch or failed or len(complete) == 9:
            break
        # Estimate the next completion boundary; 240s when an epoch timing is not yet known.
        remaining = []
        for records in snapshots.values():
            for record in records.values():
                if 'result' not in record and 'exit' not in record:
                    status = record.get('status', {})
                    if 'latest' in status:
                        remaining.append((50 - status['epoch']) * status['latest']['seconds'])
        delay = max(240, min(remaining) - 180) if remaining else 240
        print('NEXT_OBSERVATION_SECONDS', round(delay), flush=True)
        time.sleep(delay)
    if args.watch:
        command(['git', 'add', 'docs/实验交接.md', 'results/first_comparison'], cwd=PROJECT)
        command(['git', 'commit', '-m', 'Record actual first comparison terminal status'], cwd=PROJECT)
        command(['git', 'push', 'origin', 'main'], cwd=PROJECT)
        print('TERMINAL_PUBLISHED', flush=True)


if __name__ == '__main__':
    main()
