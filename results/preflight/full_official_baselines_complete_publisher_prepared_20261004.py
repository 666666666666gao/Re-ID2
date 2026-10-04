"""Publish all six completed full-official baseline references and paired analysis."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, DESKTOP, HOSTS, OPTIONS, command, remote_python

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
root = PROJECT / 'results/full_official_baselines_20261004'
report = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
analysis = json.loads((root / 'baseline_pair_analysis.json').read_text(encoding='utf-8'))
assert report['status'] == 'SIX_FULL_OFFICIAL_BASELINES_COMPLETE_ALL294_CPU_AUDITED'
assert analysis['status'] == 'THREE_COMPLETE_OFFICIAL_BASELINE_PAIRS_ALL294_CASES_ANALYZED'
assert report['cases'] == analysis['model_metric_cases'] == 294
assert report['repeated_condition_query_rows'] == analysis['repeated_condition_query_rows'] == 307916
assert report['paired_identity_sampling_exact'] and analysis['training_heldout_identities'] == 0
assert report['goal_complete'] is analysis['goal_complete'] is False
metrics = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
for dataset, record in report['datasets'].items():
    paired = analysis['datasets'][dataset]
    assert paired['paired_identity_sampling_exact'] and len(paired['conditions']) == 49
    for variant in ('demo', 'demo_shared'):
        original = record['references'][variant]
        assert all(abs(original['normal'][m] - paired['references'][variant]['normal'][m]) < 1e-8 for m in metrics)
    assert record['new_full_protocol_original_DeMo_plus2_thresholds'] == paired['original_DeMo_plus2_thresholds']
    assert all(abs(record['augmented_minus_original_normal'][m] - paired['normal']['delta_pp'][m]) < 1e-8 for m in metrics)
pf = PROJECT / 'results/preflight'
proof = pf / 'full_official_baselines_complete_publication_20261004.json'
assert not proof.exists()
files = [path.relative_to(PROJECT).as_posix() for path in root.rglob('*') if path.is_file()]
assert all((PROJECT / name).suffix in ('.json', '.csv', '.jsonl', '.log', '.py') for name in files)
archive = Path(__file__).with_suffix('.tar.gz')
assert not archive.exists()
now = datetime.now().isoformat(timespec='seconds')
goal_path = pf / 'research_goal_optimized_20261003.json'
goal = json.loads(goal_path.read_text(encoding='utf-8'))
goal['updated_at'] = now
goal['current_evidence']['full_official_protocol'].update(
    status='SIX_FULL_OFFICIAL_BASELINES_COMPLETE_ALL294_GT_CPU_AUDITS_PUBLISHED',
    completed_full_training=6, completed_all49=6, completed_independent_GT_audits=6,
    pending_full_training=0, pending_all49=0, pending_independent_GT_audits=0,
    final_baseline_summary='results/full_official_baselines_20261004/summary.json',
    final_paired_analysis='results/full_official_baselines_20261004/baseline_pair_analysis.json',
    original_DeMo_plus2_thresholds={d: r['new_full_protocol_original_DeMo_plus2_thresholds'] for d, r in report['datasets'].items()})
goal['next_action'].update(milestone='FULL_OFFICIAL_M3B_REAL_PREFLIGHT', status='BASELINES_COMPLETE_NEW_CUDA_CONTRACT_AND_SIX_SMOKES_PENDING',
    action='使用已审查一次性部署器启动新完整MSVR310 M3a/M3b × 双轴/普通频域/普通双专家共6组对照。真实零控制等价/梯度隔离合同及全部6模型3次实际更新/严格内存重载先通过，再从公开CLIP训练50轮。固定best评测全49和294四状态/基础块诊断及GT CPU复算。未证明推理增益与公平胜出前，不扩展为统一三数据集方法。')
goal_path.write_text(json.dumps(goal, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
document = PROJECT / 'docs/实验交接.md'
text = document.read_text(encoding='utf-8')
marker = '<!-- FULL_OFFICIAL_PROTOCOL_OVERRIDE_START -->'
heading = '### 三数据集完整官方基线阶段完成'
assert text.count(marker) == 1 and heading not in text
addition = f'''

{heading}（{now}）

原控制器所有6模型完成50轮、固定best重载、全49组合推理和安装文件GT独立CPU复算，共294模型-条件案例/307916重复条件-query行。全部官方训练身份与记录可用且实际累计访问，无人工fit/dev留出；同数据集两模型真实采样顺序一致。全部normal六指标、CMC1..50、逐query AP/INP/首末匹配、身份/相机/scene分组与正负缺失结果完整归档。以下所有百分数都来自同一完整官方评测mAP-best权重，最高mAP并列保留最早，benchmark参与选点须披露。

| 数据集 | 基线 | best epoch | mAP | mINP | R1 | R5 | R10 | R20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
'''
for dataset, row in report['datasets'].items():
    for variant, value in row['references'].items():
        addition += '| ' + dataset + ' | ' + variant + ' | ' + str(value['selected_epoch']) + ' | ' + ' | '.join(f'{value["normal"][m]:.6f}' for m in metrics) + ' |\n'
    addition += '| ' + dataset + ' | 同增强−原始 | — | ' + ' | '.join(f'{row["augmented_minus_original_normal"][m]:+.6f}' for m in metrics) + ' |\n'
addition += '\n同增强DeMo新增的是公共身份接口、可用性处理及缺失训练，没有新增M/F双轴专家。所有涨跌只表明这一通用基线组合的影响，不是双轴条件协作的论文证据。49组中同时达到mAP/R1各+2的数量及重要分组（重复查询不能视作独立样本，分组存在重叠）如下：\n\n'
addition += '| 数据集 | 组 | 条件数 | ΔmAP | ΔR1 | 首位误伤/救回重复query行 | 双+2条件数 |\n|---|---|---:|---:|---:|---:|---:|\n'
pair_notes = []
for dataset, row in analysis['datasets'].items():
    for group, value in row['groups'].items():
        addition += f'| {dataset} | {group} | {value["conditions"]} | {value["equal_condition_mean_delta_pp"]["mAP"]:+.6f} | {value["equal_condition_mean_delta_pp"]["Rank-1"]:+.6f} | {value["Rank1_harm_repeated_query_rows"]}/{value["Rank1_rescue_repeated_query_rows"]} | {value["both_metrics_plus2_conditions"]} |\n'
    pair_notes.append(f'{dataset} 全49双+2：{row["both_metrics_plus2_conditions"]}/49；normal误伤/救回：{row["normal"]["Rank1_harm_queries"]}/{row["normal"]["Rank1_rescue_queries"]}。')
addition += '\n' + '\n\n'.join(pair_notes) + '\n'
addition += '\n新完整协议统一方法的原始DeMo +2/+2数值参照：\n\n| 数据集 | mAP门槛 | Rank-1门槛 |\n|---|---:|---:|\n'
for dataset, row in report['datasets'].items():
    threshold = row['new_full_protocol_original_DeMo_plus2_thresholds']
    addition += f'| {dataset} | {threshold["mAP"]:.9f} | {threshold["Rank-1"]:.9f} |\n'
addition += '\n完整原始记录、50轮曲线、真实采样与更新计数、全49六指标/CMC/逐query/分组、CPU凭证和所有正负差值在results/full_official_baselines_20261004。baseline_pair_analysis.json及baseline_pair_condition_deltas.csv进一步对齐真实query，报告全部条件的差值/误伤/救回。原始距离NPZ及每实验一个best权重留在2026，不传GitHub。历史fit/dev指标保留但不用于这里的达标计算。\n'
addition += '\n基线阶段完成不等于研究目标完成。下一步仍按已审查完整协议做M3a/M3b测量—控制分离机制对照，真实CUDA合同与全部6组smokes先通过；同一统一方法三个数据集/+2/+2、至少3配对种子、公平普通专家胜出、缺失模态及推理协作净收益仍未建立。只使用2026物理GPU2/3、每卡串行、最多2个NN，不设温度或功率条件。唯一交接在project/Desktop/2026/2027逐字节核对，2025既有I/O故障镜像待恢复。Goal ACTIVE/UNMET。\n'
document.write_bytes(text.replace(marker, marker + addition, 1).encode('utf-8'))
DESKTOP.write_bytes(document.read_bytes())
digest = hashlib.sha256(document.read_bytes()).hexdigest()
files += ['results/preflight/research_goal_optimized_20261003.json', 'analyze_full_official_baseline_pairs.py']
with tarfile.open(archive, 'w:gz') as tar:
    for name in files: tar.add(PROJECT / name, arcname=name)
for host in ('2026', '2027'):
    remote_root, _ = HOSTS[host]
    target = remote_root + '/full_official_baselines_complete_text_20261004.tar.gz'
    command(['scp', *OPTIONS, str(archive), host + ':' + target])
    remote_python(host, 'import tarfile;from pathlib import Path;p=Path(' + repr(target) + ');tarfile.open(p).extractall(' + repr(remote_root) + ',filter="data");p.unlink()')
    command(['scp', *OPTIONS, str(document), host + ':' + remote_root + '/docs/实验交接.md'])
    expected = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in files}
    expected['docs/实验交接.md'] = digest
    extra = ('results/preflight/research_goal_optimized_20261003.json', 'analyze_full_official_baseline_pairs.py', 'docs/实验交接.md')
    actual = json.loads(remote_python(host, 'import hashlib,json;from pathlib import Path;root=Path(' + repr(remote_root) + ');files=[p for p in (root/"results/full_official_baselines_20261004").rglob("*") if p.is_file()];files += [root/n for n in ' + repr(extra) + '];print(json.dumps({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))'))
    assert actual == expected
assert DESKTOP.read_bytes() == document.read_bytes()
proof.write_text(json.dumps(dict(status='SIX_FULL_OFFICIAL_BASELINES_AND_ALL294_GT_AUDITS_PUBLISHED',
    published_at=datetime.now().isoformat(timespec='seconds'), document_sha256=digest,
    verified_locations=['project', 'Desktop', '2026', '2027'], pending_mirror='2025_IO',
    text_files=len(files), model_metric_cases=294, repeated_condition_query_rows=307916,
    new_dual_axis_trials=0, goal_complete=False), indent=2) + '\n', encoding='utf-8')
for host in ('2026', '2027'):
    remote_root, _ = HOSTS[host]
    command(['scp', *OPTIONS, str(proof), host + ':' + remote_root + '/results/preflight/' + proof.name])
files += ['docs/实验交接.md', proof.relative_to(PROJECT).as_posix()]
pathspec = Path(__file__).with_suffix('.pathspec')
pathspec.write_bytes(b'\0'.join(name.encode('utf-8') for name in files) + b'\0')
command(['git', 'add', '--pathspec-from-file=' + str(pathspec), '--pathspec-file-nul'], cwd=PROJECT)
command(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], cwd=PROJECT)
command(['git', 'commit', '-m', 'Publish all full official DeMo references and complete paired missing evaluations'], cwd=PROJECT)
command(['git', 'push', 'origin', 'main'], cwd=PROJECT)
head = command(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).strip()
assert command(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=PROJECT).split()[0] == head
print('SIX_FULL_OFFICIAL_BASELINES_PUBLISHED', json.dumps(dict(head=head, document_sha256=digest,
    cases=294, repeated_rows=307916, goal_complete=False)), flush=True)
