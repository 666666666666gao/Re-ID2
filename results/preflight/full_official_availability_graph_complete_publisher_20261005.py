"""Publish complete graph results and verified raw retention in ONE handoff."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, DESKTOP, HOSTS, OPTIONS, command, remote_python
from analyze_full_official_baseline_pairs import METRICS
from analyze_full_official_control_trial import VARIANTS

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
root = PROJECT / 'results/full_official_availability_graph_m5_20261005'
pf = PROJECT / 'results/preflight'
proof = pf / 'full_official_availability_graph_complete_publication_20261005.json'
archive = Path(__file__).with_suffix('.tar.gz')
assert not proof.exists() and not archive.exists()
report = json.loads((root / 'analysis.json').read_text(encoding='utf-8'))
retention_path = pf / 'full_official_availability_graph_distances_local_archive_20261005.json'
retention = json.loads(retention_path.read_text(encoding='utf-8'))
assert report['status'] == 'FOUR_FULL_OFFICIAL_GRAPH_RUNS196_FROZEN_AND882_STATE_CASES_ANALYZED'
assert (report['frozen_metric_cases'], report['enhanced_state_metric_cases'], report['comparison_condition_rows']) == (196, 882, 392)
assert (report['repeated_condition_query_rows'], report['contribution_rows']) == (521262, 86877)
assert report['training_heldout_identities'] == report['added_model_parameters'] == 0
assert report['paired_identity_and_partial_sampling_exact'] and report['goal_complete'] is False
assert set(report['models']) == {'demo_shared', *VARIANTS}
assert retention['status'] == 'ALL343_GRAPH_DISTANCE_FILES_LOCAL_SHA256_VERIFIED_SERVER_COPIES_CLEARED'
assert retention['distance_files'] == 343 and len(retention['completed_runs']) == 4
assert retention['removed_primary_bytes'] == retention['total_bytes']
with (root / 'comparison_condition_deltas.csv').open(encoding='utf-8') as handle:
    assert len(list(csv.DictReader(handle))) == 392
review_paths = ['results/preflight/full_official_availability_graph_intake_review_20261005.json',
    'results/preflight/full_official_availability_graph_analysis_review_20261005.json',
    'results/preflight/full_official_availability_graph_distance_archive_review_20261005.json',
    'results/preflight/full_official_availability_graph_complete_publisher_review_20261005.json']
for path in review_paths:
    review = json.loads((PROJECT / path).read_text(encoding='utf-8'))
    assert review['status'] == 'PASS' and not review['blocking_findings']
    sources = review['sources_sha256'] if path == review_paths[-1] else {review['source']: review['source_sha256']}
    for file, digest in sources.items():
        assert hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() == digest
assert command(['git', 'diff', '--cached', '--name-only'], cwd=PROJECT).strip() == ''
now = datetime.now().isoformat(timespec='seconds')
goal_path = pf / 'research_goal_optimized_20261003.json'
goal = json.loads(goal_path.read_text(encoding='utf-8')); assert goal['status'] == 'ACTIVE_UNMET'
goal['updated_at'] = now
goal['current_evidence']['full_official_availability_graph_m5'].update(
    status='FOUR_FULL_OFFICIAL_GRAPH_RUNS196_FROZEN882_STATE_CASES_ANALYZED_PUBLISHED',
    completed_fresh50_runs=4, actual_final_frozen_cases=196, actual_final_enhanced_state_cases=882,
    actual_text_intake=True, actual_analysis=True, analysis='results/full_official_availability_graph_m5_20261005/analysis.json',
    signed_condition_deltas='results/full_official_availability_graph_m5_20261005/comparison_condition_deltas.csv',
    distance_archive=dict(status=retention['status'], files=343, bytes=retention['total_bytes'],
        local_root=retention['local_root'], evidence=retention_path.relative_to(PROJECT).as_posix()))
goal['next_action'] = dict(milestone='FULL_OFFICIAL_GRAPH_RESULT_DECISION',
    status='DECIDE_FROM_COMPLETE_FAIR_COMPARISONS_AND_FROZEN_EXPERT_UTILITY',
    action='核对完整输入、49缺失条件和预声明分组、同目标普通对照、与M4父版本的变化及专家误伤/救回；只有证据支持后才冻结候选进入三数据集和配对多种子。保留原+2验收目标。',
    hardware_dependency='Only2026GPU2/3,max2NN,no temperature/powerconditions',
    transport_policy='归档传输与独立准备/预检/训练并行；仅删除已本地大小/SHA验证的距离副本。')
goal_path.write_bytes((json.dumps(goal, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
document = PROJECT / 'docs/实验交接.md'; text = document.read_text(encoding='utf-8')
marker = '<!-- FULL_OFFICIAL_PROTOCOL_OVERRIDE_START -->'
heading = '### 全官方跨可用集合监督：四对照完整终态与全部缺失评测'
assert text.count(marker) == 1 and heading not in text
addition = f'\n\n{heading}（{now}）\n\n'
addition += '四个模型均从公开CLIP全新训练50轮；MSVR310完整1032训练/591查询/1055图库、0人工身份留出、seed42、原生B64/P16/K4。仅新增0.1跨可用集合公共身份目标（完整与原部分查询各0.05，温度0.07，停止梯度的七种真实来源参考）。各模型实际身份/观测/部分集采样与原父版本一致；49训练配对覆盖由实际batch记录重算，未用计划代替覆盖。原模型、FFT、专家、路由、主要损失及检索接口均保留，增强三模型同参数/有效可训练参数/5632D。50轮依官方benchmark mAP最高且最早并列选best，严格重载同一权重评测；不是独立未参与选择的最终测试。\n\n'
addition += '| 模型 | best轮 | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Rank-20 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for label, model in [*report['references'].items(), *[('M5/' + v, m) for v, m in report['models'].items()]]:
    addition += '| ' + label + ' | ' + str(model['selected_epoch']) + ' | ' + ' | '.join(f'{model["normal"][m]:.6f}' for m in METRICS) + ' |\n'
addition += '\n全部比较均保留正负号（after−before，百分点）；49均值仅为等条件描述统计：\n\n| 比较 | 完整ΔmAP/ΔR1 | 首位误伤/救回 | 全49平均ΔmAP/ΔR1 |\n|---|---:|---:|---:|\n'
for label, comparison in report['comparisons'].items():
    row, mean = comparison['normal'], comparison['all49_equal_condition_mean_delta_pp']
    addition += f'| {label} | {row["delta_pp"]["mAP"]:+.6f}/{row["delta_pp"]["Rank-1"]:+.6f} | {row["Rank1_harm_queries"]}/{row["Rank1_rescue_queries"]} | {mean["mAP"]:+.6f}/{mean["Rank-1"]:+.6f} |\n'
addition += '\n专家实际作用（同一新模型11−00；00不是独立训练DeMo）：\n\n| 模型 | 完整ΔmAP/ΔR1 | 首位误伤/救回 | AP改善/下降/不变 | 全49平均ΔmAP/ΔR1 |\n|---|---:|---:|---:|---:|\n'
for variant in VARIANTS:
    effect = report['models'][variant]['inference_effects']['full11_minus00']
    row, mean = effect['normal'], effect['all49_equal_condition_mean_delta_pp']
    addition += f'| {variant} | {row["delta_pp"]["mAP"]:+.6f}/{row["delta_pp"]["Rank-1"]:+.6f} | {row["Rank1_harm_queries"]}/{row["Rank1_rescue_queries"]} | {row["AP_improved_queries"]}/{row["AP_worsened_queries"]}/{row["AP_unchanged_queries"]} | {mean["mAP"]:+.6f}/{mean["Rank-1"]:+.6f} |\n'
addition += '\n训练与成本：\n\n| 模型 | 更新/AMP跳过 | 参数/可训练/维度 | 50轮进程秒 | 峰值显存bytes |\n|---|---:|---:|---:|---:|\n'
for variant, model in report['models'].items():
    addition += f'| {variant} | {model["optimizer_steps"]}/{model["amp_skipped_steps"]} | {model["parameters"]}/{model["trainable_parameters"]}/{model["descriptor_dim"]} | {model["phase_wall_seconds"]["training"]:.3f} | {model["peak_training_memory_bytes"]} |\n'
addition += '\n四组固定best的全49组合共196例，以及三增强模型00/10/01/11/base_private/base_shared六状态×49共882例，均经安装数据集GT的独立CPU复核。完整CMC50、逐查询六指标、身份/摄像头/场景分组、公共/私有/专家头、贡献目标均值/标准差/MAE/零预测参照、11−10/01变化等全部保留在results/full_official_availability_graph_m5_20261005。八种比较×49=392行有符号差值，分组含相同可用性7、重叠错配30、来源不相交12、部分查询到完整图库6、部分到部分36。882条件重复591查询共521262行，不当作独立样本；49等条件平均不当作官方统一mAP。训练result的runtime是选中checkpoint全输入提取时间，进程phase_wall_seconds另记完整阶段，均不冒充纯GPU延迟或完整FLOPs。\n'
addition += f'\n原始196冻结NPZ+147六状态raw.npz共343个/{retention["total_bytes"]}字节，已在{retention["local_root"]}逐文件大小/SHA核对，远端重核后仅清理对应副本。四组best.pth、best_official_arrays.npz、头数组、全部文本/日志/GT审计保留；原始距离再审计先从本地恢复。此次归档传输期间仍可同步准备和启动独立实验。\n'
addition += '\n结论须同时看原DeMo、同目标DeMo、同容量普通频域/双专家、M4父版本及推理关闭收益。所有负结果保留，不拼接不同版本最好点，也不把局部改善写成三个数据集成功。三数据集统一方法mAP/Rank-1各+2、缺失公平优势、配对多种子及实际专家协作仍需分别验证。代码/文本上传GitHub，唯一交接project/Desktop/2026/2027逐字节一致；2025既有I/O待恢复。Goal ACTIVE_UNMET。\n'
document.write_bytes(text.replace(marker, marker + addition, 1).encode('utf-8')); DESKTOP.write_bytes(document.read_bytes())
extras = ['analyze_full_official_availability_graph_trial.py',
    'results/preflight/full_official_availability_graph_intake_20261005.py',
    'results/preflight/full_official_availability_graph_distances_archive_20261005.py',
    'results/preflight/full_official_availability_graph_complete_publisher_20261005.py',
    'results/preflight/research_goal_optimized_20261003.json', retention_path.relative_to(PROJECT).as_posix(), *review_paths]
files = [p.relative_to(PROJECT).as_posix() for p in root.rglob('*') if p.is_file() and p.suffix in ('.json', '.csv', '.jsonl', '.log', '.py')] + extras
assert len(files) == len(set(files))
with tarfile.open(archive, 'w:gz') as packed:
    for file in files:
        packed.add(PROJECT / file, arcname=file)
digest = hashlib.sha256(document.read_bytes()).hexdigest()
expected = {file: hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() for file in files}
expected['docs/实验交接.md'] = digest
for host in ('2026', '2027'):
    remote_root, _ = HOSTS[host]; target = remote_root + '/full_official_graph_complete_text_20261005.tar.gz'
    command(['scp', *OPTIONS, str(archive), host + ':' + target])
    remote_python(host, 'import tarfile;from pathlib import Path;p=Path(' + repr(target) + ');tarfile.open(p).extractall(' + repr(remote_root) + ',filter="data");p.unlink()')
    command(['scp', *OPTIONS, str(document), host + ':' + remote_root + '/docs/实验交接.md'])
    code = 'import hashlib,json;from pathlib import Path;root=Path(' + repr(remote_root) + ');selected=set(' + repr(extras + ['docs/实验交接.md']) + ');folder=root/' + repr(root.relative_to(PROJECT).as_posix()) + ';selected.update(p.relative_to(root).as_posix() for p in folder.rglob("*") if p.is_file() and p.suffix in (".json",".csv",".jsonl",".log",".py"));print(json.dumps({file:hashlib.sha256((root/file).read_bytes()).hexdigest() for file in selected}))'
    assert json.loads(remote_python(host, code)) == expected
assert DESKTOP.read_bytes() == document.read_bytes()
proof.write_bytes((json.dumps(dict(status='FOUR_COMPLETE_GRAPH_RESULTS_AND343_LOCAL_DISTANCES_PUBLISHED',
    published_at=now, document_sha256=digest, verified_locations=['project', 'Desktop', '2026', '2027'],
    pending_mirror='2025_IO', completed_fresh50_runs=4, frozen_cases=196, enhanced_state_cases=882,
    signed_comparison_rows=392, local_distance_files=343, local_distance_bytes=retention['total_bytes'],
    goal_complete=False), indent=2) + '\n').encode('utf-8'))
for host in ('2026', '2027'):
    command(['scp', *OPTIONS, str(proof), host + ':' + HOSTS[host][0] + '/results/preflight/' + proof.name])
files += ['docs/实验交接.md', proof.relative_to(PROJECT).as_posix()]
pathspec = Path(__file__).with_suffix('.pathspec'); pathspec.write_bytes(b'\0'.join(f.encode('utf-8') for f in files) + b'\0')
command(['git', 'add', '--pathspec-from-file=' + str(pathspec), '--pathspec-file-nul'], cwd=PROJECT)
command(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], cwd=PROJECT)
command(['git', 'commit', '-m', 'Report all four full-data graph trials and verified complete missing-modality diagnostics'], cwd=PROJECT)
command(['git', 'push', 'origin', 'main'], cwd=PROJECT)
head = command(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).strip()
assert command(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=PROJECT).split()[0] == head
print('FULL_OFFICIAL_GRAPH_COMPLETE_PUBLISHED', json.dumps(dict(head=head, document_sha256=digest, goal_complete=False)), flush=True)
