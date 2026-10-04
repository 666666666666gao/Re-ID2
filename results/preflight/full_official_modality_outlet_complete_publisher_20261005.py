"""Publish actual M4 full-data results, verified local archives and one handoff."""
from datetime import datetime
import csv
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
root = PROJECT / 'results/full_official_modality_outlet_m4_20261005'
pf = PROJECT / 'results/preflight'
report = json.loads((root / 'analysis.json').read_text(encoding='utf-8'))
retention_path = pf / 'full_official_modality_outlet_distances_local_archive_20261005.json'
retention = json.loads(retention_path.read_text(encoding='utf-8'))
assert report['status'] == 'THREE_FULL_OFFICIAL_M4_RUNS_AND882_STATE_CASES_ANALYZED'
assert (report['state_metric_cases'], report['repeated_condition_query_rows'], report['contribution_rows']) == (882, 521262, 86877)
assert report['comparison_condition_rows'] == 343 and report['goal_complete'] is False
assert report['training_heldout_identities'] == report['added_model_parameters'] == 0
assert report['paired_identity_and_partial_sampling_exact'] and set(report['models']) == set(VARIANTS)
assert retention['status'] == 'ALL294_M4_DISTANCE_FILES_LOCAL_SHA256_VERIFIED_SERVER_COPIES_CLEARED'
assert retention['distance_files'] == 294 and len(retention['completed_runs']) == 3
assert retention['removed_primary_bytes'] == retention['total_bytes']
with (root / 'comparison_condition_deltas.csv').open(encoding='utf-8') as handle:
    assert len(list(csv.DictReader(handle))) == 343
for review_name in ('full_official_modality_outlet_reporting_review_20261005.json',
                    'full_official_modality_outlet_complete_publisher_review_20261005.json'):
    review = json.loads((pf / review_name).read_text(encoding='utf-8'))
    assert review['status'] == 'PASS'
    for name, digest in review['sources_sha256'].items():
        assert hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == digest
proof = pf / 'full_official_modality_outlet_complete_publication_20261005.json'
archive = Path(__file__).with_suffix('.tar.gz')
assert not proof.exists() and not archive.exists()
assert command(['git', 'diff', '--cached', '--name-only'], cwd=PROJECT).strip() == ''
now = datetime.now().isoformat(timespec='seconds')
goal_path = pf / 'research_goal_optimized_20261003.json'
goal = json.loads(goal_path.read_text(encoding='utf-8'))
assert goal['status'] == 'ACTIVE_UNMET'
goal['updated_at'] = now
goal['current_evidence']['full_official_modality_outlet_m4'].update(
    status='THREE_FULL_OFFICIAL_M4_RUNS_AND882_STATE_CASES_ANALYZED_PUBLISHED',
    actual_fresh50_runs=3, completed_fresh50_runs=3, actual_state_metric_cases=882,
    repeated_condition_query_rows=521262, contribution_rows=86877, actual_text_intake=True, actual_analysis=True,
    analysis='results/full_official_modality_outlet_m4_20261005/analysis.json',
    signed_condition_deltas='results/full_official_modality_outlet_m4_20261005/comparison_condition_deltas.csv',
    distance_archive=dict(status=retention['status'], files=294, bytes=retention['total_bytes'],
        local_root=retention['local_root'], evidence=retention_path.relative_to(PROJECT).as_posix()))
goal['next_action'] = dict(milestone='FULL_OFFICIAL_M4_RESULT_DECISION',
    status='DECIDE_NEXT_SINGLE_FACTOR_FROM_COMPLETE_FAIR_AND_FROZEN_DIAGNOSTICS',
    action='根据本轮完整六指标、49条件、同variant M3a/同参数普通专家和推理关闭诊断决定下一单因素；机制未成立前不盲目扩展三数据集。后续传输期间也并行准备/预检/训练。Goal ACTIVE/UNMET。',
    hardware_dependency='Only2026GPU2/3,max2NN,no temperature/powerconditions')
goal_path.write_text(json.dumps(goal, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
document = PROJECT / 'docs/实验交接.md'
text = document.read_text(encoding='utf-8')
marker = '<!-- FULL_OFFICIAL_PROTOCOL_OVERRIDE_START -->'
heading = '### 全量官方M4：模态专家实际出口监督终态与完整对照'
assert text.count(marker) == 1 and heading not in text
addition = f'\n\n{heading}（{now}）\n\n'
addition += '相对M3a仅增加实际PM出口0.1公共身份监督（温度0.07），普通频域/普通双专家也接受相同监督；原FFT、专家、路由、门控、频域监督、描述子和缺失训练不变，无新增参数。完整官方MSVR310：1032训练/591查询/1055图库、0人工留出身份、公开CLIP重新初始化、seed42、原生batch64/P16/K4、50轮。三组与各自M3a父版本的身份/观测/缺失集合顺序逐步一致；2026物理GPU2运行axis，GPU3顺序frequency/twins。固定最早并列mAP-best严格重载用于全部条件，checkpoint由官方benchmark mAP选择，不是未参与选择的独立测试。\n\n'
addition += '| 模型 | best轮 | mAP | mINP | R1 | R5 | R10 | R20 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for label, model in report['references'].items():
    addition += '| ' + label + ' | ' + str(model['selected_epoch']) + ' | ' + ' | '.join(f'{model["normal"][m]:.6f}' for m in METRICS) + ' |\n'
for variant, model in report['models'].items():
    addition += '| M4/' + variant + ' | ' + str(model['selected_epoch']) + ' | ' + ' | '.join(f'{model["normal"]["metrics"]["11"][m]:.6f}' for m in METRICS) + ' |\n'
addition += '\n全部对照差值（after−before，百分点）：\n\n| 比较 | 完整ΔmAP/ΔR1 | 首位误伤/救回 | 全49等条件平均ΔmAP/ΔR1 |\n|---|---:|---:|---:|\n'
for label, comparison in report['comparisons'].items():
    row, mean = comparison['normal'], comparison['all49_equal_condition_mean_delta_pp']
    addition += f'| {label} | {row["delta_pp"]["mAP"]:+.6f}/{row["delta_pp"]["Rank-1"]:+.6f} | {row["Rank1_harm_queries"]}/{row["Rank1_rescue_queries"]} | {mean["mAP"]:+.6f}/{mean["Rank-1"]:+.6f} |\n'
addition += '\n推理11−00诊断（同一训练模型关闭专家，不是重训消融）：\n\n| 模型 | 完整ΔmAP/ΔR1 | 首位误伤/救回 | AP改善/下降/不变 | 全49平均ΔmAP/ΔR1 |\n|---|---:|---:|---:|---:|\n'
for variant, model in report['models'].items():
    effect = model['inference_effects']['full11_minus00']
    row, mean = effect['normal'], effect['all49_equal_condition_mean_delta_pp']
    addition += f'| {variant} | {row["delta_pp"]["mAP"]:+.6f}/{row["delta_pp"]["Rank-1"]:+.6f} | {row["Rank1_harm_queries"]}/{row["Rank1_rescue_queries"]} | {row["AP_improved_queries"]}/{row["AP_worsened_queries"]}/{row["AP_unchanged_queries"]} | {mean["mAP"]:+.6f}/{mean["Rank-1"]:+.6f} |\n'
addition += '\n更新/成本/校准：\n\n| 模型 | 真实优化更新/AMP跳步 | 参数/可训练参数/维度 | 50轮训练进程秒数 | 训练显存峰值bytes | PM对齐raw均值 | 贡献MAE优于零预测的M/F/I条件数 |\n|---|---:|---:|---:|---:|---:|---:|\n'
for variant, model in report['models'].items():
    addition += f'| {variant} | {model["optimizer_steps"]}/{model["amp_skipped_steps"]} | {model["parameters"]}/{model["trainable_parameters"]}/{model["descriptor_dim"]} | {model["phase_wall_seconds"]["training"]:.3f} | {model["peak_training_memory_bytes"]} | {model["alignment_mean_raw"]:.6f} | {model["calibration_MAE_better_than_zero_conditions"]} |\n'
addition += '\n三组各49冻结条件、00/10/01/11/base_private/base_shared六状态×49，共882状态指标实例/521262重复condition-query行、86877贡献行均已由安装数据集GT独立CPU复核。11逐query与冻结49一致。stdlib分析再次核对六指标、误伤/救回/AP变化，7种比较×49共343行。全部CMC50、逐query/身份/摄像头/scene分组、贡献均值/标准差/MAE及零预测、运行时间、GT凭证、完整正负变化保存在results/full_official_modality_outlet_m4_20261005；analysis.json/comparison_condition_deltas.csv包含same_availability7、overlap_mismatch30、source_disjoint12、partial_query_full_gallery6、both_partial36五组。组间重复使用591查询，均值是诊断统计，不是官方统一mAP或独立样本。training result中的runtime是选定checkpoint的完整输入提取时间，含DataLoader/解码，不是50轮训练时间；phase_wall_seconds按实际进程launch/exit时间记录全部七阶段，不是纯GPU单批延迟。关闭专家00仍是本方法训练后的基础路径，不能替代独立DeMo。\n'
addition += f'\n147冻结NPZ与147六状态raw.npz共294份/{retention["total_bytes"]}字节已归档到{retention["local_root"]}，每份大小/SHA校验后再次核对服务器原件才清理2026相同副本。凭证results/preflight/{retention_path.name}。best.pth、best_official_arrays.npz、原始日志、文字结果、heads和GT凭证保留；重做原始距离CPU审计须先从本地恢复。上一批传输期间已准备并启动本轮，后续传输与独立准备/预检/训练也并行。\n'
addition += '\n本轮是单种子MSVR受控改动，未证明统一三数据集/多种子成功。必须同时考察原始/同增强DeMo、同参数普通专家、关闭状态净收益；不拼接不同版本最好成绩、不隐藏负结果。内部三数据集mAP/Rank-1各+2、缺失覆盖、公平对照及真实协作目标仍未完成。下一单因素按本轮完整证据决定，机制不成立不盲目扩展三数据集。唯一交接project/Desktop/2026/2027逐字节一致；2025实际I/O故障待同步。Goal ACTIVE/UNMET。\n'
document.write_bytes(text.replace(marker, marker + addition, 1).encode('utf-8'))
DESKTOP.write_bytes(document.read_bytes())
extras = ['analyze_full_official_modality_outlet_trial.py', 'results/preflight/research_goal_optimized_20261003.json',
    'results/preflight/full_official_modality_outlet_primary_observer_20261005.jsonl',
    'results/preflight/full_official_modality_outlet_primary_observer_terminal_20261005.json',
    'results/preflight/full_official_modality_outlet_intake_20261005.py',
    'results/preflight/full_official_modality_outlet_distances_archive_20261005.py',
    'results/preflight/full_official_modality_outlet_distances_local_archive_20261005.json',
    'results/preflight/full_official_modality_outlet_reporting_review_20261005.json',
    'results/preflight/full_official_modality_outlet_complete_publisher_20261005.py',
    'results/preflight/full_official_modality_outlet_complete_publisher_review_20261005.json']
files = [p.relative_to(PROJECT).as_posix() for p in root.rglob('*') if p.is_file()] + extras
assert all((PROJECT / name).suffix in ('.json', '.csv', '.jsonl', '.log', '.py') for name in files)
with tarfile.open(archive, 'w:gz') as handle:
    for name in files: handle.add(PROJECT / name, arcname=name)
digest = hashlib.sha256(document.read_bytes()).hexdigest()
expected = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in files}
expected['docs/实验交接.md'] = digest
for host in ('2026', '2027'):
    remote_root, _ = HOSTS[host]
    target = remote_root + '/full_official_modality_outlet_complete_text_20261005.tar.gz'
    command(['scp', *OPTIONS, str(archive), host + ':' + target])
    remote_python(host, 'import tarfile;from pathlib import Path;p=Path(' + repr(target) + ');tarfile.open(p).extractall(' + repr(remote_root) + ',filter="data");p.unlink()')
    command(['scp', *OPTIONS, str(document), host + ':' + remote_root + '/docs/实验交接.md'])
    actual = json.loads(remote_python(host, 'import hashlib,json;from pathlib import Path;root=Path(' + repr(remote_root) + ');files=[p for p in (root/"results/full_official_modality_outlet_m4_20261005").rglob("*") if p.is_file()];files += [root/n for n in ' + repr(tuple(extras) + ('docs/实验交接.md',)) + '];print(json.dumps({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))'))
    assert actual == expected
assert DESKTOP.read_bytes() == document.read_bytes()
proof.write_text(json.dumps(dict(status='THREE_FULL_OFFICIAL_M4_RUNS_AND882_GT_CASES_WITH_LOCAL_ARCHIVE_PUBLISHED',
    published_at=datetime.now().isoformat(timespec='seconds'), document_sha256=digest,
    verified_locations=['project', 'Desktop', '2026', '2027'], pending_mirror='2025_IO',
    text_files=len(files), cases=882, repeated_condition_query_rows=521262,
    raw_distance_files_local=294, goal_complete=False), indent=2) + '\n', encoding='utf-8')
for host in ('2026', '2027'):
    remote_root, _ = HOSTS[host]
    command(['scp', *OPTIONS, str(proof), host + ':' + remote_root + '/results/preflight/' + proof.name])
files += ['docs/实验交接.md', proof.relative_to(PROJECT).as_posix()]
pathspec = Path(__file__).with_suffix('.pathspec')
pathspec.write_bytes(b'\0'.join(name.encode('utf-8') for name in files) + b'\0')
command(['git', 'add', '--pathspec-from-file=' + str(pathspec), '--pathspec-file-nul'], cwd=PROJECT)
command(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], cwd=PROJECT)
command(['git', 'commit', '-m', 'Publish full-data M4 modality outlet utility and matched control evidence'], cwd=PROJECT)
command(['git', 'push', 'origin', 'main'], cwd=PROJECT)
head = command(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).strip()
assert command(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=PROJECT).split()[0] == head
print('FULL_OFFICIAL_M4_AND_LOCAL_RAW_ARCHIVE_PUBLISHED', json.dumps(dict(head=head, document_sha256=digest, cases=882, goal_complete=False)), flush=True)
