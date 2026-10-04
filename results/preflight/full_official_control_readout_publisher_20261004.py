"""Publish the completed frozen stage diagnostics and verified local raw archive."""
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, DESKTOP, HOSTS, OPTIONS, command, remote_python
from analyze_full_official_control_readout import MODES, METRICS, READOUTS, COMPARISONS

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
root = PROJECT / 'results/full_official_control_readout_20261004'
pf = PROJECT / 'results/preflight'
report = json.loads((root / 'readout_analysis.json').read_text(encoding='utf-8'))
retention_path = pf / 'full_official_control_readout_distances_local_archive_20261004.json'
retention = json.loads(retention_path.read_text(encoding='utf-8'))
assert report['status'] == 'BOTH_FROZEN_FULL_OFFICIAL_READOUTS_AND1176_CASES_ANALYZED'
assert report['metric_cases'] == 1176 and report['repeated_condition_query_rows'] == 695016
assert report['paired_diagnostic_condition_rows'] == 784 and report['goal_complete'] is False
assert report['training_heldout_identities'] == report['neural_reruns'] == report['training_updates'] == report['new_weights'] == 0
assert set(report['modes']) == set(MODES)
assert retention['status'] == 'ALL98_READOUT_DISTANCE_FILES_LOCAL_SHA256_VERIFIED_SERVER_COPIES_CLEARED'
assert retention['distance_files'] == 98 and len(retention['completed_modes']) == 2
assert retention['removed_primary_bytes'] == retention['total_bytes']
with (root / 'readout_condition_deltas.csv').open(encoding='utf-8') as handle:
    assert len(list(csv.DictReader(handle))) == 784
for model in report['modes'].values():
    assert model['deployed_all49_previous_perquery_exact'] and model['max_cpu_sixmetric_error_pp'] < 1e-8
    assert set(model['normal']) == set(READOUTS) and set(model['comparisons']) == set(COMPARISONS)
proof = pf / 'full_official_control_readout_publication_20261004.json'
archive = Path(__file__).with_suffix('.tar.gz')
assert not proof.exists() and not archive.exists()
assert command(['git', 'diff', '--cached', '--name-only'], cwd=PROJECT).strip() == ''
now = datetime.now().isoformat(timespec='seconds')
goal_path = pf / 'research_goal_optimized_20261003.json'
goal = json.loads(goal_path.read_text(encoding='utf-8'))
goal['updated_at'] = now
goal['current_evidence']['full_official_control_readout'].update(
    status='BOTH_FROZEN_FULL_OFFICIAL_READOUTS_AND1176_CASES_ANALYZED_PUBLISHED',
    actual_text_intake=True, actual_analysis=True, actual_metric_cases=1176,
    repeated_condition_query_rows=695016, training_updates=0, new_weights=0,
    analysis='results/full_official_control_readout_20261004/readout_analysis.json',
    all49_stage_delta_csv='results/full_official_control_readout_20261004/readout_condition_deltas.csv',
    distance_archive=dict(status=retention['status'], files=98, bytes=retention['total_bytes'],
        local_root=retention['local_root'], evidence=retention_path.relative_to(PROJECT).as_posix()))
goal_path.write_text(json.dumps(goal, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
document = PROJECT / 'docs/实验交接.md'
text = document.read_text(encoding='utf-8')
marker = '<!-- FULL_OFFICIAL_PROTOCOL_OVERRIDE_START -->'
heading = '### 完整官方双轴出口分阶段核查与原始距离本地归档终态'
assert text.count(marker) == 1 and heading not in text
addition = f'''\n\n{heading}（{now}）

直接读取上述已完成M3a/M3b axis_shared固定best，分别E32/E27，真实CUDA两次64-query×7可用集合smoke后，遍历全部官方591查询/1055图库、7×7模态组合。每模型8个同出口独立检索及4个PM/PF—公共坐标有向交叉检索，共588案例；两组1176案例、695016重复条件-query行。原始安装GT独立CPU复算六指标/CMC50/逐query/分组，误差均为0；全部49 deployed指标及逐query与之前结果完全一致。没有训练更新、模型参数改动或新增权重，原始best/特征数组/结果输入未改动。0人工训练身份留出，仍是全量官方协议；checkpoint此前按官方benchmark mAP选择，不是未参与选择的独立测试。

完整三模态六指标（%）：

| 模式 | readout | mAP | mINP | R1 | R5 | R10 | R20 |
|---|---|---:|---:|---:|---:|---:|---:|
'''
for mode, model in report['modes'].items():
    for stage, values in model['normal'].items():
        addition += '| ' + mode + ' | ' + stage + ' | ' + ' | '.join(f'{values[m]:.6f}' for m in METRICS) + ' |\n'
addition += '\n逐query对应比较（after−before）；正负变化全部保留：\n\n| 模式 | 比较 | 完整ΔmAP/ΔR1 | 完整首位误伤/救回 | 全49等条件均值ΔmAP/ΔR1 |\n|---|---|---:|---:|---:|\n'
for mode, model in report['modes'].items():
    for label, comparison in model['comparisons'].items():
        normal, mean = comparison['normal'], comparison['all49_equal_condition_mean_delta_pp']
        addition += f'| {mode} | {label} | {normal["delta_pp"]["mAP"]:+.6f}/{normal["delta_pp"]["Rank-1"]:+.6f} | {normal["Rank1_harm_queries"]}/{normal["Rank1_rescue_queries"]} | {mean["mAP"]:+.6f}/{mean["Rank-1"]:+.6f} |\n'
addition += '\n这些是同一已训练模型的出口检索诊断，不是公平重训消融或新增方法成绩。M_aux/F_aux为实际独立辅助监督表示；M_pre为实际关系质量、身份anchor和合法来源加权后的路由聚合，M_post仅在同样聚合前增加PM；F_pre/F_post仅相差PF。公共base_common为512D；deployed为5632D，其余7个同出口表示为512D，不以不同维度直接宣称新方法增益。有向交叉readout仅衡量投影方向与当前公共身份坐标的经验可比较性，不能证明信息被严格破坏、语义解耦、统计独立或因果效应。\n'
addition += '\n结论：两组M投影后的独立mAP都明显低于投影前，M—公共坐标交叉检索尤其弱；F投影也仍有下降，其有向比较好于M。当前M2关系保持及M2b公共对齐只作用于PF，不能声称同样保护了PM出口，也不能用这份冻结诊断证明F公共监督的因果作用。下一轮优先单因素检验实际路由PM聚合出口的公共身份监督，普通频域/普通双专家接受同样监督和预算；先审查、真实路径/梯度检查再重训，不能把这一待验证方向预写为已改善。本次冻结诊断本身没有训练更新，后续实验进度另按实际启动/完成凭证记录。\n'
addition += '\n原始全49/12readout六指标、CMC、逐query、完整GT/相机/scene/身份分组、7可用集合尺度CSV、提取runtime及CPU凭证在results/full_official_control_readout_20261004；stdlib分析readout_analysis.json和784行readout_condition_deltas.csv保留8类变化及same_availability7/overlap_mismatch30/source_disjoint12/partial_query_full_gallery6/both_partial36。条件/重叠分组反复使用591个查询，不当作独立样本或官方统一mAP。runtime含图片解码/DataLoader首batch及全部探针阶段提取，不是纯GPU单batch延迟。\n'
addition += f'\n该诊断98份raw.npz（每份12个距离矩阵），共{retention["total_bytes"]}字节，已逐份本地SHA256/大小核验并再次核对服务器原件后移除对应2026距离副本。本地路径：{retention["local_root"]}，archive_manifest.json及项目results/preflight/full_official_control_readout_distances_local_archive_20261004.json记录全部原始路径/哈希/大小；每个原始best和官方特征数组、源码、全量文字结果及GT凭证保留。加上之前基线294份和六组对照588份，本批累计980份距离原件已归档本地。复算CPU距离审计须先从本地恢复，不能把旧凭证写成新执行。只删除列明的自产距离文件，无权重二进制或NPZ上传GitHub。唯一交接project/Desktop/2026/2027逐字节一致；2025既有I/O故障镜像仍待恢复。三数据集同一方法/+2/+2、公平普通专家、多种子和真实协作净收益仍未成立，Goal ACTIVE/UNMET。\n'
document.write_bytes(text.replace(marker, marker + addition, 1).encode('utf-8'))
DESKTOP.write_bytes(document.read_bytes())
extras = [
    'analyze_full_official_control_readout.py', 'diagnose_full_official_control_readout.py',
    'audit_full_official_control_readout.py', 'launch_full_official_control_readout.py',
    'results/preflight/research_goal_optimized_20261003.json',
    'results/preflight/full_official_control_readout_source_review_20261004.json',
    'results/preflight/full_official_control_readout_plan_20261004.json',
    'results/preflight/full_official_control_readout_deployer_20261004.py',
    'results/preflight/full_official_control_readout_launch_20261004.json',
    'results/preflight/full_official_control_readout_observer_entrypoint_20261004.py',
    'results/preflight/full_official_control_readout_primary_observer_20261004.jsonl',
    'results/preflight/full_official_control_readout_primary_observer_terminal_20261004.json',
    'results/preflight/full_official_control_readout_intake_20261004.py',
    'results/preflight/full_official_control_readout_distances_archive_20261004.py',
    'results/preflight/full_official_control_readout_distances_local_archive_20261004.json',
    'results/preflight/full_official_control_readout_intake_archive_source_review_20261004.json',
    'results/preflight/full_official_control_readout_publisher_20261004.py',
    'results/preflight/full_official_control_readout_publisher_source_review_20261004.json']
files = [p.relative_to(PROJECT).as_posix() for p in root.rglob('*') if p.is_file()] + extras
assert all((PROJECT / name).suffix in ('.json', '.csv', '.jsonl', '.log', '.py') for name in files)
with tarfile.open(archive, 'w:gz') as handle:
    for name in files: handle.add(PROJECT / name, arcname=name)
digest = hashlib.sha256(document.read_bytes()).hexdigest()
expected = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in files}
expected['docs/实验交接.md'] = digest
for host in ('2026', '2027'):
    remote_root, _ = HOSTS[host]
    target = remote_root + '/full_official_control_readout_text_20261004.tar.gz'
    command(['scp', *OPTIONS, str(archive), host + ':' + target])
    remote_python(host, 'import tarfile;from pathlib import Path;p=Path(' + repr(target) + ');tarfile.open(p).extractall(' + repr(remote_root) + ',filter="data");p.unlink()')
    command(['scp', *OPTIONS, str(document), host + ':' + remote_root + '/docs/实验交接.md'])
    actual = json.loads(remote_python(host, 'import hashlib,json;from pathlib import Path;root=Path(' + repr(remote_root) + ');files=[p for p in (root/"results/full_official_control_readout_20261004").rglob("*") if p.is_file()];files += [root/n for n in ' + repr(tuple(extras) + ('docs/实验交接.md',)) + '];print(json.dumps({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))'))
    assert actual == expected
assert DESKTOP.read_bytes() == document.read_bytes()
proof.write_text(json.dumps(dict(status='BOTH_FULL_OFFICIAL_READOUTS_AND1176_GT_CASES_WITH_LOCAL_ARCHIVE_PUBLISHED',
    published_at=datetime.now().isoformat(timespec='seconds'), document_sha256=digest,
    verified_locations=['project', 'Desktop', '2026', '2027'], pending_mirror='2025_IO',
    text_files=len(files), cases=1176, repeated_condition_query_rows=695016,
    raw_distance_files_local=98, new_NN=0, training_updates=0, goal_complete=False), indent=2) + '\n', encoding='utf-8')
for host in ('2026', '2027'):
    remote_root, _ = HOSTS[host]
    command(['scp', *OPTIONS, str(proof), host + ':' + remote_root + '/results/preflight/' + proof.name])
files += ['docs/实验交接.md', proof.relative_to(PROJECT).as_posix()]
pathspec = Path(__file__).with_suffix('.pathspec')
pathspec.write_bytes(b'\0'.join(name.encode('utf-8') for name in files) + b'\0')
command(['git', 'add', '--pathspec-from-file=' + str(pathspec), '--pathspec-file-nul'], cwd=PROJECT)
command(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], cwd=PROJECT)
command(['git', 'commit', '-m', 'Archive raw distances locally and publish complete frozen outlet utility diagnostics'], cwd=PROJECT)
command(['git', 'push', 'origin', 'main'], cwd=PROJECT)
head = command(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).strip()
assert command(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=PROJECT).split()[0] == head
print('FULL_OFFICIAL_READOUT_AND_LOCAL_RAW_ARCHIVE_PUBLISHED', json.dumps(dict(head=head, document_sha256=digest, cases=1176, goal_complete=False)), flush=True)
