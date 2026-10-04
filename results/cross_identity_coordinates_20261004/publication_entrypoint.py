from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python, sync_handoff

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
p = PROJECT / 'results/preflight'
root = PROJECT / 'results/cross_identity_coordinates_20261004'
report = json.loads((root / 'analysis.json').read_text(encoding='utf-8'))
audit = json.loads((root / 'independent_cpu_audit.json').read_text(encoding='utf-8'))
decision = json.loads((root / 'mechanism_decision.json').read_text(encoding='utf-8'))
assert audit['status'] == 'PASS' and audit['cases'] == 1029 and audit['perquery_count'] == 216090
assert report['cases'] == 1029 and report['condition_query_rows'] == 216090
assert report['optimizer_updates'] == report['new_weights'] == report['official_test_uses'] == 0
assert decision['goal_complete'] is False
proof = p / 'cross_identity_coordinate_complete_publication.json'
assert not proof.exists()
now = datetime.now().isoformat(timespec='seconds')
metrics = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
variants = ('axis_shared', 'frequency_shared', 'twins_shared')
pairs = ('common_common', 'F_pre_F_pre', 'F_post_F_post', 'F_pre_common', 'common_F_pre', 'F_post_common', 'common_F_post')

def table(records, first='模型/检索坐标'):
    text = '| ' + first + ' | mAP | mINP | R1 | R5 | R10 | R20 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for label, values in records:
        text += '| ' + label + ' | ' + ' | '.join(f'{values[m]:.6f}' for m in metrics) + ' |\n'
    return text

normal = [(variant + '/' + pair, report['runs'][variant]['normal']['metrics'][pair]) for variant in variants for pair in pairs]
group_rows = []
for variant in variants:
    for group, row in report['runs'][variant]['groups'].items():
        for pair in pairs:
            group_rows.append(dict(variant=variant, group=group, conditions=row['conditions'], pair=pair,
                **row['sixmetrics_equal_condition_mean'][pair]))
assert len(group_rows) == 147
with (root / 'availability_groups_sixmetrics.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['variant', 'group', 'conditions', 'pair', *metrics])
    writer.writeheader(); writer.writerows(group_rows)
group_table = table([(r['variant'] + '/' + r['group'] + '/' + r['pair'], r) for r in group_rows
    if r['pair'] in ('common_common', 'F_post_common', 'common_F_post')])
harm_table = '| 模型 | 交叉方向 | 正常首位误伤/恢复 | 正常AP改善/下降/不变 |\n|---|---|---:|---:|\n'
for variant in variants:
    for pair, key in (('F_post_common', 'F_post_query_vs_common_query'), ('common_F_post', 'F_post_gallery_vs_common_gallery')):
        row = audit['runs'][variant]['conditions']['q_RNT_g_RNT'][key]
        harm_table += f'| {variant} | {pair} | {row["Rank1_harm_queries"]}/{row["Rank1_rescue_queries"]} | {row["AP_improved_queries"]}/{row["AP_worsened_queries"]}/{row["AP_unchanged_queries"]} |\n'
errors = {v: dict(matrix=report['runs'][v]['max_matrix_error'], sixmetrics_pp=report['runs'][v]['max_sixmetric_error']) for v in variants}

doc = PROJECT / 'docs/实验交接.md'
s = doc.read_text(encoding='utf-8')
begin_marker = '<!-- CURRENT_CROSS_COORDINATE_STATUS_START -->'
end_marker = '<!-- CURRENT_CROSS_COORDINATE_STATUS_END -->'
assert s.count(begin_marker) == s.count(end_marker) == 1
begin = s.index(begin_marker) + len(begin_marker)
end = s.index(end_marker, begin)
text = f'''
## 最新终态：冻结M2的跨身份坐标诊断1029组已独立核算（{now}）

控制器1704381实际COMPLETE，三组smoke、三组full均exit0；双轴和普通双专家顺序使用2026 GPU2，普通频域使用GPU3。M2原最佳权重固定且strict重载：正常部署5632D特征最大差异0、原fuse重构逐元素相同、state tensor版本和既有五个输入文件SHA不变、三个同坐标控制全部六指标/逐查询与前轮精确一致。没有optimizer更新、新权重、正式测试或门控/幅值干预。

每个模型49种可用性条件×七坐标配对=343组，三模型共1029组、216090条重复条件—查询记录。独立CPU从2026安装身份/相机/场景GT重新排除同身份同场景，重算六指标、CMC1..50、逐查询AP/INP/首末匹配、身份/相机/场景分组和误伤/恢复全部PASS。归一化512D缓存重建交叉距离矩阵误差上限2e−6，实际每模型最大误差为{json.dumps(errors)}；排序重算采用精确导出的原距离，不将NumPy/Torch浮点差异冒称逐元素相同。

完整三模态下七种坐标配对的六指标全部如下。箭头前是查询表示、后是图库表示；common为原公共基础坐标，F_pre为实际路由聚合后/投影前表示，F_post为实际PF投影后表示。三者各自归一化为512D，只用来诊断，不是新的部署描述子或训练结果。

{table(normal)}

按可用性分组，下面完整报告公共同坐标与两个实际PF交叉方向。三同坐标控制与F_pre两个交叉方向的完整147行分组六指标见results/cross_identity_coordinates_20261004/availability_groups_sixmetrics.csv；全部1029行原值见all1029_sixmetrics.csv。相同集合7、重叠错配30、不相交12为互斥49分组；部分查询→完整图库6与双方部分36是可重叠诊断组。条件等权均值不等于官方总体mAP，216090记录也不是独立样本数。

{group_table}

将实际PF替换公共查询/图库时，与公共同坐标相比的逐查询误伤和恢复如下。这里是冻结替换诊断，不是模型正常11−00的专家关闭因果证据；后者仍以M2原四状态结果为准。

{harm_table}

观察与后续判断：{decision['finding']} 本诊断表明的是经验身份可比较性，不能独立归因为全局旋转、路由信息丢失或贡献预测失准。关系保持对旋转不敏感；自检索有效与能加入原公共坐标是不同证据。下一单因素：{decision['next_step']}

不扩展为三数据集/多种子候选。M2双轴正常48.887510 mAP/60.952381 R1，相对原协议DeMo+1.279717/+1.904762，双+2未达标；其49条件仍比普通双专家低2.666757mAP/4.023324R1，完整11−00仅+0.000938mAP且R1不变，不能用此轮诊断降低验收或宣称推理协作有效。原有M2三best及必要基线/控制依赖保留，本轮新增权重0，无需重复清理。

用户“不管功率和温度了”已覆盖旧硬件约束，未采集或限制功率温度；神经计算仅2026 GPU2/3，2025/2027仅文本镜像。原始特征/距离NPZ留2026，不上传数据图像或权重。源码、结果文本和唯一人类交接写入Re-ID2，repo/Desktop/2025/2026/2027五份交接按字节一致。Goal继续ACTIVE/UNMET。

'''
doc.write_bytes((s[:begin] + text + s[end:]).encode('utf-8'))
goal = p / 'research_goal_optimized_20261003.json'
g = json.loads(goal.read_text(encoding='utf-8'))
g['status'] = 'ACTIVE_UNMET'; g['updated_at'] = now
g['revision'] = '20261004_cross_coordinate_1029_CPU_PASS_single_factor_decision'
g['current_evidence']['cross_identity_coordinates'] = dict(status='COMPLETE_INDEPENDENT_INSTALLED_GT_CPU_PASS',
    controller_pid=1704381, actual_smokes=3, actual_full=3, CPU_cases=1029, condition_query_rows=216090,
    optimizer_updates=0, new_weights=0, official_test_uses=0,
    analysis='results/cross_identity_coordinates_20261004/analysis.json',
    decision='results/cross_identity_coordinates_20261004/mechanism_decision.json', finding=decision['finding'])
g['next_action'] = dict(milestone=decision['next_milestone'], status='SINGLE_FACTOR_DEFINED_REQUIRES_SOURCE_REVIEW_AND_REAL_PREFLIGHT_BEFORE_TRAINING',
    action=decision['next_step'], hardware_dependency='Only2026 physicalGPU2/3,max2 NN,no temperature/power conditions.')
goal.write_text(json.dumps(g, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
entry = root / 'publication_entrypoint.py'; assert not entry.exists(); entry.write_bytes(Path(__file__).read_bytes())
files = [f for f in root.rglob('*') if f.is_file()]
assert all(f.suffix in ('.json', '.csv', '.log', '.jsonl', '.py') for f in files)
files.extend([goal, p / 'cross_identity_coordinate_observer_terminal.json'])
manifest = {f.relative_to(PROJECT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
manifest_path = root / 'mirror_manifest.json'; assert not manifest_path.exists()
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8'); files.append(manifest_path)
archive = Path('C:/Users/gb/.codex_tmp/cross_identity_coordinate_complete_mirror_text.tar.gz'); assert not archive.exists()
with tarfile.open(archive, 'w:gz') as tar:
    for f in files: tar.add(f, arcname=f.relative_to(PROJECT).as_posix(), recursive=False)
archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
for host, (remote_root, _) in HOSTS.items():
    remote_archive = remote_root + '/results/cross_identity_coordinate_complete_mirror_text.tar.gz'
    command(['scp', *OPTIONS, str(archive), host + ':' + remote_archive])
    code = f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({remote_root!r});a=Path({remote_archive!r})
assert hashlib.sha256(a.read_bytes()).hexdigest()=={archive_sha!r}
with tarfile.open(a,'r:gz') as tar:tar.extractall(r,filter='data')
m=json.loads((r/'results/cross_identity_coordinates_20261004/mirror_manifest.json').read_text())
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in m.items())
print(json.dumps(dict(status='EXACT_TEXT_MIRROR',host={host!r},files=len(m))))
'''
    print(remote_python(host, code).strip(), flush=True)
digest = sync_handoff()
proof.write_text(json.dumps(dict(status='EXACT_THREE_TEXT_MIRRORS_FIVE_HANDOFFS', at=now,
    files=len(manifest), archive_sha256=archive_sha, doc_sha256=digest, cases=1029,
    condition_query_rows=216090, optimizer_updates=0, new_weights=0, goal_complete=False,
    temperature_power_control=False), indent=2) + '\n', encoding='utf-8')
for host, (remote_root, _) in HOSTS.items():
    command(['scp', *OPTIONS, str(proof), host + ':' + remote_root + '/results/preflight/'])
owned = [root.relative_to(PROJECT).as_posix(), goal.relative_to(PROJECT).as_posix(),
    'results/preflight/cross_identity_coordinate_observer_terminal.json', proof.relative_to(PROJECT).as_posix(), 'docs/实验交接.md']
command(['git', 'add', '--', *owned], cwd=PROJECT)
command(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], cwd=PROJECT)
command(['git', 'commit', '-m', 'Publish frozen cross-identity coordinate compatibility audit and research decision'], cwd=PROJECT)
command(['git', 'push', 'origin', 'HEAD:main'], cwd=PROJECT)
print('CROSS_COORDINATE_COMPLETE_PUBLISHED', json.dumps(dict(head=command(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).strip(),
    doc_sha256=digest, files=len(manifest), goal_complete=False)), flush=True)
