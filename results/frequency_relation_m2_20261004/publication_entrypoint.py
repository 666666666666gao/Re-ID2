from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tarfile

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python, sync_handoff

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
p = PROJECT / 'results/preflight'
root = PROJECT / 'results/frequency_relation_m2_20261004'
report = json.loads((root / 'analysis.json').read_text(encoding='utf-8'))
audit = json.loads((root / 'independent_cpu_audit.json').read_text(encoding='utf-8'))
old = json.loads((PROJECT / 'results/common_outlet_m1_v3_complete/independent_cpu_audit.json').read_text(encoding='utf-8'))
old_utility = json.loads((PROJECT / 'results/trained_outlet_utility_20261004/analysis.json').read_text(encoding='utf-8'))
baseline = json.loads((PROJECT / 'results/axis_collaboration_v4_missing_development27/MSVR310_demo_s42/full/clean.json').read_text(encoding='utf-8'))
assert audit['status'] == 'PASS' and audit['cases'] == 2058 and audit['perquery_count'] == 432180
assert report['cases'] == 2058 and report['condition_query_rows'] == 432180
proof = p / 'frequency_relation_m2_complete_publication.json'
assert not proof.exists()
now = datetime.now().isoformat(timespec='seconds')
metrics = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
variants = ('axis_shared', 'frequency_shared', 'twins_shared')
groups = dict(normal=lambda q, g: q == g == 'RNT', all49=lambda q, g: True,
    same_availability=lambda q, g: q == g,
    overlap_mismatch=lambda q, g: q != g and bool(set(q) & set(g)),
    source_disjoint=lambda q, g: not bool(set(q) & set(g)),
    partial_query_full_gallery=lambda q, g: q != 'RNT' and g == 'RNT',
    both_partial=lambda q, g: q != 'RNT' and g != 'RNT')
conditions = audit['runs']['axis_shared']['states']['conditions']
assert len(conditions) == 49
decision = dict(status='AUDITED_SINGLE_FACTOR_DECISION_NOT_FINAL_METHOD', at=now,
    goal_complete=False, normal_delta_original_demo_pp={}, fair_group_comparisons={},
    projection_identity_relations={}, limits=[
        'Single MSVR310 identity-heldout fit/dev, seed42; no official-test, multiseed or three-dataset result.',
        'Equal-condition means and repeated condition-query counts are diagnostics, not independent samples.',
        'Distance preservation is invariant to global rotation and does not prove common identity orientation or calibrated control.',
        'The reference is stopped per forward; the student route remains learnable, so this is not a frozen whole teacher.'
    ])
for variant in variants:
    measured = report['runs'][variant]['normal']['metrics']['11']
    decision['normal_delta_original_demo_pp'][variant] = {m: measured[m] - baseline[m] for m in metrics}
decision['axis_normal_both_plus2_original_demo'] = all(
    decision['normal_delta_original_demo_pp']['axis_shared'][m] >= 2 for m in ('mAP', 'Rank-1'))
rows = []
for variant in variants:
    current = audit['runs'][variant]['states']['conditions']
    previous = old['runs']['original_mean/' + variant]['conditions']
    assert set(current) == set(previous)
    for condition in current:
        rows.append(dict(comparison='M2_minus_M1_same_variant', model=variant, control=variant,
            condition=condition, **{m: current[condition]['metrics']['11'][m] - previous[condition]['metrics']['11'][m] for m in metrics}))
for control in ('frequency_shared', 'twins_shared', 'M1_augmented_demo'):
    comparison = audit['runs'][control]['states']['conditions'] if control != 'M1_augmented_demo' else old['runs']['original_mean/demo_shared']['conditions']
    decision['fair_group_comparisons'][control] = {}
    for group, predicate in groups.items():
        selected = [c for c in conditions if predicate(c.split('_')[1], c.split('_')[3])]
        delta = {m: statistics.mean(conditions[c]['metrics']['11'][m] - comparison[c]['metrics']['11'][m] for c in selected) for m in metrics}
        passed = [c for c in selected if all(conditions[c]['metrics']['11'][m] - comparison[c]['metrics']['11'][m] >= 2 for m in ('mAP', 'Rank-1'))]
        degraded = [c for c in selected if any(conditions[c]['metrics']['11'][m] < comparison[c]['metrics']['11'][m] for m in ('mAP', 'Rank-1'))]
        decision['fair_group_comparisons'][control][group] = dict(conditions=len(selected), delta_pp=delta,
            both_plus2_conditions=passed, either_primary_degraded_conditions=degraded)
    for condition in conditions:
        rows.append(dict(comparison='M2_axis_minus_control', model='axis_shared', control=control,
            condition=condition, **{m: conditions[condition]['metrics']['11'][m] - comparison[condition]['metrics']['11'][m] for m in metrics}))
assert len(rows) == 294
with (root / 'all49_paired_comparisons_sixmetrics.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['comparison', 'model', 'control', 'condition', *metrics])
    writer.writeheader()
    writer.writerows(rows)
for variant in variants:
    decision['projection_identity_relations'][variant] = {}
    for group in groups:
        stages = report['runs'][variant]['groups'][group]['utility_sixmetrics_equal_condition_mean']
        measured = dict(F_pre=stages['F_pre'], F_post=stages['F_post'],
            post_minus_pre_pp={m: stages['F_post'][m] - stages['F_pre'][m] for m in metrics},
            full_vs_base=report['runs'][variant]['groups'][group]['full_vs_base_equal_condition_delta'])
        if variant in ('axis_shared', 'frequency_shared'):
            prior = old_utility['runs']['original_mean/' + variant]['groups'][group]['sixmetrics_equal_condition_mean']
            measured['M2_minus_M1_F_pre_pp'] = {m: stages['F_pre'][m] - prior['F_pre'][m] for m in metrics}
            measured['M2_minus_M1_F_post_pp'] = {m: stages['F_post'][m] - prior['F_post'][m] for m in metrics}
        decision['projection_identity_relations'][variant][group] = measured
normal_full_gain = report['runs']['axis_shared']['normal']['full_vs_base']['delta_pp']
decision['normal_full_vs_base_pp'] = normal_full_gain
decision['no_candidate_expansion'] = not decision['axis_normal_both_plus2_original_demo']
decision['next_step'] = '完整解释实际F_pre/PF、11−00及各缺失组结果；若只改善PF独立方向而协作仍无净收益，下一轮单独检查公共身份朝向或分离贡献估计与控制，保持跨集合采样不变；若F_pre也下降，不把关系loss下降认作保留了身份信息。未形成稳定公平优势前不扩展三数据集/多种子，不增加门控温度或盲目放大残差。'
(root / 'mechanism_decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def six_table(records):
    text = '| 模型/状态 | mAP | mINP | R1 | R5 | R10 | R20 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for label, values in records:
        text += '| ' + label + ' | ' + ' | '.join(f'{values[m]:.6f}' for m in metrics) + ' |\n'
    return text

normal_records = [('旧同协议DeMo', baseline), ('M1同接口/同增强DeMo', old['runs']['original_mean/demo_shared']['conditions']['q_RNT_g_RNT']['metrics']['11'])]
for variant in variants:
    normal_records.append(('M1/' + variant, old['runs']['original_mean/' + variant]['conditions']['q_RNT_g_RNT']['metrics']['11']))
    normal_records.append(('M2/' + variant, report['runs'][variant]['normal']['metrics']['11']))
states_records = [(variant + '/' + state, run['normal']['metrics'][state]) for variant, run in report['runs'].items() for state in ('00', '10', '01', '11')]
stage_records = [(variant + '/' + stage, run['normal_utility']['metrics'][stage]) for variant, run in report['runs'].items() for stage in ('M_pre', 'M_post', 'F_pre', 'F_post', 'M_aux', 'F_aux')]
comparisons = '| 对照 | 条件组 | 条件数 | ΔmAP | ΔmINP | ΔR1 | ΔR5 | ΔR10 | ΔR20 | 双+2条件数 | 任一主指标下降条件数 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
for control, values in decision['fair_group_comparisons'].items():
    for group, row in values.items():
        comparisons += f'| {control} | {group} | {row["conditions"]} | ' + ' | '.join(f'{row["delta_pp"][m]:+.6f}' for m in metrics) + f' | {len(row["both_plus2_conditions"])} | {len(row["either_primary_degraded_conditions"])} |\n'
harm = '| 模型 | 完整11−00 ΔmAP/R1 | 首位误伤/救回 | AP改善/下降/不变 | 全49 ΔmAP/R1 | 全49误伤/救回重复次数 |\n|---|---:|---:|---:|---:|---:|\n'
for variant, run in report['runs'].items():
    full = run['normal']['full_vs_base']
    all49 = run['groups']['all49']
    pair = lambda value: f'{value["mAP"]:+.6f}/{value["Rank-1"]:+.6f}'
    harm += f'| {variant} | {pair(full["delta_pp"])} | {full["Rank1_harm_queries"]}/{full["Rank1_rescue_queries"]} | {full["AP_improved_queries"]}/{full["AP_worsened_queries"]}/{full["AP_unchanged_queries"]} | {pair(all49["full_vs_base_equal_condition_delta"])} | {all49["harm_condition_query_occurrences"]}/{all49["rescue_condition_query_occurrences"]} |\n'
training = {variant: dict(**run['training'], best_epoch=run['best_epoch'], relation_raw_first=run['relation_raw_first'], relation_raw_last=run['relation_raw_last']) for variant, run in report['runs'].items()}
doc = PROJECT / 'docs/实验交接.md'
s = doc.read_text(encoding='utf-8')
begin_marker = '<!-- CURRENT_DEMO_STATUS_START -->\n'
end_marker = '<!-- CURRENT_M2_STATUS_END -->'
assert s.count(begin_marker) == s.count(end_marker) == 1
begin = s.index(begin_marker) + len(begin_marker)
end = s.index(end_marker, begin)
delta = decision['normal_delta_original_demo_pp']['axis_shared']
text = f'''## 最新终态：M2三组50轮、全部49条件和2058组独立核算完成（{now}）

真实控制器1519220已结束，三个fresh公开CLIP/seed42/B64/K4模型各50轮、全部49条件及六状态/八阶段均实际完成，独立安装GT CPU重新排序核算2058组、432180条重复条件—查询记录，六指标、CMC1..50、逐查询AP/INP/首末匹配及身份/相机/场景分组全部PASS。三组各自真实更新、AMP跳过和best选择为{json.dumps(training, ensure_ascii=False)}。均按开发mAP最高且最早并列选择同一checkpoint；官方测试未用于训练或选择。

此轮仅full-view实际PF相对同输入、已路由F_pre.detach归一化非对角距离的SmoothL1，固定权重0.1。原FFT/专家/路由/贡献与门控、身份损失、部分查询→完整图库训练、5632D及采样预算不变；旧M1同模型逐批采样核对一致。三个新模型各100263558参数、100251270有效可训练参数；无新增头/参数。真实CUDA和三个三更新smoke先PASS才训练，初始state/七来源四状态/零权重AMP退化实际验证。

以下是完整输入的六指标。M2双轴相对旧DeMo为mAP{delta['mAP']:+.6f}、Rank-1 {delta['Rank-1']:+.6f}点，双+2={decision['axis_normal_both_plus2_original_demo']}。这不是官方全训练复现、最终统一版本或多种子证据，Goal仍ACTIVE/UNMET。

{six_table(normal_records)}

公平比较的全部诊断组如下；同集合7、重叠错配30、不相交12为互斥分组，其余组可重叠。条件等权平均不等于官方mAP，不能掩盖下降条件。双+2列仅对该行对照计数，不能冒认为旧DeMo所有49条件均达标。逐条件六指标和294行配对差值见results/frequency_relation_m2_20261004/all2058_sixmetrics.csv及all49_paired_comparisons_sixmetrics.csv，原始逐查询、分组、CMC和独立审计在同目录。

{comparisons}

四状态用于同一训练模型的冻结干预，00不是独立训练DeMo；所有分支关闭时也关闭相应条件消息与交互项。必须区分训练期辅助作用和推理期新增价值。

{six_table(states_records)}

{harm}

实际身份证据出口独立检索如下。M/F pre和post分别保持同一实际输入/聚合，仅跨PM/PF；辅助到路由同时改变条件/选择/聚合，不单归因于路由。部署为5632D，独立方向为512D，不能冒称它们输出维度相同。

{six_table(stage_records)}

关系目标在每次前向停止梯度，学生路线仍正常学习，并非冻结完整教师。距离关系对整体旋转不敏感，不能单独证明公共朝向或收益校准。M2及M1的F_pre/F_post绝对六指标、投影差值、所有组11−00和下一因素判断在mechanism_decision.json；不能仅根据关系loss下降或梯度非零宣布成功。下一步：{decision['next_step']}

仅2026物理GPU2/3，最多两NN；用户已取消温度和功率约束，未采集或调节这些设置。当前只留下本轮三份best.pth，无init/last/smoke权重；必要旧基线/对照/后续依赖保留，本次未删除权重。2025/2027仅文本镜像，原始FP32距离NPZ和best留2026。此文件仍是唯一人类交接文档，repo/Desktop/25/26/27字节一致。三个数据集同一方法双+2、公平结构必要性、至少三个配对种子及独立确认尚未完成，不降低验收或拼接不同版本峰值。

'''
doc.write_bytes((s[:begin] + text + s[end:]).encode('utf-8'))
goal = p / 'research_goal_optimized_20261003.json'
g = json.loads(goal.read_text(encoding='utf-8'))
g['status'] = 'ACTIVE_UNMET'
g['updated_at'] = now
g['revision'] = '20261004_M2_full2058_CPU_PASS_decision_pending_next_single_factor'
next(m for m in g['milestones'] if m['id'] == 'M2')['status'] = 'COMPLETE_SINGLE_FACTOR_AUDITED_NOT_FINAL_CANDIDATE'
g['current_evidence']['M2'] = dict(status='COMPLETE_INDEPENDENT_INSTALLED_GT_CPU_PASS', fresh50=3,
    full_conditions=147, controlled_states=882, utility_stages=1176, CPU_cases=2058,
    condition_query_rows=432180, analysis='results/frequency_relation_m2_20261004/analysis.json',
    decision='results/frequency_relation_m2_20261004/mechanism_decision.json',
    axis_normal_delta_original_demo_pp=delta, axis_normal_both_plus2=decision['axis_normal_both_plus2_original_demo'])
g['next_action'] = dict(milestone='M2_or_M3_single_factor_after_audited_evidence', status='ANALYZE_ACTUAL_OUTLET_AND_COOPERATION_BEFORE_NEXT_NN',
    action=decision['next_step'], hardware_dependency='Only 2026 physical GPU2/3, max two NN tasks; no temperature/power constraint.')
goal.write_text(json.dumps(g, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
entry = root / 'publication_entrypoint.py'
assert not entry.exists()
entry.write_bytes(Path(__file__).read_bytes())
files = [f for f in root.rglob('*') if f.is_file()]
assert all(f.suffix in ('.json', '.csv', '.log', '.jsonl', '.py') for f in files)
files.extend([goal, p / 'frequency_relation_m2_observer_terminal.json'])
manifest = {f.relative_to(PROJECT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
manifest_path = root / 'mirror_manifest.json'
assert not manifest_path.exists()
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
files.append(manifest_path)
archive = Path('C:/Users/gb/.codex_tmp/frequency_relation_m2_complete_mirror_text.tar.gz')
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as tar:
    for f in files:
        tar.add(f, arcname=f.relative_to(PROJECT).as_posix(), recursive=False)
archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
for host, (remote_root, _) in HOSTS.items():
    remote_archive = remote_root + '/results/frequency_relation_m2_complete_mirror_text.tar.gz'
    command(['scp', *OPTIONS, str(archive), host + ':' + remote_archive])
    code = f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({remote_root!r});a=Path({remote_archive!r})
assert hashlib.sha256(a.read_bytes()).hexdigest()=={archive_sha!r}
with tarfile.open(a,'r:gz') as tar:tar.extractall(r,filter='data')
m=json.loads((r/'results/frequency_relation_m2_20261004/mirror_manifest.json').read_text())
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in m.items())
print(json.dumps(dict(status='EXACT_TEXT_MIRROR',host={host!r},files=len(m))))
'''
    print(remote_python(host, code).strip(), flush=True)
digest = sync_handoff()
proof.write_text(json.dumps(dict(status='EXACT_THREE_TEXT_MIRRORS_FIVE_HANDOFFS', at=now,
    files=len(manifest), archive_sha256=archive_sha, doc_sha256=digest, cases=2058,
    condition_query_rows=432180, fresh50=3, retained_best=3, goal_complete=False,
    temperature_power_control=False), indent=2) + '\n', encoding='utf-8')
for host, (remote_root, _) in HOSTS.items():
    command(['scp', *OPTIONS, str(proof), host + ':' + remote_root + '/results/preflight/'])
owned = [root.relative_to(PROJECT).as_posix(), goal.relative_to(PROJECT).as_posix(),
    'results/preflight/frequency_relation_m2_observer_terminal.json', proof.relative_to(PROJECT).as_posix(), 'docs/实验交接.md']
command(['git', 'add', '--', *owned], cwd=PROJECT)
command(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], cwd=PROJECT)
command(['git', 'commit', '-m', 'Publish matched frequency relation results and complete independent utility audit'], cwd=PROJECT)
command(['git', 'push', 'origin', 'HEAD:main'], cwd=PROJECT)
print('M2_COMPLETE_PUBLISHED', json.dumps(dict(head=command(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).strip(),
    doc_sha256=digest, files=len(manifest), normal_axis_delta=delta, goal_complete=False)), flush=True)
