from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python, sync_handoff

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
p = PROJECT / 'results/preflight'
root = PROJECT / 'results/measurement_gate_m3a_20261004'
report = json.loads((root / 'analysis.json').read_text(encoding='utf-8'))
audit = json.loads((root / 'independent_cpu_audit.json').read_text(encoding='utf-8'))
comparisons = json.loads((root / 'comparison_summary.json').read_text(encoding='utf-8'))
decision = json.loads((root / 'mechanism_decision.json').read_text(encoding='utf-8'))
parent = json.loads((PROJECT / 'results/identity_alignment_m2b_20261004/analysis.json').read_text(encoding='utf-8'))
baseline = json.loads((PROJECT / 'results/axis_collaboration_v4_missing_development27/MSVR310_demo_s42/full/clean.json').read_text(encoding='utf-8'))
assert audit['status'] == 'PASS' and audit['cases'] == 3087 and audit['perquery_count'] == 648270
assert audit['M3a_gate_gradient_contract_verified']
assert report['cases'] == 3087 and decision['goal_complete'] is False
assert comparisons['status'] == 'COMPLETE_AUDITED_COMPARISONS_NOT_FINAL_METHOD'
assert comparisons['evidence']['M3a']['sha256'] == hashlib.sha256((root / 'independent_cpu_audit.json').read_bytes()).hexdigest()
proof = p / 'measurement_gate_m3a_complete_publication.json'
assert not proof.exists()
now = datetime.now().isoformat(timespec='seconds')
metrics = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
variants = ('axis_shared', 'frequency_shared', 'twins_shared')


def table(records):
    text = '| 模型/状态 | mAP | mINP | R1 | R5 | R10 | R20 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for label, values in records:
        text += '| ' + label + ' | ' + ' | '.join(f'{values[m]:.6f}' for m in metrics) + ' |\n'
    return text


normal = [('原同协议DeMo', baseline), ('同公共接口/同缺失增强DeMo', comparisons['augmented_DeMo_normal'])]
states, stages, cross, group_records = [], [], [], []
harm = '| 模型 | 正常11−00 ΔmAP/R1 | 首位误伤/救回 | AP改善/下降/不变 | 全49 ΔmAP/R1 |\n|---|---:|---:|---:|---:|\n'
for variant in variants:
    run = report['runs'][variant]
    normal.extend([('M2b/' + variant, parent['runs'][variant]['normal']['metrics']['11']), ('M3a/' + variant, run['normal']['metrics']['11'])])
    states.extend((variant + '/' + s, run['normal']['metrics'][s]) for s in ('00', '10', '01', '11'))
    stages.extend((variant + '/' + s, run['normal_utility']['metrics'][s]) for s in ('base_common', 'M_pre', 'M_post', 'M_aux', 'F_pre', 'F_post', 'F_aux'))
    cross.extend((variant + '/' + s, run['normal_cross']['metrics'][s]) for s in ('common_common', 'F_pre_F_pre', 'F_post_F_post', 'F_pre_common', 'common_F_pre', 'F_post_common', 'common_F_post'))
    full, group = run['normal']['full_vs_base'], run['groups']['all49']
    harm += f'| {variant} | {full["delta_pp"]["mAP"]:+.6f}/{full["delta_pp"]["Rank-1"]:+.6f} | {full["Rank1_harm_queries"]}/{full["Rank1_rescue_queries"]} | {full["AP_improved_queries"]}/{full["AP_worsened_queries"]}/{full["AP_unchanged_queries"]} | {group["full_vs_base_equal_condition_delta"]["mAP"]:+.6f}/{group["full_vs_base_equal_condition_delta"]["Rank-1"]:+.6f} |\n'
    for name, values in run['groups'].items():
        group_records.append((variant + '/' + name, values['sixmetrics_equal_condition_mean']))
for control, values in report['fair_comparisons'].items():
    for name, delta in values.items():
        group_records.append(('双轴−' + control + '/' + name, delta))
for name, values in comparisons['comparisons']['M3a_axis_minus_augmented_DeMo']['groups'].items():
    group_records.extend([('同增强DeMo/' + name, values['reference_mean']), ('双轴−同增强DeMo/' + name, values['delta_pp'])])
counts = '| 全49比较 | ΔmAP/R1 | 两指标各+2条件数 | mAP退步条件数 | R1退步条件数 | 任一主指标退步条件数 |\n|---|---:|---:|---:|---:|---:|\n'
for name, values in comparisons['comparisons'].items():
    v = values['groups']['all49']
    counts += f'| {name} | {v["delta_pp"]["mAP"]:+.6f}/{v["delta_pp"]["Rank-1"]:+.6f} | {len(v["both_plus2_conditions"])}/49 | {len(v["mAP_degraded_conditions"])} | {len(v["Rank1_degraded_conditions"])} | {len(v["either_primary_degraded_conditions"])} |\n'
complement = '| 模型/组 | PF独立救回基础错误 | 其中联合救回/未救回 | F_pre救回经PF保留/丢失 | 四状态M独对/F独对/同对/同错 |\n|---|---:|---:|---:|---:|\n'
for variant, values in comparisons['query_complementarity'].items():
    for group in ('normal', 'all49', 'source_disjoint', 'partial_query_full_gallery'):
        c = values['groups'][group]['counts']
        complement += f'| {variant}/{group} | {c["F_post_rescues_base"]} | {c["F_post_rescues_realized_by_joint"]}/{c["F_post_rescues_missed_by_joint"]} | {c["F_pre_rescues_retained_by_PF"]}/{c["F_pre_rescues_lost_by_PF"]} | {c["M_only_correct"]}/{c["F_only_correct"]}/{c["both_correct"]}/{c["both_wrong"]} |\n'
calibration = '| 模型/组/通道 | 目标均值 | 目标标准差 | 预测均值 | 预测标准差 | MAE | 零预测MAE | 门控均值 | 门控标准差 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
for variant, values in comparisons['contribution_calibration'].items():
    for group in ('normal', 'all49'):
        for channel, c in values[group].items():
            calibration += '| ' + variant + '/' + group + '/' + channel + ' | ' + ' | '.join(f'{c[k]:.8f}' for k in ('target_mean', 'target_std', 'prediction_mean', 'prediction_std', 'MAE', 'zero_prediction_MAE', 'implied_gate_mean', 'implied_gate_std')) + ' |\n'
doc = PROJECT / 'docs/实验交接.md'
s = doc.read_text(encoding='utf-8')
start, end = '<!-- CURRENT_M3A_STATUS_START -->', '<!-- CURRENT_M3A_STATUS_END -->'
assert s.count(start) == s.count(end) == 1
begin, finish = s.index(start) + len(start), s.index(end)
text = f'''
## 当前终态：M3a 三组fresh50和完整3087项独立审计已完成（{now}）

这一轮只切断融合门控的检索梯度；同一收益估计器仍接受原贡献回归。没有新增独立控制头。M2b关系保持0.1、身份对齐0.1/温度0.07、初始化、FFT、专家、交互、联合路由、尺度、损失、缺失采样、参数和5632D不变。部分训练没有贡献回归目标的限制保留。源码审查same-family/provisional PASS；真实CUDA初始化、全部7×4关闭状态和AMP前向/损失值相同，全/部分检索到估计器梯度全部切断、回归梯度每参数有限非零；三组短训练3次真实更新、全有效梯度、strict reload通过后才训练50轮。

三组50轮全部完成；实际训练统计：{json.dumps({v:r['training'] | dict(best_epoch=r['best_epoch'], alignment=r['alignment_training']) for v,r in report['runs'].items()}, ensure_ascii=False)}。采样顺序和部分可用集合逐批与原M2b一致。固定开发mAP最高且最早并列checkpoint重载，完整49/六状态/八阶段/七交叉坐标全完成。安装GT独立CPU审计PASS3087项、648270条重复条件—查询记录，覆盖六指标、CMC1..50、逐查询AP/INP/首末匹配、身份/相机/场景分组及误伤/救回。官方测试用于训练/选择次数为0。

正常完整六指标及同协议原DeMo、同公共接口/缺失增强DeMo、原M2b配对如下。全部为单MSVR fit/dev/seed42开发结果，不能作为三数据集或正式官方协议方法成绩。

{table(normal)}

相对原DeMo双指标+2检查：{json.dumps(comparisons['normal_original_DeMo_gate']['axis_shared'], ensure_ascii=False)}。缺失评测使用正确屏蔽来源的同增强DeMo，不使用旧零输入污染制造优势；增强DeMo没有频域目标，不是等参数专家。普通频域/普通双专家才是同轮同有效参数、同5632D对照。

{table(group_records)}

{counts}

同集合7、重叠错配30、不相交12构成49的划分；另外列部分查询→完整图库6和双方部分36。完整294项公平差值、各条件退步/+2列表见comparison_summary.json及all49_baseline_and_expert_comparisons.csv。等条件均值不是官方统一mAP，重复查询也不是独立样本。

{table(states)}

{harm}

00是经新方法训练后的基础路径，不是独立DeMo。关闭专家同时关闭对应条件消息及联合交互，冻结干预用于诊断；训练必要性仍需要匹配重训。

{complement}

独立PF正确的查询只说明潜在互补，不保证联合应该救回。四状态M/F独对和PF独立检索是不同统计。收益估计诊断从GT审计过的逐查询contributions.csv重算，并比较固定零收益预测，检查是否只消除了偏差而未学到样本差异。门控按原sigmoid20映射从实际预测重算，没有改温度或重跑网络。固定图库目标与训练当前batch目标不同，不能把较小训练回归loss直接解释为已完成开发校准；用开发目标均值构造的常数基线只是乐观诊断，未用于训练或实际门控。

{calibration}

{table(stages)}

{table(cross)}

单因素结论：{decision['finding']} 下一步：{decision['next_step']}。完整三数据集mAP、Rank-1各超过DeMo至少2点、缺失、公平对照、真实协同、多种子和独立确认仍按原范围验收，Goal ACTIVE/UNMET。

源文件和文本发布Re-ID2；原始NPZ和binary最佳权重留2026。只保留三份新best和必要历史参照，没有smoke/initial/last。只用2026 GPU2/3、每卡串行、最多两个NN；用户取消温度和功率限制，不采集或控制它们。2025/2027只作文本镜像。仍仅这一份人类交接，repo/Desktop/25/26/27逐字节相同。

'''
doc.write_bytes((s[:begin] + text + s[finish:]).encode('utf-8'))
goal = p / 'research_goal_optimized_20261003.json'
g = json.loads(goal.read_text(encoding='utf-8'))
g['status'], g['updated_at'] = 'ACTIVE_UNMET', now
g['revision'] = '20261004_M3a_three50_all3087_CPU_PASS_single_factor_decision'
g['current_evidence']['M3a'] = dict(status='COMPLETE_INDEPENDENT_INSTALLED_GT_CPU_PASS', fresh50=3, CPU_cases=3087,
    repeated_condition_query_rows=648270, analysis='results/measurement_gate_m3a_20261004/analysis.json',
    decision='results/measurement_gate_m3a_20261004/mechanism_decision.json', finding=decision['finding'], official_test_uses=0)
g['next_action'] = dict(milestone=decision['next_milestone'], status='NEXT_STEP_FROM_AUDITED_SINGLE_FACTOR_EVIDENCE',
    action=decision['next_step'], hardware_dependency='Only2026GPU2/3,max2NN,no temperature/powerconditions')
goal.write_text(json.dumps(g, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
entry = root / 'publication_entrypoint.py'
assert not entry.exists()
entry.write_bytes(Path(__file__).read_bytes())
files = [f for f in root.rglob('*') if f.is_file()]
assert all(f.suffix in ('.json', '.csv', '.log', '.jsonl', '.py') for f in files)
files.extend([goal, p / 'measurement_gate_m3a_observer_terminal.json',
    p / 'measurement_gate_m3a_first_pair_primary_export.json',
    p / 'measurement_gate_m3a_first_pair_export_entrypoint_20261004.py',
    p / 'measurement_gate_m3a_first_pair_calibration.json',
    p / 'measurement_gate_m3a_first_pair_calibration_entrypoint_20261004.py'])
manifest = {f.relative_to(PROJECT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
manifest_path = root / 'mirror_manifest.json'
assert not manifest_path.exists()
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
files.append(manifest_path)
archive = Path('C:/Users/gb/.codex_tmp/measurement_gate_m3a_complete_mirror_text.tar.gz')
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as tar:
    for f in files:
        tar.add(f, arcname=f.relative_to(PROJECT).as_posix(), recursive=False)
archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
for host, (remote_root, _) in HOSTS.items():
    remote_archive = remote_root + '/results/measurement_gate_m3a_complete_mirror_text.tar.gz'
    command(['scp', *OPTIONS, str(archive), host + ':' + remote_archive])
    code = f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({remote_root!r});a=Path({remote_archive!r})
assert hashlib.sha256(a.read_bytes()).hexdigest()=={archive_sha!r}
with tarfile.open(a,'r:gz') as tar:tar.extractall(r,filter='data')
m=json.loads((r/'results/measurement_gate_m3a_20261004/mirror_manifest.json').read_text())
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in m.items())
print(json.dumps(dict(status='EXACT_TEXT_MIRROR',host={host!r},files=len(m))))
'''
    print(remote_python(host, code).strip(), flush=True)
digest = sync_handoff()
proof.write_text(json.dumps(dict(status='EXACT_THREE_TEXT_MIRRORS_FIVE_HANDOFFS', at=now, files=len(manifest),
    archive_sha256=archive_sha, doc_sha256=digest, cases=3087, condition_query_rows=648270, fresh50=3,
    retained_best=3, goal_complete=False, temperature_power_control=False), indent=2) + '\n', encoding='utf-8')
for host, (remote_root, _) in HOSTS.items():
    command(['scp', *OPTIONS, str(proof), host + ':' + remote_root + '/results/preflight/'])
command(['git', 'add', '--', root.relative_to(PROJECT).as_posix(), goal.relative_to(PROJECT).as_posix(),
    'results/preflight/measurement_gate_m3a_observer_terminal.json', proof.relative_to(PROJECT).as_posix(), 'docs/实验交接.md'], cwd=PROJECT)
command(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], cwd=PROJECT)
command(['git', 'commit', '-m', 'Publish measurement-only gate experiment and complete independent retrieval audit'], cwd=PROJECT)
command(['git', 'push', 'origin', 'HEAD:main'], cwd=PROJECT)
print('M3A_COMPLETE_PUBLISHED', json.dumps(dict(head=command(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).strip(),
    doc_sha256=digest, files=len(manifest), goal_complete=False)), flush=True)
