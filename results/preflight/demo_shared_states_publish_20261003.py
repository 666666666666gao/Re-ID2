from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python, sync_handoff

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
root = PROJECT / 'results/shared_identity_v11_frozen_states_20261003'
analysis = json.loads((PROJECT / 'results/preflight/shared_states_analysis.json').read_text())
assert analysis['status'] == 'ACTUAL_FROZEN_GT_CPU_RECOUNT_ANALYSIS' and analysis['metric_cases'] == 1176
assert json.loads((root / 'independent_cpu_audit_exit.json').read_text())['exit_code'] == 0
assert json.loads((root / 'controller_result.json').read_text())['status'] == 'COMPLETE'
now = datetime.now().isoformat(timespec='seconds')
metrics = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
table = '| 模型 | 状态 | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Rank-20 |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
for variant, values in analysis['variants'].items():
    for state, value in values['normal']['metrics'].items():
        table += '| ' + variant + ' | ' + state + ' | ' + ' | '.join(f'{value[key]:.6f}' for key in metrics) + ' |\n'
axis = analysis['variants']['axis_shared']
ordinary = analysis['variants']['frequency_shared']
change = axis['normal']['full_vs_base']
mean = axis['expert_changes']['full_vs_base']
old = json.loads((PROJECT / 'results/anytoany49_demo_20261003/MSVR310_demo_s42/full/result.json').read_text())['measurements']['q_RNT_g_RNT']['metrics']
base_delta = {key: axis['normal']['metrics']['00'][key] - old[key] for key in metrics}
axis_raw = json.loads((root / 'MSVR310_axis_shared_s42/full/result.json').read_text())
freq_raw = json.loads((root / 'MSVR310_frequency_shared_s42/full/result.json').read_text())
bad = 'q_T_g_NT'
bad_base_gap = axis_raw['measurements'][bad]['00']['mAP'] - freq_raw['measurements'][bad]['00']['mAP']
private_pairs = axis['private_disjoint_conditions']
phase = f'''### V11 固定最佳权重的四状态与公共/私有坐标诊断（{now}）

本轮实际完成：2026 四卡各跑一组 seven-bank smoke 和固定最佳 checkpoint 的 full49×六状态；四组均正常退出，controller3497679 已终态退出，observer32728 已关闭。共1176案例、每例六指标、CMC1..50、相机/场景/身份分组及210个逐查询记录；逐查询共246960条。原full11的196种模型—条件六指标与整个逐查询表精确复现V11；模型tensor版本和best/原始数组/训练终态/退出/旧49结果受保护。没有优化器步骤、没有新权重、没有官方测试使用。公共/私有块使用2−2cosine诊断，四状态沿用原归一化平方欧氏距离，完整11使用已验证同特征的原保存距离，不改基准数值。

独立CPU从安装的MSVR训练文件名重新解析身份、场景和相机，核对原固定dev身份/图库顺序/查询索引，以原FP32距离和独立lexsort重算全部六指标、CMC、分组和逐查询，1176案例全部PASS、六指标最大误差0。贡献表的query_index为图库/dev绝对行号，检索表query_index为查询序号，已用raw query_indices映射；同一full11图库下合法异场景正例/异身份负例和三项贡献算术通过1e−7检查。该检查不是位级FP32点积复算，也没有独立重选最难参考，因此不夸大为贡献预测器已校准。完整原始JSON/CSV在results/shared_identity_v11_frozen_states_20261003，汇总1176行表为MSVR310_all4_sixstates_full49_metrics_1176.csv，独立审计为independent_cpu_audit.json，分析为results/preflight/shared_states_analysis.json。四个原始距离NPZ共288694938字节仅保留远端私有证据，不上传Git。

正常三模态各状态如下。00=本次新方法训练后的基础私有+公共身份路径；10=独立M；01=独立F；11=完整协作。00不是独立训练的DeMo，不能把11−00解释成整个训练方法收益。base_private/base_shared仅为诊断分块，不是另一个重新训练的部署方法。

{table}

主要发现：完整三模态双轴11−00：mAP{change['delta_pp']['mAP']:+.6f}、mINP{change['delta_pp']['mINP']:+.6f}、Rank-1{change['delta_pp']['Rank-1']:+.6f}；首位误伤{change['Rank1_harm_queries']}、恢复{change['Rank1_rescue_queries']}，AP改善{change['AP_improved_queries']}、下降{change['AP_worsened_queries']}。因此新增专家已能改变实际排序，但当前完整协作在完整输入上有负作用，不能主张推理协同稳定有效。00相对原DeMo的完整六指标差为{json.dumps(base_delta,ensure_ascii=False)}；mAP仍未到+2，不能把关闭专家后的49.500661当作达标。

全部49条件的等条件诊断平均，双轴11−00六指标为{json.dumps(mean['mean_delta_pp'],ensure_ascii=False)}；首位恢复{mean['Rank1_rescue_queries']}、误伤{mean['Rank1_harm_queries']}；AP改善{mean['AP_improved_queries']}、下降{mean['AP_worsened_queries']}；0/49条件达到其自身00的mAP/Rank-1双+2。这些计数为重复查询在不同可用性条件下的条件—查询对，不是互不重复的个人查询；等条件均值不是官方汇总mAP。完整四状态关闭是在查询和图库双方同时执行，条件贡献目标则使用固定11图库只切换查询状态，两个诊断定义已分开记录。

12种模态来源完全不相交的查询/图库组合：{', '.join(private_pairs)}。这些组合中base00私有5120D的原始cosine距离逐元素恒为2，说明该坐标没有跨来源身份相似度信号；其AP只是固定图库顺序在全平局下的结果，不能解释为有效身份检索。公共512D在轴模型49等条件平均mAP{axis['equal_condition_means']['base_shared']['mAP']:.6f}/Rank-1{axis['equal_condition_means']['base_shared']['Rank-1']:.6f}，普通频域对应{ordinary['equal_condition_means']['base_shared']['mAP']:.6f}/{ordinary['equal_condition_means']['base_shared']['Rank-1']:.6f}。T查询→NT图库：在关闭专家00时双轴相对普通频域已经低{bad_base_gap:.6f}点mAP，开启专家只补回部分差距，故不能把最终缺失退步全归因于残差幅值或推理融合；训练得到的基础表示本身也有明显可用性差异。

实现/问题记录：新diagnose_shared_identity_states.py与四卡launcher经fresh Astra/max SOURCE/AST/stdlib审查PASS；新增独立CPU审计也source PASS，均same-family/provisional，实际运行证据另列。CPU启动helper第一次outer f-string换行转义导致远端代码解析失败，审计日志/退出/结果均未创建，未执行CPU审计；原helper与失败记录保存在trace。只删除exit JSON后不必要的换行，fresh reviewer复核两个payload与成功/失败退出保存PASS，修正后实际CPU exit0并完整复算成功。没有因此重跑神经网络或改指标阈值。

下一步决策：不扩展当前V11配方为最终三数据集主表。优先单因素检查M/F/I如何进入共同身份坐标：保持完整DeMo私有路径、同一主干/FFT/专家访问/联合路由/损失/采样/50轮与5632D，将新增检索增量在公共身份坐标中受相同度量训练；普通频域与普通双专家使用相同新接口和活跃参数，排除坐标接口造成的比较偏差。这是待实现和验证的假设，不是已证明能修复上述基础表征差异；仍必须重训、做关闭状态和全部49条件、保留负结果，再决定是否扩大三数据集和多种子。若只能带来训练期作用，应按训练期辅助方法定位，不强行称为推理协作。

原目标三个数据集mAP/Rank-1均≥同协议DeMo+2并覆盖全部缺失组合继续ACTIVE_UNMET。V11仍只有MSVR单种子；没有把多个版本最好结果拼成最终统一模型。训练预训练仍publicCLIP/fresh50，不使用这些开发诊断结果重新挑旧checkpoint。四个V11各实验best及基线/公平对照保留，诊断零新权重；此前已授权清理记录不重复执行。所有实验、完整六指标、错误与发现仍只写本交接文档，Desktop和2025/26/27对应同文件字节校验后同步。

'''
document = PROJECT / 'docs/实验交接.md'
text = document.read_text(encoding='utf-8')
assert '### V11 固定最佳权重的四状态与公共/私有坐标诊断' not in text
start = text.index('## 当前状态（')
end = text.index('\n', start)
text = text[:start] + f'## 当前状态（{now}）' + text[end:]
point = text.index('### V11 共享身份坐标与部分查询—完整图库训练')
document.write_bytes((text[:point] + phase + text[point:]).encode('utf-8'))

trace = PROJECT / '.aris/traces/experiment-bridge/2026-10-03_shared_states'
for name in ('cpu_audit_source_review.json', 'cpu_launcher_rescue_review.json', 'cpu_launcher_initial_failure.json'):
    (PROJECT / 'results/preflight' / ('shared_states_' + name)).write_bytes((trace / name).read_bytes())
helper_names = ('deploy', 'observe', 'collect', 'cpu', 'collect_cpu', 'analyze', 'publish')
for short in helper_names:
    name = f'demo_shared_states_{short}_20261003.py'
    (PROJECT / 'results/preflight' / name).write_bytes((Path('C:/Users/gb/.codex_tmp') / name).read_bytes())
for host in HOSTS:
    server_root, _ = HOSTS[host]
    sources = ('diagnose_shared_identity_states.py', 'launch_shared_identity_states.py', 'audit_shared_identity_states.py')
    for name in sources:
        command(['scp', *OPTIONS, str(PROJECT / name), host + ':' + server_root + '/' + name])
    expected = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in sources}
    code = 'import hashlib,json;from pathlib import Path;print(json.dumps({name:hashlib.sha256((Path(' + repr(server_root) + ')/name).read_bytes()).hexdigest() for name in ' + repr(sources) + '}))'
    assert json.loads(remote_python(host, code)) == expected
digest = sync_handoff()
sync = PROJECT / 'results/preflight/shared_states_handoff_sync.json'
assert not sync.exists()
sync.write_text(json.dumps(dict(observed_at=now,sha256=digest,copies=['repository','Desktop','2025','2026','2027'],goal='ACTIVE_UNMET'),indent=2) + '\n')
stage = ['diagnose_shared_identity_states.py','launch_shared_identity_states.py','audit_shared_identity_states.py',
    'docs/实验交接.md','results/shared_identity_v11_frozen_states_20261003',
    'results/preflight/shared_states_plan.json','results/preflight/shared_states_review.json',
    'results/preflight/shared_states_analysis.json','results/preflight/shared_states_2026_launch.json',
    'results/preflight/shared_states_observer_terminal.json','results/preflight/shared_states_handoff_sync.json']
stage += ['results/preflight/shared_states_' + name for name in ('cpu_audit_source_review.json','cpu_launcher_rescue_review.json','cpu_launcher_initial_failure.json')]
stage += [f'results/preflight/demo_shared_states_{short}_20261003.py' for short in helper_names]
command(['git','add',*stage], cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'], cwd=PROJECT)
command(['git','commit','-m','Audit1176 frozen expert states with independent raw GT recount and full query changes'], cwd=PROJECT)
command(['git','push','origin','main'], cwd=PROJECT)
head = command(['git','rev-parse','HEAD'], cwd=PROJECT).strip()
assert command(['git','ls-remote','origin','refs/heads/main'], cwd=PROJECT).split()[0] == head
with Path('C:/Users/gb/memory/2026-10-03.md').open('a', encoding='utf-8') as handle:
    handle.write(f'\nDeMo {now} goal PROGRESS after prior pelican-only turn NO_REID_PROGRESS: actual four frozen1176 cases passedall196 old11 exact; independent installedGT/rawdistances six/CMC/groups/perquery error0,246960rows. Fourstate normalaxis11−00mAP{change["delta_pp"]["mAP"]}/R1{change["delta_pp"]["Rank-1"]}, harm2/rescue0; all49mean+0.233002/+0.417881,86harm129rescue.12private-disjoint exactdistance2; axis T→NT00gapvsfreq{bad_base_gap}. No optimizer/newweights/test; controller3497679/observer32728 terminal, CPU86278/collector14313 consumed0. FirstCPUlauncherescapefailure preserved/reviewed minimalrepair thenCPUexit0. Git{head};ONEdoc5SHA{digest}; goalACTIVE_UNMET, next shared-coordinate increment trial is hypothesis notresult.\n')
print('FROZEN_SHARED_STATES_PUBLISHED', json.dumps(dict(head=head,sha256=digest,cases=1176,goal='ACTIVE_UNMET')), flush=True)
