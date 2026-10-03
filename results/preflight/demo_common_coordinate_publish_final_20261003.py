from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, command, remote_python, sync_handoff

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
root = PROJECT / 'results/common_coordinate_v12_trial_20261003'
analysis = json.loads((PROJECT / 'results/preflight/common_coordinate_complete_analysis.json').read_text())
states = json.loads((PROJECT / 'results/preflight/common_coordinate_states_analysis.json').read_text())
launch = json.loads((PROJECT / 'results/preflight/common_coordinate_2026_launch.json').read_text())
plan = json.loads((PROJECT / 'results/preflight/common_coordinate_plan.json').read_text())
preflight = json.loads((PROJECT / 'results/preflight/common_coordinate_actual_preflight.json').read_text())
cpu = json.loads((root / 'controlled_states/independent_cpu_audit.json').read_text())
assert analysis['status'] == 'ACTUAL_MATCHED_COMMON_COORDINATE_WAVE_AUDIT'
assert states['metric_cases'] == cpu['cases'] == 1176 and cpu['status'] == 'PASS'
assert json.loads((root / 'controller_result.json').read_text())['status'] == 'COMPLETE'
assert json.loads((root / 'controlled_states/independent_cpu_audit_exit.json').read_text())['exit_code'] == 0
assert preflight['optimizer_updates'] == 12 and preflight['amp_skips'] == 0
assert launch['selected_gpus'] == [2, 3]
assert sum(r['epochs'] for r in analysis['runs'].values()) == 200
assert sum(r['optimizer_steps'] for r in analysis['runs'].values()) == 1812
assert all(r['amp_skipped_steps'] == 0 for r in analysis['runs'].values())
for name, row in preflight['files'].items():
    saved = PROJECT / 'results/preflight/common_coordinate_actual_preflight' / name
    collected = root / name
    assert saved.read_bytes() == collected.read_bytes()
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == row['sha256']

code = f'''import hashlib,json
from pathlib import Path
root=Path({launch['output']!r})
process=Path('/proc/{launch['pid']}')
assert not process.exists() or (process/'stat').read_text().split()[2]=='Z'
assert json.loads((root/'controller_result.json').read_text())['status']=='COMPLETE'
sources={plan['sources']!r}
assert all(hashlib.sha256((root.parents[1]/name).read_bytes()).hexdigest()==sha for name,sha in sources.items())
weights=sorted(root.rglob('*.pth'))
assert len(weights)==4 and all(p.name=='best.pth' for p in weights)
print(json.dumps(dict(status='TERMINAL_BEST_ONLY',controller_absent=True,source_files=len(sources),
 weights=[dict(relative=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in weights])))
'''
inventory = json.loads(remote_python('2026', code))
inventory_file = PROJECT / 'results/preflight/common_coordinate_terminal_inventory.json'
assert not inventory_file.exists()
inventory_file.write_text(json.dumps(inventory, indent=2) + '\n')

metrics = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
variants = ('axis_shared', 'frequency_shared', 'twins_shared', 'demo_shared')
now = datetime.now().isoformat(timespec='seconds')
normal_table = '| 模型 | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Rank-20 | 最佳轮 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for variant in variants:
    run = analysis['runs'][variant]
    normal_table += '| ' + variant + ' | ' + ' | '.join(f'{run["full_metrics"][key]:.6f}' for key in metrics) + f' | {run["best_epoch"]} |\n'
old = json.loads((PROJECT / 'results/anytoany49_demo_20261003/MSVR310_demo_s42/full/result.json').read_text())['measurements']['q_RNT_g_RNT']['metrics']
normal_table += '| 原协议DeMo | ' + ' | '.join(f'{old[key]:.6f}' for key in metrics) + ' | 原固定best |\n'
state_table = '| 双轴状态 | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Rank-20 |\n|---|---:|---:|---:|---:|---:|---:|\n'
axis = states['variants']['axis_shared']
for state, values in axis['normal']['metrics'].items():
    state_table += '| ' + state + ' | ' + ' | '.join(f'{values[key]:.6f}' for key in metrics) + ' |\n'
missing_table = '| 查询→图库 | mAP | mINP | R1 | R5 | R10 | R20 | ΔmAP/原DeMo | ΔR1/原DeMo | ΔmAP/普通频域 | ΔR1/普通频域 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
raw = json.loads((root / 'frozen49/MSVR310_axis_shared_s42/result.json').read_text())
for condition, values in raw['measurements'].items():
    original = analysis['comparisons']['original_demo']['conditions'][condition]['delta_pp']
    frequency = analysis['comparisons']['frequency_shared']['conditions'][condition]['delta_pp']
    missing_table += '| ' + condition + ' | ' + ' | '.join(f'{values["metrics"][key]:.6f}' for key in metrics)
    missing_table += ' | ' + ' | '.join(f'{v:+.6f}' for v in (original['mAP'], original['Rank-1'], frequency['mAP'], frequency['Rank-1'])) + ' |\n'
comparison_table = '| 比较参照 | 49条件均值ΔmAP | 均值ΔR1 | 同时+2条件数 | 最差ΔmAP条件 |\n|---|---:|---:|---:|---|\n'
for reference, values in analysis['comparisons'].items():
    mean_ap = statistics.mean(v['delta_pp']['mAP'] for v in values['conditions'].values())
    mean_r1 = statistics.mean(v['delta_pp']['Rank-1'] for v in values['conditions'].values())
    worst = min(values['conditions'], key=lambda k: values['conditions'][k]['delta_pp']['mAP'])
    comparison_table += f'| {reference} | {mean_ap:+.6f} | {mean_r1:+.6f} | {values["double_plus2_conditions"]}/49 | {worst}: {values["conditions"][worst]["delta_pp"]["mAP"]:+.6f} |\n'
normal_delta = analysis['comparisons']['original_demo']['normal']['delta_pp']
change = axis['normal']['full_vs_base']
mean_change = axis['expert_changes']['full_vs_base']
axis_full = json.loads((root / 'controlled_states/MSVR310_axis_shared_s42/full/result.json').read_text())
contribution = axis_full['contributions']['q_RNT_g_RNT']
freq_v11 = PROJECT / 'results/shared_identity_v11_trial_20261003/development/MSVR310_frequency_shared_s42'
freq_v12 = root / 'development/MSVR310_frequency_shared_s42'
old_orders = [json.loads(line) for line in (freq_v11 / 'batch_orders.jsonl').read_text().splitlines()]
new_orders = [json.loads(line) for line in (freq_v12 / 'batch_orders.jsonl').read_text().splitlines()]
sampling_fields = ('epoch', 'step', 'names', 'partial_set')
assert all(all(a[k] == b[k] for k in sampling_fields) for a, b in zip(old_orders, new_orders)) and len(old_orders) == len(new_orders) == 453
repro = dict(status='OBSERVED_TRAINING_DIFFERENCE_NOT_LOCALIZED',sampling_and_partial_masks_exact=True,
    initial_CUDA_forward_exact_across_7_availability_sets=True,
    V11_frequency_normal=json.loads((freq_v11 / 'result.json').read_text())['full_metrics'],
    V12_frequency_normal=analysis['runs']['frequency_shared']['full_metrics'],
    limit='Initial forward equality does not establish identical autograd or training trajectories; cause of changed trained scores is unproven. Use this fresh matched campaign controls.')
(PROJECT / 'results/preflight/common_coordinate_frequency_repeat_record.json').write_text(json.dumps(repro, indent=2) + '\n')
phase = f'''### V12 终态：公共坐标接口完成同预算比较，仍未达标（{now}）

本轮仅使用2026服务器物理GPU2、3，两条串行队列完成全部四组fresh50。controller3565271和各训练/评测子进程已退出，observer24294已关闭；tensor28可用性检查通过，四组三步smoke合计12次真实更新、0次AMP跳步。正式训练合计200轮、1812次真实更新、0次AMP跳步；四组逐batch文件名顺序和部分模态集合完全一致，实际B64/K4。每组只保留最早并列开发mAP最佳best.pth，全部指标使用该固定权重；未用官方测试调参。共同接口将M/I的七关系增量均值投到已有512D公共坐标，私有5120D在四状态中保持00值；FFT、专家访问范围、联合路由、损失、缺失训练和5632D保持本轮计划。三个增强对照均100263046参数/100250758可训练参数，DeMo共享身份对照99396608/99385344。

**结果判断：双轴完整模态相对原DeMo的mAP{normal_delta['mAP']:+.6f}点、Rank-1{normal_delta['Rank-1']:+.6f}点，mAP仍未达到+2。相对同接口普通频域，mAP下降，49种组合也没有任何一组同时达到mAP/Rank-1各+2。原三数据集与缺失模态目标仍ACTIVE_UNMET。**

{normal_table}

49种查询/图库可用集合共196组完整模型评测，四组各49×6状态共1176组，全部完成。六项指标、CMC1..50、身份/相机/场景分组、逐查询AP/INP/首末匹配以及误伤/恢复均保存。独立CPU直接读取安装的MSVR训练文件名和固定dev身份，重建360条图库与210条查询的GT/顺序，对保存的FP32原始距离独立lexsort重排：1176组、246960条查询记录全部PASS，六指标最大误差0，CMC/分组/逐查询一致；原196组full11与状态评测11逐项精确重现。CPU不运行模型，无优化器更新。四个原始距离NPZ只保留远端，不上传权重或数据图像。

各条件等权均值仅为诊断，不是官方总体mAP。28/49条件超过原DeMo双+2不能代表所有缺失条件通过；相对同增强DeMo为17/49，同容量普通双专家为11/49，同容量普通频域为0/49。表中所有条件都报告，无单独缺失checkpoint选择。

{comparison_table}

{missing_table}

专家实际作用：00是同一双轴训练后的基础路径，不是独立DeMo；10只开M、01只开F、11完整合作，关闭状态同时关闭条件消息及交互，查询和图库均使用对应状态。公共/私有单块只是诊断，不能当成独立重训方法或按其指标改选checkpoint。

{state_table}

完整模态11−00：mAP{change['delta_pp']['mAP']:+.6f}点，Rank-1不变，首位恢复{change['Rank1_rescue_queries']}、误伤{change['Rank1_harm_queries']}，AP改善{change['AP_improved_queries']}、下降{change['AP_worsened_queries']}、不变{change['AP_unchanged_queries']}。普通公共接口减少了V11完整模态的融合伤害，但并未建立完整输入下推理协作增益。全49条件11−00等权平均mAP{mean_change['mean_delta_pp']['mAP']:+.6f}、Rank-1{mean_change['mean_delta_pp']['Rank-1']:+.6f}，累计条件—查询对首位恢复{mean_change['Rank1_rescue_queries']}、误伤{mean_change['Rank1_harm_queries']}，AP改善{mean_change['AP_improved_queries']}、下降{mean_change['AP_worsened_queries']}，0/49相对自身00双+2。不能将跨条件重复查询数解释为独立个人数。

贡献预测仍未校准：完整模态三目标标准差={json.dumps(contribution['target_std'])}，MAE={json.dumps(contribution['MAE'])}，误差远大于目标变化。相同合法GT参考及停止梯度算术正确，不等于准确预测联合价值；当前结果不能把高门控解释为有效贡献。12种来源完全不相交的私有库距离仍全为2，不具有跨来源检索信号；可比性依赖公共库，但双轴公共库49条件均值mAP/Rank-1={axis['equal_condition_means']['base_shared']['mAP']:.6f}/{axis['equal_condition_means']['base_shared']['Rank-1']:.6f}，仍需解决身份表征与互补目标。

复现边界如实保留：V12普通频域在7种可用集合的初始CUDA四状态前向与V11逐元素一致，453步数据顺序和缺失集合也相同，但最终训练成绩与V11不同。初始前向相同不能证明反向图或训练轨迹一致；原因未定位，不归因于某个未经验证的因素。比较以当前四组fresh50为准，不混入历史最佳控制。共享DeMo本轮终态与V11一致。

发布过程有一项纯整理失败：前一次发布helper按RUNNING状态读取epoch/latest，而已完成状态采用epochs和顶层更新数，产生KeyError:epoch；此前只收集11份不可变预检文件，未写交接、未提交Git，模型训练不受影响。原脚本、失败JSON与11份文件均保留；本终态整理直接使用完整result.json，无重训、无删除失败证据。预检通过记录与实际评测分别保存，不能互相代替。

结果目录results/common_coordinate_v12_trial_20261003；196行完整模型表MSVR310_all4_full49_metrics_196.csv，1176行状态表controlled_states/MSVR310_all4_sixstates_full49_metrics_1176.csv；独立复算controlled_states/independent_cpu_audit.json；分析为results/preflight/common_coordinate_complete_analysis.json及common_coordinate_states_analysis.json。逐组实际epoch耗时和峰值显存详见分析，未把不完整profiler结果包装为完整FLOPs。

远端新campaign实际只存在4个best.pth，均已清点并记录大小/校验值，无initial/last/smoke/epoch历史权重，本轮无需额外删除。历史基线和必要对照最佳权重保留，不重复执行旧清理。下一步依据已完成负结果选择一个最小机制改动；本配方不直接扩展三数据集或多种子，不把局部缺失均值进步当作方法已成功。最新V12推理延迟/吞吐尚未单独测量，不能拿旧V5速度冒充。

用户后续研究建议已接收，三个候选主张为：可用性一致的公共身份槽位、身份关系保持的频域细化、跨可用集合的校准协同。V12结果落在“普通频域仍更好”这一决策分支，优先检查共享出口与分工是否有额外价值，先测量固定七关系平均、合法集合数量与anchor幅值的实际关系，不未经测量改分母；随后按同容量/同预算比较原平均、有效集合平均和共享查询槽位，重点包含12个来源不相交组合。RKD式同源关系保持与部分→部分训练覆盖是后续独立因素，暂不同时堆入。估计头/控制头拆分也需单因素验证，当前贡献估计误差大，不能称已校准。清晰正向候选出现后冻结结构/超参，再至少3个配对种子及一个独立确认划分，按身份分组统计；官方最终测试继续隔离。这些为后续待实现/待验证计划，本轮没有新增神经实验，不自动降低原+2/+2验收目标，也不要求在完整DeMo上机械追加10点。所有后续神经任务仍只使用2026物理GPU2/3。本节与历史记录整合在同一交接文档，Desktop和2025/26/27对应文件保持相同字节。

'''
document = PROJECT / 'docs/实验交接.md'
text = document.read_text(encoding='utf-8')
assert '### V12 终态：公共坐标接口完成同预算比较' not in text
start = text.index('## 当前状态（')
end = text.index('\n', start)
text = text[:start] + f'## 当前状态（{now}）' + text[end:]
point = text.index('### V12 同一公共身份坐标的检索增量试验')
document.write_bytes((text[:point] + phase + text[point:]).encode('utf-8'))
digest = sync_handoff()
sync = PROJECT / 'results/preflight/common_coordinate_final_handoff_sync.json'
assert not sync.exists()
sync.write_text(json.dumps(dict(observed_at=now,sha256=digest,copies=['repository','Desktop','2025','2026','2027'],goal='ACTIVE_UNMET'),indent=2) + '\n')
helpers = ('cpu','collect','analyze','states_analyze','publish_final','publish_verified')
for short in helpers:
    name = f'demo_common_coordinate_{short}_20261003.py'
    (PROJECT / 'results/preflight' / name).write_bytes((Path('C:/Users/gb/.codex_tmp') / name).read_bytes())
stage = ['docs/实验交接.md','results/common_coordinate_v12_trial_20261003']
stage += ['results/preflight/common_coordinate_' + name for name in (
    'actual_preflight.json','actual_preflight','complete_analysis.json','states_analysis.json',
    'observer_terminal.json','publisher_initial_failure.json','terminal_inventory.json',
    'frequency_repeat_record.json','final_handoff_sync.json')]
stage += [f'results/preflight/demo_common_coordinate_{short}_20261003.py' for short in helpers]
command(['git','add',*stage], cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'], cwd=PROJECT)
command(['git','commit','-m','Report V12 matched50 full49 and1176 raw GT recount including negative expert utility'], cwd=PROJECT)
command(['git','push','origin','main'], cwd=PROJECT)
head = command(['git','rev-parse','HEAD'], cwd=PROJECT).strip()
assert command(['git','ls-remote','origin','refs/heads/main'], cwd=PROJECT).split()[0] == head
with Path('C:/Users/gb/memory/2026-10-03.md').open('a', encoding='utf-8') as handle:
    handle.write(f'\nDeMo {now} V12 actualterminal all4fresh50/200epochs/1812real0AMPskips; GPU2026physical2/3ONLY. Controller3565271/observer24294 closed. CPU78428 andcollector14543 consumed0; rawGT1176cases246960rows maxerror0,196full11exact. 3062texts88383622B collected once. Axisnormal48.947877/33.991023/62.857143/73.809524/78.571429/83.809524 ep22; vsoriginalDeMo mAP{normal_delta["mAP"]:+.6f}/R1{normal_delta["Rank-1"]:+.6f} FAILmAP+2. Matchedfreq49.492761/61.428571 ep30,twins49.068869/61.428571 ep19,demo47.512826/55.238095 ep19. All49double+2orig28/augdemo17/freq0/twins11; meanvsfreq-1.667915mAP/-1.448008R1. Normal11-00mAP-.004505/R1zero; all49mean+.276019/+.330418,101rescue67harm. ContributionMAEhigh notcalibrated. Frequencyinitialforward/samplingexact V11 but traineddifferencecauseunproven. Only4best actual, no useless newweights. Purepublication KeyErrorepoch preserved, no NNimpact; final usescompletedresults. Git{head};ONEdoc5SHA{digest}; goalACTIVE_UNMET, no blind expansion. All old exclusivehelpers neverrerun.\n')
print('V12_FINAL_PUBLISHED', json.dumps(dict(head=head,sha256=digest,cases=1176,goal='ACTIVE_UNMET',selected_gpus=[2,3])), flush=True)
