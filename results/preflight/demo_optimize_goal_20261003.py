from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, command, sync_handoff

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
now = datetime.now().isoformat(timespec='seconds')
objective = '以完整DeMo为强基线，建立可归因、可复现、跨模态可用集合一致的双轴互补Re-ID方法；先证明相对同增强、同容量和同维度普通专家的独特检索价值，再完成三个数据集、多种子、独立确认划分、完整缺失评测与真实部署成本验证。保留三个数据集完整输入mAP和Rank-1均至少提升2个百分点的内部验收目标。'
milestones = [
    dict(id='M0',status='COMPLETE_NEGATIVE',work='封存V12四组fresh50、196完整模型条件和1176状态的GT独立复算。',
         finding='双轴正常mAP48.947877/R1 62.857143；相对原DeMo+1.340085/+3.809524，mAP未达+2；49条件相对同容量普通频域均值-1.667915/-1.448008，0/49双+2。'),
    dict(id='M1',status='NEXT_NOT_IMPLEMENTED',work='先测量固定七关系均值、合法关系数量与anchor幅值的实际关系，再单因素比较原平均、有效集合平均与共享查询槽位公共出口；保留私有DeMo、FFT及原损失。',
         evidence='同预算/同维度/同有效参数预算的公共出口对照，特别检查12个来源不相交条件与公共块独立检索。禁止用未参与前向的参数凑容量。'),
    dict(id='M2',status='PENDING',work='在M1结果明确后，单因素验证同可用来源的停止梯度频域身份关系保持，参照自身须先证明有稳定身份结构。',
         evidence='辅助→路由→投影→最终融合逐阶段逐查询AP、首位恢复及误伤；公共身份出口约束朝向。不能用关系距离损失单独证明坐标对齐。'),
    dict(id='M3',status='PENDING',work='预先固定相同集合、部分重叠、完全不相交的集合对采样，覆盖部分→部分和部分→完整；所有对照共享覆盖、预算和采样。',
         evidence='公共身份可比较性及联合路由优于独立路由的重训证据；贡献估计与门控控制分开验证，并比较固定门控/普通注意力。目标只用fit训练身份。'),
    dict(id='M4',status='GATED',work='出现明确正向候选后冻结结构、超参和选择规则，至少3个配对种子，加一个预先登记的独立身份确认划分，再扩展统一方法到三个数据集。',
         evidence='按身份分组置信区间、配对变化、seed均值及标准差；49条件重复查询不算49倍独立样本。正式测试不用于版本选择。'),
    dict(id='M5',status='GATED',work='最终重训模块消融、独立/联合路由、单/双专家、同增强DeMo、普通频域和普通双专家；冻结关闭状态作为补充诊断。',
         evidence='每个核心模块有可解释的实际净检索价值，不能仅靠非零梯度、响应图或完整模型超过弱基线。'),
    dict(id='M6',status='GATED',work='统一版本完成正式评测与真实成本测量并收尾。',
         evidence='三个数据集mAP/mINP/R1/5/10/20、CMC1..50、全部7×7模态组合、逐查询和身份/相机/场景分组、误伤/恢复；参数、有效参数、维度、真实延迟/吞吐/峰值显存及FLOPs覆盖说明。'),
]
goal = dict(status='ACTIVE_UNMET',updated_at=now,execution_objective=objective,
    platform_goal_original_objective='监控训练正常进行，指标要全面超过DeMo两个点，包括缺失模态也要做',
    platform_objective_edit_supported=False,
    scope='Execution goal is refined here and in the single human handoff; the existing platform goal stays active and unfinished.',
    acceptance=dict(normal='同一个统一方法，在RGBNT201、RGBNT100、MSVR310的完整输入mAP和Rank-1各超过同协议完整DeMo至少2个百分点；多种子和独立确认验证稳定性。',
        missing='完整覆盖49种查询—图库集合条件，预先登记来源不相交、重叠、仅查询缺失、双方缺失等关键组；报告全部正负结果，并与正确屏蔽/同增强/同容量普通专家比较。不得用单个均值或污染基线替代全面证据。',
        mechanism='公平普通专家对照稳定胜出，实际协作在预定义关键组带来可重复净收益，模块必要性由重训消融与逐查询机制证据支撑。',
        quantitative_interpretation='保留原+2/+2内部验收；不额外要求每个模块都+2或在完整DeMo上再提高10点。49条件中的双+2达标数量持续报告，不能冒称全条件通过。'),
    milestones=milestones,
    constraints=dict(neural_host='2026',physical_gpus=[2,3],max_parallel_neural_jobs=2,
        training='公开预训练CLIP；开发比较每组fresh50、B64、数据集原K、同身份采样和同缺失集合序列；最早并列开发mAP最佳checkpoint。',
        change_control='每轮一个明确假设；保留负结果，不原样扩展已失败配方，不同时堆入三个新模块。',
        retention='每实验只留固定最佳权重，保留必要基线/控制最佳及当前依赖；及时删除确认为无用的自产权重，保留文本结果和失败证据。',
        publication='GitHub Re-ID2；只维护docs/实验交接.md，镜像Desktop/document及2025/26/27同名交接；五份字节一致。',
        observation='按预计结束时间或240秒观察，不因观察超时重启训练。'))
target = PROJECT / 'results/preflight/research_goal_optimized_20261003.json'
assert not target.exists()
target.write_text(json.dumps(goal,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

phase = f'''### 优化后的执行Goal与阶段验收（用户明确要求，{now}）

**目标：{objective}** 当前仍ACTIVE_UNMET，V12终态已发布于f7b666d0，四组50轮/全部缺失/独立复算完成属于阶段进展，方法尚未定型。平台接口仅支持读取目标和更改状态，不能改写活跃目标标题；本节和research_goal_optimized_20261003.json明确更新实际执行规范，原平台goal继续保持active。

验收分三层：首先是同增强、同可用性处理、同预算、同有效参数和同输出维度的普通专家对照，证明双轴分工的额外价值；其次是专家真正改变检索并在预定义关键组产生可重复净收益，逐查询同时报告误伤和恢复，重训消融与冻结关闭状态分别解释；最后才是统一版本三个数据集、多种子、独立确认及正式测试。保留完整输入mAP/Rank-1各+2点的原内部验收，不机械要求完整DeMo再涨10点或每个模块都涨2点。缺失评测完整报告7×7条件及来源不相交/重叠/仅查询缺失/双方缺失关键组，不用平均掩盖退步，双+2条件计数不冒充全部通过。

| 阶段 | 状态 | 下一步与退出条件 |
|---|---|---|
| M0 V12封存 | 已完成，负结果 | 四组50轮、196完整条件、1176状态已GT复算；不原样扩展这套失败配方 |
| M1 公共身份出口 | 下一优先，尚未实现 | 先测量合法关系数量、固定七关系均值与anchor实际影响；再单因素比较原平均/有效集合平均/共享查询槽位，特别检验12个来源不相交条件 |
| M2 频域关系保持 | 待M1结论 | 保留FFT和专家容量，同来源参照先证实稳定；验证路由、投影到融合的身份邻域及逐查询互补是否保留 |
| M3 跨集合协作 | 独立下一因素 | 固定同集合/重叠/不相交配对覆盖，普通对照同增强同预算；分开验证收益估计和门控控制，不重复V10所有联合状态必须更强的目标 |
| M4 泛化确认 | 清晰正向候选后 | 冻结结构/超参/选择规则，至少3配对种子与独立身份确认划分，统一方法扩展三个数据集，身份分组统计 |
| M5 机制消融 | 候选通过后 | 重训单/双专家、独立/联合路由、模块去除及同增强普通对照；关闭状态只作补充，不等同重训消融 |
| M6 最终收尾 | 证据通过后 | 六指标、CMC、全部缺失矩阵、逐查询/分组/误伤恢复，以及最新真实速度、吞吐、显存、参数/维度与FLOPs覆盖；最终测试保持隔离 |

执行约束：每轮只改一个可验证因素，记录负结果及下一决策；公共槽位、RKD式关系保护和跨集合目标是待验证设计，不能靠命名预判创新成立。所有神经任务仅2026物理GPU2/3，最多两个并行任务，其他服务器只做授权文本同步；预训练和50轮开发预算保持可比。每组指标使用同一开发最佳权重，不拼不同版本的数据集最优点；及时清理确定无用的自产权重，保留必要最佳基线/对照/当前依赖与失败文本。实验代码和结果推送Re-ID2，始终只有这份人类交接文档，Desktop与2025/26/27文件相同字节。

'''
document = PROJECT / 'docs/实验交接.md'
text = document.read_text(encoding='utf-8')
assert '### 优化后的执行Goal与阶段验收' not in text
start = text.index('## 当前状态（')
end = text.index('\n',start)
text = text[:start] + f'## 当前状态（{now}）' + text[end:]
point = text.index('### V12 终态：公共坐标接口完成同预算比较')
document.write_bytes((text[:point] + phase + text[point:]).encode('utf-8'))
digest = sync_handoff()
sync = PROJECT / 'results/preflight/research_goal_optimized_handoff_sync.json'
assert not sync.exists()
sync.write_text(json.dumps(dict(observed_at=now,sha256=digest,copies=['repository','Desktop','2025','2026','2027'],goal='ACTIVE_UNMET'),indent=2)+'\n')
helper = PROJECT / 'results/preflight/demo_optimize_goal_20261003.py'
helper.write_bytes(Path(__file__).read_bytes())
command(['git','add','docs/实验交接.md',str(target.relative_to(PROJECT)),str(sync.relative_to(PROJECT)),str(helper.relative_to(PROJECT))],cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=PROJECT)
command(['git','commit','-m','Refine active research goal with fair controls mechanism gates and independent confirmation'],cwd=PROJECT)
command(['git','push','origin','main'],cwd=PROJECT)
head = command(['git','rev-parse','HEAD'],cwd=PROJECT).strip()
assert command(['git','ls-remote','origin','refs/heads/main'],cwd=PROJECT).split()[0] == head
note = f'\nDeMo {now} user explicitly asked optimizegoal basedon V11/V12 critique. Executiongoal now fairness+real expertutility+availability-consistency BEFOREall3/multiseed/confirmation/finalcost. Originalnormalall3mAP/R1+2 preserved; no mandatory+10/permodule+2. M0V12sealednegativef7b666d0; M1legalrelation/anchor measurement thencommonpoolingcomparison, M2frequencyidentityrelations, M3partial-partialcoverage+separatedcalibration independentfactors; notyetNNimplemented. Clearpositivefreeze then3pairedseeds+independentIDsplit. Physical2026GPU2/3ONLYmax2, fresh50/GT/all49/best-only/ONEdoc. Platform API cannoteditactiveobjective, originalgoalkeptACTIVE_UNMET, executioncontractinJSON+samehumanhandoff updated. Git{head};5docSHA{digest}; allsamebytes.\n'
with Path('C:/Users/gb/memory/2026-10-03.md').open('a',encoding='utf-8') as handle:
    handle.write(note)
with Path('C:/Users/gb/MEMORY.md').open('a',encoding='utf-8') as handle:
    handle.write(note)
print('GOAL_EXECUTION_CONTRACT_PUBLISHED',json.dumps(dict(head=head,sha256=digest,platform_goal='active unchanged title',execution_goal='refined ACTIVE_UNMET',physical_gpus=[2,3])),flush=True)
