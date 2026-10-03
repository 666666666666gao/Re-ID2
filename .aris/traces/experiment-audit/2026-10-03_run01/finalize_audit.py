"""Assemble the read-only review and its forensic trace from observed evidence."""
from pathlib import Path
import datetime
import hashlib
import json
import sys

TRACE=Path(__file__).parent
ROOT=TRACE.parents[3]
snapshot=json.loads((TRACE/'inputs.snapshot.json').read_text(encoding='utf-8'))
records={r['path']:r for r in snapshot['files']}
verified=json.loads((TRACE/'deterministic_verification.json').read_text(encoding='utf-8'))
assert not verified['failures']
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
meta=json.loads((TRACE/'run.meta.json').read_text(encoding='utf-8'))
development='results/axis_collaboration_v4_development/'
diag='results/axis_collaboration_v4_diagnostic/MSVR310_axis_scaled_fullref_s42_cross26/'
pre='results/preflight/'
metrics=['mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20']

def evidence(path, line, detail):
    assert path in records and 1<=line<=len(records[path]['text'].splitlines()),(path,line)
    return {'path':path,'line':line,'reference':path+':'+str(line),'detail':detail}

def item(status, conclusion, facts, references, limitations):
    return {'status':status,'conclusion':conclusion,'details':facts,'evidence':references,'limitations':limitations,'integrity_blockers':[]}

checks={}
checks['A']=item('PASS','检索 GT 沿数据集文件名解析；未发现模型输出充当身份或相机/场景 GT。',[
    'RGBNT201 从文件名取六位身份与相机，RGBNT100 用身份/camera 正则，MSVR310 从文件名取身份、camera 与 scene；source_records 排序后按固定 dev_ids 隔离，只有 fit 身份重编码。',
    '14 张 CSV 的 14,145 行 query_index 均连续，姓名文件名解析的 identity/camera/scene 全部一致。V4 与独立 DeMo/V3 的配对 query、GT、有效标记、相关图库数及保留图库数完全一致。',
    '5 份训练 batch_orders 覆盖各自全部声明 fit 文件名，身份数分别为 MSVR310 103、RGBNT201 141、RGBNT100 33，全部与各自 dev_ids 不相交。',
    'RGBNT201/RGBNT100 排除 same-ID + same-camera；MSVR310 排除 same-ID + same-scene。MSVR310 的 360 个 dev triplet 中仅 210 个有跨场景正例并被选为 query；invalid_queries=0 不表示其余150个也做了 query。',
    '历史 metric_computation 与四状态程序在模型创建/计算前核对完整安装数据的 query indices、ids、cameras、scenes、names 顺序；本次核验这些回执与调用源码及本地 CSV。'
],[
    evidence('data/datasets/RGBNT201.py',79,'身份/相机文件名解析'),evidence('data/datasets/RGBNT100.py',63,'正则解析官方目录记录'),evidence('data/datasets/msvr310.py',81,'身份、camera、scene 解析'),
    evidence('experiment_data.py',24,'真实数据目录及稳定排序'),evidence('experiment_data.py',33,'身份隔离与可用query选择'),evidence('run_experiment.py',65,'loader GT 与 feature 同序积累'),
    evidence('C:/Users/gb/.codex_tmp/demo_axis_v4_complete_dev_metrics_20261003.py',34,'冻结数组对安装记录的五字段顺序核对'),evidence('diagnose_axis_collaboration.py',103,'诊断前完整安装GT顺序断言'),
    evidence(development+'MSVR310_axis_scaled_fullref_s42/development_metrics/metric_computation.json',5,'实际保存的安装GT顺序核对回执')
],['未读取远端数据集图片或原始 NPY/NPZ 标签数组；真实安装数据顺序这一层是历史运行回执支持，而不是审查者重新访问数据集所得。','未在线复审数据集官方评测实现；本地是标准 same-ID/camera-or-scene 排除规则的自有 ReID 评测适配器。'])

checks['B']=item('PASS','未发现以预测自身 max/min/mean 为分母篡改检索指标。',[
    'descriptor 的 L2 归一化与平方欧氏距离用于排序；AP、INP、CMC 的分母分别为真实相关数/排名或有效 query 数，再乘100表示百分点。',
    'V4 的投影方向归一化和停止梯度的 base 范数只改变模型 descriptor 构造，不是对最终 mAP/Rank 分数的自归一化。',
    '贡献校准的四状态 margin 来自归一化模型 descriptor；它是单独标注的模型派生诊断/训练目标，不是 ReID 性能 GT。',
    '全部六指标与逐 query 汇总相符；高分 RGBNT100 也保留较低 mINP 和负差值，未见用自身最佳值缩放到100。'
],[evidence('run_experiment.py',69,'descriptor L2与距离'),evidence('full_evaluation.py',22,'稳定排序和GT过滤'),evidence('full_evaluation.py',33,'AP/INP定义'),evidence('full_evaluation.py',43,'有效query均值×100'),evidence('utils/reid_evaluation.py',33,'CMC/AP原始定义'),evidence('scaled_axis_collaboration.py',48,'模型内部残差方向与幅度'),evidence('scaled_axis_collaboration.py',18,'独立的四状态校准目标')],['未重新运行神经网络或从距离数组重建排序；结论针对已读评测链及可复算文本。'])

checks['C']=item('PASS','固定五项结果、对应调用/退出回执及已引用数字均有本地文件支持。',[
    '初始128个要求输入全部存在；补入两份V3已列对照的4个JSON/CSV、冻结统计JSON/脚本、实际metric helper与采集调用链，最终137个输入全部冻结并保存SHA。',
    '5项 result/status/run/best 一致，退出码均0，完整log的50个EPOCH行与epochs.csv逐字段一致，TRAIN_COMPLETE内容与result一致；81份归档文件的大小/SHA与intake逐一相同。',
    '四份开发汇总内的六指标、六项差值、误伤/救回、AP提高/下降/相同及相应比例均由本次标准库重新计算并匹配。',
    'CURRENT和最后V4首两项章节的表格按6位小数匹配；当时06:46:31观察确有3项训练结束而2项已六指标审计，06:58:35冻结快照则有固定5项完成。历史文档时间戳保留，不把动态快照当成全11项终态。',
    '文档频域尾部0.024633/能量0.000609有新增读取的现存 frozen_stats JSON 支持；其权重SHA与诊断、数组SHA与metric_computation/跨机launch相同。该证据是冻结保存数组的历史CPU统计，不是expert-off结果。',
    '旧54项诊断源码审查与25项部署审查均明确same-family/provisional和静态/CPU mock范围；本次没有将其当作新的GPU执行证明。'
],[evidence(pre+'axis_collaboration_v4_first5_development_analysis.json',5,'固定五项汇总'),evidence(development+'RGBNT100_axis_scaled_fullref_s42/result.json',43,'COMPLETE50、best22、严格重载'),evidence(development+'RGBNT100_plain_scaled_fullref_s42/result.json',43,'COMPLETE50、best32、严格重载'),evidence(diag+'controller_result.json',2,'smoke/full均exit0'),evidence(diag+'intake.json',40,'21个归档文件的SHA/bytes'),evidence('docs/实验交接.md',10,'首两项汇总表'),evidence('docs/实验交接.md',775,'四状态数值表'),evidence(pre+'axis_collaboration_v4_msvr_frozen_stats.json',5,'原权重和数组SHA'),evidence(pre+'axis_collaboration_v4_msvr_frozen_stats.json',13,'冻结tail norm/energy历史统计'),evidence('C:/Users/gb/.codex_tmp/demo_axis_v4_msvr_frozen_stats_20261003.py',18,'CPU加载及tail统计真实源码')],['文档当前顶区仍是06:47历史快照；下一次发布需更新至明确的已审集合，不能把本审查外的新结束运行自动纳入。','原权重、原始数组、states.npz及remote/Desktop同步只核验本地保存的回执与哈希链；没有本次远端二进制重读或跨机同步实测。'])

checks['D']=item('PASS','full_metrics、mINP、CMC与Ranks有真实调用链和结果文件；可用文本的逐query复算通过。',[
    'V4补充metrics helper在CUDA隐藏后调用full_evaluation.full_metrics，写出full_metrics JSON/CSV与metric_computation，再按SHA收集；这与full_evaluation.main的旧baseline官方test流程分开。',
    'MSVR310四状态主程序对00/10/01/11逐个调用distance和full_metrics，并写状态CSV与完整metric报告；smoke/full命令、exit0、log和同SHA输出相互对应。',
    '本次实际标准库复算14张表、14,145个query、595个camera/scene/identity分组。六指标最大误差1.4210854715202004e-14pp；50点CMC和逐query INP误差0；分组最大误差4.263256414560601e-14pp。',
    '每条INP由 relevant_gallery/last_match 重算，Ranks与50点CMC由 first_match 重算；mAP由CSV所存AP均值重算。',
    '两份贡献CSV共420条，按记录R00/R10/R01/R11的float32逐次差分复算三目标全部误差0；六组校准均值/标准差/MAE/RMSE/符号/协方差最大差4.412695053107596e-10。'
],[evidence('C:/Users/gb/.codex_tmp/demo_axis_v4_complete_dev_metrics_20261003.py',29,'导入实际full_metrics'),evidence('C:/Users/gb/.codex_tmp/demo_axis_v4_complete_dev_metrics_20261003.py',44,'CPU实际调用与原指标/重载断言'),evidence('full_evaluation.py',20,'六指标及分组函数定义'),evidence('full_evaluation.py',48,'evaluate_reid交叉核对与CMC输出'),evidence('diagnose_axis_collaboration.py',140,'四状态实际metric调用'),evidence('diagnose_axis_collaboration.py',146,'两个固定参考贡献调用'),evidence(diag+'full.log',14,'实际完成日志含四状态六指标'),evidence(diag+'full/state_11.csv',2,'逐query原始文本'),evidence(diag+'full/full_reference_contributions.csv',2,'逐query四状态R及目标/预测')],['CSV未保存所有正例位置，因此不能仅凭first_match/last_match完整重建单query AP；本次不声称从原始距离矩阵独立重算AP。','未重新执行GPU、torch/numpy导入、严格加载权重、feature逐bit比较或hard-positive/negative搜索；这几项只能按已保存回执和源码判断。'])

checks['E']=item('WARN','证据支持固定五项单seed开发结果；用户效果目标未达成，完整对照与最终协议仍未覆盖。',[
    '固定5项是MSVR310 axis、RGBNT201 axis/plain、RGBNT100 axis/plain，均seed42、50轮，总250 epoch/12723次batch尝试=12723次optimizer更新、AMP跳过0。best epoch依次37/42/48/22/32，全部是唯一最大dev mAP；源码严格>保证并列时保留最早。',
    '5份(epoch,step,names)与同数据集DeMo完整逐项一致，每batch64；日志、逐batch和累计epoch计数互相吻合。RGBNT100 YAML原为30轮/128batch，但configuration明确覆写到50/64，实际run config与250轮CSV均相符。',
    'RGBNT201 axis/plain总参数100355206、可训练100343430；RGBNT100二者98857094/98845318；MSVR310 axis99681414/99669638。5项都是5632维。保存的预检有11个3-step smoke、5515个可训练tensor梯度检查全true；这是历史运行记录，不是本审查重新计数模型权重或执行梯度。',
    '三项axis对独立DeMo的mAP/Rank1差值依次为MSVR -1.724306952/-0.952380952pp、RGBNT201 +0.226174961/+2.060606061pp、RGBNT100 +0.653117096/+1.632pp。没有一个同时达到两项+2，三数据集整体目标为FAIL_EFFICACY，不是FAIL_INTEGRITY。',
    'axis相对同参数plain：RGBNT201 mAP/Rank1 -1.072672947/-3.272727273pp；RGBNT100 +1.792459127/+0.48pp。不能据两个数据集一个seed宣称统一优于普通专家或稳定改善。',
    '训练使用固定identity-heldout fit子集；不是完整官方train论文复现，也不是官方test最终结论。当前审查不涵盖其余6项完整对照、新版V4重复seed或缺失模态性能、训练独立routing消融。',
    'V1的27个checkpoint/351条件/seed42,43,44仅核对既有汇总标签及1/13、0/13、0/13门槛标签，不重算旧351条件，不把它归属V4。',
    '用户验收是三个数据集mAP和Rank-1各至少+2并覆盖缺失模态；完整六指标必须报告，不能把辅助all_six_2pp或every_dataset_and_condition标志自动改写为用户要求全部六指标/每一条件都+2。',
    '可审成本是参数量、训练wall_seconds、训练peak_memory及MSVR诊断耗时。固定输入不包含完整V4 FLOPs/吞吐/推理延迟对照profile；文档“全部成本”应限定为实际已保存成本。'
],[evidence('run_experiment.py',37,'有效batch/epoch固定'),evidence('run_experiment.py',200,'实际50轮loop与同seed采样'),evidence('run_experiment.py',220,'最高dev mAP且并列保最早'),evidence('run_experiment.py',228,'strict=True重载和四指标相等断言'),evidence('run_experiment.py',99,'native GradScaler与更新/跳过计数'),evidence('configs/RGBNT100/DeMo.yml',33,'原YAML配置由runner覆写'),evidence(pre+'axis_collaboration_v4_first5_development_analysis.json',368,'三数据集axis真实门槛'),evidence(pre+'axis_collaboration_v4_first5_development_analysis.json',403,'同参数plain对照'),evidence(pre+'axis_collaboration_v4_first5_development_analysis.json',430,'单seed开发与缺失验证范围'),evidence('docs/实验交接.md',17,'V1与V4版本区分'),evidence('docs/实验交接.md',771,'现存成本描述须按具体字段解释')],['目前为开发阶段观察，不能支持稳定、完整官方train、完整新版缺失模态或全部V4对照结论。','开发集用于checkpoint选择与方法调试，不能作为完全未参与选择的最终泛化评测。','本审查不暂停或重启后台任务；06:58:35快照为读取时的历史状态，后来运行进度排除在固定范围外。'])

checks['F']=item('PASS','真实GT检索性能与模型派生贡献诊断已区分；没有可接受的因果/信息论/独立路由优越性结论。',[
    '五项开发检索、三项独立DeMo开发对照、两项V3已引用对照和MSVR四状态检索分类为real_gt。',
    '四状态margin贡献目标分类为synthetic_proxy（模型派生的经验margin，使用真实identity限制参考正负样本）；它不是伪造检索标签，也不是信息论协同量。',
    '训练fullref表示停止梯度的当前batch full11 gallery，同身份非自身为正例，不排除同camera/scene；冻结诊断的完整dev reference分别使用00和11并加入相机/场景排除。这是不同目标分布，不能据诊断校准差直接断言训练损失实现错误。',
    '00是同一jointly-trained checkpoint内部base路径加零尾部，不是独立DeMo；10/01用未交换条件的m0/f0，只开单专家，11保留条件消息、psi和交互投影。它们是冻结功能干预，不能替代重新训练消融，亦不能把bundled差异归因到某一个模块。',
    '11相对00/10/01的mAP为-0.012381678/-0.082757610/-0.043033103pp，Rank1为0/0/+0.476190476pp。00->11有16 AP提高/50下降/144相同，Rank1误伤/救回0/0；实际改变排序不等于联合收益。',
    '路由L1=0.014377150684595108表明joint分布与其边缘乘积有差异，本身不证明比独立routing训练更有效。plain_twins本身仍用同一joint router，因此不是independent-router ablation。',
    '完整dev11参考的Pearson(M,F,I)=-0.106541974/+0.051685510/-0.104222243，符号一致率0.504761905/0.409523810/0.476190476；F预测100%为负而目标59.047619%为正。数值由本次CSV复算，解释上仅支持校准失配诊断。'
],[evidence('scaled_axis_collaboration.py',18,'训练current-batch full11派生目标'),evidence('axis_collaboration.py',218,'00/10/01/11定义'),evidence('axis_collaboration.py',259,'joint消息与路由真实组合'),evidence('diagnose_axis_collaboration.py',63,'frozen dev参考及跨相机/场景正例'),evidence('diagnose_axis_collaboration.py',195,'路由对自身边缘乘积L1'),evidence('diagnose_axis_collaboration.py',208,'不同参考与非因果/非信息论边界'),evidence(diag+'full/diagnostic.json',6,'训练参考与frozen诊断参考分别记录'),evidence(diag+'full/diagnostic.json',2550,'full11参考校准报告'),evidence('docs/实验交接.md',782,'同模型00与冻结干预边界'),evidence('docs/实验交接.md',784,'校准失配与解释边界')],['没有独立训练的无消息/无psi/独立routing对照，不能宣称联合路由已验证优于独立路由。','路由张量、gates及残差norm平均值只与保存报告对应，未从远端states.npz重新计算。'])

changed=[]
for p,r in records.items():
    actual=Path(p) if Path(p).is_absolute() else ROOT/p
    current=hashlib.sha256(actual.read_bytes()).hexdigest()
    if current!=r['sha256']:
        changed.append({'path':p,'audited_sha256':r['sha256'],'current_sha256':current,'observed_at':now,'treatment':'The frozen audit snapshot remains authoritative; no newer contents incorporated.'})

report={
 'audit_skill':'experiment-audit','verdict':'WARN','overall_verdict':'WARN','integrity_status':'warn',
 'reason_code':'fixed_single_seed_development_scope_and_raw_binary_verification_not_repeated',
 'summary':'本地文本证据与标准库数值复算通过；固定五项效果门槛未达成，完整范围不足。没有发现伪GT、自归一化检索分数或虚构结果，不能据此声称完整远端原始数组/GPU复验。',
 'generated_at':now,'date':'2026-10-03','execution_started_at':meta['started_at'],'execution_timestamp_basis':'First saved input capture and actual finalization clock; native spawn dispatch timestamp was not exposed.',
 'agent_id':'/root/audit_v4_frozen_evaluation','verdict_id':'/root/audit_v4_frozen_evaluation','agent_id_kind':'canonical task name; native UUID unavailable',
 'executor_model':meta['executor_model'],'executor_family':'openai','reviewer_model':'gpt-6-astra','reviewer_reasoning':'max','reviewer_reasoning_effort':'max','reviewer_family':'openai','reviewer_runtime_attribution':'Native spawn parameters confirmed by parent; fork_turns=none. No alternate reviewer or child agent.',
 'review_independence':'same-family','acceptance_status':'provisional','trace_path':TRACE.relative_to(ROOT).as_posix(),
 'human_report_policy':'No EXPERIMENT_AUDIT.md or other handoff was created. Parent will merge a human summary into the sole docs/实验交接.md.',
 'checks':checks,'blocking_items':[],
 'acceptance_goal':{'user_requirement':'三个数据集的 mAP、Rank-1 均提升至少2点，并覆盖缺失模态；完整报告mAP/mINP/Rank-1/5/10/20和全部对照。','efficacy_status':'FAIL_FOR_THE_THREE_AUDITED_V4_AXIS_DEVELOPMENT_GATES','overall_user_goal':'NOT_ACHIEVED','must_not_infer':'Do not replace this with all six metrics +2 or every missing condition +2 solely because auxiliary strict flags exist.','remaining_scope':'Unaudited remaining V4 controls, V4 missing-modality efficacy, repeated seeds, full official-train/final-test protocol, independent-routing trained ablation, and full V4 cost profile.'},
 'actual_actions':{'standard_library_CPU_recomputation':True,'experiment_module_imports':0,'torch_or_numpy_imports':0,'package_installs':0,'GPU_queries_or_forwards':0,'optimizer_updates':0,'SSH_or_SCP_calls':0,'remote_binary_reads':0,'experiment_source_edits':0,'training_or_monitor_restarts':0,'subagents_spawned':0,'GitHub_publications':0,'handoff_edits':0},
 'deterministic_verification':{'status':'PASS','checks':verified['counts'],'path':(TRACE/'deterministic_verification.json').relative_to(ROOT).as_posix(),'program_path':(TRACE/'verify_local_text.py').relative_to(ROOT).as_posix(),'program_sha256':hashlib.sha256((TRACE/'verify_local_text.py').read_bytes()).hexdigest(),'python':sys.executable,'python_version':sys.version,'table_count':len(verified['metric_tables']),'query_rows':sum(t['queries'] for t in verified['metric_tables'].values()),'groups':sum(t['groups'] for t in verified['metric_tables'].values()),'max_six_metric_abs_error_pp':max(t['max_metric_error'] for t in verified['metric_tables'].values()),'max_CMC_abs_error_pp':max(t['max_cmc_error'] for t in verified['metric_tables'].values()),'max_group_abs_error_pp':max(t['max_group_error'] for t in verified['metric_tables'].values()),'max_per_query_INP_abs_error':max(t['max_per_query_INP_error'] for t in verified['metric_tables'].values()),'intake_hash_matches':len(verified['intake_hash_checks'])},
 'completed_run_evidence':verified['runs'],'metric_table_recomputations':verified['metric_tables'],'paired_query_recomputations':verified['query_pair_comparisons'],'contribution_recomputations':verified['calibration'],
 'dynamic_snapshot':verified['snapshots']['axis_collaboration_v4_latest_snapshot.json'],'input_changes_after_frozen_capture':changed,
 'audited_input_hashes':{p:'sha256:'+r['sha256'] for p,r in records.items()},'audited_input_manifest':(TRACE/'inputs.sha256.json').relative_to(ROOT).as_posix(),
 'scope_limits':['Only fixed five V4 training outcomes and one completed MSVR diagnostic were audited; subsequent live completions are excluded.','Each scalar AP source rank list and original distance/feature/weight arrays were unavailable to this local text-only audit.','Source-reviewed historical remote execution receipts are not a fresh GPU run.','V1 351-condition raw evaluation was not recomputed.','Baseline official_test metrics are present in source JSON but were not independently re-evaluated or used for V4 model acceptance.']
}
target=ROOT/'results/preflight/axis_collaboration_v4_first5_integrity_audit.json'
assert not target.exists()
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

prompt='''按 C:/Users/gb/.codex/skills/experiment-audit/SKILL.md 做一次 fresh 只读证据完整性审查。必须先读该技能及引用的 local-codex-policy、review-tracing 指令。用户要求只有一个人工交接文档，故不要新建 EXPERIMENT_AUDIT.md；可只写 JSON 和机器 trace，我会把人工审查摘要合并到现有 docs/实验交接.md。禁止改实验源码/训练/SSH/GPU/启动其他代理。只审查以下路径及其列出的输入，独立读取、判断，不依赖父代理结论。目录 C:/Users/gb/projects/demo_dual_axis_20261002。
源码：run_experiment.py；axis_collaboration.py；scaled_axis_collaboration.py；diagnose_axis_collaboration.py；full_evaluation.py；utils/reid_evaluation.py；experiment_data.py；splits.json；configs/RGBNT201/DeMo.yml；configs/RGBNT100/DeMo.yml；configs/MSVR310/DeMo.yml；data/datasets/RGBNT201.py；data/datasets/RGBNT100.py；data/datasets/msvr310.py。
结果路径：results/axis_collaboration_v4_development/{MSVR310_axis_scaled_fullref_s42,RGBNT201_axis_scaled_fullref_s42,RGBNT201_plain_scaled_fullref_s42,RGBNT100_axis_scaled_fullref_s42,RGBNT100_plain_scaled_fullref_s42}/（全部已有 result/run/status/best/epochs.csv/batch_orders.jsonl/intake.json/development_metrics/{full_metrics.json,full_metrics.csv,metric_computation.json,intake.json}）；各 run 的外层 _exit.json/_launch.json/.log；results/axis_collaboration_v4_diagnostic/MSVR310_axis_scaled_fullref_s42_cross26/（controller/launch/exit/log/intake、smoke、full 所有 JSON/CSV）；results/full_suite/{RGBNT201_demo_s42,RGBNT100_demo_s42,MSVR310_demo_s42}/full_evaluation/{metrics.json,dev_per_query.csv} 和相应 batch_orders.jsonl。
机器分析：results/preflight/axis_collaboration_v4_first2_development_analysis.json；axis_collaboration_v4_first_plain_control_analysis.json；axis_collaboration_v4_rgbnt100_development_analysis.json；axis_collaboration_v4_msvr_mechanism_analysis.json；axis_collaboration_v4_msvr_cross26_launch.json；axis_collaboration_v4_four_state_review.json；axis_collaboration_v4_cross26_deploy_review.json；axis_collaboration_v4_first2_publication_observation.json；axis_collaboration_v4_first2_handoff_sync.json（都在 results/preflight）。
人工叙述：docs/实验交接.md 的 CURRENT 标记区和最后一个 V4首两项章节；机器现状另看 results/preflight/axis_collaboration_v4_latest_snapshot.json（后台仍写入，记录读取时的 SHA/时间，不把它当终态）。版本限定的旧缺失汇总只用于核对标签：results/missing_modalities/verified_aggregate.json；不需要重算旧的351条件。
审查 A-F：GT是否真实数据标签且顺序一致；分数是否按模型统计自归一化（区分合法descriptor L2与篡改metric）；所有声明是否有实际文件/调用和数字对应；full_metrics/mINP/CMC/ranks是否真实调用且逐query可复算；50轮/最早最大dev mAP/严格重载/采样/AMP计数/同参数同维是否有证据；开发身份隔离 vs 完整官方train、单seed vs 稳定、多模态缺失旧V1 vs 新V4、关闭专家 vs 重新训练消融、训练current-batch参考 vs frozen完整dev参考、00同模型基底 vs 独立DeMo是否区分。联合贡献/路由不可写成信息论/因果或已验证优于独立路由。读所有列出核心评测脚本，必要纯标准库CPU复算，严禁假装读取远端二进制/安装torch/numpy。
写 results/preflight/axis_collaboration_v4_first5_integrity_audit.json，A-F各给 PASS/WARN/FAIL、精确路径行号、数字复算证据、局限、阻塞项；review_independence=same-family、acceptance_status=provisional、真实 model/reasoning/执行时刻。按 trace 指令保存机器 trace 和输入SHA。给我短结果，诚实区分本地实际文本验证和没有实际重新执行的GPU/原始远端数组校验。不要发布 GitHub 或写唯一交接文档。'''
request={'call_number':1,'purpose':'frozen-evidence-integrity','timestamp':meta['started_at'],'timestamp_basis':'First saved trace capture; native dispatch timestamp unavailable','tool':'spawn_agent','model':'gpt-6-astra','reasoning_effort':'max','fork_turns':'none','files_referenced':list(records),'prompt':prompt,'supplemental_instructions':[
 'Parent confirmed actual spawn parameters model=gpt-6-astra, reasoning_effort=max, fork_turns=none; native return exposes canonical task only, no UUID. Parent specific model unavailable beyond GPT-6 Codex.',
 'Added preexisting results/preflight/axis_collaboration_v4_first5_development_analysis.json; read directly and independently.',
 'User acceptance is three datasets mAP and Rank-1 at least +2 and coverage of missing modality; all six metrics must be reported. Auxiliary stricter all_six/every_condition gates do not replace that requirement. Keep fixed five-run scope.',
 'Added preexisting frozen_stats JSON/script after reviewer requested traceable support for the tail-norm/energy sentence.',
 'Parent supplied actual CPU metrics helper and read-only observer/collect_results call-chain paths. These sources were read, never executed. Other suggested summary-generator scripts were not needed because the reviewer recomputed directly from primary local texts.'
 ]}
(TRACE/'001-frozen-evidence-integrity.request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

response='''已完成审查并写入 [审查 JSON](C:/Users/gb/projects/demo_dual_axis_20261002/results/preflight/axis_collaboration_v4_first5_integrity_audit.json)。总体 **WARN**；A/B/C/D/F PASS，E WARN；无证据完整性阻塞项。

标准库实际复算 1,249 项全部通过：250 轮、12,723 次更新、0 AMP 跳过；14 张表/14,145 query/595 分组，六指标最大误差 1.42e-14，CMC 与逐 query INP 误差0。

三项 V4 双轴的开发 mAP/Rank-1 同时+2均未达成。仅支持单seed开发与冻结干预结论；未复验GPU、远端原始数组/权重，也未重算旧V1的351条件。机器 trace 和输入SHA已保存，未修改源码或唯一交接文档。'''
(TRACE/'001-frozen-evidence-integrity.response.md').write_text(response+'\n',encoding='utf-8')
duration=(datetime.datetime.fromisoformat(now)-datetime.datetime.fromisoformat(meta['started_at'])).total_seconds()*1000
review_meta={'call_number':1,'purpose':'frozen-evidence-integrity','timestamp':now,'agent_id':'/root/audit_v4_frozen_evaluation','agent_id_kind':'canonical task name; UUID unavailable','model':'gpt-6-astra','reasoning_effort':'max','reviewer_family':'openai','review_independence':'same-family','acceptance_status':'provisional','duration_ms_since_first_trace_capture':round(duration),'status':'ok','verdict':'WARN','audit_file':target.relative_to(ROOT).as_posix(),'audit_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'execution_notes':[{'command':'PATH python -','exit_code':1,'output':'No pyvenv.cfg file','effect':'No audit Python code executed; direct existing uv-managed CPython used after uv python find.'},{'command':'Inline V1 label display','exit_code':1,'output':"KeyError: 'mAP_Rank1_2pp'",'effect':'Exploratory display used the wrong key; corrected to the actual mAP_Rank1_at_least_2pp field. Final deterministic verifier passes.'},{'command':'verify_local_text.py initial','exit_code':0,'checks_PASS':1191},{'command':'verify_local_text.py after added input/cross-reference checks','exit_code':0,'checks_PASS':1249}]}
(TRACE/'001-frozen-evidence-integrity.meta.json').write_text(json.dumps(review_meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
meta.update(completed_at=now,status='ok',verdict='WARN',audit_file=target.relative_to(ROOT).as_posix(),trace_privacy='Project-local forensic trace; do not commit .aris/traces.')
(TRACE/'run.meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
events=ROOT/'.aris/meta/events.jsonl'; events.parent.mkdir(parents=True,exist_ok=True)
with events.open('a',encoding='utf-8') as stream:
    stream.write(json.dumps({'event':'review_trace','skill':'experiment-audit','purpose':'frozen-evidence-integrity','agent_id':'/root/audit_v4_frozen_evaluation','trace_path':TRACE.relative_to(ROOT).as_posix()+'/', 'status':'ok','verdict':'WARN','timestamp':now},ensure_ascii=False)+'\n')
check=json.loads(target.read_text(encoding='utf-8')); assert list(check['checks'])==list('ABCDEF') and check['deterministic_verification']['checks']=={'PASS':1249}
print(json.dumps({'audit_path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'verdict':check['verdict'],'checks':{k:v['status'] for k,v in checks.items()},'inputs':len(records),'source_changes_since_capture':changed,'trace':str(TRACE)},ensure_ascii=False))
