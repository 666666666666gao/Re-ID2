from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS,command,remote_python,sync_handoff

HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
p=PROJECT/'results/preflight';root=PROJECT/'results/identity_alignment_m2b_20261004'
report=json.loads((root/'analysis.json').read_text(encoding='utf-8'))
audit=json.loads((root/'independent_cpu_audit.json').read_text(encoding='utf-8'))
comparisons=json.loads((root/'comparison_summary.json').read_text(encoding='utf-8'))
decision=json.loads((root/'mechanism_decision.json').read_text(encoding='utf-8'))
parent=json.loads((PROJECT/'results/frequency_relation_m2_20261004/analysis.json').read_text(encoding='utf-8'))
baseline=json.loads((PROJECT/'results/axis_collaboration_v4_missing_development27/MSVR310_demo_s42/full/clean.json').read_text(encoding='utf-8'))
assert audit['status']=='PASS' and audit['cases']==3087 and audit['perquery_count']==648270
assert report['cases']==3087 and decision['goal_complete'] is False
assert comparisons['status']=='COMPLETE_AUDITED_COMPARISONS_NOT_FINAL_METHOD'
assert comparisons['evidence']['M2b']['sha256']==hashlib.sha256((root/'independent_cpu_audit.json').read_bytes()).hexdigest()
proof=p/'identity_alignment_m2b_complete_publication.json';assert not proof.exists()
now=datetime.now().isoformat(timespec='seconds')
metrics=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
variants=('axis_shared','frequency_shared','twins_shared')

def table(records):
 text='| 模型/状态 | mAP | mINP | R1 | R5 | R10 | R20 |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for label,values in records:text+='| '+label+' | '+' | '.join(f'{values[m]:.6f}' for m in metrics)+' |\n'
 return text

normal=[('原同协议DeMo',baseline),('同公共接口/同缺失增强DeMo',comparisons['augmented_DeMo_normal'])]
states=[];stages=[];cross=[];group_records=[]
harm='| 模型 | 正常11−00 ΔmAP/R1 | 首位误伤/救回 | AP改善/下降/不变 | 全49 ΔmAP/R1 |\n|---|---:|---:|---:|---:|\n'
for variant in variants:
 run=report['runs'][variant]
 normal.extend([('M2/'+variant,parent['runs'][variant]['normal']['metrics']['11']),('M2b/'+variant,run['normal']['metrics']['11'])])
 states.extend((variant+'/'+s,run['normal']['metrics'][s]) for s in ('00','10','01','11'))
 stages.extend((variant+'/'+s,run['normal_utility']['metrics'][s]) for s in ('base_common','F_pre','F_post','F_aux'))
 cross.extend((variant+'/'+s,run['normal_cross']['metrics'][s]) for s in ('common_common','F_pre_F_pre','F_post_F_post','F_pre_common','common_F_pre','F_post_common','common_F_post'))
 full=run['normal']['full_vs_base'];group=run['groups']['all49']
 harm+=f'| {variant} | {full["delta_pp"]["mAP"]:+.6f}/{full["delta_pp"]["Rank-1"]:+.6f} | {full["Rank1_harm_queries"]}/{full["Rank1_rescue_queries"]} | {full["AP_improved_queries"]}/{full["AP_worsened_queries"]}/{full["AP_unchanged_queries"]} | {group["full_vs_base_equal_condition_delta"]["mAP"]:+.6f}/{group["full_vs_base_equal_condition_delta"]["Rank-1"]:+.6f} |\n'
 for name,values in run['groups'].items():
  group_records.append((variant+'/'+name,values['sixmetrics_equal_condition_mean']))
for control,values in report['fair_comparisons'].items():
 for name,delta in values.items():group_records.append(('双轴−'+control+'/'+name,delta))
for name,values in comparisons['comparisons']['M2b_axis_minus_augmented_DeMo']['groups'].items():
 group_records.extend([('同增强DeMo/'+name,values['reference_mean']),('双轴−同增强DeMo/'+name,values['delta_pp'])])
counts='| 全49比较 | ΔmAP/R1 | 双指标各+2条件数 | mAP退步条件数 | R1退步条件数 | 任一主指标退步条件数 |\n|---|---:|---:|---:|---:|---:|\n'
for name,values in comparisons['comparisons'].items():
 g=values['groups']['all49']
 counts+=f'| {name} | {g["delta_pp"]["mAP"]:+.6f}/{g["delta_pp"]["Rank-1"]:+.6f} | {len(g["both_plus2_conditions"])}/49 | {len(g["mAP_degraded_conditions"])} | {len(g["Rank1_degraded_conditions"])} | {len(g["either_primary_degraded_conditions"])} |\n'
complement='| 模型/组 | PF独立救回基础错误 | 其中联合实际救回/未救回 | F_pre救回经PF保留/丢失 | 单状态M独对/F独对/同对/同错 |\n|---|---:|---:|---:|---:|\n'
for variant,values in comparisons['query_complementarity'].items():
 for group in ('normal','all49','source_disjoint','partial_query_full_gallery'):
  c=values['groups'][group]['counts']
  complement+=f'| {variant}/{group} | {c["F_post_rescues_base"]} | {c["F_post_rescues_realized_by_joint"]}/{c["F_post_rescues_missed_by_joint"]} | {c["F_pre_rescues_retained_by_PF"]}/{c["F_pre_rescues_lost_by_PF"]} | {c["M_only_correct"]}/{c["F_only_correct"]}/{c["both_correct"]}/{c["both_wrong"]} |\n'
calibration='| 模型/组/估计 | 目标均值 | 目标标准差 | 预测均值 | 预测标准差 | MAE | RMSE |\n|---|---:|---:|---:|---:|---:|---:|\n'
for variant,values in comparisons['contribution_calibration'].items():
 for group in ('normal','all49'):
  for channel,c in values[group].items():
   calibration+='| '+variant+'/'+group+'/'+channel+' | '+' | '.join(f'{c[k]:.8f}' for k in ('target_mean','target_std','prediction_mean','prediction_std','MAE','RMSE'))+' |\n'
doc=PROJECT/'docs/实验交接.md';s=doc.read_text(encoding='utf-8')
start='<!-- CURRENT_M2B_STATUS_START -->';end='<!-- CURRENT_M2B_STATUS_END -->'
assert s.count(start)==s.count(end)==1
begin=s.index(start)+len(start);finish=s.index(end,begin)
text=f'''
## 最新终态：M2b三fresh50与3087组独立核算完成（{now}）

三模型公开CLIP/fresh50/B64/K4/seed42/nativeAMP512、devmAP最佳最早并列checkpoint；全部49部署、六状态、八阶段及七坐标对实际完成。独立安装GT CPU3087组、648270重复条件—查询记录核算六指标、CMC1..50、逐查询AP/INP/首末匹配、身份/相机/场景及误伤/恢复PASS。实际训练与选择为{json.dumps({v:r['training']|dict(best_epoch=r['best_epoch'],alignment=r['alignment_training']) for v,r in report['runs'].items()},ensure_ascii=False)}。原M2实际采样/部分集合精确相同，参数及5632D不变，官方测试未用于训练或选择。

仅新增full-view实际PF→停止梯度公共身份图库的跨观测身份对比，weight0.1/tau0.07；原M2关系0.1、专家/路由/门控/幅值/FFT/所有旧损失与部分训练不变。真实重复文件名从正例和分母排除，无合法正例anchor不计入平均；不是逐样本特征复制或冻结整模型教师。首次部署CPU导入失败在NN前停止，原始证据保留；搜索路径修复复审后只从失败位置继续。

正常部署六指标与原M2配对如下；不是完整官方训练集的论文主表，不是三数据集统一版本。

{table(normal)}

双轴正常输入相对原DeMo的精确差值为{json.dumps(comparisons['normal_original_DeMo_gate']['axis_shared'],ensure_ascii=False)}。全49缺失比较使用已正确屏蔽来源的同公共接口/同缺失增强DeMo，避免以旧置零来源污染制造提升；DeMo没有频域辅助目标，不是等参数专家对照，等有效参数比较仍以本轮普通频域/普通双专家为准。

完整缺失分组和公平专家差值如下。同集合7、重叠错配30、不相交12组成互斥49；其余组可重叠。全部原值all3087_sixmetrics.csv及逐查询/分组/CMC在results/identity_alignment_m2b_20261004；全49共294条同版本M2、普通专家、同增强DeMo配对差值为all49_baseline_and_expert_comparisons.csv。各比较双+2与退步的完整条件名单在comparison_summary.json。条件等权均值不冒充官方mAP，重复查询不作为独立统计样本。

{table(group_records)}

{counts}

冻结四状态与真正部署11−00如下。00是新方法训练后的基础路径，不是独立DeMo；关闭专家时也关闭相应条件消息和联合交互，完整重训必要性仍需候选明确后检验。

{table(states)}

{harm}

逐查询潜在互补与实际兑现如下。PF独立首位正确、基础错误只表示潜在互补；不保证混合后应该恢复。四状态M/F正确集合与PF独立检索是不同对象。全49行计数为重复条件—查询出现次数，非独立样本或统计显著性。

{complement}

收益估计误差由已核算GT参考的contributions.csv逐条重新计算；原float32均值与Python双精度均值以1e−6容差校验。报告准备时原M2实际出现1.43e−7差异，原1e−7断言失败后已记录并修正；六指标独立GT核算继续原1e−8标准。正常和全49如下，其余关键组在comparison_summary.json。开发固定图库参考与训练当前batch参考不同；这里是经验预测诊断，不把误差或梯度耦合直接当作因果解释。

{calibration}

独立公共/频域身份证据和完整七坐标对的正常六指标如下，缺失全部49值同目录。PF自检索与公共坐标兼容、实际融合是不同量，不可将其一涨点当作另两个已成立。

{table(stages)}

{table(cross)}

观察与判断：{decision['finding']} 后续单因素：{decision['next_step']} 仍以三个数据集完整mAP/R1各+2、全面缺失及公平机制/多种子/独立确认验收；本轮不能替代这些证据，Goal ACTIVE/UNMET。

仅2026物理GPU2/3，每卡串行、最多2NN；用户取消功率和温度约束，未采集或修改相关设置。三新best与必要旧依赖保留，smoke/initial/last权重0；原始NPZ/binaryweight留26，25/27仅文本。源码与文本上传Re-ID2，唯一交接repo/Desktop/25/26/27五份字节一致。

'''
doc.write_bytes((s[:begin]+text+s[finish:]).encode('utf-8'))
goal=p/'research_goal_optimized_20261003.json';g=json.loads(goal.read_text(encoding='utf-8'))
g['status']='ACTIVE_UNMET';g['updated_at']=now;g['revision']='20261004_M2b_three50_all3087_CPU_PASS_single_factor_decision'
g['current_evidence']['M2b']=dict(status='COMPLETE_INDEPENDENT_INSTALLED_GT_CPU_PASS',fresh50=3,CPU_cases=3087,
 repeated_condition_query_rows=648270,analysis='results/identity_alignment_m2b_20261004/analysis.json',
 decision='results/identity_alignment_m2b_20261004/mechanism_decision.json',finding=decision['finding'],official_test_uses=0)
g['next_action']=dict(milestone=decision['next_milestone'],status='NEXT_STEP_FROM_AUDITED_SINGLE_FACTOR_EVIDENCE',
 action=decision['next_step'],hardware_dependency='Only2026GPU2/3,max2NN,no temperature/powerconditions')
goal.write_text(json.dumps(g,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
entry=root/'publication_entrypoint.py';assert not entry.exists();entry.write_bytes(Path(__file__).read_bytes())
files=[f for f in root.rglob('*') if f.is_file()]
assert all(f.suffix in ('.json','.csv','.log','.jsonl','.py') for f in files)
files.extend([goal,p/'identity_alignment_m2b_observer_terminal.json',p/'identity_alignment_m2b_first_pair_primary_export.json',p/'identity_alignment_m2b_report_precision_issue.json'])
manifest={f.relative_to(PROJECT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
manifest_path=root/'mirror_manifest.json';assert not manifest_path.exists()
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');files.append(manifest_path)
archive=Path('C:/Users/gb/.codex_tmp/identity_alignment_m2b_complete_mirror_text.tar.gz');assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for f in files:tar.add(f,arcname=f.relative_to(PROJECT).as_posix(),recursive=False)
archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
for host,(remote_root,_) in HOSTS.items():
 remote_archive=remote_root+'/results/identity_alignment_m2b_complete_mirror_text.tar.gz'
 command(['scp',*OPTIONS,str(archive),host+':'+remote_archive])
 code=f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({remote_root!r});a=Path({remote_archive!r})
assert hashlib.sha256(a.read_bytes()).hexdigest()=={archive_sha!r}
with tarfile.open(a,'r:gz') as tar:tar.extractall(r,filter='data')
m=json.loads((r/'results/identity_alignment_m2b_20261004/mirror_manifest.json').read_text())
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in m.items())
print(json.dumps(dict(status='EXACT_TEXT_MIRROR',host={host!r},files=len(m))))
'''
 print(remote_python(host,code).strip(),flush=True)
digest=sync_handoff()
proof.write_text(json.dumps(dict(status='EXACT_THREE_TEXT_MIRRORS_FIVE_HANDOFFS',at=now,files=len(manifest),
 archive_sha256=archive_sha,doc_sha256=digest,cases=3087,condition_query_rows=648270,fresh50=3,
 retained_best=3,goal_complete=False,temperature_power_control=False),indent=2)+'\n',encoding='utf-8')
for host,(remote_root,_) in HOSTS.items():command(['scp',*OPTIONS,str(proof),host+':'+remote_root+'/results/preflight/'])
command(['git','add','--',root.relative_to(PROJECT).as_posix(),goal.relative_to(PROJECT).as_posix(),
 'results/preflight/identity_alignment_m2b_observer_terminal.json','results/preflight/identity_alignment_m2b_first_pair_primary_export.json','results/preflight/identity_alignment_m2b_report_precision_issue.json',proof.relative_to(PROJECT).as_posix(),'docs/实验交接.md'],cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=PROJECT)
command(['git','commit','-m','Publish matched identity alignment experiment and complete independent retrieval audit'],cwd=PROJECT)
command(['git','push','origin','HEAD:main'],cwd=PROJECT)
print('M2B_COMPLETE_PUBLISHED',json.dumps(dict(head=command(['git','rev-parse','HEAD'],cwd=PROJECT).strip(),
 doc_sha256=digest,files=len(manifest),goal_complete=False)),flush=True)
