from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, command, sync_handoff

HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
report=json.loads((PROJECT/'results/preflight/shared_identity_complete_analysis.json').read_text())
assert report['status']=='ACTUAL_MATCHED_SHARED_IDENTITY_WAVE_AUDIT'
root=PROJECT/'results/shared_identity_v11_trial_20261003'
assert json.loads((root/'controller_result.json').read_text())['status']=='COMPLETE'
now=datetime.now().isoformat(timespec='seconds')
metrics=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
table='| 模型 | mAP | mINP | Rank-1 | Rank-5 | Rank-10 | Rank-20 | 最优轮 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for variant,value in report['runs'].items():
    table+='| '+variant+' | '+' | '.join(f"{value['full_metrics'][key]:.6f}" for key in metrics)+' | '+str(value['best_epoch'])+' |\n'
old=json.loads((PROJECT/'results/anytoany49_demo_20261003/MSVR310_demo_s42/full/result.json').read_text())['measurements']['q_RNT_g_RNT']['metrics']
table+='| 原协议DeMo | '+' | '.join(f'{old[key]:.6f}' for key in metrics)+' | 原固定best |\n'
counts={key:value['double_plus2_conditions'] for key,value in report['comparisons'].items()}
normal={key:value['normal']['delta_pp'] for key,value in report['comparisons'].items()}
gate=report['comparisons']['original_demo']['normal']['double_plus2']
phase=f'''V11 四组终态及完整缺失矩阵（{now}）：原controller四组fresh50全部完成，四组冻结49组合共196条件均完成；逐查询聚合复核六指标、CMC1..50、相机/场景/身份分组，原GT和查询/图库顺序一致。四组50轮原始批次顺序与部分集合逐步完全匹配；真实更新/AMP跳步、公共/私有接口、完整batch停止引用和原始模型状态保持证据见 shared_identity_complete_analysis.json 与完整原始目录。所有成绩使用每组按开发mAP最早并列规则选择的同一个checkpoint，缺失条件不另选权重，无官方测试调参。

{table}

双轴完整模态相对原DeMo差值为mAP{normal['original_demo']['mAP']:+.6f}、Rank-1{normal['original_demo']['Rank-1']:+.6f}个百分点，当前单MSVR双+2门槛={gate}；相对同增强DeMo、普通频域、普通双专家完整六指标差值分别为{json.dumps({key:value for key,value in normal.items() if key!='original_demo'},ensure_ascii=False)}。全部49条件同时达到双+2的数量（比较基准）为{json.dumps(counts,ensure_ascii=False)}，不能将少数条件拼成全面优势。各基准全49逐查询首位误伤/恢复、AP升降均完整保留，不只报告涨点条件。完整196行六指标表 MSVR310_all4_full49_metrics_196.csv 与原始JSON/CSV一并上传。

缺失矩阵的公平对照尤其重要：对49条件等权平均（仅诊断，非官方单项mAP），双轴相对普通频域mAP{report['comparisons']['frequency_shared']['equal_condition_mean_delta_pp']['mAP']:+.6f}、Rank-1{report['comparisons']['frequency_shared']['equal_condition_mean_delta_pp']['Rank-1']:+.6f}；相对普通双专家mAP{report['comparisons']['twins_shared']['equal_condition_mean_delta_pp']['mAP']:+.6f}、Rank-1{report['comparisons']['twins_shared']['equal_condition_mean_delta_pp']['Rank-1']:+.6f}。其中T查询→NT图库相对普通频域mAP-8.219165、Rank-1-15.238095，不能用完整模态的局部优势覆盖这些退步。当前共同接口/部分训练相对旧DeMo的缺失改善是事实，但不足以证明双轴优于同增强、同容量普通专家；此配方不直接扩展三数据集或宣称论文主结论成立。下一步优先固定checkpoint做M/F/交互关闭的四状态诊断，辨别公共身份、模态私有坐标和频域增量在排序中的实际作用，再决定修改。

本轮只是单数据集seed42机制开发，尚不能声明三个数据集均达标或稳定通用。共享接口、部分训练、最终融合监督同时更新的收益需通过本轮同增强对照判断；推理时M/F的实际排序增量仍须受控00/10/01/11评测，不能只用非零梯度推断协作有效。RGBNT201/RGBNT100、其他种子、独立路由重训及最终测试均未由本轮完成。目标仍ACTIVE_UNMET。新代码从始至终只保存每组best.pth，没有产生非最优initial/last/smoke权重，无需删除仍用于评测的best。

'''
document=PROJECT/'docs/实验交接.md'
text=document.read_text(encoding='utf-8')
assert 'V11 四组终态及完整缺失矩阵（' not in text
marker='### V11 共享身份坐标与部分查询—完整图库训练'
point=text.index('\n\n',text.index(marker))+2
document.write_bytes((text[:point]+phase+text[point:]).encode('utf-8'))
digest=sync_handoff()
sync=PROJECT/'results/preflight/shared_identity_complete_handoff_sync.json'
assert not sync.exists()
sync.write_text(json.dumps(dict(observed_at=now,sha256=digest,copies=['repository','Desktop','2025','2026','2027'],goal='ACTIVE_UNMET'),indent=2)+'\n')
helper=PROJECT/'results/preflight/demo_shared_identity_publish_complete_20261003.py'
assert not helper.exists()
helper.write_bytes(Path(__file__).read_bytes())
for name in ('demo_shared_identity_collect_20261003.py','demo_shared_identity_comparison_summary_20261003.py'):
    (PROJECT/'results/preflight'/name).write_bytes((Path('C:/Users/gb/.codex_tmp')/name).read_bytes())
command(['git','add','docs/实验交接.md','results/shared_identity_v11_trial_20261003',
         'results/preflight/shared_identity_complete_analysis.json','results/preflight/shared_identity_observer_terminal.json',
         'results/preflight/shared_identity_complete_handoff_sync.json','results/preflight/demo_shared_identity_publish_complete_20261003.py',
         'results/preflight/demo_shared_identity_collect_20261003.py','results/preflight/demo_shared_identity_comparison_summary_20261003.py'],cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=PROJECT)
command(['git','commit','-m','Report four matched shared identity fresh50 runs and all196 availability conditions'],cwd=PROJECT)
command(['git','push','origin','main'],cwd=PROJECT)
head=command(['git','rev-parse','HEAD'],cwd=PROJECT).strip()
assert command(['git','ls-remote','origin','refs/heads/main'],cwd=PROJECT).split()[0]==head
with Path('C:/Users/gb/memory/2026-10-03.md').open('a',encoding='utf-8') as handle:
    handle.write(f'\nDeMo {now}: V11 all4fresh50+196full49 complete and sixmetric/CMC/perquery/groups matched-sampling/partialmasks/stdrecount actualPASS. Allfinalscores/updates inshared_identity_complete_analysis.json, normaldualvsorig{normal["original_demo"]}, normaldouble+2{gate}. Comparisonsfull49counts{counts}. Onlybest weights produced. Git{head};ONEdoc5SHA{digest}. Originalcontroller3446554 terminal, observer2842 closure mustconsume; no restart. Actual4stateutility stillrequired; all3/seeds/formal remainunmet. GoalACTIVE_UNMET/currentturnPROGRESS.\n')
print('SHARED_IDENTITY_TERMINAL_PUBLISHED',json.dumps(dict(head=head,sha256=digest,normal_gate=gate,normal_delta=normal['original_demo'])),flush=True)
