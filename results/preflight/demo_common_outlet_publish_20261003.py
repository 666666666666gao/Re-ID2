from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python, sync_handoff

HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
root=PROJECT/'results/common_outlet_scale_v12_frozen_20261003'
analysis=json.loads((PROJECT/'results/preflight/common_outlet_scale_analysis.json').read_text())
review=json.loads((PROJECT/'results/preflight/common_outlet_scale_review.json').read_text())
plan=json.loads((PROJECT/'results/preflight/common_outlet_scale_plan.json').read_text())
launch=json.loads((PROJECT/'results/preflight/common_outlet_scale_2026_launch.json').read_text())
assert analysis['cases']==882 and analysis['independent_raw_GT_recount']=='PASS'
assert review['status']=='PASS' and not review['blockers']
assert json.loads((root/'controller_result.json').read_text())['status']=='COMPLETE'
assert json.loads((root/'independent_cpu_audit_exit.json').read_text())['exit_code']==0
assert launch['selected_gpus']==[2,3]
inventory=json.loads(remote_python('2026',f'''import hashlib,json
from pathlib import Path
root=Path({launch['output']!r});process=Path('/proc/{launch['pid']}')
assert not process.exists() or (process/'stat').read_text().split()[2]=='Z'
assert not list(root.rglob('*.pth'))
print(json.dumps(dict(controller_absent=True,new_weights=0,raw_distances_bytes=sum(p.stat().st_size for p in root.rglob('raw_distances.npz')))))
'''))
(PROJECT/'results/preflight/common_outlet_scale_terminal_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
now=datetime.now().isoformat(timespec='seconds')
metrics=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
table='| 模型 | 条件组 | 表示阶段 | mAP | mINP | R1 | R5 | R10 | R20 |\n|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
scale_table='| 模型 | 可用集合 | 合法关系数 | 加权anchor/公共norm | M增量/公共norm | F增量/公共norm | I增量/公共norm | 总增量/公共norm | M取消比 | 门控M/F/I |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n'
projection_table='| 模型 | 组 | 投影 | ΔmAP | ΔR1 | 首位恢复 | 首位误伤 | AP改善 | AP下降 |\n|---|---|---|---:|---:|---:|---:|---:|---:|\n'
for variant,run in analysis['runs'].items():
    for group in ('normal','source_disjoint','all49'):
        values=run['groups'][group]
        for stage,value in values['equal_condition_metrics'].items():
            table+='| '+variant+' | '+group+' | '+stage+' | '+' | '.join(f'{value[m]:.6f}' for m in metrics)+' |\n'
        for stage,value in values['projection_changes'].items():
            projection_table+=f'| {variant} | {group} | {stage} | {value["mean_delta_pp"]["mAP"]:+.6f} | {value["mean_delta_pp"]["Rank-1"]:+.6f} | {value["rank1_rescue"]} | {value["rank1_harm"]} | {value["AP_improved"]} | {value["AP_worsened"]} |\n'
    for availability,values in run['scales'].items():
        keys=('legal_relations','weighted_anchor_relative_to_shared','M_common_increment_relative_to_shared','F_common_increment_relative_to_shared','I_common_increment_relative_to_shared','total_increment_relative_to_shared','M_cancellation_ratio')
        scale_table+='| '+variant+' | '+availability+' | '+' | '.join(f'{values[k]["mean"]:.7f}' for k in keys)+' | '
        scale_table+='/'.join(f'{values["gate_"+k]["mean"]:.6f}' for k in ('M','F','I'))+' |\n'
axis=analysis['runs']['axis_shared']
normal=axis['groups']['normal']['projection_changes']
all49=axis['groups']['all49']['projection_changes']
full_scale=axis['scales']['RNT']
single_scale=axis['scales']['R']
phase=f'''### M1实际测量：V12公共出口量级与投影前后身份检索（{now}）

上一目标回合已完成V12终态及执行Goal优化，本回合继续M1而非原样重训失败配方。2026物理GPU2/3执行三组固定V12开发最佳：双轴epoch22、普通频域epoch30、普通双专家epoch19。只读加载，不改变池化、分母、权重或损失，没有优化器更新、没有新权重、没有官方测试。三组均先通过七可用集合/前64观测smoke的全局屏障，再执行全49；controller{launch['pid']}已结束。原生产fuse由真实PM/PF/PI输出hook重新构造，逐元素精确一致；原147组部署指标及逐查询CSV与V12完全重现，模型状态tensor版本与原输入最佳权重/数组/结果/退出/旧评测文件保持不变。

实际测量的缩放链为：关系mass×合法关系数k，乘DeMo对应关系anchor，再按固定七位置平均；F使用公共anchor，M/I使用关系anchor；三门控与学习残差尺度共同作用，最后公共/私有块归一化为.25/.75度量权重。不能只根据k=1/3/7推断需要把除7改成除k。CSV记录每个观测的合法关系、七anchor/mass、M/F/I/MI/总增量相对公共norm、向量抵消比、门控及投影前后norm；下表为360个观测均值，完整min/median/std/max在分析JSON中。M取消比为范数(向量和)/各向量范数尺度和，不是身份信息协同量；legal_mean_amplitude_factor仅标注7/k算术倍数，没有真正部署或训练该改动。

{scale_table}

六阶段为：deployed=原5632D最终描述子；base_common=原公共512D；M_pre/M_post=实际路由M在PM前/后使用同一relation mass、anchor和固定mean7的512D汇总；F_pre/F_post=实际路由F在PF前/后的512D。所有阶段独立归一化检索，前后M只改变PM投影，未把不同池化混在比较中。它们不是独立重训方法，不得拿其中最佳值改选checkpoint或宣称部署提升。本轮base_common直接归一化原common并使用一般平方欧氏距离；旧base_shared是归一化00切片并使用2−2cosine，有限精度实现并不相同，故本轮原结果精确复现约束针对147组deployed，不能宣称旧公共单块逐元素复现。

{table}

投影前后逐查询变化如下；来源不相交为12条件，all49为49条件，数值为等条件平均/累计条件—查询对。重复查询与重叠组不能视作更多独立样本，平均不等于官方总体mAP。

{projection_table}

关键发现：双轴完整输入的M投影前后ΔmAP{normal['M_projection']['mean_delta_pp']['mAP']:+.6f}/ΔRank-1{normal['M_projection']['mean_delta_pp']['Rank-1']:+.6f}，F对应{normal['F_projection']['mean_delta_pp']['mAP']:+.6f}/{normal['F_projection']['mean_delta_pp']['Rank-1']:+.6f}；全49等权平均M的mAP变化{all49['M_projection']['mean_delta_pp']['mAP']:+.6f}、F为{all49['F_projection']['mean_delta_pp']['mAP']:+.6f}。这是实际路径投影后独立检索变弱的证据，不是严格的信息损失证明。完整输入F门控均值{full_scale['gate_F']['mean']:.6f}，RGB单模态{single_scale['gate_F']['mean']:.6f}；相应F增量/公共norm为{full_scale['F_common_increment_relative_to_shared']['mean']:.6f}/{single_scale['F_common_increment_relative_to_shared']['mean']:.6f}，说明门控存在强可用集合依赖，不能仅以七关系分母解释实际差别。双轴完整输入M向量取消比{full_scale['M_cancellation_ratio']['mean']:.6f}，没有显示大规模相互抵消。普通双专家在来源不相交组的F投影mAP有小幅正向变化，也表明不能强迫所有集合接受同一种教师或一概断言投影有害；反例已完整保留。

三组49×6阶段合计882项，185220条查询。独立CPU重新从安装的MSVR训练文件名与固定dev身份建立GT/查询图库顺序，对私有保存的FP32原始距离独立lexsort复算，六指标、CMC1..50、身份/相机/场景分组和逐查询全部通过；标量行序/来源标签/合法关系数/无效anchor与mass为零也检查通过。该检查证明本轮测量与评分一致，不证明共享槽位优于平均或某个因素具有因果作用。

源代码diagnose_common_outlet_scale.py、launch_common_outlet_scale.py、audit_common_outlet_scale.py经fresh Astra/max源审查；同模型家族provisional，实际GPU和CPU结果另行提供。结果目录results/common_outlet_scale_v12_frozen_20261003，882行汇总MSVR310_all3_sixstages_full49_metrics_882.csv，每个可用集合scale_*.csv，各条件六指标/CMC/分组/逐查询JSON/CSV齐全，分析results/preflight/common_outlet_scale_analysis.json；原距离NPZ合计{inventory['raw_distances_bytes']}B仅远端私有保存。此诊断不生成权重，无额外清理；必要V12四个best继续保留。

M1的测量阶段已完成，公共出口的平均/有效集合平均/共享查询槽位同容量重训仍待实现。下一改动必须依据本轮量级及投影检索证据，保持一个因素；RKD式关系保持、跨集合配对覆盖与门控估计分离仍作为独立后续因素。原三数据集+2/+2及全面缺失目标保持ACTIVE_UNMET，最新速度仍需另行同协议测量。本轮不能被包装成新方法通过验收。

'''
document=PROJECT/'docs/实验交接.md'
text=document.read_text(encoding='utf-8')
assert '### M1实际测量：V12公共出口量级与投影前后身份检索' not in text
start=text.index('## 当前状态（');end=text.index('\n',start)
text=text[:start]+f'## 当前状态（{now}）'+text[end:]
point=text.index('### M1公共出口诊断启动')
document.write_bytes((text[:point]+phase+text[point:]).encode('utf-8'))
goal_path=PROJECT/'results/preflight/research_goal_optimized_20261003.json'
goal=json.loads(goal_path.read_text(encoding='utf-8'))
goal['updated_at']=now
goal['milestones'][1]['status']='MEASUREMENT_COMPLETE_POOLING_RETRAIN_PENDING'
goal['milestones'][1]['actual_evidence']='Frozen V12 outlet scale / projection utility:882 raw-GT cases and185220 query records audited; no training or pooling intervention.'
goal_path.write_text(json.dumps(goal,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for host in ('2025','2027'):
    remote_root,_=HOSTS[host]
    command(['scp',*OPTIONS,*[str(PROJECT/n) for n in plan['sources']],host+':'+remote_root+'/'])
for host in HOSTS:
    remote_root,_=HOSTS[host]
    code='import hashlib,json;from pathlib import Path;print(json.dumps({n:hashlib.sha256((Path('+repr(remote_root)+')/n).read_bytes()).hexdigest() for n in '+repr(list(plan['sources']))+'}))'
    assert json.loads(remote_python(host,code))==plan['sources']
digest=sync_handoff()
sync=PROJECT/'results/preflight/common_outlet_scale_handoff_sync.json'
assert not sync.exists()
sync.write_text(json.dumps(dict(observed_at=now,sha256=digest,copies=['repository','Desktop','2025','2026','2027'],goal='ACTIVE_UNMET'),indent=2)+'\n')
helpers=('deploy','observe','cpu','collect','analyze','publish')
for short in helpers:
    name=f'demo_common_outlet_{short}_20261003.py'
    (PROJECT/'results/preflight'/name).write_bytes((Path('C:/Users/gb/.codex_tmp')/name).read_bytes())
stage=list(plan['sources'])+['docs/实验交接.md','results/common_outlet_scale_v12_frozen_20261003',str(goal_path.relative_to(PROJECT))]
stage+=['results/preflight/common_outlet_scale_'+name for name in ('plan.json','review.json','2026_launch.json','observer_terminal.json','analysis.json','terminal_inventory.json','handoff_sync.json','actual_neural_summary.json')]
stage+=[f'results/preflight/demo_common_outlet_{short}_20261003.py' for short in helpers]
command(['git','add',*stage],cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=PROJECT)
command(['git','commit','-m','Measure frozen common outlet scale and M F projection retrieval utility with882 GT recounts'],cwd=PROJECT)
command(['git','push','origin','main'],cwd=PROJECT)
head=command(['git','rev-parse','HEAD'],cwd=PROJECT).strip()
assert command(['git','ls-remote','origin','refs/heads/main'],cwd=PROJECT).split()[0]==head
with Path('C:/Users/gb/memory/2026-10-03.md').open('a',encoding='utf-8') as handle:
    handle.write(f'\nDeMo {now} actualM1 measurement complete:3fixedV12best epoch22/30/19,882stagesfull49/185220queryrows independentinstalledGT/rawdistanceCPU PASS;147deploymentexact, productionfuseexact/read-onlyversions/protectedinputs. No optimizer/newweights/poolingchange/officialtest. Physics2026GPU2/3ONLY; controller{launch["pid"]}/observerterminal. ScalarCSV21banks legalcounts/anchor/mass*k/mean7/gates/cancellation recorded; samepoolMprepost/Fprepost all49 utility. Git{head};ONEdoc5SHA{digest}; M1measurementdone poolingretrainpending; goalACTIVE_UNMET. Neverrerun exclusivehelpers.\n')
print('COMMON_OUTLET_MEASUREMENT_PUBLISHED',json.dumps(dict(head=head,sha256=digest,cases=882,goal='ACTIVE_UNMET')),flush=True)
