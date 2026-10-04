from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS,command,remote_python,sync_handoff

HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
p=PROJECT/'results/preflight'
plan=json.loads((p/'trained_outlet_utility_plan.json').read_text(encoding='utf-8'))
review=json.loads((p/'trained_outlet_utility_review.json').read_text(encoding='utf-8'))
launch=json.loads((p/'trained_outlet_utility_2026_launch.json').read_text(encoding='utf-8'))
snapshot=json.loads((p/'trained_outlet_utility_latest_snapshot.json').read_text(encoding='utf-8'))
assert review['verdict']=='PASS_SOURCE_REVIEW' and not review['blockers']
assert snapshot['pid']==launch['pid'] and snapshot['live']
first=p/'trained_outlet_utility_first_snapshot.json'
assert not first.exists()
first.write_text(json.dumps(snapshot,indent=2)+'\n',encoding='utf-8')
for stem in ('deploy','observe','complete','publish_start'):
 source=Path('C:/Users/gb/.codex_tmp')/('demo_trained_outlet_'+stem+'_20261004.py')
 target=p/('trained_outlet_utility_'+stem+'_entrypoint_20261004.py')
 assert not target.exists()
 target.write_bytes(source.read_bytes())
goal=p/'research_goal_optimized_20261003.json'
g=json.loads(goal.read_text(encoding='utf-8'))
now=datetime.now().isoformat(timespec='seconds')
g['updated_at']=now;g['status']='ACTIVE_UNMET'
g['scope']='Refined execution objective persists in this JSON and the single handoff. The platform original objective is unchanged and was actually observed ACTIVE on 2026-10-04; completion still requires all specified evidence.'
g['revision']='20261004_M1_allten_complete_frozen_outlet_utility_active'
g['current_evidence']['M1_trained']=dict(status='COMPLETE_NEGATIVE_FOR_UNIFIED_SUPERIORITY',fresh50_runs=10,
 full_model_conditions=490,controlled_state_conditions=2940,installed_GT_CPU_rows=617400,
 analysis='results/common_outlet_m1_v3_complete/analysis.json',published_commit='8a9f478415ac817d97a954923952e4bda0e07853')
g['current_evidence']['trained_outlet_utility']=dict(status='DISPATCHED_SIX_SMOKES_NOT_ALL_ACCEPTED',
 plan='results/preflight/trained_outlet_utility_plan.json',controller_pid=launch['pid'],
 observed_at=snapshot['observed_at'],actual_smokes=len(snapshot['smokes']),actual_full=len(snapshot['completed']),
 planned_cases=2352,planned_query_rows=493920,optimizer_updates=0,new_weights=0)
g['milestones'][1]['decision']='All three trained outlets have negligible complete-input11minus00 and inconsistent fair-control benefits. Frozen actual auxiliary/routed/projected identity utility across all49 is now required before selecting an M2 relation-preservation target.'
goal.write_text(json.dumps(g,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
doc=PROJECT/'docs/实验交接.md';content=doc.read_text(encoding='utf-8')
marker='<!-- CURRENT_DEMO_STATUS_START -->\n'
assert content.count(marker)==1
text=f'''## 最新执行：冻结已训练出口的身份证据诊断（{now}）

原M1十组fresh50、490个完整模型条件、2940个六状态条件及617400条独立CPU核算已完成并发布8a9f478。原始/合法平均/共享查询均未形成稳定公平优势，三个双轴完整输入11−00的mAP仅−0.008084/−0.013660/−0.008099，不能把辅助训练收益写成推理协同成功。原Goal仍ACTIVE/UNMET：不降低统一三数据集各mAP与Rank-1超过完整DeMo2点、缺失公平比较、机制与多种子要求。

当前仅在2026物理GPU2/3启动冻结诊断控制器{launch['pid']}；GPU2依次三个双轴最佳权重，GPU3依次相同出口的普通频域最佳权重，每卡串行、最多两项并行。温度与功率限制已取消。实际观察{snapshot['observed_at']}：smoke {len(snapshot['smokes'])}/6，完整诊断{len(snapshot['completed'])}/6；六个64条记录×七种可用集合smoke必须全部通过才可开始full。此处是启动状态，不能写成2352条件已完成。

每权重49种查询—图库组合，读取部署描述子、公共基础块、M/F实际路由前后投影和现有独立辅助监督表征，共八阶段，计划2352组六指标/CMC1..50/逐查询/身份相机场景分组，493920条重复条件—查询记录。M_pre/M_post共享真实训练出口、anchor和关系mass，F_pre/F_post仅差PF；M_aux/F_aux来自相同可观测来源的原独立辅助表征。辅助→路由包含条件更新、路由和汇聚，不能据此单独归因路由。运行先调用原fuse，再要求重建逐元素完全一致、294条部署条件与既有六指标/逐查询完全一致，state版本和五项输入SHA保持。无optimizer更新、新权重或官方测试。

新源码独立same-family/provisional审核发现并修复重复gpu关键字（真实表达式TypeError）和CPU标量审计漏查来源支持/关系mass/逐项系数的问题；未修改旧57份源。query出口审计合法simplex和导出算术，不声称独立重算未导出的query logits。三份新源、审核和私有部署入口SHA已绑定。后续先依据同来源辅助→实际路由→投影的真实身份结构决定一个M2关系保持目标，不同时增加三模块或默认RKD已能修复坐标朝向。

仅保留每实验best.pth和必要基线/依赖，当前诊断读取六个M1best，因此没有删除这些依赖；本阶段新增权重0。计划、审核、实际启动及首个观察在results/preflight/trained_outlet_utility_*，人类交接仍仅本文件，repo/Desktop/25/26/27按SHA保持一致。下方为已完成M1及历史记录。

'''
doc.write_bytes(content.replace(marker,marker+text,1).encode('utf-8'))
digest=sync_handoff()
names=[*plan['sources'], 'results/preflight/research_goal_optimized_20261003.json']
names.extend(f.relative_to(PROJECT).as_posix() for f in p.glob('trained_outlet_utility_*') if f.name not in ('trained_outlet_utility_latest_snapshot.json','trained_outlet_utility_observer_started.json'))
assert len(names)==len(set(names))
sha={n:hashlib.sha256((PROJECT/n).read_bytes()).hexdigest() for n in names}
for host,(root,_) in HOSTS.items():
 command(['scp',*OPTIONS,*[str(PROJECT/n) for n in names if n.startswith('results/preflight/')],host+':'+root+'/results/preflight/'])
 code='import hashlib,json;from pathlib import Path;r=Path('+repr(root)+');print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in '+repr(names)+'}))'
 assert json.loads(remote_python(host,code))==sha
proof=p/'trained_outlet_utility_start_publication.json'
assert not proof.exists()
proof.write_text(json.dumps(dict(status='EXACT_THREE_TEXT_MIRRORS_AND_FIVE_HANDOFFS',at=now,
 doc_sha256=digest,files_sha256=sha,controller_pid=launch['pid'],temperature_power_control=False),indent=2)+'\n',encoding='utf-8')
for host,(root,_) in HOSTS.items():command(['scp',*OPTIONS,str(proof),host+':'+root+'/results/preflight/'])
owned=[*names,proof.relative_to(PROJECT).as_posix(),'docs/实验交接.md']
command(['git','add','--',*owned],cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=PROJECT)
command(['git','commit','-m','Diagnose frozen trained outlets before identity relation preservation'],cwd=PROJECT)
command(['git','push','origin','HEAD:main'],cwd=PROJECT)
head=command(['git','rev-parse','HEAD'],cwd=PROJECT).strip()
print('TRAINED_OUTLET_START_PUBLISHED',json.dumps(dict(head=head,doc_sha256=digest,files=len(names))),flush=True)
