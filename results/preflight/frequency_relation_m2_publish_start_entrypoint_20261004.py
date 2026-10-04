from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS,command,remote_python,sync_handoff

HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
p=PROJECT/'results/preflight';plan=json.loads((p/'frequency_relation_m2_plan.json').read_text(encoding='utf-8'))
review=json.loads((p/'frequency_relation_m2_review.json').read_text(encoding='utf-8'))
launch=json.loads((p/'frequency_relation_m2_2026_launch.json').read_text(encoding='utf-8'))
snapshot=json.loads((p/'frequency_relation_m2_latest_snapshot.json').read_text(encoding='utf-8'))
assert review['verdict']=='PASS_SOURCE_REVIEW' and snapshot['pid']==launch['pid'] and snapshot['live']
first=p/'frequency_relation_m2_first_snapshot.json';assert not first.exists()
first.write_text(json.dumps(snapshot,indent=2)+'\n',encoding='utf-8')
for stem in ('prepare','deploy','observe','publish_start'):
 source=Path('C:/Users/gb/.codex_tmp')/('demo_frequency_relation_'+stem+'_20261004.py')
 target=p/('frequency_relation_m2_'+stem+'_entrypoint_20261004.py');assert not target.exists()
 target.write_bytes(source.read_bytes())
now=datetime.now().isoformat(timespec='seconds')
goal=p/'research_goal_optimized_20261003.json';g=json.loads(goal.read_text(encoding='utf-8'))
g['status']='ACTIVE_UNMET';g['updated_at']=now;g['revision']='20261004_M2_frequency_relation_preflight_active'
g['milestones'][2]['status']='M2_SOURCE_REVIEWED_ACTUAL_PREFLIGHT_RUNNING_NOT_YET_ACCEPTED'
g['current_evidence']['M2']=dict(status='DISPATCHED_PREFLIGHT',plan='results/preflight/frequency_relation_m2_plan.json',
 controller_pid=launch['pid'],observed_at=snapshot['observed_at'],tensor_contract_present=snapshot['tensor'] is not None,
 actual_smokes=len(snapshot['smokes']),actual_training=snapshot['training'],planned_fresh50=3,planned_CPU_cases=2058)
goal.write_text(json.dumps(g,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
doc=PROJECT/'docs/实验交接.md';s=doc.read_text(encoding='utf-8');marker='<!-- CURRENT_DEMO_STATUS_START -->\n';assert s.count(marker)==1
text=f'''## 最新执行：M2实际频域出口关系保持预检已启动（{now}）

上一轮冻结身份证据2352条件/493920查询记录已CPU独立核算并发布8951fc2542b02f7f6dac8778fb4773c953f88419，完整48行八阶段六指标、全49条件、标量和所有误伤/救回见下方及results/trained_outlet_utility_20261004。六个模型实际PF后独立mAP均比PF前下降，三双轴为−9.760963/−12.775647/−7.407435点，完整输入F增量也弱；这些证据支持一个受控训练假设，不能提前宣称PF保持已经有效。

新源码通过fresh same-family/provisional审核，旧60源精确不变、新增6源。2026物理GPU2/3控制器{launch['pid']}已实际启动，当前观察{snapshot['observed_at']}：tensor结果存在={snapshot['tensor'] is not None}，smoke {len(snapshot['smokes'])}/3，已有训练状态{json.dumps(snapshot['training'],ensure_ascii=False)}。先进行真实CUDA初始state/参数/七种来源四状态完全一致、零权重AMP训练输出一致、target停止梯度和实际PF损失梯度预检，再三组各3真实optimizer更新且strict reload的smoke全部通过，才可fresh50。这里记录启动/预检状态，不是3×50或2058评测已完成。

单因素为full-view实际PF输出与同输入、已路由F_pre.detach的批内归一化非对角距离SmoothL1，固定权重0.1。原original_mean出口、FFT、专家访问、条件更新、联合路由、贡献预测和门控、原身份/贡献损失、部分查询→完整图库训练和采样、参数及5632D均不变。full仅加一次，partial/关闭状态/eval不加。目标距离停止梯度，学生路径仍正常学习；这不是冻结完整教师，也不强迫联合状态对所有样本超过单专家。

双轴、普通频域、普通双专家三种增强模型均用相同新目标、公开CLIP、seed42/B64/K4/fresh50，GPU2串行axis→twins、GPU3frequency，每卡最多一个NN、共最多两个。已完成三种original_mean无该目标的M1对照保留，不重跑；终态逐(epoch/step/names/partial_set)核对既有采样。固定开发mAP最高且最早并列best，统一49条件、6状态和8阶段，计划147完整条件、882六状态、1176八阶段、2058组独立CPU核算；必须完整报告六指标、CMC、分组和逐查询的负结果。

关系保持对整体旋转不敏感，不能单独证明公共身份朝向一致或门控校准。若只提升PF独立检索而真实11−00/误伤救回仍不改善，再分别检验身份朝向或估计—控制分离；此轮不同时加入这些改动。每实验仅best，smoke权重在内存strict reload，未新增init/last/smoke落盘权重；旧必要基线和依赖继续保护。用户已取消温度/功率限制，2025/2027仅文本镜像。本Goal仍ACTIVE/UNMET，三个数据集双+2、公平专家必要性、多种子独立确认未完成。人类文档仍唯一此文件，五份按SHA一致。

'''
doc.write_bytes(s.replace(marker,marker+text,1).encode('utf-8'));digest=sync_handoff()
names=[*plan['sources'],'results/preflight/research_goal_optimized_20261003.json']
names.extend(f.relative_to(PROJECT).as_posix() for f in p.glob('frequency_relation_m2_*')
 if f.name not in ('frequency_relation_m2_latest_snapshot.json','frequency_relation_m2_observer_started.json'))
assert len(names)==len(set(names));sha={n:hashlib.sha256((PROJECT/n).read_bytes()).hexdigest() for n in names}
for host,(root,_) in HOSTS.items():
 command(['scp',*OPTIONS,*[str(PROJECT/n) for n in names if n.startswith('results/preflight/')],host+':'+root+'/results/preflight/'])
 code='import hashlib,json;from pathlib import Path;r=Path('+repr(root)+');print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in '+repr(names)+'}))'
 assert json.loads(remote_python(host,code))==sha
proof=p/'frequency_relation_m2_start_publication.json';assert not proof.exists()
proof.write_text(json.dumps(dict(status='EXACT_THREE_TEXT_MIRRORS_FIVE_HANDOFFS',at=now,files_sha256=sha,
 doc_sha256=digest,controller_pid=launch['pid'],temperature_power_control=False),indent=2)+'\n',encoding='utf-8')
for host,(root,_) in HOSTS.items():command(['scp',*OPTIONS,str(proof),host+':'+root+'/results/preflight/'])
command(['git','add','--',*names,proof.relative_to(PROJECT).as_posix(),'docs/实验交接.md'],cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=PROJECT)
command(['git','commit','-m','Launch matched frequency relation preservation after audited outlet evidence'],cwd=PROJECT)
command(['git','push','origin','HEAD:main'],cwd=PROJECT)
print('M2_START_PUBLISHED',json.dumps(dict(head=command(['git','rev-parse','HEAD'],cwd=PROJECT).strip(),doc_sha256=digest,files=len(names))),flush=True)
