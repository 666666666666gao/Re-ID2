from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS,command,remote_python,sync_handoff

HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
p=PROJECT/'results/preflight'
plan=json.loads((p/'identity_alignment_m2b_plan.json').read_text(encoding='utf-8'))
review=json.loads((p/'identity_alignment_m2b_review.json').read_text(encoding='utf-8'))
rescue=json.loads((p/'identity_alignment_m2b_deploy_resume_review.json').read_text(encoding='utf-8'))
launch=json.loads((p/'identity_alignment_m2b_2026_launch.json').read_text(encoding='utf-8'))
snapshot=json.loads((p/'identity_alignment_m2b_latest_snapshot.json').read_text(encoding='utf-8'))
assert review['verdict']=='PASS_SOURCE_REVIEW' and not review['blockers']
assert rescue['verdict']=='PASS_SOURCE_REVIEW' and not rescue['blockers']
assert snapshot['pid']==launch['pid'] and snapshot['live'] and snapshot['tensor']['status']=='PASS_IDENTITY_ALIGNMENT_CONTRACT'
assert len(snapshot['smokes'])==3 and all(v['status']=='SMOKE_PASS' and v['steps']==3 and v['strict_reload_equal'] for v in snapshot['smokes'].values())
assert snapshot['training']
first=p/'identity_alignment_m2b_start_snapshot.json';assert not first.exists()
first.write_text(json.dumps(snapshot,indent=2)+'\n',encoding='utf-8')
for name in ('prepare','deploy','deploy_failure','deploy_resume','observe','complete','publish_start'):
 target=p/('identity_alignment_m2b_'+name+'_entrypoint_20261004.py');assert not target.exists()
 target.write_bytes(Path('C:/Users/gb/.codex_tmp/demo_identity_alignment_'+name+'_20261004.py').read_bytes())
now=datetime.now().isoformat(timespec='seconds')
goal=p/'research_goal_optimized_20261003.json';g=json.loads(goal.read_text(encoding='utf-8'))
g['status']='ACTIVE_UNMET';g['updated_at']=now;g['revision']='20261004_M2b_actual_alignment_contract_and_three_smokes_pass_fresh50_active'
g['current_evidence']['M2b']=dict(status='ACTUAL_CONTRACT_THREE_REAL_UPDATE_SMOKES_PASS_FRESH50_ACTIVE',
 controller_pid=launch['pid'],observed_at=snapshot['observed_at'],actual_smokes=3,
 actual_smoke_updates=sum(v['steps'] for v in snapshot['smokes'].values()),training=snapshot['training'],
 plan='results/preflight/identity_alignment_m2b_plan.json',planned_fresh50=3,planned_CPU_cases=3087,
 planned_repeated_condition_query_rows=648270,official_test_uses=0)
g['next_action']=dict(milestone='M2b_actual_PF_public_identity_alignment_single_factor',status='EXISTING_REVIEWED_CONTROLLER_RUNNING_NO_RESTART',
 action='完成原M2单因素身份匹配监督的三fresh50、全49/六状态/八阶段/七坐标对，独立3087组核算后判断兼容与真正融合收益。保留原跨集合采样/门控/关系损失；不将新loss下降或cross涨点当作达标。',
 hardware_dependency='Only2026GPU2/3,max2NN,no temperature/powerconditions')
goal.write_text(json.dumps(g,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
doc=PROJECT/'docs/实验交接.md';s=doc.read_text(encoding='utf-8');marker='<!-- CURRENT_DEMO_STATUS_START -->\n'
assert s.count(marker)==1 and '<!-- CURRENT_M2B_STATUS_START -->' not in s
text=f'''<!-- CURRENT_M2B_STATUS_START -->
## 最新执行：M2b实际频域出口身份匹配监督已通过真实预检并启动（{now}）

上一轮冻结M2跨坐标诊断1029组/216090重复查询独立GT核算PASS，发布363ff924d9d929fd363d784bb3ce6bdb3bb6f8a8。双轴正常PF自检索32.687296mAP/41.428571R1，PF→公共19.142618/10.000000、公共→PF15.832914/12.857143；普通对照也有经验兼容差，来源不相交存在混合结果，不能严格归因于旋转或把它称为双轴独有问题。

M2b只新增一个目标：full-view实际路由后PF查询，对停止梯度的同full-view公共身份图库做跨样本身份对比，固定weight0.1/temperature0.07。原M2关系保持0.1、FFT、专家、一次交互、联合路由、收益估计/门控、残差幅值、所有旧损失和部分查询对完整图库训练、采样、有效参数和5632D不变。公共参照每次前向停止，不是整模型冻结教师；不逐样本复制公共特征，不以缺失输入重建完整细节。

现有M2三组同序453batch中436含重复观测，120身份组在batch内只有一个不同观测。新增正例须相同训练身份且文件名不同，分母也排除同一观测；只对有合法正例的anchor取均值，原采样与其他loss不改。每batch记录合法/排除anchor、正例对数、原始loss和停止梯度标记，终态CPU独立按实际文件名/身份重算。

新七源码fresh same-family/provisional审核PASS，旧69源不变、总76源三服务器精确一致。真实CUDA初始化state/参数、全部七可用集合×四状态逐元素与M2一致，zero-alignment-weight AMP训练输出精确退化到保留M2关系loss的父配方；合成重复观测的独立CE重算、排除anchor查询梯度为零、公共参照无梯度及实际PF梯度非零均通过。三组smoke各3真实更新、全部有效梯度和内存strict reload通过，才放行fresh50。

首次部署已完成三服务器源码同步后，在远程CPU门槛处因执行目录为/home/gaob且缺少项目搜索路径而ModuleNotFoundError，未创建NN、输出目录或根日志。原失败入口、stderr与审核证据保留；修复只加入项目sys.path，经fresh same-family/provisional复审PASS后从CPU检查和唯一launch位置继续，未重跑原完整部署，76源码/训练配方/环境未改。

控制器{launch['pid']}，实际{snapshot['observed_at']}训练状态：{json.dumps(snapshot['training'],ensure_ascii=False)}。GPU2双轴→普通双专家顺序，GPU3普通频域，每卡一个NN、最大2。每模型公开CLIP/fresh50/B64/K4/seed42/nativeAMP512，最高devmAP最早并列同一checkpoint；原M2三组无alignment最佳权重留作控制。上一轮每模型训练约14分钟，预计含评测总35–40分钟。唯一240秒观察器和完成收集器沿用此控制器，超时不重启。

计划三fresh50、147完整部署条件、882六状态、1176八阶段、1029跨坐标条件，独立3087组/648270重复条件—查询记录核算六指标mAP/mINP/R1/5/10/20、CMC1..50、逐查询AP/INP/首末匹配、身份/相机/场景分组及误伤/恢复。上述全量仍是计划，当前只能确认预检及训练已启动。终态同时报告M2b−M2同模型、双轴−普通频域/双专家、公同接口DeMo与原同协议DeMo；只有兼容改善、独立PF信息保留、11−00净收益和公平对照优势共同成立才考虑候选扩展。

三数据集mAP和R1均+2及完整缺失/多种子目标仍ACTIVE/UNMET，未用官方测试训练或选择。只保存每模型best，smoke不落权重，必要M1/M2依赖保留，不重复清理。用户已取消温度/功率约束，2025/2027仅文本镜像；神经计算只2026 GPU2/3。继续唯一人类交接，repo/Desktop/25/26/27五份字节一致，GitHub仅源码和文本结果。

<!-- CURRENT_M2B_STATUS_END -->

'''
doc.write_bytes(s.replace(marker,marker+text,1).encode('utf-8'))
files=[PROJECT/n for n in plan['sources']]+[goal]
files.extend(f for f in p.glob('identity_alignment_m2b_*') if f.name not in ('identity_alignment_m2b_latest_snapshot.json',
 'identity_alignment_m2b_observer_started.json','identity_alignment_m2b_completion_wait_started.json'))
assert len(files)==len(set(files))
manifest={f.relative_to(PROJECT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
for host,(root,_) in HOSTS.items():
 selected=[f for f in files if f.parent==p]
 command(['scp',*OPTIONS,*[str(f) for f in selected],host+':'+root+'/results/preflight/'])
 code='import hashlib,json;from pathlib import Path;r=Path('+repr(root)+');print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in '+repr(list(manifest))+'}))'
 assert json.loads(remote_python(host,code))==manifest
digest=sync_handoff();proof=p/'identity_alignment_m2b_start_publication.json';assert not proof.exists()
proof.write_text(json.dumps(dict(status='EXACT_THREE_TEXT_MIRRORS_FIVE_HANDOFFS',at=now,files_sha256=manifest,
 doc_sha256=digest,controller_pid=launch['pid'],actual_smokes=3,goal_complete=False,temperature_power_control=False),indent=2)+'\n',encoding='utf-8')
for host,(root,_) in HOSTS.items():command(['scp',*OPTIONS,str(proof),host+':'+root+'/results/preflight/'])
command(['git','add','--',*manifest,proof.relative_to(PROJECT).as_posix(),'docs/实验交接.md'],cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=PROJECT)
command(['git','commit','-m','Launch matched actual PF public identity alignment after CUDA preflight'],cwd=PROJECT)
command(['git','push','origin','HEAD:main'],cwd=PROJECT)
print('M2B_START_PUBLISHED',json.dumps(dict(head=command(['git','rev-parse','HEAD'],cwd=PROJECT).strip(),
 doc_sha256=digest,files=len(manifest),goal_complete=False)),flush=True)
