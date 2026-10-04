from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS,command,remote_python,sync_handoff

HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
p=PROJECT/'results/preflight';root=PROJECT/'results/trained_outlet_utility_20261004'
report=json.loads((root/'analysis.json').read_text(encoding='utf-8'))
decision=json.loads((root/'next_decision.json').read_text(encoding='utf-8'))
assert report['cases']==2352 and report['condition_query_rows']==493920
assert json.loads((root/'independent_cpu_audit.json').read_text(encoding='utf-8'))['status']=='PASS'
assert decision['status']=='FROZEN_DIAGNOSTIC_INTERPRETATION_NOT_TRAINING_RESULT'
now=datetime.now().isoformat(timespec='seconds');metrics=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
table='| 出口/模型 | 阶段 | mAP | mINP | R1 | R5 | R10 | R20 |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
for key,run in report['runs'].items():
 for stage,values in run['normal']['metrics'].items():table+='| '+key+' | '+stage+' | '+' | '.join(f'{values[m]:.6f}' for m in metrics)+' |\n'
changes='| 出口/模型 | F路由相对辅助ΔmAP/R1 | PF后相对前ΔmAP/R1 | PF误伤/救回 | 49均值PFΔmAP/R1 | 不相交PFΔmAP/R1 |\n|---|---:|---:|---:|---:|---:|\n'
for key,run in report['runs'].items():
 normal=run['normal'];proj=normal['F_projection'];route=normal['F_aux_to_route']['delta_pp']
 values=[route,proj['delta_pp'],run['groups']['all49']['comparisons']['F_projection']['delta_pp'],run['groups']['source_disjoint']['comparisons']['F_projection']['delta_pp']]
 pair=lambda v:f'{v["mAP"]:+.6f}/{v["Rank-1"]:+.6f}'
 changes+=f'| {key} | {pair(values[0])} | {pair(values[1])} | {proj["Rank1_harm_queries"]}/{proj["Rank1_rescue_queries"]} | {pair(values[2])} | {pair(values[3])} |\n'
doc=PROJECT/'docs/实验交接.md';s=doc.read_text(encoding='utf-8');marker='<!-- CURRENT_DEMO_STATUS_START -->\n'
assert s.count(marker)==1
paragraphs='\n\n'.join(decision['findings'])
text=f'''## 最新终态：训练后八阶段身份证据诊断全量通过（{now}）

只读控制器1403791已实际结束：六个64记录×七集合smoke通过后，六个full均exit0/COMPLETE，各392阶段—条件，共2352组六指标、CMC1..50、身份/相机/场景分组及493920条重复条件—查询记录，安装GT独立CPU重新排序/核算全部PASS。旧57源不变；各最佳权重、best开发数组、训练终态、退出文件及既有49条件终态的输入SHA不变，state tensor版本不变，部署特征逐元素0差异、294条部署条件六指标和逐查询与原评测完全一致。GPU仅2026:2/3，无功率温度限制；新增训练更新/权重/官方测试使用均0。

以下是同一固定fit/dev、seed42、原M1开发mAP所选最佳权重的完整RNT→RNT阶段结果。部署5632D，其余阶段独立归一化512D；阶段独立检索不是直接替代正式部署成绩，不能横向误认描述子同维度。全部49条件完整六指标见results/trained_outlet_utility_20261004/all2352_sixmetrics.csv，原始逐查询/分组/标量/终态日志及独立核算在同目录，FP32原始距离NPZ仍在2026。

{table}

{changes}

{paragraphs}

M_aux/F_aux是相同可用来源下现有独立辅助监督表征；辅助→实际路由同时改变条件更新、路由和汇聚，不能单独归因路由。M_pre/M_post使用相同真实出口系数、anchor及关系mass，仅差PM；F_pre/F_post仅差PF。即使某阶段独立mAP下降，也不能宣称严格信息论损失。query系数只独立审核合法支持/simplex及导出算术，未导出logits不伪称重算softmax。原M1全十组、490完整/2940六状态的公平负结果继续保留，不挑选不同出口或不同阶段的峰值拼成新方法。

下一步登记判断：{decision['next_step']} 尚未把该建议记为已训练收益，不默认关系蒸馏能解决坐标朝向或门控。原Goal仍ACTIVE/UNMET：统一三个数据集mAP与Rank-1各超过完整DeMo2点、缺失公平比较、机制必要性与多种子独立确认尚未成立。每实验只留best及必要依赖，本诊断新增权重0、删除权重0。人类交接仍唯一此文档，repo/Desktop/25/26/27五份保持同字节；下方所有启动与旧结论是历史证据。

'''
doc.write_bytes(s.replace(marker,marker+text,1).encode('utf-8'))
goal=p/'research_goal_optimized_20261003.json';g=json.loads(goal.read_text(encoding='utf-8'))
g['status']='ACTIVE_UNMET';g['updated_at']=now;g['revision']='20261004_M1_trained_outlet2352_CPU_PASS_next_single_factor'
g['current_evidence']['trained_outlet_utility']=dict(status='COMPLETE_INDEPENDENT_INSTALLED_GT_CPU_PASS',cases=2352,condition_query_rows=493920,
 actual_smokes=6,actual_full=6,analysis='results/trained_outlet_utility_20261004/analysis.json',decision='results/trained_outlet_utility_20261004/next_decision.json',optimizer_updates=0,new_weights=0)
g['milestones'][2]['decision']=decision['next_step']
goal.write_text(json.dumps(g,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(p/'trained_outlet_utility_completion_entrypoint_20261004.py').write_bytes(Path(__file__).read_bytes())
files=[f for f in root.rglob('*') if f.is_file()]
assert all(f.suffix in ('.json','.csv','.log','.py') for f in files)
files.extend([goal,p/'trained_outlet_utility_observer_terminal.json',p/'trained_outlet_utility_completion_entrypoint_20261004.py'])
manifest={f.relative_to(PROJECT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
manifest_path=root/'mirror_manifest.json';assert not manifest_path.exists()
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');files.append(manifest_path)
archive=Path('C:/Users/gb/.codex_tmp/trained_outlet_utility_complete_mirror_text.tar.gz');assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
 for f in files:tar.add(f,arcname=f.relative_to(PROJECT).as_posix(),recursive=False)
archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
for host,(remote_root,_) in HOSTS.items():
 remote_archive=remote_root+'/results/trained_outlet_utility_complete_mirror_text.tar.gz'
 command(['scp',*OPTIONS,str(archive),host+':'+remote_archive])
 code=f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({remote_root!r});a=Path({remote_archive!r})
assert hashlib.sha256(a.read_bytes()).hexdigest()=={archive_sha!r}
with tarfile.open(a,'r:gz') as tar:tar.extractall(r,filter='data')
m=json.loads((r/'results/trained_outlet_utility_20261004/mirror_manifest.json').read_text())
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in m.items())
print(json.dumps(dict(status='EXACT_TEXT_MIRROR',host={host!r},files=len(m))))
'''
 print(remote_python(host,code).strip(),flush=True)
digest=sync_handoff()
proof=p/'trained_outlet_utility_complete_publication.json';assert not proof.exists()
proof.write_text(json.dumps(dict(status='EXACT_THREE_TEXT_MIRRORS_FIVE_HANDOFFS',at=now,files=len(manifest),
 doc_sha256=digest,archive_sha256=archive_sha,cases=2352,condition_query_rows=493920,optimizer_updates=0,new_weights=0,goal_complete=False),indent=2)+'\n',encoding='utf-8')
for host,(remote_root,_) in HOSTS.items():command(['scp',*OPTIONS,str(proof),host+':'+remote_root+'/results/preflight/'])
owned=[root.relative_to(PROJECT).as_posix(),goal.relative_to(PROJECT).as_posix(),
 'results/preflight/trained_outlet_utility_observer_terminal.json',
 'results/preflight/trained_outlet_utility_completion_entrypoint_20261004.py',proof.relative_to(PROJECT).as_posix(),'docs/实验交接.md']
command(['git','add','--',*owned],cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=PROJECT)
command(['git','commit','-m','Audit all trained outlet stages and record next evidence-based intervention'],cwd=PROJECT)
command(['git','push','origin','HEAD:main'],cwd=PROJECT)
head=command(['git','rev-parse','HEAD'],cwd=PROJECT).strip()
print('TRAINED_OUTLET_COMPLETE_PUBLISHED',json.dumps(dict(head=head,doc_sha256=digest,files=len(manifest))),flush=True)
