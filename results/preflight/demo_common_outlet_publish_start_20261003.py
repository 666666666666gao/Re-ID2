from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python, sync_handoff

HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
plan=json.loads((PROJECT/'results/preflight/common_outlet_scale_plan.json').read_text())
review=json.loads((PROJECT/'results/preflight/common_outlet_scale_review.json').read_text())
launch=json.loads((PROJECT/'results/preflight/common_outlet_scale_2026_launch.json').read_text())
assert review['status']=='PASS' and not review['blockers'] and launch['selected_gpus']==[2,3]
snapshot=json.loads(remote_python('2026',f'''import json
from pathlib import Path
root=Path({launch['output']!r});process=Path('/proc/{launch['pid']}')
live=process.exists() and (process/'stat').read_text().split()[2]!='Z'
result=json.loads((root/'controller_result.json').read_text()) if (root/'controller_result.json').exists() else None
print(json.dumps(dict(pid={launch['pid']},live=live,controller_result=result)))
'''))
(PROJECT/'results/preflight/common_outlet_scale_initial_actual.json').write_text(json.dumps(snapshot,indent=2)+'\n')
now=datetime.now().isoformat(timespec='seconds')
phase=f'''### M1公共出口诊断启动：固定V12权重、GPU2/3、无训练（{now}）

按优化后的Goal优先做实际测量，而不是直接堆三个新模块。三新source与launcher/CPU审计/deployer已由fresh Astra/max源审查PASS，报告common_outlet_scale_review.json；同模型家族provisional，仅源审查，不是GPU或指标通过。核对48项源SHA、实际融合/投影调用链、147前序CSV、smoke全局屏障/失败mock、物理GPU隔离、CPU纯numpy导入链和五段远端payload AST。

实际部署2026 controller{launch['pid']}，输出{launch['output']}，本次只读观察controller live={snapshot['live']}，终态文件是否已出现={snapshot['controller_result'] is not None}。物理GPU2串行双轴/普通双专家，GPU3普通频域；每卡一次只运行一个模型，三组七集合/前64 smoke全部通过后才进行完整评测。固定V12最佳epoch22/30/19，不更新参数、不新增权重、不使用官方测试、不改变生产池化或分母。

计划测量七可用集合的合法关系数1/3/7、anchor、mass×k、mean7、向量抵消、实际M/F/I和公共增量，以及同池化/同anchor下M在PM前后、实际路由F在PF前后的身份检索。六阶段×49组合×三模型共882项，为待完成/待核查的计划，不能提前宣称完成；原147项部署指标与逐查询须精确复现，随后对私有FP32距离进行安装GT独立CPU复算。后台observer76800每240秒观察具体PID；观察期限到达不代表训练失败或允许重启。实际无训练任务，本轮不会产生可清理的checkpoint。

M1共享槽位/有效集合平均的同容量重训尚未实现，本测量不能宣称哪种池化更好。原完整三数据集+2/+2及全面缺失目标ACTIVE_UNMET；全部后续神经任务仍仅2026物理GPU2/3。源码、计划、审查及真实部署receipt现同步GitHub与这份唯一交接，其他服务器只同步文本。

'''
document=PROJECT/'docs/实验交接.md'
text=document.read_text(encoding='utf-8')
assert '### M1公共出口诊断启动' not in text
start=text.index('## 当前状态（');end=text.index('\n',start)
text=text[:start]+f'## 当前状态（{now}）'+text[end:]
point=text.index('### 优化后的执行Goal与阶段验收')
document.write_bytes((text[:point]+phase+text[point:]).encode('utf-8'))
goal_path=PROJECT/'results/preflight/research_goal_optimized_20261003.json'
goal=json.loads(goal_path.read_text(encoding='utf-8'))
goal['updated_at']=now;goal['milestones'][1]['status']='MEASUREMENT_LAUNCHED_PENDING_AUDIT'
goal_path.write_text(json.dumps(goal,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for host in ('2025','2027'):
    remote_root,_=HOSTS[host]
    command(['scp',*OPTIONS,*[str(PROJECT/n) for n in plan['sources']],host+':'+remote_root+'/'])
for host in HOSTS:
    remote_root,_=HOSTS[host]
    code='import hashlib,json;from pathlib import Path;print(json.dumps({n:hashlib.sha256((Path('+repr(remote_root)+')/n).read_bytes()).hexdigest() for n in '+repr(list(plan['sources']))+'}))'
    assert json.loads(remote_python(host,code))==plan['sources']
digest=sync_handoff()
sync=PROJECT/'results/preflight/common_outlet_scale_start_handoff_sync.json'
assert not sync.exists()
sync.write_text(json.dumps(dict(observed_at=now,sha256=digest,copies=['repository','Desktop','2025','2026','2027'],goal='ACTIVE_UNMET'),indent=2)+'\n')
helpers=('deploy','observe','cpu','collect','analyze','publish_start')
for short in helpers:
    name=f'demo_common_outlet_{short}_20261003.py'
    (PROJECT/'results/preflight'/name).write_bytes((Path('C:/Users/gb/.codex_tmp')/name).read_bytes())
stage=list(plan['sources'])+['docs/实验交接.md',str(goal_path.relative_to(PROJECT))]
stage+=['results/preflight/common_outlet_scale_'+name for name in ('plan.json','review.json','2026_launch.json','initial_actual.json','start_handoff_sync.json')]
stage+=[f'results/preflight/demo_common_outlet_{short}_20261003.py' for short in helpers]
command(['git','add',*stage],cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=PROJECT)
command(['git','commit','-m','Start reviewed frozen outlet scale measurement on2026 GPUs2 and3'],cwd=PROJECT)
command(['git','push','origin','main'],cwd=PROJECT)
head=command(['git','rev-parse','HEAD'],cwd=PROJECT).strip()
assert command(['git','ls-remote','origin','refs/heads/main'],cwd=PROJECT).split()[0]==head
with Path('C:/Users/gb/memory/2026-10-03.md').open('a',encoding='utf-8') as handle:
    handle.write(f'\nDeMo {now} M1frozenoutlet SOURCEPASS48source/147oldCSV/mocks/5payloadAST, noTorch/numpy/SSHreview. Actualcontroller{launch["pid"]} on2026GPU2/3ONLY; initialPIDlive={snapshot["live"]}, nativeobserver76800 livefirst240s. All3bestepoch22/30/19 no optimizer/newweights; smokeall3barrier thenplanned882sixstagesfull49+GTCPU. Sharedslots/poolingretrainnotyetimplemented. Git{head};ONEdoc5SHA{digest}; source3exact25/26/27. GoalACTIVE_UNMET; do notrerun exclusivehelpers.\n')
print('OUTLET_START_PUBLISHED',json.dumps(dict(head=head,sha256=digest,pid=launch['pid'],actual_live=snapshot['live'])),flush=True)
