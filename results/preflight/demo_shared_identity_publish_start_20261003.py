from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python, sync_handoff

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
plan = json.loads((PROJECT / 'results/preflight/shared_identity_plan.json').read_text(encoding='utf-8'))
review = json.loads((PROJECT / 'results/preflight/shared_identity_review.json').read_text(encoding='utf-8'))
receipt = json.loads((PROJECT / 'results/preflight/shared_identity_2026_launch.json').read_text())
assert review['status'] == 'PASS' and not review['blockers']
root, pid = receipt['output'], receipt['pid']
code = f'''import json
from pathlib import Path
root=Path({root!r});stages={{}}
for path in root.glob('**/*_exit.json'):stages[str(path.relative_to(root))]=json.loads(path.read_text())
for path in root.glob('**/status.json'):stages[str(path.relative_to(root))]=json.loads(path.read_text())
for path in (root/'preflight/tensor/result.json',root/'preflight_result.json',root/'controller_result.json'):
 if path.exists():stages[str(path.relative_to(root))]=json.loads(path.read_text())
live=Path('/proc/{pid}').exists()
if live:live=Path('/proc/{pid}/stat').read_text().split()[2]!='Z'
print(json.dumps(dict(live=live,pid={pid},stages=stages)))
'''
snapshot = json.loads(remote_python('2026', code))
now = datetime.now().isoformat(timespec='seconds')
snapshot['observed_at'] = now
target = PROJECT / 'results/preflight/shared_identity_start_snapshot.json'
assert not target.exists()
target.write_text(json.dumps(snapshot, indent=2) + '\n')
phase = f'''### V11 共享身份坐标与部分查询—完整图库训练（{now}）

目标保持 ACTIVE_UNMET：同一方法在三个数据集的 mAP、Rank-1 均超过同协议 DeMo 至少2个百分点，覆盖缺失条件，并通过同参数/同维度普通专家对照。上一目标轮仅完成用户临时 HTML 动画，不构成研究目标进展；本轮重新核实 fb8808d、前一阶段两个已完成 controller、实际GPU容量后实施新训练代码。

本轮重新配对了双方都采用来源屏蔽的冻结 V5/DeMo 全49结果，RGBNT201/RGBNT100/MSVR310 达到双+2的条件分别为13/9/5（每数据集49）；完整模态仍只有RGBNT201达标。原始六指标差值全147行保存在 all3_masked_V5_minus_masked_DeMo_147.csv；不能只挑改善条件，也不能把来源屏蔽本身声称为最终方案。

新接口保留完整 DeMo 的私有5120维，同时将实际可用模态的共享 CLIP 原始 CLS512 与 patch均值512先平均，再经同一轻量投影得到公共身份512维。描述子固定5632维：私有块/公共块分别归一化，其距离权重为0.75/0.25，使用平方根幅值。频域增量修正非零公共身份方向，模态/交互增量修正合法关系坐标。缺失来源跳过主干与独立 reduction，非法关系 gate/output为0；训练时跳过这些关系的专家BN，避免全零更新。未更换FFT/视觉主干，也未复制21套专家。

训练每个batch先完整输入前向/反向，保留原全局与关系身份监督；再从六种非空部分集合均匀抽取一个，以相同增强图像作为部分查询，完整batch的描述子停止梯度作为图库，使用训练GT身份选择非自身正例与异身份负例。部分损失为标签平滑CE×0.25＋跨图库软Triplet×0.5。两次反向、一次optimizer更新，避免同时保留两次B64主干激活。最终融合ReID监督权重为1，公共身份辅助0.25；增强模型的M/F辅助各0.1及贡献项0.05相同。不会重新使用已失败V10的max四状态hinge。优化器仍是原配置Adam；初始计划误写AdamW已在神经执行前更正并保留原始记录。

第一批固定MSVR310、seed42、公开CLIP、B64、native AMP512、fresh50、最早并列开发mAP最优checkpoint，四卡分别双轴联合路由/普通频域/普通双专家/同接口同部分训练DeMo。前三者要求初始化张量、state keys、参数量、可训练参数和描述子维度完全匹配，普通双专家读取全频普通流、普通频域分支保留DeMo坐标而将额外容量置于新增分支。DeMo对照也接受相同公共身份与缺失训练，不能将通用增强收益归于双轴。当前仅单数据集单种子机制试验；不构成三数据集验收。只保存每组best.pth；smoke在内存严格重载，不保存smoke/last/initial权重。

源码审查为 fresh gpt-6-astra/max、same-family/provisional，见 shared_identity_review.json；审查不等于模型通过。已启动原controller PID {pid}，本次观察 live={snapshot['live']}，实际阶段为 {json.dumps({name: value.get('epoch', value.get('status', value.get('exit_code'))) for name,value in snapshot['stages'].items()},ensure_ascii=False)}。CUDA tensor合同需先通过全部可用性/合法坐标/公共度量/BN/停止图库/初始化容量检查，再并行四个3步smoke，全部通过后才允许50轮；训练结束自动固定best完整评测49种查询/图库组合，每组六指标、CMC1..50、逐查询和分组。未产生的训练成绩不得提前写成结果。240秒观察同一PID，超时不重启，其他项目任务不抢占。

'''
document = PROJECT / 'docs/实验交接.md'
text = document.read_text(encoding='utf-8')
assert '### V11 共享身份坐标与部分查询' not in text
before, rest = text.split('<!-- CURRENT_DEMO_STATUS_START -->', 1)
current, after = rest.split('<!-- CURRENT_DEMO_STATUS_END -->', 1)
point = current.index('\n\n')
current = '\n## 当前状态（' + now + '）\n\n' + phase + current[point + 2:]
document.write_bytes((before + '<!-- CURRENT_DEMO_STATUS_START -->' + current + '<!-- CURRENT_DEMO_STATUS_END -->' + after).encode('utf-8'))
for host in ('2025', '2027'):
    remote_root, _ = HOSTS[host]
    for name in plan['sources']:
        command(['scp', *OPTIONS, str(PROJECT / name), host + ':' + remote_root + '/' + name])
    remote_python(host, f'import hashlib\nfrom pathlib import Path\nroot=Path({remote_root!r})\nassert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {plan["sources"]!r}.items())')
digest = sync_handoff()
sync = PROJECT / 'results/preflight/shared_identity_start_handoff_sync.json'
assert not sync.exists()
sync.write_text(json.dumps(dict(observed_at=now,sha256=digest,copies=['repository','Desktop','2025','2026','2027'],goal='ACTIVE_UNMET'),indent=2)+'\n')
helpers = ['demo_shared_identity_prepare_20261003.py','demo_shared_identity_deploy_20261003.py',
           'demo_shared_identity_observe_20261003.py','demo_shared_identity_publish_start_20261003.py',
           'demo_available_base_matched_comparison_20261003.py']
copied=[]
for name in helpers:
    destination=PROJECT/'results/preflight'/name
    assert not destination.exists()
    destination.write_bytes((Path('C:/Users/gb/.codex_tmp')/name).read_bytes())
    copied.append(destination.relative_to(PROJECT).as_posix())
names=['shared_identity_plan','shared_identity_review','shared_identity_2026_launch','shared_identity_start_snapshot',
       'shared_identity_start_handoff_sync','availability_base_matched_comparison']
command(['git','add','docs/实验交接.md',*plan['sources'],*copied,*['results/preflight/'+name+'.json' for name in names],
         'results/availability_base_comparison_20261003/all3_masked_V5_minus_masked_DeMo_147.csv'],cwd=PROJECT)
trace='.aris/traces/experiment-bridge/2026-10-03_shared_identity'
command(['git','add','-f',trace],cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=PROJECT)
command(['git','commit','-m','Add availability-consistent shared identity and matched asymmetric training trial'],cwd=PROJECT)
command(['git','push','origin','main'],cwd=PROJECT)
head=command(['git','rev-parse','HEAD'],cwd=PROJECT).strip()
assert command(['git','ls-remote','origin','refs/heads/main'],cwd=PROJECT).split()[0]==head
with Path('C:/Users/gb/memory/2026-10-03.md').open('a',encoding='utf-8') as handle:
    handle.write(f'\nDeMo {now}: Previousgoalturn HTMLside task no research progress; currentturn revalidatedoldcontrollers terminal/no ownNN/capacity26all4 and implemented sharedidentity5632 private.75/common.25, legalavailability/skippedinvalidBN, fullthenpartial backwards/oneAdamstep, stoppedfullgalleryGTtriplet. Matched4MSVRfresh50planned and actualcontroller{pid}live={snapshot["live"]}; exactsnapshotstages inshared_identity_start_snapshot. Fresh Astra/max source-review{review["status"]}samefamilyprovisional; noGPUchecksclaimedbysource. FrozenmaskedV5vsmaskedDeMo147CSV double+2count20113/1009/MSVR5 normal100/MSVRstillFAIL. Newcode+plan+startGit{head};ONEdoc5SHA{digest};onlybestweightpolicy. GoalACTIVE_UNMET.\n')
print('SHARED_IDENTITY_START_PUBLISHED',json.dumps(dict(head=head,sha256=digest,pid=pid,live=snapshot['live'])),flush=True)
