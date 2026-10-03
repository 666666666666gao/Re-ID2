from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, command, remote_python, copy_file, sync_handoff

HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
receipt=json.loads((PROJECT/'results/preflight/shared_identity_2026_launch.json').read_text())
snapshot=json.loads((PROJECT/'results/preflight/shared_identity_latest_snapshot.json').read_text())
assert snapshot['live'] and snapshot['pid']==receipt['pid']
assert snapshot['stages']['preflight_result.json']['status']=='PASS'
tensor=snapshot['stages']['preflight/tensor/result.json']
assert tensor['status']=='PASS_SHARED_IDENTITY_TENSOR_CONTRACT' and len(tensor['availability_checks'])==28
code=f'''import hashlib,json
from pathlib import Path
root=Path({receipt['output']!r});pre=root/'preflight'
rows=[]
for path in sorted(pre.rglob('*')):
 if path.is_file() and path.suffix in ('.json','.log'):
  rows.append(dict(relative=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
print(json.dumps(rows))
'''
rows=json.loads(remote_python('2026',code))
local=PROJECT/'results/shared_identity_v11_trial_20261003'
for row in rows:
    destination=local/row['relative']
    copy_file('2026',receipt['output']+'/'+row['relative'],destination)
    assert destination.stat().st_size==row['bytes'] and hashlib.sha256(destination.read_bytes()).hexdigest()==row['sha256']
for variant in ('axis_shared','frequency_shared','twins_shared','demo_shared'):
    result=json.loads((local/'preflight'/variant/'smoke.json').read_text())
    assert result['status']=='SMOKE_PASS' and result['steps']==3 and result['strict_reload_equal']
    assert all(result['gradients'].values()) and all(d['optimizer_updated'] for d in result['details'])
    assert not list((local/'preflight'/variant).glob('*.pth'))
static=PROJECT/'results/preflight/shared_identity_training_verified_snapshot.json'
assert not static.exists()
static.write_bytes(json.dumps(snapshot,indent=2).encode())
(local/'preflight_intake.json').write_text(json.dumps(dict(files=rows,source=receipt['output'],neural_contract_pass=True,all4_smokes_pass=True),indent=2)+'\n')
now=datetime.now().isoformat(timespec='seconds')
actual={name.split('/')[-2]:dict(epoch=value['epoch'],steps=value['steps'],optimizer_steps=value['latest']['optimizer_steps'],
                              amp_skipped_steps=value['latest']['amp_skipped_steps'])
        for name,value in snapshot['stages'].items() if name.endswith('status.json')}
paragraph=f'''V11 实际运行更新（{now}，观察来源{snapshot['observed_at']}）：CUDA tensor合同PASS，四模型×七种可用集合共28项来源/维度/公共度量检查通过；三种增强模型均100263046参数、100250758可训练参数、5632D，初始化张量/state keys完全相同，DeMo公共头初始化相同。非法专家BN不更新，冻结state版本不变，跨图库训练引用停止梯度、正例排除自身与GT负例检查通过。四组各3步smoke均真实更新、梯度有限非零、内存严格重载等同，无smoke权重文件。原controller{receipt['pid']}仍存活，实际训练进度为{json.dumps(actual,ensure_ascii=False)}；尚未完成50轮及49矩阵，不报终态成绩。实际原始preflight和静态运行快照已入库，后续按剩余轮数接近预计终点观察，原observer240秒不变。

'''
document=PROJECT/'docs/实验交接.md'
text=document.read_text(encoding='utf-8')
assert 'V11 实际运行更新（' not in text
marker='### V11 共享身份坐标与部分查询—完整图库训练'
point=text.index('\n\n',text.index(marker))+2
text=text[:point]+paragraph+text[point:]
document.write_bytes(text.encode('utf-8'))
digest=sync_handoff()
sync=PROJECT/'results/preflight/shared_identity_verified_handoff_sync.json'
assert not sync.exists()
sync.write_text(json.dumps(dict(observed_at=now,sha256=digest,copies=['repository','Desktop','2025','2026','2027'],goal='ACTIVE_UNMET'),indent=2)+'\n')
helpers=['demo_shared_identity_publish_verified_20261003.py','demo_shared_identity_collect_20261003.py','demo_shared_identity_analyze_20261003.py']
for name in helpers:
    target=PROJECT/'results/preflight'/name
    assert not target.exists()
    target.write_bytes((Path('C:/Users/gb/.codex_tmp')/name).read_bytes())
command(['git','add','docs/实验交接.md','results/shared_identity_v11_trial_20261003/preflight',
         'results/shared_identity_v11_trial_20261003/preflight_intake.json',
         'results/preflight/shared_identity_training_verified_snapshot.json','results/preflight/shared_identity_verified_handoff_sync.json',
         *['results/preflight/'+name for name in helpers]],cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=PROJECT)
command(['git','commit','-m','Record actual shared identity contracts and four matched training smoke passes'],cwd=PROJECT)
command(['git','push','origin','main'],cwd=PROJECT)
head=command(['git','rev-parse','HEAD'],cwd=PROJECT).strip()
assert command(['git','ls-remote','origin','refs/heads/main'],cwd=PROJECT).split()[0]==head
with Path('C:/Users/gb/memory/2026-10-03.md').open('a',encoding='utf-8') as handle:
    handle.write(f'\nDeMo {now}: V11 actualCUDA28availabilitycontracts+all4smoke3real finite nonzero gradients/strict in-memory reload PASS; aug3param100263046/trainable100250758/5632exactinitstates. Fourfresh50 liveoriginalcontroller3446554; static snapshot{snapshot["observed_at"]}progress{actual}/all0AMPskip. Observer native2842 running/240secs/deadline~19:12; NEVERrestart on observation timeout. Publish native64538 completed0 startGit2e8c835; actualgate+runningpublishGit{head};ONEdoc5SHA{digest};oldothersdirtyuntracked snapshots untouched. Newterminal-only collect/analyze stdlibhelpers prepared NOTexecuted before50+196conditionterminal. Next waitnear~18:57trainingend/frozen49~19:00, collectsamecontroller, aggregate all6/CMC50/groups/sampling+partialmasks/pairedharmrescue, actual utility fourstates stillneeded before mechanism claim. Onlybest saved. GoalACTIVE_UNMET/currentturnPROGRESS.\n')
print('SHARED_IDENTITY_NEURAL_GATES_PUBLISHED',json.dumps(dict(head=head,sha256=digest,pid=receipt['pid'],progress=actual)),flush=True)
