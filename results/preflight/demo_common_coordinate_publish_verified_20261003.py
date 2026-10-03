from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, copy_file, command, remote_python, sync_handoff

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
launch = json.loads((PROJECT / 'results/preflight/common_coordinate_2026_launch.json').read_text())
root = launch['output']
variants = ('axis_shared','frequency_shared','twins_shared','demo_shared')
files = ['preflight/tensor/result.json','preflight_result.json','preflight/tensor_exit.json']
files += [f'preflight/{variant}/smoke.json' for variant in variants]
files += [f'preflight/{variant}_exit.json' for variant in variants]
code = f'''import hashlib,json
from pathlib import Path
root=Path({root!r})
assert json.loads((root/'preflight_result.json').read_text())['status']=='PASS'
assert all(json.loads((root/name).read_text())['exit_code']==0 for name in {files!r} if name.endswith('_exit.json'))
print(json.dumps({{name:dict(bytes=(root/name).stat().st_size,sha256=hashlib.sha256((root/name).read_bytes()).hexdigest()) for name in {files!r}}}))
'''
inventory = json.loads(remote_python('2026', code))
dest = PROJECT / 'results/preflight/common_coordinate_actual_preflight'
assert not dest.exists()
for name, value in inventory.items():
    path = dest / name
    copy_file('2026', root + '/' + name, path)
    assert path.stat().st_size == value['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == value['sha256']
contract = json.loads((dest / 'preflight/tensor/result.json').read_text())
assert contract['common_increments_only'] and contract['ordinary_frequency_legacy_parity']
assert len(contract['availability_checks']) == 28
smokes = {variant:json.loads((dest / f'preflight/{variant}/smoke.json').read_text()) for variant in variants}
assert all(r['status']=='SMOKE_PASS' and r['strict_reload_equal'] and all(r['gradients'].values()) for r in smokes.values())
steps = sum(sum(int(d['optimizer_updated']) for d in r['details']) for r in smokes.values())
assert steps == 12
report = dict(status='ACTUAL_COMMON_COORDINATE_PREFLIGHT_PASS',selected_gpus=[2,3],optimizer_updates=steps,amp_skips=0,
    tensor=contract,smokes={variant:dict(parameters=r['parameters'],trainable_parameters=r['trainable_parameters'],
      gradients_checked=len(r['gradients']),strict_reload_equal=r['strict_reload_equal'],peak_memory=r['peak_memory']) for variant,r in smokes.items()},
    files=inventory,goal='ACTIVE_UNMET',limit='Preflight success does not establish final retrieval advantage; fresh50/full49/controlled cases/GT recount pending.')
proof = PROJECT / 'results/preflight/common_coordinate_actual_preflight.json'
assert not proof.exists()
proof.write_text(json.dumps(report,indent=2)+'\n')
snapshot = json.loads((PROJECT / 'results/preflight/common_coordinate_latest_snapshot.json').read_text())
progress = {name:dict(epoch=value['epoch'],steps=value['steps'],optimizer_steps=value['latest']['optimizer_steps'],amp_skips=value['latest']['amp_skipped_steps'])
            for name,value in snapshot['stages'].items() if name.endswith('status.json')}
now = datetime.now().isoformat(timespec='seconds')
document = PROJECT / 'docs/实验交接.md'
text = document.read_text(encoding='utf-8')
assert 'V12 实际两卡预检已通过' not in text
marker = '### V12 同一公共身份坐标的检索增量试验'
point = text.index('\n\n',text.index(marker))+2
phase = f'''V12 实际两卡预检已通过（{now}）：28种模型—可用性CUDA契约全部通过，所有四状态private5120逐元素等于00；新普通频域的四状态在七种初始可用性下与旧工厂逐元素相等；三个增强模型初始state/参数一致，DeMo公共头初始相同，非法private坐标/非法专家BN/停止梯度图库等原有契约通过。四组真实full+partial短训练共12次optimizer更新、0次AMP skip，所有trainable梯度有限非零，各组8样本内存strict checkpoint重载输出逐元素一致。参数/梯度计数/峰值内存及逐步原始结果见common_coordinate_actual_preflight.json和common_coordinate_actual_preflight目录。

正式训练观察（{snapshot['observed_at']}）：controller{launch['pid']}真实存活，无失败退出，GPU2/3负载正常；状态{json.dumps(progress,ensure_ascii=False)}。中间epoch不是终态验收，也不据此报告完整六指标或宣布+2成功；后续队列仍包含普通双专家和DeMo对照。普通频域的初始前向精确等于旧接口，不意味着50轮训练轨迹必须位级相同，本轮对照必须以自身完整终态为准，不能替换成旧版本最佳指标。

'''
start,end = text.index('## 当前状态（'),text.index('\n',text.index('## 当前状态（'))
text=text[:start]+f'## 当前状态（{now}）'+text[end:]
point=text.index('\n\n',text.index(marker))+2
document.write_bytes((text[:point]+phase+text[point:]).encode('utf-8'))
digest=sync_handoff()
sync=PROJECT/'results/preflight/common_coordinate_verified_handoff_sync.json'
assert not sync.exists()
sync.write_text(json.dumps(dict(observed_at=now,sha256=digest,copies=['repository','Desktop','2025','2026','2027'],goal='ACTIVE_UNMET'),indent=2)+'\n')
helper=PROJECT/'results/preflight/demo_common_coordinate_publish_verified_20261003.py'
helper.write_bytes(Path(__file__).read_bytes())
command(['git','add','docs/实验交接.md','results/preflight/common_coordinate_actual_preflight',str(proof),str(sync),str(helper)],cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=PROJECT)
command(['git','commit','-m','Verify twoGPU CUDA contracts and12 real updates before matched50 wave'],cwd=PROJECT)
command(['git','push','origin','main'],cwd=PROJECT)
head=command(['git','rev-parse','HEAD'],cwd=PROJECT).strip()
assert command(['git','ls-remote','origin','refs/heads/main'],cwd=PROJECT).split()[0]==head
with Path('C:/Users/gb/memory/2026-10-03.md').open('a',encoding='utf-8') as handle:
    handle.write(f'\nDeMo {now} V12 actual28CUDA/privateinvariance/oldfrequencyinitialparity +12realupdates0skip/nonzeroalltrainable/memorystrictreload PASSED; currentcontroller3565271 andnativeobserver24294live (do NOT restart), actualsnapshot{snapshot["observed_at"]} progress{progress}. Only26GPU2/3, secondqueues automatically after each first50+49+294cases. Git{head};ONEdoc5SHA{digest}; goalACTIVE_UNMET. New output /runs/common_coordinate_v12_trial_20261003; currentgoalturn PROGRESS.\n')
print('ACTUAL_TWO_GPU_PREFLIGHT_PUBLISHED',json.dumps(dict(head=head,sha256=digest,updates=12,progress=progress)),flush=True)
