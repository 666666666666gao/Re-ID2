"""Publish M4 source and its actual observed startup, preserving other completed evidence."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, DESKTOP, HOSTS, OPTIONS, command, remote_python

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
pf = PROJECT / 'results/preflight'
plan = json.loads((pf / 'full_official_modality_outlet_plan_20261005.json').read_text(encoding='utf-8'))
launch = json.loads((pf / 'full_official_modality_outlet_launch_20261005.json').read_text(encoding='utf-8'))
observation = [json.loads(row) for row in (pf / 'full_official_modality_outlet_primary_observer_20261005.jsonl').read_text(encoding='utf-8').splitlines()][-1]
assert launch['status'] == 'M4_CONTROLLER_LAUNCHED_REAL_CONTRACT_AND_NINE_SMOKE_UPDATES_PENDING'
assert observation['controller_pid'] == launch['launch']['pid']
assert not observation['failures'] and (observation['controller_alive'] or observation['controller_complete'])
assert plan['P'] == 16 and plan['K'] == 4 and plan['batch'] == 64
assert all(hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() == digest for file, digest in plan['source_sha256'].items())
proof = pf / 'full_official_modality_outlet_start_publication_20261005.json'
snapshot = pf / 'full_official_modality_outlet_publication_snapshot_20261005.json'
archive = Path(__file__).with_suffix('.tar.gz')
assert not proof.exists() and not snapshot.exists() and not archive.exists()
assert command(['git', 'diff', '--cached', '--name-only'], cwd=PROJECT).strip() == ''
snapshot.write_text(json.dumps(observation, indent=2) + '\n', encoding='utf-8')
now = datetime.now().isoformat(timespec='seconds')
active = {row['variant'] for row in observation['active_neural_processes'] if row['phase'] == 'training'}
actual_started = sum(row['training_epoch'] > 0 or row['variant'] in active for row in observation['runs'])
completed = sum(row['training_complete'] for row in observation['runs'])
contracts = sum(row['contract_pass'] for row in observation['runs'])
smoke_updates = sum(row.get('smoke_updates', 0) for row in observation['runs'])
goal_path = pf / 'research_goal_optimized_20261003.json'
goal = json.loads(goal_path.read_text(encoding='utf-8'))
goal['updated_at'] = now
goal['current_evidence']['full_official_modality_outlet_m4'].update(
    status='M4_CONTROLLER_LAUNCHED_WITH_ACTUAL_OBSERVED_STARTUP',
    controller_pid=launch['launch']['pid'], observer_snapshot=snapshot.relative_to(PROJECT).as_posix(),
    launch='results/preflight/full_official_modality_outlet_launch_20261005.json',
    actual_contracts_passed=contracts, actual_smoke_updates=smoke_updates,
    preflight_pass=observation['preflight_pass'], actual_fresh50_runs=actual_started,
    completed_fresh50_runs=completed, planned_fresh50_runs=3, new_model_parameters=0)
goal['next_action'].update(milestone='FULL_OFFICIAL_M4_THREE_MATCHED_TRIALS',
    status='MONITOR_ONE_EXISTING_CONTROLLER_AND_COLLECT_COMPLETE_TRIAL_RESULTS',
    action='继续唯一240秒观察器监控既有M4控制器，不重复启动。三个版本合同与9次AMP真实更新全部通过后自动进入fresh50；每模型mAP-best严格重载后全49及GT CPU49、6状态全49及GT CPU294，完整六指标/逐query/误伤救回/校准/成本终态再分析。全部正式数据，无人工留出；原始距离逐份本地SHA/大小验证后清理。与同variant M3a及M4同参数普通专家公平比较，机制净收益成立前不扩展为最终三数据集方法。Goal ACTIVE/UNMET。')
goal_path.write_text(json.dumps(goal, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
document = PROJECT / 'docs/实验交接.md'
text = document.read_text(encoding='utf-8')
marker = '<!-- FULL_OFFICIAL_PROTOCOL_OVERRIDE_START -->'
heading = '### 完整官方M4实际PM出口监督对照启动'
assert text.count(marker) == 1 and heading not in text
addition = f'''\n\n{heading}（{now}）

按用户要求，归档原始距离的传输期间并行准备/推进实验。基于已执行1176分阶段GT核查提出单因素M4：在M3a测量头仅接受回归梯度的协议上，只给实际路由PM归一化、原DeMo关系anchor、关系质量×合法关系数加权后mean7的512D出口增加0.1公共身份监督。目标为当前同完整输入的停止梯度公共图库，正样本同身份不同观测；没有新参数、推理变更或新增频带，不改原有F关系/公共监督、贡献loss、缺失训练和路由。普通频域/普通双专家接受同一新增监督，不将通用监督增益都归因于双轴。

计划3个公开CLIP fresh50：axis_shared在2026物理GPU2；frequency_shared后接twins_shared在GPU3，每卡串行、至多2个NN。全部官方train1032/query591/gallery1055，0人工留出，seed42；沿原MSVR采样B64=P16身份×K4实例。最初准备计划的K标注误写为16，审查已修正为K4/P16，训练采样源码未改动。三版本参数及5632D与既有M3a各自相同，冻结其他共享源码；只部署新增文件。固定50轮官方benchmark mAP最佳且最早并列checkpoint，之后全49共147条件、六状态882案例/521262重复query行独立GT核查；这种benchmark选点不是未参与选择的独立测试。

实际启动凭证：控制器PID {launch['launch']['pid']}，启动时间{launch['launched_at']}；观察时间{observation['local_observed_at']}，合同已通过{contracts}/3，真实smoke更新{smoke_updates}/9，全局preflight通过{observation['preflight_pass']}，fresh50已开始{actual_started}/3、已完成{completed}/3。这个启动状态不是终态指标。独立新上下文同家族源码审查为provisional、后端未独立证明；真实CUDA零权重四状态全可用性FP32/AMP等价、PM非零梯度及公共教师/测量头梯度隔离、3次真实更新/全可训练参数梯度/内存严格重载全部通过后才允许50轮。唯一主观察器每240秒查询，不因观察超时重复启动。无温度或功率条件。

精确源码/计划/审查/真实启动与固定观察快照在results/preflight/full_official_modality_outlet_*_20261005；后续结果位于2026的runs/full_official_modality_outlet_m4_20261005，按实际完成再收集分析。目前不把计划中的50轮、147条件或882案例写成已完成，也没有新的三数据集结果或达标结论。只保留每受控实验best.pth和必要原始参照；文字正负结果永久保留，原始距离先本地核验再清理服务器副本。研究目标继续ACTIVE/UNMET。
'''
document.write_bytes(text.replace(marker, marker + addition, 1).encode('utf-8'))
DESKTOP.write_bytes(document.read_bytes())
extras = [
    'results/preflight/full_official_modality_outlet_plan_20261005.json',
    'results/preflight/full_official_modality_outlet_source_review_20261005.json',
    'results/preflight/full_official_modality_outlet_deployer_20261005.py',
    'results/preflight/full_official_modality_outlet_deployer_source_review_20261005.json',
    'results/preflight/full_official_modality_outlet_observer_20261005.py',
    'results/preflight/full_official_modality_outlet_launch_20261005.json',
    'results/preflight/full_official_modality_outlet_publication_snapshot_20261005.json',
    'results/preflight/full_official_modality_outlet_launch_publisher_20261005.py',
    'results/preflight/full_official_modality_outlet_launch_publisher_source_review_20261005.json',
    'results/preflight/research_goal_optimized_20261003.json']
files = [*plan['source_sha256'], *extras]
assert all((PROJECT / file).suffix in ('.py', '.json') for file in files)
with tarfile.open(archive, 'w:gz') as handle:
    for file in files: handle.add(PROJECT / file, arcname=file)
expected = {file: hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() for file in files}
digest = hashlib.sha256(document.read_bytes()).hexdigest(); expected['docs/实验交接.md'] = digest
for host in ('2026', '2027'):
    remote_root, _ = HOSTS[host]
    target = remote_root + '/full_official_modality_outlet_start_text_20261005.tar.gz'
    command(['scp', *OPTIONS, str(archive), host + ':' + target])
    excluded = tuple(plan['source_sha256']) if host == '2026' else ()
    remote_python(host, 'import tarfile;from pathlib import Path;p=Path(' + repr(target) + ');archive=tarfile.open(p);members=[m for m in archive.getmembers() if m.name not in ' + repr(excluded) + '];archive.extractall(' + repr(remote_root) + ',members=members,filter="data");p.unlink()')
    command(['scp', *OPTIONS, str(document), host + ':' + remote_root + '/docs/实验交接.md'])
    actual = json.loads(remote_python(host, 'import hashlib,json;from pathlib import Path;root=Path(' + repr(remote_root) + ');files=' + repr(files + ['docs/实验交接.md']) + ';print(json.dumps({file:hashlib.sha256((root/file).read_bytes()).hexdigest() for file in files}))'))
    assert actual == expected
assert DESKTOP.read_bytes() == document.read_bytes()
proof.write_text(json.dumps(dict(status='M4_SOURCE_AND_ACTUAL_STARTUP_PUBLISHED', published_at=datetime.now().isoformat(timespec='seconds'),
    controller_pid=launch['launch']['pid'], snapshot=snapshot.relative_to(PROJECT).as_posix(),
    document_sha256=digest, verified_locations=['project', 'Desktop', '2026', '2027'],
    pending_mirror='2025_IO', actual_fresh50_started=actual_started, actual_fresh50_complete=completed,
    actual_contracts_passed=contracts, actual_smoke_updates=smoke_updates, goal_complete=False), indent=2) + '\n', encoding='utf-8')
for host in ('2026', '2027'):
    remote_root, _ = HOSTS[host]
    command(['scp', *OPTIONS, str(proof), host + ':' + remote_root + '/results/preflight/' + proof.name])
command(['git', 'add', '--', *files, 'docs/实验交接.md', proof.relative_to(PROJECT).as_posix()], cwd=PROJECT)
command(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], cwd=PROJECT)
command(['git', 'commit', '-m', 'Launch matched full official PM outlet supervision trials while archiving distances'], cwd=PROJECT)
command(['git', 'push', 'origin', 'main'], cwd=PROJECT)
head = command(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).strip()
assert command(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=PROJECT).split()[0] == head
print('M4_ACTUAL_STARTUP_PUBLISHED', json.dumps(dict(head=head, document_sha256=digest, goal_complete=False)), flush=True)
