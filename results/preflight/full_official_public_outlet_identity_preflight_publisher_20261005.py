"""Publish actual native preflight evidence and prospective closeout sources."""
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
review_path = pf / 'full_official_public_outlet_identity_closeout_preparation_review_20261005.json'
actual_path = pf / 'full_official_public_outlet_identity_actual_preflight_20261005.json'
proof = pf / 'full_official_public_outlet_identity_preflight_publication_20261005.json'
archive = Path(__file__).with_suffix('.tar.gz')
assert not proof.exists() and not archive.exists()
review = json.loads(review_path.read_text(encoding='utf-8'))
assert review['status'] == 'PASS' and not review['blocking_findings']
assert all(hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() == digest for file, digest in review['sources_sha256'].items())
actual = json.loads(actual_path.read_text(encoding='utf-8'))
assert actual['status'] == 'FOUR_REAL_NATIVE_PUBLIC_OUTLET_CONTRACTS_AND_TWELVE_TRUE_SMOKE_UPDATES_PASSED'
assert actual['actual_contract_optimizer_updates'] == 16 and actual['actual_smoke_optimizer_updates'] == 12
now = datetime.now().isoformat(timespec='seconds')
goal_path = pf / 'research_goal_optimized_20261003.json'
goal = json.loads(goal_path.read_text(encoding='utf-8')); assert goal['status'] == 'ACTIVE_UNMET'
goal['updated_at'] = now
goal['current_evidence']['full_official_public_outlet_identity_m6'].update(
    status='FOUR_NATIVE_CONTRACTS_AND_ALL_SMOKES_PASSED_TWO_FRESH50_RUNNING',
    actual_contract_optimizer_updates=16, actual_smoke_optimizer_updates=12,
    actual_smoke_amp_skips=actual['actual_smoke_amp_skips'], actual_fresh50_runs_started=2,
    actual_completed_fresh50=0, preflight_evidence=actual_path.relative_to(PROJECT).as_posix(),
    training_started_snapshot=actual['training_started_snapshot'],
    closeout_sources='Prepared and reviewed only; collect/analyze/archive after all actual196+882 terminal cases.',
    caveat='The saved sole-observer snapshot proves two started fresh50 runs; no final metric or candidate superiority is inferred.')
goal['next_action'].update(status='MONITOR_EXISTING_TWO_LANE_FULL50_AND_COMPLETE_FROZEN49_STATES',
    action='Continue the same sole240-second observer and original controller. At the actual four-run196/882 terminal, collect text, analyze all six signed metrics against original/M4/current ordinary/strongM5frequency references, then archive343 distances locally by size/SHA before clearing matching server copies.')
goal_path.write_bytes((json.dumps(goal, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
document = PROJECT / 'docs/实验交接.md'; text = document.read_text(encoding='utf-8')
marker = '<!-- FULL_OFFICIAL_PROTOCOL_OVERRIDE_START -->'
heading = '### 公共出口直接身份监督：真实原生预检通过，完整训练已开始'
assert text.count(marker) == 1 and heading not in text
addition = f'\n\n{heading}（{now}）\n\n'
addition += '四组各实际完成父步骤、父重复、零权重委托、正向新CE步骤4次更新，共16次原生B64 Adam更新。零权重原损失/梯度按原容差、原缓冲区/RNG/明细一致；每个步骤按自身实际Adam矩重建更新，跨执行参数末位差异仅为诊断，沿用有原父重复证据支持的M5v3规则。正向实际捕获full/partial真实公共512、同一现有分类头、官方训练标签、原损失加0.05+0.05新CE，4次主干、2次投影，3次共享BN及相对原父多2次计数；其他缓冲区/RNG一致，更新真实发生。\n\n'
addition += '| 模型 | 合同真实更新 | AMP预检真实更新 | AMP跳步 | 尝试数 | 严格重载 |\n|---|---:|---:|---:|---:|---|\n'
for name, smoke in actual['smokes'].items():
    addition += f'| {name} | 4 | {smoke["steps"]} | {smoke["amp_skipped_steps"]} | {smoke["attempts"]} | {smoke["strict_reload_equal"]} |\n'
addition += f'\n共12次预检真实更新，AMP跳步{actual["actual_smoke_amp_skips"]}次，原始次数和明细保留，不将尝试数冒充更新数。四组全部通过后原队列才启动完整训练。保存的唯一观察器快照时间{actual["training_started_snapshot"]["local_observed_at"]}：\n\n'
for row in actual['training_started_snapshot']['runs']:
    addition += f'- GPU{row["gpu"]} / {row["variant"]}：合同/预检通过，已记录训练第{row["training_epoch"]}轮，完整终态={row["training_complete"]}。\n'
addition += '\nDeMo和普通频域正在两张卡并行训练；双轴和普通双专家随后各自串行运行。保持同一个控制器和240秒观察器，不重复启动。按上一轮实际进程时间估计，两条队列及全部冻结/六状态评测约需70分钟，实际终态为准。只有全官方1032/591/1055，0人工留出，原B64/P16/K4/seed42/publicCLIP/fresh50/唯一best，不使用历史研究权重。\n\n'
addition += '训练期间已准备并审查下一轮终态文本收集、完整六指标/49有符号比较、六状态及贡献/误伤救回分析、343距离本地大小/SHA归档代码。这些后处理尚未执行，必须等待当前四组实际196冻结+882六状态GT核查完整终态。分析将包含9组×49=441有符号比较，原DeMo/公共DeMo、同目标普通频域/双专家、M4对应父版本及M5强普通频域均保留；M5到M6不是单因素消融。若新增监督只提升普通对照或仍误伤，则如实记录，不扩大方法主张。此前343距离归档完成、远端对应副本清理，所有必要参照best保留。\n\n'
addition += '传输与独立实验准备/审查/合格预检和训练并行的规则持续有效；不把传输结束作为额外启动许可。本轮仍没有完整训练最终指标，目标ACTIVE_UNMET：统一方法三数据集mAP/Rank-1各+2、缺失公平验证、实际协作收益及多种子尚待建立。此为单种子完整官方开发、benchmark参与checkpoint选择，非独立未触碰测试。源码审查同家族临时认可、后端未认证；上述真实合同和预检记录独立于源码认可。交接只有一个，project/Desktop/2026/2027字节一致；2025 I/O仍待恢复。\n'
document.write_bytes(text.replace(marker, marker + addition, 1).encode('utf-8')); DESKTOP.write_bytes(document.read_bytes())
files = [*review['sources_sha256'], review_path.relative_to(PROJECT).as_posix(),
    actual_path.relative_to(PROJECT).as_posix(), goal_path.relative_to(PROJECT).as_posix()]
assert len(files) == len(set(files))
with tarfile.open(archive, 'w:gz') as packed:
    for file in files:
        packed.add(PROJECT / file, arcname=file)
digest = hashlib.sha256(document.read_bytes()).hexdigest()
expected = {file: hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() for file in files}
expected['docs/实验交接.md'] = digest
assert command(['git', 'diff', '--cached', '--name-only'], cwd=PROJECT).strip() == ''
for host in ('2026', '2027'):
    remote_root, _ = HOSTS[host]; target = remote_root + '/full_official_public_identity_preflight_20261005.tar.gz'
    command(['scp', *OPTIONS, str(archive), host + ':' + target])
    remote_python(host, 'import tarfile;from pathlib import Path;p=Path(' + repr(target) + ');tarfile.open(p).extractall(' + repr(remote_root) + ',filter="data");p.unlink()')
    command(['scp', *OPTIONS, str(document), host + ':' + remote_root + '/docs/实验交接.md'])
    code = 'import hashlib,json;from pathlib import Path;root=Path(' + repr(remote_root) + ');files=' + repr(list(expected)) + ';print(json.dumps({file:hashlib.sha256((root/file).read_bytes()).hexdigest() for file in files}))'
    assert json.loads(remote_python(host, code)) == expected
assert DESKTOP.read_bytes() == document.read_bytes()
proof.write_bytes((json.dumps(dict(status='ACTUAL_PUBLIC_IDENTITY_NATIVE_AND_SMOKE_PREFLIGHT_PUBLISHED',
    published_at=now, document_sha256=digest, verified_locations=['project','Desktop','2026','2027'],
    pending_mirror='2025_IO', actual_contract_updates=16, actual_smoke_updates=12,
    actual_smoke_amp_skips=actual['actual_smoke_amp_skips'], actual_completed_fresh50=0,
    closeout_sources='Prepared only, not executed', goal_complete=False), indent=2) + '\n').encode('utf-8'))
for host in ('2026','2027'):
    command(['scp', *OPTIONS, str(proof), host + ':' + HOSTS[host][0] + '/results/preflight/' + proof.name])
files += ['docs/实验交接.md', proof.relative_to(PROJECT).as_posix()]
pathspec = Path(__file__).with_suffix('.pathspec'); pathspec.write_bytes(b'\0'.join(file.encode('utf-8') for file in files) + b'\0')
command(['git','add','--pathspec-from-file=' + str(pathspec),'--pathspec-file-nul'], cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'], cwd=PROJECT)
command(['git','commit','-m','Record real native public outlet preflights and prepare complete audited readout'], cwd=PROJECT)
command(['git','push','origin','main'], cwd=PROJECT)
head = command(['git','rev-parse','HEAD'], cwd=PROJECT).strip()
assert command(['git','ls-remote','origin','refs/heads/main'], cwd=PROJECT).split()[0] == head
print('FULL_OFFICIAL_PUBLIC_IDENTITY_PREFLIGHT_PUBLISHED',json.dumps(dict(head=head,document_sha256=digest,goal_complete=False)),flush=True)
