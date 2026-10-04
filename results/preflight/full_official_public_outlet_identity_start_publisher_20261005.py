"""Publish the real controller start, reviewed sources, and one synchronized handoff."""
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
plan_path = pf / 'full_official_public_outlet_identity_plan_20261005.json'
review_path = pf / 'full_official_public_outlet_identity_execution_review_20261005.json'
training_review_path = pf / 'full_official_public_outlet_identity_training_review_20261005.json'
launch_path = pf / 'full_official_public_outlet_identity_launch_20261005.json'
proof = pf / 'full_official_public_outlet_identity_start_publication_20261005.json'
archive = Path(__file__).with_suffix('.tar.gz')
assert not proof.exists() and not archive.exists()
plan = json.loads(plan_path.read_text(encoding='utf-8'))
review = json.loads(review_path.read_text(encoding='utf-8'))
launch = json.loads(launch_path.read_text(encoding='utf-8'))
assert review['status'] == 'PASS' and not review['blocking_findings']
assert plan['sources_sha256'] == review['sources_sha256']
assert all(hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() == digest for file, digest in plan['sources_sha256'].items())
assert review['deployer_sha256'] == hashlib.sha256((pf / 'full_official_public_outlet_identity_deployer_20261005.py').read_bytes()).hexdigest()
assert review['start_publisher_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
complete = json.loads((pf / 'full_official_availability_graph_complete_publication_20261005.json').read_text(encoding='utf-8'))
assert complete['status'] == 'FOUR_COMPLETE_GRAPH_RESULTS_AND343_LOCAL_DISTANCES_PUBLISHED'
assert command(['git', 'diff', '--cached', '--name-only'], cwd=PROJECT).strip() == ''
now = datetime.now().isoformat(timespec='seconds')
goal_path = pf / 'research_goal_optimized_20261003.json'
goal = json.loads(goal_path.read_text(encoding='utf-8')); assert goal['status'] == 'ACTIVE_UNMET'
goal['updated_at'] = now
goal['current_evidence']['full_official_public_outlet_identity_m6'] = dict(
    status=launch['status'], plan=plan_path.relative_to(PROJECT).as_posix(),
    controller_pid=launch['launch']['pid'], controller_output=launch['launch']['output'],
    launch_evidence=launch_path.relative_to(PROJECT).as_posix(), source_review_status='PASS_SOURCE_ONLY',
    actual_fresh50_runs=0, completed_fresh50_runs=0,
    caveat='Launch receipt proves controller start only; native contracts, smoke success and training results require separate real evidence.',
    single_factor_parent='M4 expert parents and original shared-identity DeMo; M5-to-M6 is an exploratory replacement, not a single-factor ablation.')
goal['next_action'] = dict(milestone='FULL_OFFICIAL_PUBLIC_OUTLET_IDENTITY_M6',
    status='NATIVE_CONTRACTS_AND_FOUR_TRUE_THREE_UPDATE_SMOKES_REQUIRED_BEFORE_FRESH50',
    action='Observe the single existing controller at240-second milestones. Read real native update/head/BN/RNG evidence before any fresh50, then collect full49 and enhanced six-state GT audits and all signed fair comparisons.',
    hardware_dependency='Only2026 physicalGPU2/3,max2NN,no power or temperature conditions',
    transport_policy='Prepare, review, preflight and train eligible independent experiments while archival transfer runs; delete only local size/SHA-verified raw copies.')
goal_path.write_bytes((json.dumps(goal, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
document = PROJECT / 'docs/实验交接.md'; text = document.read_text(encoding='utf-8')
marker = '<!-- FULL_OFFICIAL_PROTOCOL_OVERRIDE_START -->'
heading = '### 公共检索出口直接身份监督：下一轮已启动原生预检'
assert text.count(marker) == 1 and heading not in text
addition = f'\n\n{heading}（{now}）\n\n'
addition += '完整M5四组50轮、196个冻结条件和882个六状态条件均完成安装数据集GT独立CPU审计。双轴完整mAP/Rank-1为47.939403/65.989848，普通频域50.034410/67.174281；双轴11−00为−0.141167/−0.338409，首位误伤3、救回1，全部49平均−0.094029/−0.328050。M5没有满足目标，不原样扩到三数据集或多种子。全部正负结果、完整六指标、逐查询变化与缺失分组继续保留。\n\n'
addition += 'M6仅在M4专家父版本/原始公共接口DeMo之上添加一个因素：对实际用于检索的fused描述子5120:5632公共出口，通过现有shared_neck/shared_classifier，完整和原均匀采样缺失输入各加0.05标签平滑身份CE，总权重0.1。只使用全官方训练身份。原完整/缺失CE、Triplet、M4 PM/F对齐、频域关系、测量门控和推理接口保留；不加参数、不增加主干前向、不改变PK/缺失采样、5632D和0.75/0.25度量接口。共享BN每个正向训练步骤额外调用2次，这是明确的训练状态变化，须实际捕获检验。M5到M6同时替换了探索目标，不能写成二者单因素消融。该监督可能仍失败，也可能仅改善普通对照；不预先归因或承诺涨点。\n\n'
addition += '传输约2.538GB原始距离期间并行准备了下一轮原生合同、评测包装、双卡串行队列和观察代码；训练代码此前已准备。执行源码审查随后完成，不将审查结束或神经训练写成发生在传输期间。后续独立预检/训练只依赖审查与实际资源，不以归档完成标志为前提。M5归档按本地大小/SHA逐文件验证后清理远端副本，完整343文件的终态凭证已单独发布。M6源码审查为同家族、临时认可、实际后端未认证，只代表源码无阻塞，不代表CUDA或科学指标通过。\n\n'
addition += f'实际控制器PID {launch["launch"]["pid"]}，输出 {launch["launch"]["output"]}。只用2026物理GPU2：DeMo→双轴，GPU3：普通频域→普通双专家，每卡串行、最多2个神经进程。四组必须各完成父步骤/重复/零权重/正向共4个真实原生更新，以及3个真实AMP更新和严格重载预检，全部通过后才可开始50轮。正向预检捕获真实公共出口、同一共享分类头、官方训练标签、原损失+新CE算式、4次主干/2次公共投影、共享BN3次/+2计数、其他缓冲区和RNG、各次实际Adam更新。启动时这些预检尚待真实结果；不能将计划16合同/12 smoke更新写为已经完成。\n\n'
addition += '随后四组均按全官方MSVR310：1032训练/591查询/1055图库、0人工留出身份、publicCLIP、seed42、原生B64/P16/K4训练50轮；按官方benchmark最高mAP且最早并列保存唯一best，再固定权重评测49组合，增强3组完整六状态×49/头/贡献/误伤与救回，全部安装GT独立CPU核查。此阶段仅单数据集单种子开发，benchmark参与checkpoint选择，不能称独立未触碰最终测试。旧M4、原DeMo/公共DeMo和强M5普通频域均是必要参考，不拼接跨版本最好分数。三数据集统一方法各mAP/Rank-1+2、缺失公平对照、实际正向协作和多种子目标继续ACTIVE_UNMET。唯一交接project/Desktop/2026/2027保持相同字节，2025仍因I/O问题待恢复。\n'
document.write_bytes(text.replace(marker, marker + addition, 1).encode('utf-8')); DESKTOP.write_bytes(document.read_bytes())
extras = [plan_path, review_path, training_review_path, launch_path, goal_path,
    pf / 'full_official_public_outlet_identity_deployer_20261005.py', Path(__file__),
    pf / 'full_official_public_outlet_identity_transfer_overlap_20261005.json',
    pf / 'full_official_availability_graph_axis_frozen_readout_20261005.py',
    pf / 'full_official_availability_graph_axis_frozen_readout_20261005.json']
files = list(plan['sources_sha256']) + [file.relative_to(PROJECT).as_posix() for file in extras]
assert len(files) == len(set(files))
with tarfile.open(archive, 'w:gz') as packed:
    for file in files:
        packed.add(PROJECT / file, arcname=file)
digest = hashlib.sha256(document.read_bytes()).hexdigest()
expected = {file: hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() for file in files}
expected['docs/实验交接.md'] = digest
for host in ('2026', '2027'):
    remote_root, _ = HOSTS[host]; target = remote_root + '/full_official_public_identity_start_20261005.tar.gz'
    command(['scp', *OPTIONS, str(archive), host + ':' + target])
    remote_python(host, 'import tarfile;from pathlib import Path;p=Path(' + repr(target) + ');tarfile.open(p).extractall(' + repr(remote_root) + ',filter="data");p.unlink()')
    command(['scp', *OPTIONS, str(document), host + ':' + remote_root + '/docs/实验交接.md'])
    code = 'import hashlib,json;from pathlib import Path;root=Path(' + repr(remote_root) + ');files=' + repr(list(expected)) + ';print(json.dumps({file:hashlib.sha256((root/file).read_bytes()).hexdigest() for file in files}))'
    assert json.loads(remote_python(host, code)) == expected
assert DESKTOP.read_bytes() == document.read_bytes()
proof.write_bytes((json.dumps(dict(status='PUBLIC_IDENTITY_REAL_CONTROLLER_START_AND_REVIEWED_SOURCES_PUBLISHED',
    published_at=now, controller_pid=launch['launch']['pid'], controller_output=launch['launch']['output'],
    document_sha256=digest, verified_locations=['project', 'Desktop', '2026', '2027'],
    pending_mirror='2025_IO', source_review_status='PASS_SOURCE_ONLY', actual_completed_fresh50=0,
    actual_training_metrics='Pending; no native or scientific success inferred from launch.', goal_complete=False), indent=2) + '\n').encode('utf-8'))
for host in ('2026', '2027'):
    command(['scp', *OPTIONS, str(proof), host + ':' + HOSTS[host][0] + '/results/preflight/' + proof.name])
files += ['docs/实验交接.md', proof.relative_to(PROJECT).as_posix()]
pathspec = Path(__file__).with_suffix('.pathspec'); pathspec.write_bytes(b'\0'.join(file.encode('utf-8') for file in files) + b'\0')
command(['git', 'add', '--pathspec-from-file=' + str(pathspec), '--pathspec-file-nul'], cwd=PROJECT)
command(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], cwd=PROJECT)
command(['git', 'commit', '-m', 'Start matched full-data public outlet identity experiments after complete graph results'], cwd=PROJECT)
command(['git', 'push', 'origin', 'main'], cwd=PROJECT)
head = command(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).strip()
assert command(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=PROJECT).split()[0] == head
print('FULL_OFFICIAL_PUBLIC_IDENTITY_START_PUBLISHED', json.dumps(dict(head=head, document_sha256=digest, goal_complete=False)), flush=True)
