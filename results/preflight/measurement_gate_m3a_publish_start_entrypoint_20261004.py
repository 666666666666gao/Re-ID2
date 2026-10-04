from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python, sync_handoff

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
p = PROJECT / 'results/preflight'
plan = json.loads((p / 'measurement_gate_m3a_plan.json').read_text(encoding='utf-8'))
review = json.loads((p / 'measurement_gate_m3a_review.json').read_text(encoding='utf-8'))
launch = json.loads((p / 'measurement_gate_m3a_2026_launch.json').read_text(encoding='utf-8'))
snapshot = json.loads((p / 'measurement_gate_m3a_latest_snapshot.json').read_text(encoding='utf-8'))
assert review['verdict'] == 'PASS_SOURCE_REVIEW' and not review['blockers']
assert review['checked_source_sha256'] == plan['sources']
assert snapshot['pid'] == launch['pid'] and snapshot['live']
assert snapshot['tensor']['status'] == 'PASS_MEASUREMENT_GATE_GRADIENT_CONTRACT'
assert len(snapshot['smokes']) == 3
assert all(v['status'] == 'SMOKE_PASS' and v['steps'] == 3 and v['strict_reload_equal'] for v in snapshot['smokes'].values())
assert snapshot['training']
first = p / 'measurement_gate_m3a_start_snapshot.json'
assert not first.exists()
first.write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')
for name in ('deploy', 'observe', 'complete', 'publish_start'):
    target = p / ('measurement_gate_m3a_' + name + '_entrypoint_20261004.py')
    assert not target.exists()
    target.write_bytes(Path('C:/Users/gb/.codex_tmp/demo_measurement_gate_' + name + '_20261004.py').read_bytes())
now = datetime.now().isoformat(timespec='seconds')
goal = p / 'research_goal_optimized_20261003.json'
g = json.loads(goal.read_text(encoding='utf-8'))
g['status'] = 'ACTIVE_UNMET'
g['updated_at'] = now
g['revision'] = '20261004_M3a_actual_gradient_contract_three_smokes_pass_fresh50_active'
g['current_evidence']['M3a'] = dict(
    status='ACTUAL_GRADIENT_CONTRACT_THREE_REAL_UPDATE_SMOKES_PASS_FRESH50_ACTIVE',
    controller_pid=launch['pid'], observed_at=snapshot['observed_at'], actual_smokes=3,
    actual_smoke_updates=sum(v['steps'] for v in snapshot['smokes'].values()),
    training=snapshot['training'], plan='results/preflight/measurement_gate_m3a_plan.json',
    planned_fresh50=3, planned_CPU_cases=3087, planned_repeated_condition_query_rows=648270,
    official_test_uses=0, partial_contribution_regression_targets_present=False)
g['next_action'] = dict(
    milestone='M3a_measurement_only_gate_gradient_single_factor',
    status='EXISTING_REVIEWED_CONTROLLER_RUNNING_NO_RESTART',
    action='完成当前三组fresh50、完整49条件、六状态、八阶段、七种交叉坐标检索及独立GT审计。只切断融合门控的检索反向梯度，收益估计器继续接受原贡献回归；不新增控制头、损失或缺失配对。与原M2b、同轮普通专家和同增强DeMo完整比较，依据真实推理收益决定下一步。',
    hardware_dependency='Only2026GPU2/3,max2NN,no temperature/powerconditions')
goal.write_text(json.dumps(g, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
doc = PROJECT / 'docs/实验交接.md'
s = doc.read_text(encoding='utf-8')
marker = '<!-- CURRENT_DEMO_STATUS_START -->\n'
assert s.count(marker) == 1 and '<!-- CURRENT_M3A_STATUS_START -->' not in s
text = f'''<!-- CURRENT_M3A_STATUS_START -->
## 当前执行：M3a 门控梯度单因素试验已通过真实预检并开始训练（{now}）

前轮 M2b 已完成三组 fresh50 和独立安装 GT 审计 PASS：3087 项、648270 条重复条件—查询记录。MSVR310 双轴完整输入为 mAP 47.630672、mINP 32.836992、Rank-1/5/10/20 59.523810/72.380952/77.142857/81.428571；相对原同协议 DeMo 仅 +0.022880 mAP、+0.476190 Rank-1。全49等权均值仍低于同轮普通频域 5.021974 mAP、普通双专家 5.903164 mAP。完整 PF 独立救回基础首位错误的11条查询，没有一条被联合模型兑现；共同坐标兼容性改善不能当作最终方法成功。完整负结果、六指标、294项公平比较和逐查询机制证据保留于下方 M2b 段及 results/identity_alignment_m2b_20261004；已发布 6a42418e0be6c443359bec32d73454e6256fb535。

这一轮只改变融合门控的反向路径：同一个收益估计器继续训练原贡献回归，供融合使用的门控执行 detach，完整和全部六种部分可用集合的检索损失均不再更新估计器。没有新建独立控制头，也不把梯度切断解释为已经校准或必然改善。原 M2b 关系保持0.1、身份对齐0.1、温度0.07、FFT、专家访问范围、一次交互、联合路由、尺度、损失、缺失集合采样及训练预算不变；部分训练原本没有贡献回归目标，这个限制保留并明确记录。

Fresh same-family/provisional 源审查 PASS、0阻塞；原76源与新7源共83项 SHA 一致。真实 CUDA 检查已完成：三模型初始化 state_dict 和参数相同，七可用集合×四关闭状态、完整 AMP 前向及全部损失数值精确一致；完整和六种部分检索梯度对估计器参数全部为 None，原加权贡献回归对每个估计器参数保持有限非零梯度。三组 smoke 各3次真实更新、全有效梯度、严格重载一致全部通过，才发出 fresh50。源审查不是独立实验验收；上述张量与梯度是实际执行结果。

实际控制器 {launch['pid']}，快照 {snapshot['observed_at']}：{json.dumps(snapshot['training'], ensure_ascii=False)}。只使用2026物理GPU2/3：GPU2双轴→普通双专家串行，GPU3普通频域，最多两个我方神经任务。每模型公共CLIP、seed42、B64/K4、50轮、nativeAMP512，按开发mAP最高且最早并列的checkpoint选择。三模型同有效参数100263558、可训练100251270、5632D。官方测试不用于训练或版本选择。

唯一240秒观察器与完成收集器已启动；预计本轮训练及机制评测约40–45分钟，以真实终态为准。计划完整147个模态组合、882个六状态、1176个八阶段、1029个交叉坐标案例，合计3087项独立GT审计、648270条重复条件—查询记录。全部报告mAP/mINP/Rank-1/5/10/20、CMC1..50、逐查询AP/INP/首末匹配、身份/相机/场景分组和误伤/救回；重复查询不能当作独立样本。当前只确认预检、短训练及启动，完整指标和最终审计尚未完成。

验收仍为同一方法在三个数据集正常mAP和Rank-1各超过同协议完整DeMo至少2个百分点，覆盖缺失模态、公平普通专家、实际协同收益、多种子和独立确认。Goal ACTIVE/UNMET。当前只是单MSVR开发划分、seed42机制实验，不能拼接历史最好版本。只保留每实验最佳权重和必要先前参照；smoke/initial/last权重不落盘。用户已取消温度和功率限制，不采集、不控制温度/功率，不询问管理员。2025/2027只作源码和文本镜像；交接仍只维护这一份，repo/Desktop/25/26/27逐字节相同，代码和文本同步 GitHub Re-ID2。

<!-- CURRENT_M3A_STATUS_END -->

'''
doc.write_bytes(s.replace(marker, marker + text, 1).encode('utf-8'))
files = [PROJECT / n for n in plan['sources']] + [goal]
files.extend(f for f in p.glob('measurement_gate_m3a_*') if f.name not in (
    'measurement_gate_m3a_latest_snapshot.json', 'measurement_gate_m3a_observer_started.json',
    'measurement_gate_m3a_completion_wait_started.json'))
assert len(files) == len(set(files))
manifest = {f.relative_to(PROJECT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
for host, (root, _) in HOSTS.items():
    selected = [f for f in files if f.parent == p]
    command(['scp', *OPTIONS, *[str(f) for f in selected], host + ':' + root + '/results/preflight/'])
    code = 'import hashlib,json;from pathlib import Path;r=Path(' + repr(root) + ');print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ' + repr(list(manifest)) + '}))'
    assert json.loads(remote_python(host, code)) == manifest
digest = sync_handoff()
proof = p / 'measurement_gate_m3a_start_publication.json'
assert not proof.exists()
proof.write_text(json.dumps(dict(status='EXACT_THREE_TEXT_MIRRORS_FIVE_HANDOFFS', at=now,
    files_sha256=manifest, doc_sha256=digest, controller_pid=launch['pid'], actual_smokes=3,
    goal_complete=False, temperature_power_control=False), indent=2) + '\n', encoding='utf-8')
for host, (root, _) in HOSTS.items():
    command(['scp', *OPTIONS, str(proof), host + ':' + root + '/results/preflight/'])
command(['git', 'add', '--', *manifest, proof.relative_to(PROJECT).as_posix(), 'docs/实验交接.md'], cwd=PROJECT)
command(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], cwd=PROJECT)
command(['git', 'commit', '-m', 'Launch measurement-only gate gradient experiment after real CUDA contract'], cwd=PROJECT)
command(['git', 'push', 'origin', 'HEAD:main'], cwd=PROJECT)
print('M3A_START_PUBLISHED', json.dumps(dict(head=command(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).strip(),
    doc_sha256=digest, files=len(manifest), goal_complete=False)), flush=True)
