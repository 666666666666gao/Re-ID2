from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python, sync_handoff

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
p = PROJECT / 'results/preflight'
evidence = p / 'frequency_relation_m2_accepted_snapshot.json'
assert not evidence.exists()
snapshot = json.loads((p / 'frequency_relation_m2_latest_snapshot.json').read_text(encoding='utf-8'))
launch = json.loads((p / 'frequency_relation_m2_2026_launch.json').read_text(encoding='utf-8'))
assert snapshot['pid'] == launch['pid'] and snapshot['live']
assert snapshot['tensor']['status'] == 'PASS_FREQUENCY_RELATION_CONTRACT'
assert len(snapshot['smokes']) == 3
smokes = {}
for name, value in snapshot['smokes'].items():
    assert value['status'] == 'SMOKE_PASS' and value['steps'] == 3
    assert value['attempts'] == 3 and value['amp_skipped_steps'] == 0
    assert value['strict_reload_equal'] and all(value['gradients'].values())
    smokes[name] = {k: value[k] for k in ('status', 'steps', 'attempts', 'amp_skipped_steps', 'strict_reload_equal', 'parameters', 'trainable_parameters', 'descriptor_dim')}
now = datetime.now().isoformat(timespec='seconds')
accepted = dict(status='REAL_CUDA_CONTRACT_AND_THREE_SMOKES_PASS_TRAINING_ACTIVE', at=now,
    observed_at=snapshot['observed_at'], controller_pid=launch['pid'], tensor=snapshot['tensor'],
    smokes=smokes, training=snapshot['training'], active=snapshot['active'],
    temperature_power_control=False, goal_complete=False)
evidence.write_text(json.dumps(accepted, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
goal = p / 'research_goal_optimized_20261003.json'
g = json.loads(goal.read_text(encoding='utf-8'))
g['status'] = 'ACTIVE_UNMET'
g['updated_at'] = now
g['revision'] = '20261004_M2_actual_preflight_pass_training_and_utility_active'
next(m for m in g['milestones'] if m['id'] == 'M2')['status'] = 'REAL_CUDA_AND_THREE_SMOKES_PASS_FRESH50_AND_DIAGNOSTICS_ACTIVE'
g['current_evidence']['M2'] = dict(status='REAL_PREFLIGHT_PASS_EXPERIMENT_ACTIVE',
    plan='results/preflight/frequency_relation_m2_plan.json',
    accepted='results/preflight/frequency_relation_m2_accepted_snapshot.json',
    controller_pid=launch['pid'], observed_at=snapshot['observed_at'], actual_smokes=3,
    actual_training=snapshot['training'], planned_fresh50=3, planned_CPU_cases=2058)
g['next_action'] = dict(milestone='M2', status='EXISTING_CONTROLLER_RUNNING_NO_DUPLICATE_LAUNCH',
    action='完成当前三组fresh50、全部49条件/六状态/八阶段，并独立CPU核算2058组、432180条重复条件—查询记录；核对M1采样、真实更新及F投影前后判别力和11−00误伤/救回。根据终态证据选择下一单因素，未形成清晰候选前不扩展三数据集或多种子。',
    excluded_this_factor=['改变跨集合训练分布', '拆分贡献估计与控制', '改变FFT或专家容量', '改动门控温度或放大残差'],
    hardware_dependency='用户已取消温度和功率限制；仅2026物理GPU2/3，每卡最多一个NN任务，共最多两个。')
g['user_resumption']['platform_status_last_read'] = 'active'
g['user_resumption']['platform_status_note'] = 'Platform goal actually read ACTIVE on 2026-10-04; user removed the historical hardware blocker. Original objective remains active and unmet.'
goal.write_text(json.dumps(g, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
doc = PROJECT / 'docs/实验交接.md'
s = doc.read_text(encoding='utf-8')
marker = '<!-- CURRENT_DEMO_STATUS_START -->\n'
begin = s.index(marker) + len(marker)
end = s.index('## 最新终态：训练后八阶段身份证据诊断全量通过', begin)
text = f'''## 最新执行：M2真实预检通过，三组训练及全量诊断进行中（{now}）

用户已明确取消温度和功率限制。仅使用2026物理GPU2/3，每卡最多一个神经网络任务；2025/2027只同步文本。已有控制器{launch['pid']}继续运行，不重复启动或因观察超时重训。

真实CUDA合同已PASS：三个模型初始state及七种非空来源的四状态推理完全一致，零权重AMP训练输出与旧M1完全一致；实际PF关系损失有非零有限梯度，目标停止梯度。三组smoke均实际更新3次、无AMP跳过、所有活跃参数梯度有限且非零、内存strict reload一致。每个增强模型100263558参数、100251270有效可训练参数、5632D；未增加任何身份头或参数。

截至{snapshot['observed_at']}的真实训练状态为{json.dumps(snapshot['training'], ensure_ascii=False)}。其中50轮已完成不代表全部独立核算已结束；这些是开发选择记录，完整六指标、49条件和机制结论待终态复算后统一公布。双轴当前mAP48.887510、Rank-1 60.952381，相对旧同协议DeMo为+1.279717/+1.904762点，两项仍未同时达到+2。不得将中间普通双专家值当成终态。

本轮唯一新因素：完整输入实际PF输出相对同输入、已路由F_pre.detach的归一化非对角距离SmoothL1，固定权重0.1，仅完整训练前向加一次。原original_mean出口、专家、FFT、联合路由、门控、身份/贡献损失、部分查询→完整图库训练、采样及维度保持一致，三模型使用相同新目标和fresh公开CLIP/seed42/B64/K4/50轮。GPU2依次双轴→普通双专家，GPU3普通频域。旧M1无关系目标对照保留并逐采样核对。

既有240秒观察器与唯一完成收集器继续等待当前控制器。计划147个完整模型条件、882组六状态和1176组八阶段；独立CPU核算2058组、432180条重复条件—查询记录，报告六指标、CMC1..50、逐查询和身份/相机/场景分组。关系保持并不保证公共坐标朝向或门控校准；必须同时检查F_pre绝对判别力、PF前后变化和11−00净收益，不能只看关系loss下降。

上一轮2352组/493920记录冻结诊断已独立通过，结果在results/trained_outlet_utility_20261004。Goal仍ACTIVE/UNMET：三个数据集同一方法双+2、公平普通专家优势、机制证据、三个配对种子及独立确认尚未完成。每实验仅保留开发mAP最佳权重，必要旧对照继续保留；本次未删除权重。此文件是唯一人类交接文档，repo/Desktop/25/26/27按字节一致。

<!-- CURRENT_M2_STATUS_END -->

'''
doc.write_bytes((s[:begin] + text + s[end:]).encode('utf-8'))
entry = p / 'frequency_relation_m2_accepted_entrypoint_20261004.py'
assert not entry.exists()
entry.write_bytes(Path(__file__).read_bytes())
files = [evidence, goal, entry]
manifest = {f.relative_to(PROJECT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
for host, (root, _) in HOSTS.items():
    command(['scp', *OPTIONS, *[str(f) for f in files], host + ':' + root + '/results/preflight/'])
    code = 'import hashlib,json;from pathlib import Path;r=Path(' + repr(root) + ');print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ' + repr(list(manifest)) + '}))'
    assert json.loads(remote_python(host, code)) == manifest
digest = sync_handoff()
proof = p / 'frequency_relation_m2_accepted_publication.json'
assert not proof.exists()
proof.write_text(json.dumps(dict(status='EXACT_THREE_TEXT_MIRRORS_FIVE_HANDOFFS', at=now,
    files_sha256=manifest, doc_sha256=digest, controller_pid=launch['pid'], temperature_power_control=False,
    goal_complete=False), indent=2) + '\n', encoding='utf-8')
for host, (root, _) in HOSTS.items():
    command(['scp', *OPTIONS, str(proof), host + ':' + root + '/results/preflight/'])
command(['git', 'add', '--', *manifest, proof.relative_to(PROJECT).as_posix(), 'docs/实验交接.md'], cwd=PROJECT)
command(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], cwd=PROJECT)
command(['git', 'commit', '-m', 'Record accepted frequency relation preflight and current goal milestone'], cwd=PROJECT)
command(['git', 'push', 'origin', 'HEAD:main'], cwd=PROJECT)
print('M2_ACCEPTED_PUBLISHED', json.dumps(dict(head=command(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).strip(), doc_sha256=digest)), flush=True)
