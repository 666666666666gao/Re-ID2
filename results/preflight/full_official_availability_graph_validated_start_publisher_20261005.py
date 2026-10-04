"""Publish actual graph preflight/start evidence and the single synchronized handoff."""
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
proof = pf / 'full_official_availability_graph_validated_start_publication_20261005.json'
snapshot_file = pf / 'full_official_availability_graph_validated_start_snapshot_20261005.json'
archive = Path(__file__).with_suffix('.tar.gz')
assert not proof.exists() and not snapshot_file.exists() and not archive.exists()
review = json.loads((pf / 'full_official_availability_graph_validated_start_publisher_review_20261005.json').read_text(encoding='utf-8'))
assert review['status'] == 'PASS' and not review['blocking_findings']
assert review['publisher_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
plan = json.loads((pf / 'full_official_availability_graph_validated_plan_20261005.json').read_text(encoding='utf-8'))
launch = json.loads((pf / 'full_official_availability_graph_validated_launch_20261005.json').read_text(encoding='utf-8'))['launch']
observation = json.loads((pf / 'full_official_availability_graph_validated_primary_observer_20261005.jsonl').read_text(encoding='utf-8').splitlines()[-1])
assert observation['preflight_pass'] and not observation['failures']
assert all(row['contract_pass'] and row['smoke_pass'] and row['smoke_updates'] == 3 for row in observation['runs'])
source = launch['output']
snapshot = json.loads(remote_python('2026', f'''import json
from pathlib import Path
root=Path({source!r});result=dict(preflight=json.loads((root/'preflight_result.json').read_text()),contracts={{}},smokes={{}})
for variant in ('demo_shared','axis_shared','frequency_shared','twins_shared'):
 name='MSVR310_graph_'+variant+'_s42'
 result['contracts'][variant]=json.loads((root/'contract'/name/'result.json').read_text())
 result['smokes'][variant]=json.loads((root/'preflight'/name/'smoke.json').read_text())
print(json.dumps(result))
'''))
assert snapshot['preflight']['status'] == 'PASS' and snapshot['preflight']['smoke_actual_optimizer_updates'] == 12
for variant in snapshot['smokes']:
    contract, smoke = snapshot['contracts'][variant], snapshot['smokes'][variant]
    assert contract['status'] == 'PASS_FULL_OFFICIAL_GRAPH_PARENT_AMP_STEP'
    assert contract['parent_step_loss_gradient_equivalence'] and contract['optimizer_update_consistency']
    assert contract['CPU_and_selected_CUDA_RNG_exact']
    assert len(contract['own_adam_update_checks']) == 4
    assert contract['BN_update_counts_exact'] and contract['parent_detail_fields_exact']
    assert contract['positive_graph_optimizer_updates'] == 1
    assert contract['positive_graph_branch']['selected_reference_input_exact']
    assert contract['positive_graph_branch']['reference_projection_no_grad']
    assert contract['positive_graph_branch']['original_full_and_partial_losses_retained']
    assert smoke['status'] == 'SMOKE_PASS' and smoke['steps'] == 3 and smoke['strict_reload_equal']
    assert all(smoke['gradients'].values()) and smoke['training_heldout_identities'] == 0
now = datetime.now().isoformat(timespec='seconds')
cleanup = json.loads((pf / 'full_official_availability_graph_obsolete_weight_cleanup_20261005.json').read_text(encoding='utf-8'))
assert cleanup['status'] == 'FIVE_OBSOLETE_NEGATIVE_CHECKPOINTS_REMOVED_FIFTEEN_CURRENT_BEST_REFERENCES_UNCHANGED'
assert cleanup['removed_files'] == 5 and cleanup['removed_bytes'] == 1995438505
assert cleanup['protected_after'] == cleanup['protected_best_checkpoints']
diagnostic = json.loads((pf / 'full_official_availability_graph_adam_diagnostic_intake_20261005.json').read_text(encoding='utf-8'))
assert not diagnostic['alive']
diag_result = json.loads(diagnostic['files']['run/result.json'])
assert diag_result['status'] == 'FOUR_ACTUAL_ADAM_DIAGNOSTIC_UPDATES_COMPLETE'
assert len(diag_result['updates']) == 4
assert diag_result['updates'][2]['phase'] == 'parent3' and diag_result['updates'][2]['culprit_state_error'] == 2.117827534675598e-6
snapshot.update(intake_at=now, observed_progress=observation,
    obsolete_checkpoint_cleanup=cleanup, adam_diagnostic=diag_result,
    scope='Actual startup contract/smoke intake only. Fresh50 and all49/882 final performance remain pending.',
    actual_new_benchmark_results=0, goal_complete=False)
snapshot_file.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
goal_path = pf / 'research_goal_optimized_20261003.json'
goal = json.loads(goal_path.read_text(encoding='utf-8'))
assert goal['status'] == 'ACTIVE_UNMET'
goal['updated_at'] = now
goal['current_evidence']['full_official_availability_graph_m5'] = dict(
    status='REAL_FOUR_PARENT_AMP_CONTRACTS_AND_TWELVE_SMOKE_UPDATES_PASS_FRESH50_IN_PROGRESS',
    completed_fresh50_runs=sum(row['training_complete'] for row in observation['runs']),
    actual_contract_updates=16, actual_parent_reference_updates=8, actual_disabled_updates=4, actual_positive_graph_updates=4, actual_smoke_updates=12,
    startup_evidence=snapshot_file.relative_to(PROJECT).as_posix(),
    decision=plan['decision'], planned_frozen_cases=196, planned_enhanced_state_cases=882,
    actual_final_frozen_cases=0, actual_final_enhanced_state_cases=0,
    experimental_revision='M5 graph objective, distinct from goal M5 ablation milestone')
goal['next_action'] = dict(milestone='FULL_OFFICIAL_AVAILABILITY_GRAPH_TRIALS',
    status='MONITOR_EXISTING_TWO_GPU_LANES_THEN_FIXEDBEST_FULL49_AND_GT_STATES',
    action='完成同一跨可用集合训练目标下DeMo/普通频域/普通双专家/双轴的4组fresh50；固定最佳权重全49和3增强模型6状态×49，以真实GT复核完整六指标、49来源配对训练覆盖及专家净救回/误伤。未证明公平优势和推理正收益前不扩展三数据集/种子。',
    hardware_dependency='Only2026 physicalGPU2/3,max2NN,no temperature/powerconditions',
    transport_policy='Archive transfer overlaps independent preparation/preflight/training; clear only locally size/SHA-verified raw copies.')
goal_path.write_text(json.dumps(goal, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
document = PROJECT / 'docs/实验交接.md'
text = document.read_text(encoding='utf-8')
marker = '<!-- FULL_OFFICIAL_PROTOCOL_OVERRIDE_START -->'
heading = '### 全量官方跨可用集合训练：真实AMP预检与四组对照启动'
assert text.count(marker) == 1 and heading not in text
addition = '\n\n' + heading + '（' + now + '）\n\n'
addition += 'M4三组已实际完成50轮与882状态GT复核并发布：双轴完整mAP49.448975/R1 65.482234，相对原DeMo +2.284286/+1.522843，未满足双指标+2；普通双专家49.744114/66.666667更好，打开双轴专家相对自身00仍为负收益。根据完整负结果，仅检验部分模态之间身份可比较性，不原样扩大三个数据集。实验修订M5不是goal内的M5消融里程碑。\n\n'
addition += '第一次真实预检已有DeMo合同与smoke、双轴合同通过的保存证据，普通频域零权重对照出现1/32768个状态元素差异约2.1178e-6，超过原容差；控制器未进入fresh50，错误日志与原源码快照保留。独立救援审查后，仅在权重为零时不将新损失连接到backward，0.1正常实验路径不变；复验增加同初始化/同批次重复父步骤，仍使用原rtol1e-5/atol1e-6并报告张量名，不直接放宽门槛。该差异的因果根源没有凭单次失败定论。\n\n'
addition += '第二次复验已在同一router.to_modality_query.1.weight元素出现相同2.1178e-6差异，重复父步骤通过但条件取消损失反传尚不足；该次DeMo/双轴合同与各3次smoke已实际通过，普通频域失败导致0轮fresh50。按审查建议重写v2验证：权重0在任何新计算前委派父步骤，只声明禁用模式等价；另外独立真实0.1的R→N更新验证选定真实来源、停梯度、正负身份、原损失/新损失算式、4次主干和3次投影调用、RNG与BN。原数值容差不变，命名梯度/状态摘要先保存后断言；不声称已证实微差根因。\n\n'
addition += '本轮单因素：在原完整损失、0.25部分CE/0.5部分→完整图库triplet及M4专家损失之外，增加0.1公共身份跨集合监督，完整和原部分查询各占0.05，温度0.07。参考B按R/N/T/RN/RT/NT/RNT确定循环，仅复用同一完整训练前向缓存的对应真实CLIP原始特征，经已有shared_projection并停止梯度；同训练身份的不同观测为正样本，其他身份为负样本。额外主干前向0、新增模型参数0、推理结构0变化；普通对照接受相同覆盖/预算。B覆盖在最终实际batch_orders逐项复核，不把预估49覆盖冒充已发生。\n\n'
addition += '完整官方MSVR310的1032训练/591查询/1055图库、0人工身份留出、公开CLIP fresh50、seed42、原生B64/P16/K4；GPU2串行DeMo→双轴，GPU3串行普通频域→普通双专家，最多2个NN，无功率/温度限制。50轮按官方完整benchmark mAP最早并列best选点，固定同一权重用于全49；不是未参与选择的独立最终测试。\n\n'
addition += '| 模型 | GPU | 实际父/禁用AMP对照及正权重支路 | smoke真实更新 | AMP跳过 | 严格重载 | 最新已完成训练轮 |\n|---|---:|---|---:|---:|---|---:|\n'
for row in observation['runs']:
    smoke = snapshot['smokes'][row['variant']]
    contract = snapshot['contracts'][row['variant']]
    addition += '| ' + row['variant'] + ' | ' + str(row['gpu']) + ' | PASS；最大状态误差' + str(contract['maximum_state_absolute_error']) + ' | 3 | ' + str(smoke['amp_skipped_steps']) + ' | exact | ' + str(row['training_epoch']) + '/50 |\n'
addition += '\n以上是实际启动检查：四个合同16次真实更新（8次父步骤、4次禁用步骤、4次正权重图步骤）、12次smoke更新；禁用模式的CPU及选定CUDA RNG、BN次数和原详情字段一致，数值梯度与各自Adam更新校验通过；跨次更新后参数差异仅记录，不宣称满足旧参数allclose；正权重图支路也独立验证了上述计算组成，不以委派对照代替。没有保存预检权重；每组只保存mAP-best.pth。完整性能、196冻结条件、3增强模型882状态条件、贡献86877重复样本行尚待完成，不能当作涨点。缺失评测使用原GT场景排除；逐条件六指标、公共/私有块、贡献相关/MAE/零预测参照和净救回/误伤均继续保留。\n\n'
addition += '传输期间实际已完成新训练代码、真实预检/评测/控制器实现与审查。M4距离294文件/2,434,750,270字节全部已在本地D账户归档目录完成大小/SHA核对后清除远端副本；累计1274份/15,104,039,284字节。先前必要best和对照保留，历史失败不删结果。审查识别新预检误差统计对bool频带mask相减会报错，已仅让误差统计遍历浮点state；所有state仍做数值比较，BN计数仍精确比较；未修改模型/训练因素。\n\n'
addition += '代码/计划/审查及实际启动证据：results/preflight/full_official_availability_graph_*20261005*；远端runs/full_official_availability_graph_m5_validated_20261005。独立源码审查为同模型家族、暂定，实际后端未证实；不宣称跨家族审查。单一交接文件同步project/Desktop/2026/2027同字节，2025已有I/O故障仍待处理，未声称5份一致。Goal ACTIVE_UNMET，原三数据集各mAP/R1+2且缺失公平对照和真实协作、多种子目标不变。\n'
addition += '\n本轮首次清洁版本部署在只读磁盘预检查中退出，未复制源码、未启动预检或训练；当时可用4,614,545,408字节，低于4500MiB门槛。之后清理已完成50轮且被全官方协议替代的V6/V7/V8/V9/V10五个旧开发负结果best权重，共1,995,438,505字节；保留全部结果记录。清理前后15个当前全官方best及必要对照的大小/SHA一致，原始距离文件未动。清理后实际可用空间' + str(cleanup['free_bytes_after']) + '字节；改变磁盘条件后才重新部署，门槛未降低。清理清单和两次预检失败均保留。\n'
addition += '\n第三次v2预检再次失败，但一次真实诊断发现：第三次直接原parent已复现同一2.1178275e-6参数差，随后disabled委派与其故障元素梯度、参数和moments相等。该元素权重衰减后的有效梯度约5.74944e-9/5.60074e-9，epsilon为1e-8；约1.48702e-10的梯度差经Adam得到约2.11825e-6的预测参数差，重建残差小于5e-10。说明旧更新后参数跨次allclose门也会拒绝原训练，并非新增图目标的特异证据；未定位具体CUDA梯度变化来源。v3明确改用原损失/梯度容差、精确RNG/采样/全部buffer和各次自身Adam更新重建校验；跨次参数误差完整保留为诊断，不扩大原容差。正式训练代码/模型/AMP/优化器不变。第三次失败、原v2源码、实际诊断及判据变更审查全部保留；当前新的实际正权重预检通过后才启动50轮。\n'
document.write_bytes(text.replace(marker, marker + addition, 1).encode('utf-8'))
DESKTOP.write_bytes(document.read_bytes())
files = list(plan['source_sha256']) + [
    'verify_full_official_availability_graph_v2.py',
    'diagnose_graph_disabled_adam.py',
    'results/preflight/full_official_availability_graph_clean_launch_20261005.json',
    'results/preflight/full_official_availability_graph_clean_plan_20261005.json',
    'results/preflight/full_official_availability_graph_clean_execution_review_20261005.json',
    'results/preflight/full_official_availability_graph_clean_preflight_failure_20261005.json',
    'results/preflight/full_official_availability_graph_failed_clean_sources_20261005.json',
    'results/preflight/full_official_availability_graph_adam_diagnostic_deployer_20261005.py',
    'results/preflight/full_official_availability_graph_adam_diagnostic_review_20261005.json',
    'results/preflight/full_official_availability_graph_adam_diagnostic_launch_20261005.json',
    'results/preflight/full_official_availability_graph_adam_diagnostic_intake_20261005.json',
    'results/preflight/full_official_availability_graph_adam_diagnostic_analysis_20261005.json',
    'results/preflight/full_official_availability_graph_rescue3_judgment_20261005.json',
    'results/preflight/full_official_availability_graph_clean_deployment_precheck_failure_20261005.json',
    'results/preflight/full_official_availability_graph_clean_space_inventory_20261005.json',
    'results/preflight/full_official_availability_graph_obsolete_negative_weights_intake_20261005.json',
    'results/preflight/full_official_availability_graph_obsolete_weight_cleanup_20261005.py',
    'results/preflight/full_official_availability_graph_obsolete_weight_cleanup_20261005.json',
    'results/preflight/full_official_availability_graph_cleanup_receipt_correction_20261005.json',
    'results/preflight/full_official_availability_graph_rescue_launch_20261005.json',
    'results/preflight/full_official_availability_graph_rescue_plan_20261005.json',
    'results/preflight/full_official_availability_graph_rescue_execution_review_20261005.json',
    'results/preflight/full_official_availability_graph_rescue_preflight_failure_20261005.json',
    'results/preflight/full_official_availability_graph_failed_rescue_sources_20261005.json',
    'results/preflight/full_official_availability_graph_rescue2_judgment_20261005.json',
    'results/preflight/full_official_availability_graph_launch_20261005.json',
    'results/preflight/full_official_availability_graph_plan_20261005.json',
    'results/preflight/full_official_availability_graph_execution_review_20261005.json',
    'results/preflight/full_official_availability_graph_first_preflight_failure_20261005.json',
    'results/preflight/full_official_availability_graph_failed_first_sources_20261005.json',
    'results/preflight/full_official_availability_graph_amp_rescue_review_20261005.json',
    'results/preflight/full_official_availability_graph_preparation_20261005.json',
    'results/preflight/full_official_availability_graph_validated_plan_20261005.json',
    'results/preflight/full_official_availability_graph_training_review_20261005.json',
    'results/preflight/full_official_availability_graph_validated_execution_review_20261005.json',
    'results/preflight/full_official_availability_graph_validated_deployer_20261005.py',
    'results/preflight/full_official_availability_graph_validated_observer_20261005.py',
    'results/preflight/full_official_availability_graph_validated_start_publisher_20261005.py',
    'results/preflight/full_official_availability_graph_validated_start_publisher_review_20261005.json',
    'results/preflight/full_official_availability_graph_validated_launch_20261005.json',
    'results/preflight/full_official_availability_graph_validated_start_snapshot_20261005.json',
    'results/preflight/research_goal_optimized_20261003.json']
with tarfile.open(archive, 'w:gz') as packed:
    for file in files:
        packed.add(PROJECT / file, arcname=file)
digest = hashlib.sha256(document.read_bytes()).hexdigest()
expected = {file:hashlib.sha256((PROJECT / file).read_bytes()).hexdigest() for file in files}
expected['docs/实验交接.md'] = digest
for host in ('2026', '2027'):
    remote_root, _ = HOSTS[host]
    target = remote_root + '/full_official_availability_graph_validated_start_text_20261005.tar.gz'
    command(['scp', *OPTIONS, str(archive), host + ':' + target])
    remote_python(host, 'import tarfile;from pathlib import Path;p=Path(' + repr(target) + ');tarfile.open(p).extractall(' + repr(remote_root) + ',filter="data");p.unlink()')
    command(['scp', *OPTIONS, str(document), host + ':' + remote_root + '/docs/实验交接.md'])
    actual = json.loads(remote_python(host, 'import hashlib,json;from pathlib import Path;root=Path(' + repr(remote_root) + ');files=' + repr(tuple(files) + ('docs/实验交接.md',)) + ';print(json.dumps({file:hashlib.sha256((root/file).read_bytes()).hexdigest() for file in files}))'))
    assert actual == expected
assert DESKTOP.read_bytes() == document.read_bytes()
proof.write_text(json.dumps(dict(status='REAL_FOUR_GRAPH_CONTRACTS_TWELVE_SMOKE_UPDATES_AND_TWO_LANES_PUBLISHED',
    published_at=now, document_sha256=digest, verified_locations=['project','Desktop','2026','2027'],
    pending_mirror='2025_IO', actual_contract_updates=16, actual_parent_reference_updates=8, actual_disabled_updates=4, actual_positive_graph_updates=4, actual_smoke_updates=12,
    actual_final_results=0, completed_fresh50_runs=sum(row['training_complete'] for row in observation['runs']), goal_complete=False),indent=2) + '\n',encoding='utf-8')
for host in ('2026','2027'):
    command(['scp',*OPTIONS,str(proof),host+':'+HOSTS[host][0]+'/results/preflight/'+proof.name])
files += ['docs/实验交接.md', proof.relative_to(PROJECT).as_posix()]
assert command(['git','diff','--cached','--name-only'],cwd=PROJECT).strip() == ''
pathspec = Path(__file__).with_suffix('.pathspec')
pathspec.write_bytes(b'\0'.join(file.encode('utf-8') for file in files) + b'\0')
command(['git','add','--pathspec-from-file='+str(pathspec),'--pathspec-file-nul'],cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=PROJECT)
command(['git','commit','-m','Launch full-data cross-availability graph objective with four matched controls'],cwd=PROJECT)
command(['git','push','origin','main'],cwd=PROJECT)
head = command(['git','rev-parse','HEAD'],cwd=PROJECT).strip()
assert command(['git','ls-remote','origin','refs/heads/main'],cwd=PROJECT).split()[0] == head
print('FULL_OFFICIAL_GRAPH_START_PUBLISHED',json.dumps(dict(head=head,document_sha256=digest,actual_smoke_updates=12,goal_complete=False)),flush=True)
