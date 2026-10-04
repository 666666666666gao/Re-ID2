from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python, sync_handoff

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
p = PROJECT / 'results/preflight'
plan = json.loads((p / 'cross_identity_coordinate_plan.json').read_text(encoding='utf-8'))
review = json.loads((p / 'cross_identity_coordinate_review.json').read_text(encoding='utf-8'))
snapshot = json.loads((p / 'cross_identity_coordinate_first_snapshot.json').read_text(encoding='utf-8'))
assert review['verdict'] == 'PASS_SOURCE_REVIEW' and not review['blocking_issues']
assert snapshot['live'] and len(snapshot['smokes']) == 3
assert all(v['status'] == 'PASS' and v['normal_feature_max_error'] == 0 for v in snapshot['smokes'].values())
for stem in ('prepare','deploy','observe','collect','publish_start'):
    source = Path('C:/Users/gb/.codex_tmp/demo_cross_coordinate_' + stem + '_20261004.py')
    target = p / ('cross_identity_coordinate_' + stem + '_entrypoint_20261004.py')
    assert not target.exists()
    target.write_bytes(source.read_bytes())
now = datetime.now().isoformat(timespec='seconds')
goal = p / 'research_goal_optimized_20261003.json'
g = json.loads(goal.read_text(encoding='utf-8'))
g['status'] = 'ACTIVE_UNMET'; g['updated_at'] = now
g['revision'] = '20261004_M2_cross_coordinate_smokes_pass_frozen_diagnostic_active'
g['current_evidence']['cross_identity_coordinates'] = dict(status='ACTUAL_THREE_SMOKES_PASS_FULL_ACTIVE',
    controller_pid=snapshot['pid'], observed_at=snapshot['observed_at'], actual_smokes=3,
    plan='results/preflight/cross_identity_coordinate_plan.json',planned_cases=1029,
    optimizer_updates=0,new_weights=0,official_test_uses=0)
g['next_action'] = dict(milestone='M2_common_identity_coordinate_compatibility', status='EXISTING_FROZEN_CONTROLLER_RUNNING_NO_RESTART',
    action='完成三M2 best的实际F_pre/PF与公共身份坐标四个交叉方向、三个原样同坐标控制，全部49条件及1029组独立CPU核算。根据坐标兼容与真实融合收益决定下一单因素，保持现有跨集合训练/门控不变。',
    hardware_dependency='Only2026 GPU2/3,max2 NN; no temperature/power conditions.')
goal.write_text(json.dumps(g,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
doc = PROJECT / 'docs/实验交接.md'
s = doc.read_text(encoding='utf-8'); marker = '<!-- CURRENT_DEMO_STATUS_START -->\n'
assert s.count(marker) == 1 and '<!-- CURRENT_CROSS_COORDINATE_STATUS_START -->' not in s
text = f'''<!-- CURRENT_CROSS_COORDINATE_STATUS_START -->
## 最新执行：冻结M2权重的公共—频域坐标兼容诊断（{now}）

上一轮三组fresh50和2058组/432180重复查询记录已独立核算并发布445c82e566e77a84545f34f28209e6779084c8a5。双轴PF独立mAP比M1改善3.869874点，但完整11−00仅+0.000938mAP/R1不变；49条件比普通双专家低2.666757mAP/4.023324R1，原三数据集双+2验收仍未达标，不能只展示赢普通频域的部分。

三新源码通过fresh same-family/provisional审核；旧66源精确不变，实际2026输入已COMPLETE、独立审计PASS、仅三best。启动前一次因审核JSON尚未写完而在本地read门槛停止，未执行SCP/SSH或NN；审核文件落盘并核对后仅一次实际部署。控制器{snapshot['pid']}现已启动，观察{snapshot['observed_at']}：三组64记录×七集合smoke全PASS，正常部署5632D特征最大差异0、原始fuse重构逐元素相同、state版本不变、optimizer更新及新权重均0；已有full在运行，不能将计划1029组写成已完成。

使用同一M2开发mAP最佳权重，不改模型、门控或幅值：三个独立归一化512D实际表示base_common、F_pre、F_post，四个交叉方向F_pre→common/common→F_pre/F_post→common/common→F_post，加三个同坐标控制。三模型各7×7×7=343组，总1029组/216090重复条件—查询记录。全部MSVR安装GT按同身份同场景排除；三个同坐标控制要求六指标及逐查询与既有M2出口诊断完全一致。

GPU2顺序双轴→普通双专家，GPU3普通频域，每卡最多一个NN。唯一240秒观察器和完成收集器沿用实际控制器，不重启；终态才独立从缓存512D特征验证交叉矩阵布线（数值容差2e−6），对精确原始距离重新排序并核算六指标、CMC1..50、逐查询AP/INP/首末匹配及身份/相机/场景分组。原始NPZ留2026；不新增或删除权重，不使用官方测试调参。

交叉坐标检索只是公共身份可比较性的诊断，不是部署新方法、严格解耦或因果效应。若兼容差，下一轮单因素检查身份朝向；若兼容已好但协作弱，再单独检查估计/控制或幅值作用。关系保持对旋转不敏感，不能预先把PF独立涨点当成融合收益；此轮不同时增加跨集合覆盖或改门控。用户已取消功率温度约束，2025/2027仅文本镜像。Goal仍ACTIVE/UNMET，人类交接文件仍唯一且五份按字节一致。

<!-- CURRENT_CROSS_COORDINATE_STATUS_END -->

'''
doc.write_bytes(s.replace(marker,marker+text,1).encode('utf-8'))
files = [PROJECT/name for name in plan['sources']] + [goal]
files.extend(f for f in p.glob('cross_identity_coordinate_*') if f.name not in
    ('cross_identity_coordinate_latest_snapshot.json','cross_identity_coordinate_observer_started.json','cross_identity_coordinate_completion_wait_started.json'))
assert len(files) == len(set(files))
manifest = {f.relative_to(PROJECT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
for host,(root,_) in HOSTS.items():
    selected = [f for f in files if f.parent == p]
    command(['scp',*OPTIONS,*[str(f) for f in selected],host+':'+root+'/results/preflight/'])
    code = 'import hashlib,json;from pathlib import Path;r=Path('+repr(root)+');print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in '+repr(list(manifest))+'}))'
    assert json.loads(remote_python(host,code)) == manifest
digest = sync_handoff()
proof = p/'cross_identity_coordinate_start_publication.json'; assert not proof.exists()
proof.write_text(json.dumps(dict(status='EXACT_THREE_TEXT_MIRRORS_FIVE_HANDOFFS',at=now,files_sha256=manifest,
    doc_sha256=digest,controller_pid=snapshot['pid'],optimizer_updates=0,new_weights=0,goal_complete=False,
    temperature_power_control=False),indent=2)+'\n',encoding='utf-8')
for host,(root,_) in HOSTS.items():
    command(['scp',*OPTIONS,str(proof),host+':'+root+'/results/preflight/'])
command(['git','add','--',*manifest,proof.relative_to(PROJECT).as_posix(),'docs/实验交接.md'],cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=PROJECT)
command(['git','commit','-m','Launch frozen frequency-public identity coordinate compatibility diagnosis'],cwd=PROJECT)
command(['git','push','origin','HEAD:main'],cwd=PROJECT)
print('CROSS_COORDINATE_START_PUBLISHED',json.dumps(dict(head=command(['git','rev-parse','HEAD'],cwd=PROJECT).strip(),doc_sha256=digest,files=len(manifest))),flush=True)
