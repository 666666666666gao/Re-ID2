from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python, sync_handoff

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
p = PROJECT / 'results/preflight'
target = p / 'm1_outlet_scope_recheck_20261004.json'
assert not target.exists()
original = PROJECT / 'results/common_outlet_m1_v3_complete'
analysis = json.loads((original / 'analysis.json').read_text(encoding='utf-8'))
audit = json.loads((original / 'independent_cpu_audit.json').read_text(encoding='utf-8'))
assert audit['status'] == 'PASS' and audit['cases'] == 2940 and audit['perquery_count'] == 617400
plan = json.loads((p / 'measurement_gate_m3a_plan.json').read_text(encoding='utf-8'))
sources = plan['previous_sources'] | plan['sources']
assert all(hashlib.sha256((PROJECT / n).read_bytes()).hexdigest() == sha for n, sha in sources.items())
metrics = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
now = datetime.now().isoformat(timespec='seconds')
rows = {}
for name, run in analysis['runs'].items():
    rows[name] = dict(best_epoch=run['best_epoch'], normal_sixmetrics=run['normal']['metrics']['11'],
        normal_full_vs_base=run['normal']['full_vs_base'],
        groups={g: run['groups'][g] for g in ('all49', 'same_availability', 'overlap_mismatch', 'source_disjoint', 'partial_query_full_gallery', 'both_partial')})
fair = {}
for pooling in ('original_mean', 'eligible_mean', 'seed_query'):
    axis = analysis['runs'][pooling + '/axis_shared']
    fair[pooling] = {}
    for variant in ('frequency_shared', 'twins_shared'):
        control = analysis['runs'][pooling + '/' + variant]
        fair[pooling][variant] = dict(
            normal_delta_pp={m: axis['normal']['metrics']['11'][m] - control['normal']['metrics']['11'][m] for m in metrics},
            all49_delta_pp={m: axis['groups']['all49']['sixmetrics_equal_condition_mean'][m] - control['groups']['all49']['sixmetrics_equal_condition_mean'][m] for m in metrics})
target.write_text(json.dumps(dict(status='EXISTING_INSTALLED_GT_RESULTS_AND_SOURCE_SCOPE_RECHECK', at=now,
    references={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (original / 'analysis.json', original / 'independent_cpu_audit.json', PROJECT / 'common_outlet_axis.py')},
    cases=2940, repeated_condition_query_rows=617400, runs=rows, matched_expert_deltas=fair,
    code_scope='eligible_mean/seed_query modify the seven-relation M/I residual outlet. The raw-CLIP base common identity aggregation still uses the inherited available-modality mean. SharedReadout.query is both the common bias and the residual pooling query, present and active in all matched controls. This is not a full replacement of the base identity aggregator by multi-slot Set Transformer.',
    finding='These three exits were already trained and GT-audited. Their complete-input joint-vs-base changes are tiny, and seed-query gains are inconsistent against the matched ordinary experts. They are not untried proposals or a demonstrated final method.',
    limits='M1 training protocol, singleMSVRfit/dev seed42 only; do not mix outlet-specific best results into one method, infer significance from49 repeated queries, or attribute all M2b differences to pooling.',
    neural_executions=0, neural_source_changes=0, goal_complete=False), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
goal = p / 'research_goal_optimized_20261003.json'
g = json.loads(goal.read_text(encoding='utf-8'))
g['status'], g['updated_at'] = 'ACTIVE_UNMET', now
g['revision'] = '20261004_reconcile_completed_outlet_scope_and_M3a_order_while_original_training_active'
for milestone in g['milestones']:
    if milestone['id'] == 'M1':
        milestone['status'] = 'COMPLETE_AUDITED_NO_FINAL_CANDIDATE'
        milestone['actual_evidence'] = 'Ten fresh50;490 full49 conditions;2940 six-state installedGT cases/617400 repeatedrows PASS. Original_mean,eligible_mean,seed_query each have matched ordinary frequency and twins; augmentedDeMo original_mean reference also completed.'
        milestone['decision'] = 'Three M/I residual outlets were tested, with tiny normal11−00 and inconsistent fair-control effects. They do not prove a full base identity slot replacement. Frozen eight-stage utility supported M2 relation preservation; audited M2 cross-coordinate mismatch led to M2b identity alignment, whose fair-control negative results led to M3a gradient isolation. Do not repeat pooling as an untried factor.'
    elif milestone['id'] == 'M3':
        milestone['status'] = 'IN_PROGRESS_M3A_MEASUREMENT_GRADIENT_ISOLATION'
        milestone['work'] = '依据M2b实测收益估计失配，先单因素验证融合门控检索梯度与贡献回归的隔离；相同/重叠/不相交集合配对覆盖仍是后续独立因素，不同时加入。所有公平对照保持相同可用性处理和训练/计算预算。'
        milestone['decision'] = 'M3a只detach门控，不新建控制头，不改原M2b损失/采样/预算。真实CUDA契约和三次真实更新smokes通过，原控制器运行中。完整校准、真实PF救回、11−00及同轮普通专家和增强DeMo比较完成前，不宣布效益、不进入大规模扩展。'
g['current_evidence']['M1_scope_recheck'] = dict(status='EXISTING_GT_AND_SOURCE_SCOPE_RECHECK_NO_NEURAL_EXECUTION',
    path=target.relative_to(PROJECT).as_posix(), original_source_sha256=sources['common_outlet_axis.py'])
goal.write_text(json.dumps(g, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
doc = PROJECT / 'docs/实验交接.md'
s = doc.read_text(encoding='utf-8')
marker = '<!-- CURRENT_M3A_STATUS_END -->'
assert s.count(marker) == 1 and '<!-- M1_OUTLET_SCOPE_RECHECK_20261004 -->' not in s
text = '''<!-- M1_OUTLET_SCOPE_RECHECK_20261004 -->
前轮出口范围复核：M1三种汇聚方案已完成十组fresh50和GT审计2940项/617400重复查询记录，不应再当作未尝试建议。original_mean、eligible_mean、seed_query修改的是七关系M/I残差进入公共坐标的汇聚；原始CLIP公共基础仍按可用模态平均。共享query既是公共偏置又是残差汇聚查询，所有匹配对照均存在并实际训练。这轮不等价于全面替换公共基础为多语义槽位网络。双轴三种正常mAP/R1依次48.289661/61.428571、47.552859/60.000000、47.976572/60.476190；seed_query全49较同出口普通频域+0.848483mAP/+1.768707R1，较普通双专家仅+0.295347mAP/−0.009718R1，正常mAP也较普通双专家−0.047665。各版本优势不一致，不能拼成统一方法。完整六项、所有关键组、逐阶段范围和原证据SHA见results/preflight/m1_outlet_scope_recheck_20261004.json；这是已有记录/源码复核，没有新增NN执行。Goal中M1完成状态和M3执行顺序已据此统一：当前仅做M3a门控梯度，跨可用集合覆盖保留为另一轮；完整三集+2和缺失、公平对照、多种子范围未改。

'''
doc.write_bytes(s.replace(marker, text + marker, 1).encode('utf-8'))
entry = p / 'm1_outlet_scope_recheck_entrypoint_20261004.py'
assert not entry.exists()
entry.write_bytes(Path(__file__).read_bytes())
files = [target, entry, goal]
manifest = {f.relative_to(PROJECT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
for host, (root, _) in HOSTS.items():
    command(['scp', *OPTIONS, *[str(f) for f in files], host + ':' + root + '/results/preflight/'])
    code = 'import hashlib,json;from pathlib import Path;r=Path(' + repr(root) + ');print(json.dumps({n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ' + repr(list(manifest)) + '}))'
    assert json.loads(remote_python(host, code)) == manifest
digest = sync_handoff()
proof = p / 'm1_outlet_scope_recheck_publication_20261004.json'
assert not proof.exists()
proof.write_text(json.dumps(dict(status='EXACT_THREE_TEXT_MIRRORS_FIVE_HANDOFFS', at=now, files_sha256=manifest,
    doc_sha256=digest, neural_executions=0, neural_source_changes=0, goal_complete=False), indent=2) + '\n', encoding='utf-8')
for host, (root, _) in HOSTS.items():
    command(['scp', *OPTIONS, str(proof), host + ':' + root + '/results/preflight/'])
command(['git', 'add', '--', *manifest, proof.relative_to(PROJECT).as_posix(), 'docs/实验交接.md'], cwd=PROJECT)
command(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], cwd=PROJECT)
command(['git', 'commit', '-m', 'Reconcile completed outlet scope and single-factor goal milestones'], cwd=PROJECT)
command(['git', 'push', 'origin', 'HEAD:main'], cwd=PROJECT)
print('M1_SCOPE_GOAL_RECONCILED', json.dumps(dict(head=command(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).strip(), doc_sha256=digest, goal_complete=False)), flush=True)
