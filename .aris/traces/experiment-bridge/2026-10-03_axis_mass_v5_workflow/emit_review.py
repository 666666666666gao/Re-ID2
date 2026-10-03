"""Persist this fresh review and its actual scope; no remote/runtime execution."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path

project = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
trace = project / '.aris/traces/experiment-bridge/2026-10-03_axis_mass_v5_workflow'
helper = Path('C:/Users/gb/.codex_tmp/demo_axis_v5_mass_deploy_20261003.py')
names = ('mass_axis_collaboration.py', 'run_mass_experiment.py', 'verify_axis_mass.py')
source_hashes = {n: hashlib.sha256((project / n).read_bytes()).hexdigest() for n in names}
helper_hash = hashlib.sha256(helper.read_bytes()).hexdigest()
assert helper_hash == '3b72e700151015a28c04a2563692333b75617d42aa58ba2dd57a0e37d25e5d2b'
for path in [helper] + [project / n for n in names]:
    compile(ast.parse(path.read_text(encoding='utf-8')), str(path), 'exec')
checks = json.loads((trace / 'checks.json').read_text(encoding='utf-8'))
assert len(checks) == 11 and all(c['status'] == 'PASS' for c in checks)
model_review = json.loads((project / 'results/preflight/axis_collaboration_v5_mass_model_review.json').read_text(encoding='utf-8'))
assert model_review['status'] == 'PASS' and not model_review['blockers']
assert all(model_review['checked_source_sha256'][n] == source_hashes[n] for n in names[:2])

source_checks = [
    dict(name='initial_controls_and_uniform_V4_fuse', status='PASS', mode='source_review',
         source=['verify_axis_mass.py:21', 'verify_axis_mass.py:30', 'verify_axis_mass.py:41', 'verify_axis_mass.py:49', 'run_experiment.py:45'],
         detail='The new model and each V4 control reset seed42 before identical registered-module initialization. Total/trainable counts and every initial state tensor are compared directly. The pre-hook captures the sole full-forward calibrator inputs (base Bx5632, M Bx7x512, F Bx512); all-eligible/gates-one fusion compares the V4 axis model with V5 uniform1/7 mass using torch.equal. This is a fuse-equivalence check, not a claim of full-forward equivalence.'),
    dict(name='fixed_evidence_aS_hook_and_actual_M_gradient', status='PASS', mode='source_review',
         source=['verify_axis_mass.py:54', 'verify_axis_mass.py:72', 'axis_collaboration.py:117', 'mass_axis_collaboration.py:13'],
         detail='U Bx7x3x512 and V Bx3x3x512 remain fixed. relation_score receives U.mean(2), returns Bx7x1 and the forward hook adds1x7x1, modifying only a_S; c and psi read unchanged evidence. Hook removal precedes the new autograd graph. M conditional vectors should remain near-equal after relation-mass cancellation, while pi.sum(2) changes. Fixed outer gates[1,0,0] suppress F/I; the1536:5120 slice is exactly the seven M descriptor blocks. autograd.grad is outside no_grad and targets the actual trainable relation_score.weight through post-normalization M mass, with finite/nonzero validation; there is no calibrator-loss or F/I path in this scalar probe.'),
    dict(name='closed_evidence_independence_and_six_missing_masks', status='PASS', mode='source_review',
         source=['verify_axis_mass.py:78', 'verify_axis_mass.py:88', 'axis_collaboration.py:218', 'mass_axis_collaboration.py:21'],
         detail='State10 recomputes the independent M pooled evidence and its own relation scores; state01 receives zero M and uses its independent F evidence. Changing only closed-branch evidence therefore must preserve outputs exactly; state00 equals base. All six single/double missing masks leave at least one sensor, build allowed Bx7 from RELATIONS, and boolean-index the Bx7x3 route correctly. Invalid mass must be exactly zero; multiplying pi marginal by eligible-count makes mean eligible weight one. No cached full-state route enters these closed states.'),
    dict(name='semantic_fit_forward_stopped_full11_zero_optimizer', status='PASS', mode='source_review',
         source=['verify_axis_mass.py:101', 'verify_axis_mass.py:107', 'scaled_axis_collaboration.py:18', 'scaled_axis_collaboration.py:42', 'experiment_data.py:83'],
         detail='The final loader uses fixed fit identities, the preserved B64 identity sampler and seed42; it performs one training-mode autocast forward. The inherited full-reference constructor binds the no_grad target function whose reference is state11 and whose indices are shared across four states. The script checks finite contribution, detached target and at least one nonzero target component. It creates no optimizer and takes no step; the isolated semantic model can update BN running buffers. Formal smoke/train launch separate processes and construct fresh models from public CLIP, so those buffers are not reused.')
]
checks.extend(source_checks)
now = datetime.now().astimezone().isoformat(timespec='seconds')
started = datetime.fromtimestamp((trace / 'controller_import_reproduction.json').stat().st_ctime).astimezone().isoformat(timespec='seconds')
fixed_issue = dict(
    issue='Generated controller file-mode launch could not import project-root modules from its campaign subdirectory.',
    initial_status='BLOCKED',
    evidence='controller_import_reproduction.json: actual isolated stdlib fixture exit1/ModuleNotFoundError with project cwd.',
    fix='The executor inserted sys.path.insert(0, root) before launch_axis_scaled and launch_runs imports in the generated controller.',
    verification='controller_native_import_fix plus controller_import_fixed.json: explicit-path fixture exit0 and import ordering verified.',
    final_status='RESOLVED')
setup = dict(
    initial_local_python_failure='PATH python resolved to E:/Scripts/python.exe and reported No pyvenv.cfg file; no review code ran.',
    existing_python_used='C:/Users/gb/AppData/Roaming/uv/python/cpython-3.10-windows-x86_64-none/python.exe',
    first_harness_failure='Happy-path expectation used integer jobs keys after a JSON round-trip; actual receipt correctly had string keys.',
    correction='Only the harness expected keys changed to strings; first_harness_checks.json and first_harness_errors.txt preserve the failure. Second run:11 groups PASS, exit0.',
    installs=0, remote_calls=0, cuda_calls=0, optimizer_updates=0)
report = dict(
    status='PASS', blockers=[], observed_at=now,
    helper_sha256=helper_hash, checked_source_sha256=source_hashes,
    checks_count=len(checks), deterministic_groups=11, source_review_groups=4,
    reviewer='gpt-6-astra', reasoning_effort='max', fork_turns='none',
    agent_id='/root/review_axis_mass_v5_workflow', review_independence='same-family', acceptance_status='provisional',
    scope='Narrow V5 sanity/deployment delta; source/AST/stdlib mock only. Separate fresh model review remains the model gate.',
    model_review_contract_matches_actual_report=True,
    checks=checks, resolved_blockers=[fixed_issue], harness_setup=setup,
    limitations=['Actual Torch/CUDA data forwards, tensor values, gradients, smoke updates and fresh50 training were not run by this review; all six real preflight gates remain mandatory.',
                 'Pool mocks verify sequencing, submission set and max_workers=3; they do not execute concurrent GPU jobs. Existing parallel jobs are not forcibly killed on another job failure.'],
    trace_path='.aris/traces/experiment-bridge/2026-10-03_axis_mass_v5_workflow')
report_path = project / 'results/preflight/axis_collaboration_v5_mass_workflow_review.json'
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(trace / 'source_review_checks.json').write_text(json.dumps(source_checks, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(trace / 'harness_setup_notes.json').write_text(json.dumps(setup, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(trace / 'resolved_blocker.json').write_text(json.dumps(fixed_issue, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(trace / 'review.result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

prompt = '''按 C:/Users/gb/.codex/skills/experiment-bridge/SKILL.md CODE_REVIEW=true，独立窄审查 V5实际sanity/部署增量，不重审模型（另一fresh agent已做）。先读skill与共享local-codex-policy/review-tracing。项目C:/Users/gb/projects/demo_dual_axis_20261002，用户简单最小/no fallback/tryexcept/兼容层，只有一个人类交接docs/实验交接.md。允许机器trace，不另加handoffMarkdown。
模型mass_axis_collaboration.py和run_mass_experiment.py已准备、未GPU运行、正在独立modelreview；新verify_axis_mass.py做实际8dev+6缺失+64fit无optimizer的初始契约：V4 axis/plain/freq参数与全部初始tensors等同；uniformfullmass严格torch.equal退化V4fuse；固定U/V/c/psi仅通过relation_score hook改变aS，conditionalM近一致、归一化后M残差明显改变，F/Ioutergates0隔离；只M残差对relation_score实际非零梯度；关闭F证据更换不能改变10，关闭M不能改变01；6mask非法route0、平均合法关系weight1；停止梯度full11 target非零。0optimizer，语义BN会更新但正式训练总从publicCLIP新建。请核对这些断言/shape/hooks/actualgradient正确，不用真实GPU。
部署helper C:/Users/gb/.codex_tmp/demo_axis_v5_mass_deploy_20261003.py（刚py_compile0，未执行）。它先读model/workflow双PASS、现有SHA绑定、已有V4缺失78实际analysis COMPLETE。物理host2026只读source15不变、旧2910174dead、controller COMPLETE78/0updates、3newfiles和output不存在、GPU0/1/3每张<500MB，然后只scp3新文件，不改旧14+评测module，不碰环境。随后再次sourceproof核对，独立output/controllerPopen。controller先MSVR tensor+3更新smoke，再201同两gate、100同两gate，全6exit0/PASS/完整全部梯度非零/严格model+optimizerreload才创建development并最大3并行fresh50：0RGB100，1RGB201，3MSVR；2保留他人任务，每stage idle等240，不重复失败。seed42/B64/loss .05/splits/devbest earliest与V4完整原runner同源，官方test0；smoke9更新与150epoch预算分开。
此scope核对helper实际remote/native路径调用、执行顺序、防重复已发生旧输出/未完成前驱/繁忙资源门禁zero mutation、model/workflowreviewSHA及source15原约束、四个当前主要实际Torch验证断言合理性、不抢GPU2、日志exit/失败停止/preflight全PASS再train。需要report错误真证据最小修复；不做宽完整旧80/51/25/17套review，不额外hash方案/服务/框架、不真实SSH/GPU/安装/信号/改源码或handoff；AST/stdlib/mock即可。可借旧freshreviewharness方式，但记录首次失败修正，不盲跑。
输出results/preflight/axis_collaboration_v5_mass_workflow_review.json status PASS/BLOCKED, blockers[],helper_sha256, checked_source_sha256（新3文件mapping与helper预计格式必须匹配），checks_count。trace .aris/traces/experiment-bridge/2026-10-03_axis_mass_v5_workflow。实际modelgpt-6-astra/max/forknone，same-family/provisional；结束简短报告。'''
response = '''PASS：15 组窄检查通过，blockers=[]，same-family/provisional。

已复现并修正 controller 导入路径问题；当前 helper SHA256 为 3b72e700151015a28c04a2563692333b75617d42aa58ba2dd57a0e37d25e5d2b。首次失败证据与 harness 的 JSON 键类型修正均已保留。

报告：[axis_collaboration_v5_mass_workflow_review.json](C:/Users/gb/projects/demo_dual_axis_20261002/results/preflight/axis_collaboration_v5_mass_workflow_review.json)。Trace 已齐备于 .aris/traces/experiment-bridge/2026-10-03_axis_mass_v5_workflow。

本审查仅用源码、AST 和本地 stdlib/mock，未执行 SSH、CUDA 或 optimizer 更新。实际六项 preflight 仍须全部通过，才能启动三套 fresh50。'''
run_meta = dict(skill='experiment-bridge',run_id='2026-10-03_axis_mass_v5_workflow',started_at=started,
                executor='codex',executor_model='gpt-6-astra',executor_family='openai',
                review_independence='same-family',acceptance_status='provisional',project_dir=project.as_posix())
request = dict(call_number=1,purpose='mass-v5-workflow-review',timestamp=started,tool='spawn_agent',
               model='gpt-6-astra',reasoning_effort='max',fork_turns='none',
               files_referenced=[helper.as_posix()]+list(names),prompt=prompt)
meta = dict(call_number=1,purpose='mass-v5-workflow-review',timestamp=now,agent_id='/root/review_axis_mass_v5_workflow',
            model='gpt-6-astra',reasoning_effort='max',reviewer_family='openai',review_independence='same-family',
            acceptance_status='provisional',status='ok',verdict='PASS')
for filename,value in [('run.meta.json',run_meta),('001-workflow.request.json',request),('001-workflow.meta.json',meta)]:
    (trace/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(trace/'001-workflow.response.md').write_text(response+'\n',encoding='utf-8')
events = project / '.aris/meta/events.jsonl'
events.parent.mkdir(parents=True,exist_ok=True)
with events.open('a',encoding='utf-8') as stream:
    stream.write(json.dumps(dict(event='review_trace',skill='experiment-bridge',purpose='mass-v5-workflow-review',
        agent_id='/root/review_axis_mass_v5_workflow',trace_path=report['trace_path'],status='ok'),ensure_ascii=False)+'\n')
print(json.dumps(dict(status=report['status'],checks_count=len(checks),helper_sha256=helper_hash,checked_source_sha256=source_hashes),ensure_ascii=False))
