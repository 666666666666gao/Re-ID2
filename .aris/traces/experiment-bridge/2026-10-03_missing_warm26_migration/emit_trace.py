from datetime import datetime, timezone
import json
from pathlib import Path

project = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
trace = project / '.aris/traces/experiment-bridge/2026-10-03_missing_warm26_migration'
report_path = project / 'results/preflight/axis_collaboration_v4_missing_development26_repaired_review.json'
report = json.loads(report_path.read_text(encoding='utf-8'))
stamp = datetime.now(timezone.utc).isoformat()
prompt = '''按 C:/Users/gb/.agents/skills/experiment-bridge/SKILL.md CODE_REVIEW，仅审最小 host/provider migration delta，不重复原80/51/25测试。新文件 C:/Users/gb/.codex_tmp/demo_axis_v4_missing_dev26_repaired_deploy_20261003.py 对已reviewPASS25的 demo_axis_v4_missing_dev27_repaired_deploy_20261003.py。项目 C:/Users/gb/projects/demo_dual_axis_20261002；固定patched module missing_development.py SHA9cbb3440b61ecf3bddfe88bf071328632c9ad3903b2d02aa0bf5a36d57cf8973 已审，不再次全面审模块。实际证据 results/preflight/axis_collaboration_v4_next_resource_inventory.json（2026 GPU0/1/3空闲，2027 GOLA占1/2/3）、axis_collaboration_v4_missing_development27_repaired_guard_probe.json/resource_probe.json、原missing27_launch/failure_snapshot/repaired_review。新helper只借用已实际长期跑实验的2026 warm env/data/CLIP、用GPU0/1；四项待评测的checkpoint输入原就在2026，因此零SCP权重。远端run-dir直接绑定original launch input_files里source_host=2026的四条source_path，baseline100内exit、axis三个外exit；16proof+COMPLETE50/exit0。旧2027 cohort死/源old15 unchanged/两baseline完整13+两fail+两未启动，在2027只读验证。随后26guard检查14源码/新module尚不存在/4输入16proof/输出未存在/0+1free，才上传patched module并launch四项52条件；26新COMPLETE带reused_host2027/reused_output/两baseline26条件，总78，未触碰14训练源码/其他项目。请执行 AST+stdlib migration delta mock：物理host和绝对paths/source绑定/零权重copy/reused不当成26本地inputs/双卡序列&失败waitgate/资源门禁零mutation。不要真实SSH/GPU/信号/安装/改源码/人类doc/进一步delegate。写 results/preflight/axis_collaboration_v4_missing_development26_repaired_review.json，status PASS/BLOCKED、blockers、helper_sha256、module_sha256、实际checks数量与trace .aris/traces/experiment-bridge/2026-10-03_missing_warm26_migration。same-family/provisional。请保持审查范围小，及时返回。'''
response = '''PASS，17/17 项迁移专项检查通过，blockers=[]；same-family / provisional。

核实 2027 只读复用验证 -> 2026 资源与原输入门禁 -> 仅上传 patched module -> 2026 单次 controller 启动。四个 run-dir 直接绑定原 source_host=2026 的绝对 source_path，16 项输入证明在门禁及启动前均读取；RGBNT100 baseline 内置 exit 与三个 axis 外置 exit 均正确。两项 2027 baseline 未进入 2026 本地输入，COMPLETE 保留 reused_host=2027 与原 reused_output，明确 52+26=78。

GPU0 MSVR310 axis -> RGBNT201 axis；GPU1 RGBNT100 demo -> RGBNT100 axis。八个阶段均 idle -> spawn -> wait，smoke 后 full。四个失败 mock 均停止自己的后续阶段/任务且不写 aggregate COMPLETE。新旧 controller AST 在仅归一化主机路径、三个 provider 参数、GPU 编号及新增 reused_host 后相同。资源忙、既有目标、无效原 terminal/exit、输入证明变化均在 mutation 前拒绝。

helper SHA256: 4ed2c5a03025c18b553ffa07167eed5ddb1a2eda19a4b691997b524f30e7c471
module SHA256: 9cbb3440b61ecf3bddfe88bf071328632c9ad3903b2d02aa0bf5a36d57cf8973

未重跑原80/51/25套件，未重新全面审 patched module。仅使用 Python3.10.19 标准库和内存 mock；真实 SSH/GPU/信号/安装、源码和人类文档修改均为零。实际运行仍需通过 live guards 和四项 smoke/full。

审查工具更正记录：PATH 的 python 指向无 pyvenv.cfg 的 E:/Scripts/python.exe，改用已安装的 Python3.10.19，无安装。首次 harness 默认 GBK 读取失败，显式 UTF-8 后运行。第一次完整检查 16/17：mock 允许写入范围遗漏新输出根目录自身；读取 traceback 后只修 harness 白名单，部署 helper 未改，最终17/17。首次结果及 traceback 已保留。
'''
common = dict(model='gpt-6-astra', reasoning_effort='max', reviewer_family='openai', review_independence='same-family', acceptance_status='provisional')
run = dict(skill='experiment-bridge', run_id='2026-10-03_missing_warm26_migration', started_at=report['observed_at'], executor='codex', executor_model='gpt-6-astra', executor_family='openai', project_dir=str(project), **common)
request = dict(call_number=1, purpose='warm26-migration-delta-code-review', timestamp=report['observed_at'], tool='spawn_agent', files_referenced=[str(report_path), 'C:/Users/gb/.codex_tmp/demo_axis_v4_missing_dev26_repaired_deploy_20261003.py', 'C:/Users/gb/.codex_tmp/demo_axis_v4_missing_dev27_repaired_deploy_20261003.py', str(project / 'missing_development.py')], prompt=prompt, **common)
meta = dict(call_number=1, purpose=request['purpose'], timestamp=stamp, agent_id='/root/review_v4_missing_warm26_migration', status='ok', verdict=report['status'], checks_count=report['checks_count'], **common)
for name, value in [('run.meta.json', run), ('001-warm26-migration.request.json', request), ('001-warm26-migration.meta.json', meta)]:
    (trace / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(trace / '001-warm26-migration.response.md').write_text(response, encoding='utf-8')
(trace / 'harness_setup_notes.json').write_text(json.dumps(dict(path_python_failure='E:/Scripts/python.exe: No pyvenv.cfg file', installed_python_used='C:/Users/gb/AppData/Roaming/uv/python/cpython-3.10-windows-x86_64-none/python.exe', first_harness_failure='Windows default GBK read_text; fixed only harness to explicit utf-8.', first_full_run='16/17; harness omitted permitted output root itself from write allowlist; read stored traceback and corrected only harness.', preserved_first_run=['first_run_harness_result.json', 'first_run_harness_errors.txt'], deployed_helper_modified=False), indent=2) + '\n', encoding='utf-8')
events = project / '.aris/meta/events.jsonl'
events.parent.mkdir(parents=True, exist_ok=True)
with events.open('a', encoding='utf-8') as handle:
    handle.write(json.dumps(dict(event='review_trace', skill='experiment-bridge', purpose=request['purpose'], agent_id=meta['agent_id'], trace_path=str(trace), status='ok', verdict=report['status']), ensure_ascii=False) + '\n')
print(json.dumps(dict(status=report['status'], trace_path=str(trace), trace_files=4, checks=report['checks_count'])))
