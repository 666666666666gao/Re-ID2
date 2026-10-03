"""Write the two migration verdicts from the completed local validation trace."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT/'.aris/traces/experiment-bridge/2026-10-03_host27_migration'
PREFLIGHT = ROOT/'results/preflight'
TMP = Path('C:/Users/gb/.codex_tmp')
VALIDATION = json.loads((TRACE/'validation.json').read_text(encoding='utf-8'))
NOW = datetime.now().astimezone().isoformat(timespec='seconds')
MODEL_SHA = '65f7eadb7fd0b05c939559626546dfe5f5686af6153958b98f10a5f552dbe91f'
SPEC_SHA = '4993690212b8f331ceac59c1cf8fe15751a69be89ffdcb2c046cdb7989cae5a0'
HELPERS = {
    'recovery': TMP/'demo_axis_v4_hardware_recovery27_20261003.py',
    'missing': TMP/'demo_axis_v4_missing_dev27_deploy_20261003.py',
}
HASHES = {
    'recovery': 'bf96e45e21c883a646007a6a5ebd0a10fbd6cf4461d4cbae6d5d6aa616768a5f',
    'missing': '88e99293e833593f59173c89339b00e4fbbef1419a20261107c1ad52f92588ba',
}
REPORTS = {
    'recovery': 'axis_collaboration_v4_hardware_recovery27_review.json',
    'missing': 'axis_collaboration_v4_missing_development27_review.json',
}
OLD_REPORTS = {
    'recovery': 'axis_collaboration_v4_hardware_recovery_review.json',
    'missing': 'axis_collaboration_v4_missing_development_review.json',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


for kind in HELPERS:
    assert sha(HELPERS[kind]) == HASHES[kind], 'Helper changed after narrow review'
assert sha(ROOT/'missing_development.py') == MODEL_SHA
assert VALIDATION['status'] == 'PASS'

attribution = {
    'agent': '/root/review_v4_host27_migration',
    'role': 'fresh native Codex reviewer of host27 migration delta',
    'model': 'gpt-6-astra',
    'reasoning_effort': 'max',
    'model_attribution': 'Provided parent launch metadata; this child cannot independently inspect the selected model or reasoning effort.',
    'further_delegation': 0,
}
limits = [
    'PASS is acceptance of the exact host/path/provider-document/transport/control-flow changes under local AST and standard-library mocks. It does not certify that server2027 is ready or that any real controller has started.',
    'No SSH, real scp, NVIDIA query, real signal, remote process spawn, torch/numpy import, neural forward, optimizer update or checkpoint deserialization was performed by this reviewer. All remote files, processes, copies, GPU answers and env27 validation documents in the executable harness are in-memory fixtures.',
    'The actual future server2027 setup/env27_doc_validation.json PASS record bound to the canonical spec SHA was not observed in this review. Its absence, pending/non-PASS status and wrong spec each fail the real generated initial guard under mocks before any source read, GPU query or mutation.',
    'Local prior-source bytes and supplied stage27 metadata agree on all14 hashes; fresh actual server2027 source checks, installed dataset/GT identity checks, package/import evidence and seeded CUDA witness remain execution evidence to collect. Stage metadata is not an independently repeated remote source or weight verification.',
    'GPU gates retain the existing memory.used <500MiB rule. The two reviewed controllers use disjoint pairs{0,1} and{2,3}; this does not reserve cards against unrelated tasks. Actual availability must pass when the helpers execute.',
    'Prior117-check recovery and80-check missing-development reviews were read and their exact helper/module hashes matched. Their old test suites and model/gradient checks were not rerun. The new tests cover only the migration delta and generated command bindings.',
    'Review context is fresh but same-family; acceptance is provisional. Model/effort fields are parent-supplied launch metadata, not independently confirmed by this child.',
]
execution_pending = {
    'status': 'EXECUTION_PENDING',
    'actual_provider_doc_PASS_for_canonical_spec': 'NOT_OBSERVED_BY_REVIEWER; required by both initial target27 guards',
    'actual_target27_source14_or15_recheck': 'EXECUTION_PENDING; only local source bytes and staged-source receipt compared here',
    'actual_installed_GT_and_exact_environment_witness': 'EXECUTION_PENDING; no actual installed datasets, package imports or CUDA witness checked by reviewer',
    'actual_GPU_pair_availability': 'EXECUTION_PENDING; only generated guard behavior tested',
    'actual_smoke_and_full_results': 'EXECUTION_PENDING; synthetic child terminal files prove orchestration only',
}

findings = {
    'recovery': [
        'The initial remote call uses2027 and reads setup/env27_doc_validation.json first. It requires PASS and canonical spec SHA499369..., then checks the same14 staged source hashes, exclusive new output and only GPUs0/1. Missing/non-PASS/wrong-spec docs stop before pause25, mkdir or spawn.',
        'The old2026 predecessor-output/PID guard is removed rather than misdirected to2027. The previously reviewed pause25 template is byte-semantically unchanged at AST level, remains a2025 call and targets only the already specified PID1408150 with SIGSTOP in the mock.',
        'The actual helper and generated entry/controller mock issue one detached2027 controller using /data/gb/Re-ID/conda-envs/tri_reid/bin/python and target27 cwd. Worker bindings remain GPU0 RGBNT100 frequency_scaled_fullref and GPU1 MSVR310 axis_raw_fullref, seed42, each smoke then train in new27 output paths. Data and public CLIP arguments use2027 paths.',
        'The two smoke/train pairs retain the old success checks; this review exercises positive command flow only and reuses the old117-check evidence for unchanged failure/wait/budget logic. Launch metadata reports host2027, availability27, original25 incident/SIGSTOP and fresh-attempt separation, with actual smoke and50 epochs still pending.',
        'No source or checkpoint transport is introduced by the recovery helper. Two successful target GPUs remain usable while the other controller pair is busy; busy500MiB on either assigned card prevents mutation.',
    ],
    'missing': [
        'The initial remote call uses2027 and requires actual provider doc PASS plus the canonical spec SHA before source14, GPU2/3 and exclusive input creation. Missing/non-PASS/wrong-spec docs stop before mkdir, transfer, upload or spawn.',
        'All three baseline proof strings bind their own original datasets on2025/2025/2026 dynamic_amp_comparison, with inner exit.json. All three axis proof strings still read2026 cross26_input, with sibling <run>_exit.json. Every loop instance was compiled and executed under stdlib fixtures.',
        'The real helper control flow produces exactly24 scp -3 input transfers,8 from2025 and16 from2026, to24 unique2027 paths. All six new run directories are created on2027. The obsolete2026 same-host copy branch is absent. Only missing_development.py is uploaded as source; the prior14 are hash-checked without source upload.',
        'Copied baseline exit files are inside each new run directory; copied axis exits remain siblings in the new input root. Launch checks15 target source entries and all24 copied target files. Same-length corruption of representative baseline/axis checkpoints and moving either exit layout prevent controller output creation/spawn.',
        'Original host/path proof metadata is retained in the receipt and controller. All12 smoke/full worker commands read the new2027 --run-dir, including all three axis runs; none reads source26 paths on2027. GPU2 runs axis and GPU3 DeMo in MSVR310,RGBNT201,RGBNT100 order, smoke before full, with2027 data/CLIP/Python paths.',
        'missing_development.py remains the exact prior80-check SHA65f7eadb.... Its model, GT, thirteen-condition and zero-update implementation is unchanged and not retested here. The aggregate78 conditions are orchestration metadata until actual neural smoke/full results arrive.',
    ],
}

for kind in HELPERS:
    checks = VALIDATION['checks'][kind]
    failed = [row for row in checks if row['status'] != 'PASS']
    result = VALIDATION['results'][kind]
    old = json.loads((PREFLIGHT/OLD_REPORTS[kind]).read_text(encoding='utf-8'))
    report = {
        'status': 'BLOCKED' if failed else 'PASS',
        'verdict': 'PASS_HOST27_MIGRATION_AST_AND_STDLIB_MOCK' if not failed else 'BLOCKED',
        'blockers': failed,
        'reviewed_at': NOW,
        'helper_path': str(HELPERS[kind]),
        'helper_sha256': HASHES[kind],
        'reviewer': attribution,
        'review_independence': 'same-family',
        'acceptance_status': 'provisional',
        'scope': 'Narrow2026-to2027 host/path/provider-doc/transfer/intake/command/metadata migration delta; no source/model edits or old full-suite rerun.',
        'reviewed_target': {
            'host': '2027',
            'project': '/data/gb/Re-ID/DeMo-DualAxis',
            'python': '/data/gb/Re-ID/conda-envs/tri_reid/bin/python',
            'gpu_set': [0,1] if kind == 'recovery' else [2,3],
            'data_root': '/data/gb/Re-ID/dataset',
            'pretrained': '/data/gb/Re-ID/pretrained/ViT-B-16.pt',
            'provider_doc': '/data/gb/Re-ID/DeMo-DualAxis/setup/env27_doc_validation.json',
            'provider_doc_required_status': 'PASS',
            'canonical_spec_sha256': SPEC_SHA,
            'spec_digest_definition': "sha256(json.dumps(parsed_spec,sort_keys=True,separators=(',',':')).encode())",
        },
        'prior_review_reused': {
            'path': 'results/preflight/'+OLD_REPORTS[kind],
            'report_sha256': sha(PREFLIGHT/OLD_REPORTS[kind]),
            'status': old['status'],
            'helper_sha256': old['helper_sha256'],
            'prior_distinct_passed_checks': 117 if kind == 'recovery' else 80,
            'old_full_suite_rerun': False,
        },
        'checks': {
            'actual_distinct_checks': len(checks),
            'passed': len(checks)-len(failed),
            'failed': len(failed),
            'harness_executions': 1,
            'generated_script_instances_compiled': result['generated_script_count'],
            'generated_script_instances_executed_with_mocks': result['generated_script_count'],
            'details': checks,
        },
        'findings': findings[kind],
        'execution_pending': execution_pending,
        'host_migration_limitations': limits,
        'actual_reviewer_actions': VALIDATION['boundary'],
        'validation': {
            'python': 'C:/Users/gb/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/python.exe',
            'command': 'python.exe -X utf8 .aris/traces/experiment-bridge/2026-10-03_host27_migration/review_host27.py',
            'process_exit_code': 0,
            'tool_chunk': '7d9217',
            'trace_root': '.aris/traces/experiment-bridge/2026-10-03_host27_migration',
            'trace_files': {
                name: sha(TRACE/name)
                for name in ('review_host27.py','validation.json',kind+'_migration.diff',kind+'_generated_scripts.json')
            },
            'generated_script_sha256': result['generated_script_sha256'],
        },
    }
    if kind == 'missing':
        report['module_path'] = str(ROOT/'missing_development.py')
        report['module_sha256'] = MODEL_SHA
        report['transfer_counts'] = dict(scp3_input_files=24, source25=8, source26=16, distinct_target27_input_files=24,
            baseline_inner_exits=3, axis_sibling_exits=3, module_uploads=1, existing_source_uploads=0,
            no_same_host_copy_branch=True)
    write(TRACE/(kind+'_review.json'), report)
    write(PREFLIGHT/REPORTS[kind], report)

request = """按experiment-bridge CODE_REVIEW做fresh窄范围2027迁移差异复核，只CPU/std-lib，不SSH/GPU/真实signal，不改源/人类文档，不再代理。Repo C:/Users/gb/projects/demo_dual_axis_20261002。只读2新helper：C:/Users/gb/.codex_tmp/demo_axis_v4_hardware_recovery27_20261003.py 与 demo_axis_v4_missing_dev27_deploy_20261003.py；对照旧26helpers同目录及已完成JSON results/preflight/axis_collaboration_v4_hardware_recovery_review.json（117检查）和 axis_collaboration_v4_missing_development_review.json（80检查），旧检查不整体重跑。module missing_development.py SHA65f7...已80PASS不改模型。实际25 kernel Xid79，26全卡被另项目占用，旧恢复26 guard line13失败发生在一切SIGSTOP/mkdir/spawn前，原25 controller1408150尚未暂停。27四3090空闲但原只有GOLA/cu118，现已stage124已审core代码、publicCLIP SHA5806...，正在复制精确3.10.14/torch2.5.1cu121环境+三数据，specSHA4993690212b8f331ceac59c1cf8fe15751a69be89ffdcb2c046cdb7989cae5a0。新27初始guard必须读取实际未来env27_doc_validation.json PASS+spec，再source14+GPUfree；未ready不可启动。Recovery27GPU0 RGBNT100 frequency/GPU1 MSVRraw仍各3smoke→50新CLIP同seed、原25已核定PID仅SIGSTOP防待任务，不改模型。Missing27GPU2主3 /GPU3DeMo3，全部6×4输入实际复制到独占27input（baseline25/25/26，axis3来源26旧cross26_input）；baseline innerexit，axis siblingexit；3axis输入原路径保metadata但实际--run-dir必须指向新27inputs，安装数据/CLIP必须27。共24scp3，不再有26本地复制branch。源upload只有missing module，其他14对target stagedsamehash。请实际AST/stdlib mock仅host/path/providerdoc-gate+transfer/input/command与metadata增量，错误spec/notPASS/missingdoc不spawn，source14/26读取proof不误路由成27，2controllersGPU集合{0,1}/{2,3}互斥，24input目标exit布局正确；编译生成字符串包括各loop绑定，不重复旧梯度/全部80/117。写2报告 results/preflight/axis_collaboration_v4_hardware_recovery27_review.json（helper_sha256）和 axis_collaboration_v4_missing_development27_review.json（helper_sha256/module_sha256），均status PASS/BLOCKED、blockers数组、实际checks、same-family/provisional、hostmigrationlimitations。机器trace .aris/traces/experiment-bridge/2026-10-03_host27_migration。模型字段可照父层launch gpt-6-astra/max但注明提供metadata，不自称独立确认。缺实际envdoc/source/GT/witness明确executionpending。本地 uv Python3.13 path C:/Users/gb/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/python.exe；禁止安装torch/numpy。"""
response = """两份27迁移审查均 PASS，blockers=[]；same-family / provisional。

- recovery：36/36项通过，helper SHA bf96e45e21c883a646007a6a5ebd0a10fbd6cf4461d4cbae6d5d6aa616768a5f。
- missing：51/51项通过，helper SHA 88e99293e833593f59173c89339b00e4fbbef1419a20261107c1ad52f92588ba；module SHA65f7eadb7fd0b05c939559626546dfe5f5686af6153958b98f10a5f552dbe91f未变。

已实际执行AST/标准库mock：环境缺文档、未PASS或spec错误均零副作用；24scp3、六组来源绑定、27新run-dir、baseline内置/axis同级exit、15源码+24输入启动复核及GPU{0,1}/{2,3}互斥通过。无真实SSH/GPU/信号，无源文件或人类文档修改，旧117/80整套未重跑。

报告：results/preflight/axis_collaboration_v4_hardware_recovery27_review.json；results/preflight/axis_collaboration_v4_missing_development27_review.json。完整trace：.aris/traces/experiment-bridge/2026-10-03_host27_migration。

实际env27_doc_validation PASS、当前27源码/GT/环境CUDA witness、smoke和full结果均为本审查的execution pending，不能将源审查PASS记为已经启动或环境ready。model/effort只按父层提供metadata记录。
"""
write(TRACE/'run.meta.json', dict(skill='experiment-bridge', run_id='2026-10-03_host27_migration',
    recorded_at=NOW, executor='codex', executor_model='gpt-6-astra', model_attribution=attribution['model_attribution'],
    executor_family='openai', review_independence='same-family', acceptance_status='provisional', project_dir=str(ROOT)))
write(TRACE/'001-host27-migration.request.json', dict(call_number=1, purpose='host27-migration-review', timestamp=NOW,
    timestamp_semantics='trace recording time; exact parent spawn timestamp unavailable in child',
    tool='spawn_agent', model='gpt-6-astra', reasoning_effort='max', model_attribution=attribution['model_attribution'],
    files_referenced=[str(p) for p in HELPERS.values()]+[str(PREFLIGHT/name) for name in OLD_REPORTS.values()], prompt=request,
    parent_followup='Canonical JSON digest was confirmed by parent; provider extraction/import/packages/splits/seeded CUDA witness was in progress and Tier3 document pending. Reviewer independently recalculated canonical499369... from supplied local spec.'))
(TRACE/'001-host27-migration.response.md').write_text(response, encoding='utf-8')
write(TRACE/'001-host27-migration.meta.json', dict(call_number=1, purpose='host27-migration-review', timestamp=NOW,
    agent_id='/root/review_v4_host27_migration', model='gpt-6-astra', reasoning_effort='max', model_attribution=attribution['model_attribution'],
    reviewer_family='openai', review_independence='same-family', acceptance_status='provisional', status='ok', substantive_verdict='PASS',
    distinct_checks=87, harness_executions=1, actual_execution_status='EXECUTION_PENDING'))
print(json.dumps({kind:dict(path=str(PREFLIGHT/name),sha256=sha(PREFLIGHT/name),status='PASS',checks=len(VALIDATION['checks'][kind])) for kind,name in REPORTS.items()},indent=2))
