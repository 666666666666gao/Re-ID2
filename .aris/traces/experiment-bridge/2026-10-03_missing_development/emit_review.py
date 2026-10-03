"""Publish the completed narrow review as JSON only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

root=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
trace=root/'.aris/traces/experiment-bridge/2026-10-03_missing_development'
helper=Path('C:/Users/gb/.codex_tmp/demo_axis_v4_missing_dev26_deploy_20261003.py')
validation=json.loads((trace/'validation.json').read_text())
supplement=json.loads((trace/'supplement_validation.json').read_text())
assert validation['status']==supplement['status']=='PASS'
shas=validation['source_hashes']
assert hashlib.sha256((root/'missing_development.py').read_bytes()).hexdigest()==shas['module_sha256']
assert hashlib.sha256(helper.read_bytes()).hexdigest()==shas['helper_sha256']
previous=json.loads((root/'results/preflight/axis_collaboration_v4_extra_cross26_launch.json').read_text())
for name,digest in previous['source_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
counts=next(item['counts'] for item in validation['checks'] if item['name'].startswith('six_real_local_terminal_texts'))
review={
    'status':'PASS',
    'verdict':'PASS_STATIC_AND_STDLIB_MOCK',
    'reviewed_at':datetime.now(timezone.utc).isoformat(),
    'review_independence':'same-family',
    'acceptance_status':'provisional',
    'reviewer':{'role':'fresh delegated native Codex source reviewer','context':'Read directly from supplied files; no external model backend; no further delegation'},
    'blockers':[],
    'required_fixes':[],
    'module_path':str(root/'missing_development.py'),
    'helper_path':str(helper),
    **shas,
    'scope':'New missing_development.py and the exact existing 2026 deployment helper, including generated guard/proof/controller/launch code. Existing functions were inspected and selectively executed with stdlib primitives. No source/human-document changes, SSH, GPU queries, installed packages, or neural execution.',
    'reviewed_target':{'host':'2026','source_hosts':['2025','2026'],'workers':{'2':['MSVR310_axis_scaled_fullref_s42','RGBNT201_axis_scaled_fullref_s42','RGBNT100_axis_scaled_fullref_s42'],'3':['MSVR310_demo_s42','RGBNT201_demo_s42','RGBNT100_demo_s42']},'gpu0_gpu1_recovery':'Independent and not invoked by this helper','host_migration':'Any later 2027 migration is outside this exact helper SHA and needs only its own changed host/path/synchronization review.'},
    'actual_counts':{
        'checks':validation['check_count']+supplement['check_count'],
        'main_harness_checks':validation['check_count'],
        'supplement_checks':supplement['check_count'],
        'generated_scripts':len(validation['generated']),
        'generated_main_families':['guard','baseline_proof','axis_proof','controller','launch'],
        'baseline_proof_instances':3,'axis_proof_instances':3,
        'generated_auxiliary_strings':{'mkdir':3,'copy':1,'entry':1},
        'models':6,'conditions_per_model':13,'conditions_in_positive_module_mocks':78,
        'positive_module_full_runs':6,'positive_module_smokes':6,'smoke_triplets':64,'masks_per_smoke':6,
        'input_proofs':24,'baseline_files_copied':12,'baseline_scp3_files_2025_to_2026':8,'baseline_local_copy_files_2026':4,
        'axis_files_reused_from_existing_frozen_cross26_input':12,'new_axis_file_copies':0,
        'source_uploads':1,'unchanged_existing_source_files':14,
        'actual_local_terminal_counts':counts,
        'real_remote_neural_conditions':0,
    },
    'findings':[
        {'status':'PASS','area':'Development split and frozen checkpoint intake','source':'missing_development.py:27',
         'detail':'COMPLETE with exactly50 epochs, the variant-appropriate exit0, and four original input-file proofs precede model build. Installed development query_indices, ids, cameras, scenes and names must all equal the saved arrays. split_records reads installed official-training directories, holds out fixed identities and preserves all dev records as gallery. No official_records or official-test selection is called. The six available local terminal/exit texts were read and are COMPLETE50/seed42 for the expected dataset/variant; MSVR retains360 gallery/210 queries, RGBNT201825/825 and RGBNT1003125/3125.'},
        {'status':'PASS','area':'FP32 strict reload, exact numerical clean parity and no update gates','source':'missing_development.py:46',
         'detail':'Existing build seeds and returns float().cuda(); weights_only checkpoint load is strict, then all modules are checked eval. The reused extraction has torch.no_grad and no autocast. Clean features use the original64 inference batch layout and must have max absolute error exactly0 against frozen saved dev features. Four clean metrics must match strict_reload. Both smoke and full compare all state_dict tensor versions after their final extraction and before their successful terminal; full also rechecks original input sizes/SHA before COMPLETE. No optimizer, backward, step, train-mode or test-time update call exists in the new module. Injected late state mutation and changed original file prevented COMPLETE.'},
        {'status':'PASS','area':'Six normalized-zero masks and original DeMo semantics','source':'missing_evaluation.py:20',
         'detail':'Actual mask_images was executed for all r/n/t/rn/rt/nt sets with primitive normalized arrays: only listed modality tensors become zeros_like. Actual verify_original_mask executed all six comparisons using the original DeMo forward mask AST with primitive arrays, returned12 forward calls, and restored miss_type=nothing; a sixth-mask mismatch failed. Real neural equality remains an execution gate, not a result of the primitive mock.'},
        {'status':'PASS','area':'Thirteen conditions, full gallery and GT metrics','source':'missing_development.py:75',
         'detail':'Actual main control flow was exercised for all6 models and6 smokes with list-backed arrays and stub feature extraction. Every full mock generated exactly clean+6 both_missing+6 query_missing. Both_missing indexes queries from the same masked full dev feature set; query_missing uses that masked query subset with the clean full dev gallery. All GT/name arrays and lengths remained aligned. MSVR uses scenes for same-ID/same-scene junk; RGBNT uses cameras. Actual inherited full_metrics executed against an independent primitive rank-counting fixture and wrote six metrics, CMC, camera/scene/identity groups and per-query CSV, including invalid-query bookkeeping.'},
        {'status':'PASS','area':'Six independent proof bindings and exact allowed transport','source':'demo_axis_v4_missing_dev26_deploy_20261003.py:33',
         'detail':'Executed the real helper outer AST with injected transports and all generated code. All3 baseline proofs and3 axis proofs bind their own dataset/run/host. Baseline MSVR310 andRGBNT201 read2025 dynamic_amp_comparison with inner exit0; RGBNT100 reads the same campaign on2026. Only best.pth,best_dev_arrays.npz,result.json,exit.json are copied for baselines:8scp -3 files and4same-host copyfile calls, all exact source/target. Axis runs directly reuse the already frozen2026 cross26_input with sibling exits, adding12 proofs and0new copies. All24 inputs are size/SHA checked before launch; originals remain unchanged. This is24verified inputs, not24new copies.'},
        {'status':'PASS','area':'Source preservation and launch gates','source':'demo_axis_v4_missing_dev26_deploy_20261003.py:20',
         'detail':'All14 prior local deployed source SHAs match the previous launch receipt. The helper checks the same14 remote digests, requires both target GPUs below500MiB and exclusive new input/output roots, uploads only missing_development.py, then rechecks15 source entries and24 input proofs before one detached controller spawn. Actual generated-code mocks rejected GPU500/busy, old source drift, existing output and same-length checkpoint corruption for each of6models before any spawn.'},
        {'status':'PASS','area':'Worker dependency order and honest aggregate completion','source':'demo_axis_v4_missing_dev26_deploy_20261003.py:80',
         'detail':'The actual controller ran with stdlib ThreadPoolExecutor and fake waited children. GPU2 receives axis andGPU3demo, each MSVR310→RGBNT201→RGBNT100. Each model must finish its64-triplet smoke exit0/PASS/zero-update/exact-parity before full. No retry loop exists. Smoke failure at each of the6positions prevented the full stage and every later job of that worker; full failure and malformed exit0 smoke outputs also blocked progress. The other worker may finish independently. A real threading.Event blocked the final child wait and proved aggregate COMPLETE was absent until release. After both workers return6runs only then are conditions78 and total COMPLETE written. Actual reused idle loop waited240seconds at2700 and500MiB and returned only at499MiB.'},
    ],
    'validation':{
        'status':'PASS','process_exit_codes':[0,0],
        'python_requested':'C:/Users/gb/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/python.exe',
        'command_prefix':'uv run --offline --no-project --python <python_requested> python -X utf8',
        'tool_chunks':['cc8d5e','f703f6'],
        'trace_root':str(trace),
        'artifacts':['review_harness.py','validation.json','review_supplement.py','supplement_validation.json','emit_review.py','review.json'],
        'checks':validation['checks']+supplement['checks'],
        'generated_string_line_counts':[{'kind':item['kind'],'lines':item['lines']} for item in validation['generated']],
    },
    'source_sha256':{'missing_development.py':shas['module_sha256'],'missing_evaluation.py':hashlib.sha256((root/'missing_evaluation.py').read_bytes()).hexdigest(),**previous['source_sha256']},
    'limitations':[
        'Same-family independent fresh context, provisional source/control-flow acceptance only; not cross-family review.',
        'No torch/numpy is installed or imported by these tests; no actual checkpoint deserialize, feature vector, CUDA calculation or neural equality was tested.',
        'Helper terminal/exit fixtures use actual local result text, but weight/NPZ bytes are distinct representative mock content. Remote input bytes, installed GT, live resource state and real FP32 parity require the actual M0/full execution.',
        'The clean parity implementation is exact numerical FP32 equality via max_abs_error==0, not a raw bytewise comparison. The original DeMo mask neural check uses torch.equal. No empirical neural equality is claimed by this review.',
        'The helper emits only STARTED_SMOKE_AND_78_CONDITIONS_PENDING at launch; this review does not certify any real launch, smoke PASS or78-condition result. Mock PID777 printed by the harness is not a real process.',
        'The reviewed helper targets2026. Parent-reported live resource failure and possible2027 migration do not alter this frozen-SHA source review; new target/path synchronization needs separate narrow review.',
        'Single seed42 frozen missing-input diagnostic does not establish missing-input training, full official-train reproduction or multi-seed acceptance.',
    ],
    'actual_reviewer_actions':validation['actual_actions'],
    'source_edits_by_reviewer':[],
    'human_document_edits_by_reviewer':[],
    'conclusion':'No concrete blockers found for the recorded module/helper SHA.80 targeted checks passed. Parent may use this provisional source review for the matching helper only, while collecting actual resource, provenance, strict neural smoke and full execution evidence separately.'
}
payload=json.dumps(review,ensure_ascii=False,indent=2)+'\n'
(trace/'review.json').write_text(payload,encoding='utf-8')
target=root/'results/preflight/axis_collaboration_v4_missing_development_review.json'
target.write_text(payload,encoding='utf-8')
print(json.dumps({'status':review['status'],'checks':review['actual_counts']['checks'],'blockers':review['blockers'],'report':str(target),**shas}))
