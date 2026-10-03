"""Write the final source-only review, bound to the actually inspected revision."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = Path(__file__).resolve().parent
HELPER = Path('C:/Users/gb/.codex_tmp/demo_available_base_deploy_20261003.py')
PLAN = ROOT / 'results/preflight/availability_base_plan.json'
first = json.loads((TRACE / '001_checks_result.json').read_text(encoding='utf-8'))
latest = json.loads((TRACE / '003_schedule_checks_result.json').read_text(encoding='utf-8'))
assert first['status'] == latest['status'] == 'PASS'
assert first['checked_source_sha256'] == latest['checked_source_sha256']
four = latest['checked_source_sha256']
assert len(four) == 4
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha for name, sha in four.items())
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == latest['helper_sha256']
assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == latest['plan_sha256']

review = dict(
    status='PASS', blockers=[], nonblocking_findings=[],
    observed_at=datetime.now(timezone.utc).isoformat(),
    reviewer_model='gpt-6-astra', reasoning_effort='max',
    review_independence='same-family', acceptance_status='provisional',
    review_type='FRESH_SOURCE_ONLY_WITH_EXECUTED_STDLIB_ORCHESTRATION_MOCKS',
    checked_source_sha256=four, helper_sha256=latest['helper_sha256'], plan_sha256=latest['plan_sha256'],
    baseline_commit='47e69f8080be26a0267e15598fb1509602725dac',
    conclusion='No remaining source-level blocker for the frozen six-job intervention and staged deployment. Actual tensor contracts, bitwise clean parity and full49 metrics remain runtime gates, not results of this review.',
    resolved_findings=[
        dict(id='scheduler_dependency_binding', file=str(HELPER), line=45,
             finding='The first helper extension omitted launch_axis_scaled.py and its launch_runs.py import, despite the controller executing idle from that chain.',
             correction='The implementation owner added exactly these two existing files to local/remote checks. Current parent maps contain neither key, so their addition does not overwrite a parent digest.',
             evidence='001_initial_concern_and_resolution.json; 001_checks_result.json; 003_schedule_checks_result.json'),
        dict(id='full_exit_precedes_final_controller_receipt', file=str(HELPER), line=31,
             finding='The first global-four wait used full_exit existence then immediately read controller_result. launch_available_base_frozen.py:51 writes the exit before lines53-59 validate artifacts/inputs and line62 writes the final per-job receipt.',
             original_observed_code="path=root/name/'full_exit.json'; if path.exists(): read zero exit; read root/name/'controller_result.json'",
             correction='Use controller_result existence as the completion marker, then require PASS and full_exit=0. full_exit=0 without a controller receipt remains pending.',
             evidence='Source-observed original ordering; corrected predicate executed in schedule_mocks_003/full_exit_before_controller_waits. No original neural run occurred.'),
        dict(id='stage_failure_cannot_remain_pending', file=str(HELPER), line=27,
             finding='The first primary-controller wait ignored nonzero smoke/full exits whenever controller_result was absent, leaving a failed job pending forever.',
             correction='Inspect each existing smoke/full exit first and raise on nonzero. Final receipt and full exit still both required before releasing two slots.',
             evidence='sources_002_pending_gate retains original helper bytes; two original-version stdlib cases reproduce released=false; corrected smoke/full failure cases raise AssertionError. Raw outputs and traces retained.')
    ],
    source_findings=[
        dict(topic='ATMoE layout and invalid relations', status='CORRECT_BY_SOURCE_INSPECTION',
             references=['availability_base_intervention.py:10', 'availability_base_intervention.py:14',
                         'availability_base_intervention.py:19', 'availability_base_intervention.py:24',
                         'modeling/moe/AttnMOE.py:94', 'modeling/moe/AttnMOE.py:153', 'dual_axis.py:8'],
             evidence='RELATIONS matches R,N,T,RN,RT,NT,RNT. HDM source inputs and relation outputs are masked. q/k preserve original B,H,1,D/H and B,H,7,D/H layout and scale. Chunking is on the feature dimension; head outputs concatenate before relation-major flattening, matching the original 7*512 descriptor. Illegal keys are masked to -inf before softmax and relation outputs are zeroed after biased experts. All three configs use 512-dimensional CLIP, HDM+ATM+GLOBAL_LOCAL and head counts 4 or 8.'),
        dict(topic='global hooks and exact full-input dispatch', status='CORRECT_BY_SOURCE_INSPECTION',
             references=['availability_base_intervention.py:33', 'availability_base_intervention.py:46',
                         'availability_base_intervention.py:51', 'modeling/make_model.py:160',
                         'mass_axis_collaboration.py:50'],
             evidence='Both DeMo and V5 call the three reduction modules before concatenating globals and before base fusion. Hooks therefore zero unavailable post-PIFE global blocks in both paths. The all-available branch returns the original hook output and calls the captured original fusion; the captured original model forward remains the entry computation. Installing closures/hooks adds no parameter or buffer keys. Actual clean feature equality is asserted against saved arrays in smoke and full, but was not executed by this reviewer.'),
        dict(topic='V5 additions remain the original V5 computations', status='CORRECT_BY_SOURCE_INSPECTION',
             references=['mass_axis_collaboration.py:60', 'mass_axis_collaboration.py:71',
                         'mass_axis_collaboration.py:81', 'availability_base_intervention.py:58'],
             evidence='Intervention changes only the base fusion callable and reduction outputs. Existing V5 bands, experts, cross-conditioning, joint router and residual fusion code/parameters remain parent-bound. V5 calibrator/residual outputs can change because their base input and base-norm anchors change; this is a downstream consequence of the controlled base intervention, not a claimed independently isolated V5 residual.'),
        dict(topic='dev-only fixed descriptor banks and ground-truth metrics', status='CORRECT_BY_SOURCE_INSPECTION',
             references=['missing_available_base_development.py:44', 'missing_available_base_development.py:82',
                         'missing_available_base_development.py:92', 'missing_available_base_development.py:102',
                         'full_evaluation.py:20', 'experiment_data.py:33'],
             evidence='Identity-heldout development rows and query indices are aligned with all saved IDs/cameras/scenes/names. Seven banks are encoded once and reused for all49 query/gallery pairs. RGBNT uses same-ID/same-camera junk exclusion; MSVR uses same-ID/same-scene. full_metrics writes six metrics, CMC1-50, per-query CSV and groups. Saved clean distances are used only after exact clean feature parity. No official-test loader or optimizer is called.'),
        dict(topic='frozen artifacts and stage gating', status='CORRECT_BY_SOURCE_AND_STDLIB_MOCK',
             references=['missing_available_base_development.py:38', 'missing_available_base_development.py:53',
                         'missing_available_base_development.py:109', 'launch_available_base_frozen.py:27',
                         'launch_available_base_frozen.py:39', 'launch_available_base_frozen.py:51'],
             evidence='Strictly reloads best checkpoints from COMPLETE 50-epoch runs with zero exit. Records keys/parameter objects and state versions; only eval/no_grad feature extraction and no_grad tensor verifier run. Original checkpoint/array/result/exit file bytes are checked before and after both stages by the controller. Smoke must produce PASS, exact clean parity, tensor-contract PASS and unchanged state/inputs before full. Nonzero child exits and failed/missing artifacts halt the affected worker; exclusive output/log creation prevents automatic reruns and preserves failure logs.'),
        dict(topic='source closure and global scheduling', status='CORRECT_BY_SOURCE_AND_STDLIB_MOCK',
             references=[str(HELPER) + ':20', str(HELPER) + ':27', str(HELPER) + ':43',
                         'launch_axis_scaled.py:23', 'results/preflight/availability_base_plan.json'],
             evidence='Current sources match all18 parent keys on2026 and16 on2027. Four disjoint existing dependencies and exactly four new NN files yield26/24 bound source maps, with no parent digest overwrite. Review binds helper and plan. 2026 starts four fixed GPUs; 2027 starts two only after both specified2026 V5 jobs have final PASS and zero full exit, keeping at most four NN jobs. Busy GPU polling and cross-host release polling use240 seconds. RGBNT100 DeMo uses its original2026 dynamic_amp_comparison path.'),
    ],
    executed_checks=dict(
        first_trace='001_checks_result.json', later_trace='003_schedule_checks_result.json',
        source_hash_ast_checks=6, current_stdlib_mock_cases=21,
        original_defect_reproductions=2,
        total_recorded_checks=first['check_count'] + latest['check_count'],
        actual_neural_tensor_checks=0, actual_model_imports=0,
        actual_torch_or_numpy_imports=0, actual_ssh_calls=0, actual_gpu_jobs=0,
        qualification='27 current source/stdlib checks passed; two additional recorded checks reproduce the preserved old wait defect. Source reasoning about tensors is not a runtime PASS.'),
    limitations=[
        'Same-family fresh-context review is provisional. No cross-family acceptance is claimed.',
        'Real-module tensor checks, saved-feature bitwise equality, all-mask finite/zero-coordinate checks and294 condition evaluations have not run in this review. They must pass their actual deployment stages before any runtime completion claim.',
        'This is frozen seed42 identity-heldout development, with zero optimizer updates and zero official-test uses. It does not establish a trained availability model, multi-seed superiority or the final +2mAP/+2Rank1 target.',
        'Strict masked DeMo retains separate source/relation coordinates. Disjoint retained sets, such as R-query and N-gallery, have disjoint surviving blocks and can yield orthogonal descriptors and tied distances. This expected diagnostic limitation can harm retrieval; report all49 pairs unchanged and preserve negative outcomes. Shared identity coordinates and asymmetric training remain future work.',
        'Source binding here is the explicit parent maps plus the four evidenced additional dependencies and four new files; it does not claim a recursively hashed Python environment.'
    ],
    trace_directory=str(TRACE),
    raw_sources='sources_001, sources_002_pending_gate, sources_003_final',
    raw_diffs='001_*.diff and003_*_helper.diff',
    raw_checks='001_checks_raw.log,003_schedule_checks_raw.log,controller_mocks_001,schedule_mocks_003'
)
payload = (json.dumps(review, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
archive = TRACE / '004_review.response.json'
target = ROOT / 'results/preflight/availability_base_review.json'
assert not archive.exists() and not target.exists()
archive.write_bytes(payload)
target.write_bytes(payload)
assert archive.read_bytes() == target.read_bytes()
print(json.dumps(dict(status=review['status'], blockers=review['blockers'], target=str(target),
    current_checks=27, preserved_original_defect_reproductions=2,
    helper_sha256=review['helper_sha256'], plan_sha256=review['plan_sha256'],
    output_sha256=hashlib.sha256(payload).hexdigest()), indent=2))
