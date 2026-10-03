"""Prepare a source-only review record and isolated stdlib harnesses."""
import ast
from datetime import datetime
import difflib
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_retrieval_utility_frozen49'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_retrieval_utility_frozen26_deploy_20261003.py')
OLD_HELPER = HELPER.with_name('demo_relation_frequency_frozen26_deploy_20261003.py')
FILES = ('diagnose_retrieval_utility_axis.py', 'missing_retrieval_utility_development.py', 'launch_retrieval_utility_frozen_evaluation.py')
PARENTS = ('diagnose_relation_frequency_axis.py', 'missing_relation_frequency_development.py', 'launch_relation_frequency_frozen_evaluation.py')
PROMPT = """Fresh same-family provisional review per experiment-bridge CODE_REVIEW. Project C:/Users/gb/projects/demo_dual_axis_20261002. SOURCE/AST/stdlib mocks ONLY: NO torch/numpy import, SSH/GPU/NN/install or implementation edits. Review exact3 NEW leaves diagnose_retrieval_utility_axis.py, missing_retrieval_utility_development.py, launch_retrieval_utility_frozen_evaluation.py and helper C:/Users/gb/.codex_tmp/demo_retrieval_utility_frozen26_deploy_20261003.py. Compare exact parents diagnose_relation_frequency_axis.py / missing_relation_frequency_development.py / launch_relation_frequency_frozen_evaluation.py and previous helper. Intended ONLY builder/variant/output/names changes. Parent run newRetrievalUtilityDeMo extends B with training-only stopped joint margin gain; inference interface+controlledstates identical B. Current actual parent launch results/preflight/retrieval_utility_launch.json controller3340643 GPU1 source55; actual tensor+3AMP PASS and fresh50epoch16 at16:39:57, parent may incomplete while reviewing. Helper must fail before SCP/NN if parent not COMPLETE50 exit0/matching variant. Never run helper in review. Once complete gate same GPU1idle, original4inputs protected, source55 binding +exact3 leaves, smoke->full4states->smoke->full49. Full49 = sevencached per-sample descriptor banks, 49cartesianquery/gallery, savednormal distances only after exactnormalfeatures parity; exactinstalledGT/order, all six/CMC50/groups/perquery, no optimizer, no test or trainedablation claims. Fourstates close conditional messages+joint term foroffexperts usingactual currentclasscontrolledstates, hooks no stale fused math. No changes to model/loss/trainedweights/runtime. Check Python3.10 syntax and stdlib gate/controller mocks. Write machineJSON ONLY results/preflight/retrieval_utility_frozen_review.json status PASS/FAIL/blockers/non_blocking/checks, checked_source_sha256 three leaves, helper_sha256. Save request/response/stdout/stderr/source diff traces .aris/traces/experiment-bridge/2026-10-03_retrieval_utility_frozen49; no extra humanhandoff MD. Skill-requested spawn is gpt-6-astra/max/forknone. Report concreteblocker immediately; avoid speculative compatibility/wrappers/fallback/hash schemes. Actual CUDA frozen smoke/full remains later root runtime; yourreview isprovisional."""


def write(name, value):
    (TRACE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parsed(name):
    return ast.parse((ROOT / name).read_text(encoding='utf-8'), filename=name, feature_version=(3, 10))


def function(module, name):
    return next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == name)


now = datetime.now().astimezone().isoformat(timespec='seconds')
write('run.meta.json', dict(skill='experiment-bridge', run_id=TRACE.name, started_at=now,
      executor='codex', executor_model='gpt-6-astra', reasoning_effort='max', executor_family='openai',
      reviewer_agent='/root/review_retrieval_utility_frozen49', review_independence='same-family',
      acceptance_status='provisional', project_dir=str(ROOT), scope='SOURCE_AST_STDLIB_MOCKS_ONLY'))
write('001-code-review.request.json', dict(call_number=1, purpose='frozen49-code-review', timestamp=now,
      tool='spawn_agent', model='gpt-6-astra', reasoning_effort='max', fork_turns='none',
      files_referenced=[*FILES, str(HELPER), *PARENTS, str(OLD_HELPER)], prompt=PROMPT))
write('000-tooling-note.json', dict(
      default_python_probe={'command':'python --version', 'result':'No pyvenv.cfg file'},
      existing_interpreter='C:/Users/gb/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/python.exe',
      execution='Existing standalone Python with -I -S; no package install or neural imports',
      diff_note='Initial raw git diff was dominated by LF/CRLF; retained trace uses normalized line endings and snapshots retain original bytes.'))

checks = []
snapshot = TRACE / 'reviewed_sources'
snapshot.mkdir(exist_ok=True)
for path in [*(ROOT / name for name in FILES + PARENTS), HELPER, OLD_HELPER]:
    (snapshot / path.name).write_bytes(path.read_bytes())
    ast.parse(path.read_text(encoding='utf-8'), filename=path.name, feature_version=(3, 10))
checks.append(dict(name='python310_source_syntax', status='PASS', files=list(FILES) + [HELPER.name],
                   evidence='ast.parse(feature_version=(3, 10)) on the three leaves and helper; no production module import.'))
diffs = []
for old, new in zip([*(ROOT / n for n in PARENTS), OLD_HELPER], [*(ROOT / n for n in FILES), HELPER]):
    before, after = old.read_text(encoding='utf-8'), new.read_text(encoding='utf-8')
    expected = before
    if new == HELPER:
        expected = expected.replace('relation_frequency_interface_launch.json', 'retrieval_utility_launch.json')
        expected = expected.replace('relation_frequency', 'retrieval_utility').replace('v9_', 'v10_')
        expected = expected.replace('RELATION_FREQUENCY_FROZEN26_STARTED', 'RETRIEVAL_UTILITY_FROZEN26_STARTED')
    else:
        for left, right in (
            ('run_relation_frequency_experiment', 'run_retrieval_utility_experiment'),
            ('axis_relation_frequency_fullref', 'axis_retrieval_utility_fullref'),
            ('diagnose_relation_frequency_axis.py', 'diagnose_retrieval_utility_axis.py'),
            ('missing_relation_frequency_development.py', 'missing_retrieval_utility_development.py'),
        ):
            expected = expected.replace(left, right)
    assert expected == after, new.name
    diffs.append(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=old.name, tofile=new.name)))
    checks.append(dict(name='exact_parent_delta_' + new.name, status='PASS',
          evidence='Only enumerated builder, variant, script, receipt, output and launch-label substitutions; CRLF normalization ignored.'))
(TRACE / '001-source.diff').write_text('\n'.join(diffs), encoding='utf-8')
launch = json.loads((ROOT / 'results/preflight/retrieval_utility_launch.json').read_text(encoding='utf-8'))
assert launch['pid'] == 3340643 and launch['host'] == '2026' and launch['gpu'] == 1
assert len(launch['source_sha256']) == 55
assert set(FILES).isdisjoint(launch['source_sha256'])
assert all(sha(ROOT / name) == digest for name, digest in launch['source_sha256'].items())
(snapshot / 'retrieval_utility_launch.json').write_bytes((ROOT / 'results/preflight/retrieval_utility_launch.json').read_bytes())
checks.append(dict(name='actual_parent_launch_and_all55_local_sources', status='PASS',
          evidence='Receipt PID3340643 / host2026 / GPU1; all55 current local files equal training receipt; exact3 new leaves are disjoint.'))

diagnose = parsed(FILES[0])
extract = function(diagnose, 'extract_states')
calls = [ast.unparse(node.func) for node in ast.walk(extract) if isinstance(node, ast.Call)]
assert calls.count('model.controlled_states') == 1 and 'model.fuse' not in calls
assert ast.dump(extract, include_attributes=False) == ast.dump(function(parsed(PARENTS[0]), 'extract_states'), include_attributes=False)
checks.append(dict(name='four_state_hooks_use_actual_current_method', status='PASS', evidence=
      'extract_states is AST-identical to B parent: captures independent reads0 and calibrator base, evaluates actual controlled_states, requires read counts2+2 / calibrator1 then3, restores hooks; no copied fuse math.'))

model = parsed('retrieval_utility_axis.py')
controlled = function(model, 'controlled_states')
forward = function(model, 'forward')
assert ast.unparse(controlled).count('if self.training:') == 1
assert 'states = super().controlled_states(*args, **kwargs)' in ast.unparse(controlled)
assert 'if not self.training:\n        return output' in ast.unparse(forward)
assert len([n for n in model.body if isinstance(n, ast.ClassDef)]) == 1
assert ast.unparse(next(n for n in model.body if isinstance(n, ast.ClassDef)).bases[0]) == 'RelationFrequencyInterfaceDeMo'
checks.append(dict(name='retrieval_utility_eval_inherits_B_interface_and_states', status='PASS', evidence=
      'Subclass controlled_states returns inherited states; _gain_states storage is training-only. forward returns superclass output immediately in eval before gain computation. Inference adds no learned module or descriptor interface.'))

missing = parsed(FILES[1])
body = function(missing, 'main').body
unparsed = ast.unparse(function(missing, 'main'))
assert unparsed.index('assert parity == 0') < unparsed.index("distances = saved['distances']")
for name in ('query_indices', 'ids', 'cameras', 'scenes', 'names'):
    assert "np.array_equal(saved['" + name + "']" in unparsed
assert unparsed.index("np.array_equal(saved['names']") < unparsed.index('model = build(')
assert "exclusion = scenes if arguments.dataset == 'MSVR310' else cams" in unparsed
checks.append(dict(name='installed_GT_order_and_exact_parity_before_saved_distance', status='PASS', evidence=
      'Current dev split query indices, IDs, cameras, scenes and names checked exactly before model construction; full clean feature parity==0 precedes any reuse of saved normal distances. MSVR excludes same identity+scene.'))
for name in FILES[:2]:
    f = parsed(name)
    callnames = [ast.unparse(n.func) for n in ast.walk(f) if isinstance(n, ast.Call)]
    assert not any(n.endswith(('.backward', '.step')) or n == 'official_records' for n in callnames)
    assert 'split_records' in callnames and 'full_metrics' in callnames
checks.append(dict(name='frozen_development_only_no_optimizer_or_test', status='PASS', evidence=
      'Both leaves use dev split, strict dev-best reload, eval/no_grad extraction, state version checks and no step/backward/official_records invocation; explicit single-seed frozen diagnostic limits.'))
metric = ast.unparse(function(parsed('full_evaluation.py'), 'full_metrics'))
assert "('camera', 'scene', 'identity')" in metric and "summary['CMC_1_to_50']" in metric
assert 'writer.writerows(rows)' in metric and "'mINP'" in metric and "(1, 5, 10, 20)" in metric
checks.append(dict(name='all_six_metrics_CMC50_groups_perquery', status='PASS', evidence=
      'Bound full_metrics computes mAP/mINP/Rank1/5/10/20, cross-checks original evaluate_reid, emits CMC1..50, identity/camera/scene groups and per-query CSV; same GT filtering for every pair.'))
write('001-static-checks.json', dict(status='PASS', checks=checks,
      checked_source_sha256={name:sha(ROOT/name) for name in FILES}, helper_sha256=sha(HELPER),
      current_training_source_sha256=launch['source_sha256']))

old_harness = ROOT / '.aris/traces/experiment-bridge/2026-10-03_relation_frequency_frozen49/002-mock-harness.json'
source = json.loads(old_harness.read_text(encoding='utf-8'))['source']
source = source.replace('relation_frequency_interface_launch.json', 'retrieval_utility_launch.json')
source = source.replace('relation_frequency', 'retrieval_utility').replace('v9_', 'v10_')
source = source.replace('retrieval_utility_interface.py', 'retrieval_utility_axis.py')
source = source.replace("'mock_helper', 'exec'", "'isolated_helper_AST_with_all_IO_replaced', 'exec'")
ast.parse(source, feature_version=(3, 10))
(TRACE/'002-mock-harness.py').write_text(source, encoding='utf-8')
write('002-mock-harness-origin.json', dict(adapted_from=str(old_harness), adapted_from_sha256=sha(old_harness),
      scope='Execute only import-stripped AST in MemoryFS; command, remote_python, subprocess, path and timing are mocks. No actual helper module entrypoint or imports.'))
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print(json.dumps(dict(status='PASS', static_checks=len(checks), source_count=55, new_leaves=3, no_neural_imports=True)))
