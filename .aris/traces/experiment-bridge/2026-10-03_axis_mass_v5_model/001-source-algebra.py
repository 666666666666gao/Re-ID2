"""Narrow V5 review: stdlib AST, inherited-call spies, and independent algebra only."""
import ast
import copy
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(r'C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_axis_mass_v5_model'
NOW = datetime.now(timezone.utc).isoformat()
CHECKS = []


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def check(name, passed, evidence):
    CHECKS.append({'name': name, 'passed': bool(passed), 'evidence': evidence})
    write_json(TRACE / '001-checks-progress.json', CHECKS)
    print(('PASS ' if passed else 'FAIL ') + name, flush=True)


def dump(node):
    return ast.dump(node, include_attributes=False)


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def cls(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


FILES = ['mass_axis_collaboration.py', 'run_mass_experiment.py', 'axis_collaboration.py',
         'scaled_axis_collaboration.py', 'run_experiment.py', 'dual_axis.py']
TREES = {p: ast.parse((ROOT / p).read_text(encoding='utf-8-sig'), filename=p) for p in FILES}
plan = read_json('results/preflight/axis_collaboration_v5_mass_plan.json')
frozen_receipt = read_json('results/preflight/axis_collaboration_v4_missing_development26_repaired_launch.json')
bindings = {**frozen_receipt['source_sha256'], **plan['new_sources']}
actual_hashes = {p: sha(p) for p in bindings}
check('reviewed_new_sources_match_supplied_receipt',
      all(actual_hashes[p] == expected for p, expected in plan['new_sources'].items()), plan['new_sources'])
check('frozen_v4_and_active_missing_sources_unchanged',
      all(actual_hashes[p] == expected for p, expected in frozen_receipt['source_sha256'].items())
      and all(actual_hashes[p] == expected for p, expected in plan['frozen_v4_sources'].items()),
      {'receipt': 'results/preflight/axis_collaboration_v4_missing_development26_repaired_launch.json',
       'original_frozen_sources': 14, 'existing_missing_module': 1,
       'mismatches': [p for p, expected in frozen_receipt['source_sha256'].items() if actual_hashes[p] != expected]})
check('new_and_direct_parent_source_syntax', len(TREES) == 6, {'ast_parsed': FILES, 'module_imports_executed': 0})

new_class = cls(TREES['mass_axis_collaboration.py'], 'MassAxisCollaborationDeMo')
scaled_class = cls(TREES['scaled_axis_collaboration.py'], 'ScaledAxisCollaborationDeMo')
axis_class = cls(TREES['axis_collaboration.py'], 'AxisCollaborationDeMo')
init = function(new_class, '__init__')
expected_init = ast.parse("def __init__(self, classes, cfg, cameras):\n    super().__init__(classes, cfg, cameras, 'axis_scaled_fullref')").body[0]
check('same_constructor_capacity_state_and_rng_sequence',
      dump(init) == dump(expected_init)
      and [ast.unparse(b) for b in new_class.bases] == ['ScaledAxisCollaborationDeMo']
      and [n.name for n in new_class.body if isinstance(n, ast.FunctionDef)] == ['__init__', 'fuse', 'forward'],
      {'source': 'mass_axis_collaboration.py:9',
       'reason': 'Exactly the V4 axis_scaled_fullref constructor; no added registered parameter, buffer, RNG operation or init assignment. Actual tensor equality remains the planned GPU gate.'})

new_forward = copy.deepcopy(function(new_class, 'forward'))
old_forward = function(axis_class, 'forward')
fused = next(n for n in new_forward.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'fused')
extra_argument = ast.unparse(fused.value.args.pop())
old_reference = next(n.value for n in ast.walk(old_forward) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith('one stopped-gradient current-batch'))
reference_nodes = [n for n in ast.walk(new_forward) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith('one stopped-gradient current-batch')]
new_reference = reference_nodes[0].value
reference_nodes[0].value = old_reference
check('forward_exact_ast_except_explicit_mass_and_reference_metadata',
      extra_argument == 'route.sum(2)' and len(reference_nodes) == 1 and dump(new_forward) == dump(old_forward),
      {'new_argument': extra_argument, 'new_reference': new_reference,
       'preserved': 'backbone, availability masks, m0/f0, simultaneous conditions, m1/f1, routing, gates, detached diagnostics, controlled states, auxiliary CE/triplet taps, contribution loss, output ordering'})

new_fuse = copy.deepcopy(function(new_class, 'fuse'))
old_fuse = copy.deepcopy(function(scaled_class, 'fuse'))
removed_arg = new_fuse.args.args.pop().arg
removed_default = ast.unparse(new_fuse.args.defaults.pop())
old_fuse.body.pop(0)  # Fixed V5 constructor selects the normalized V4 branch.
new_body = new_fuse.body[0].body
old_body = old_fuse.body[0].body
new_m = next(n for n in new_body if isinstance(n, ast.If) and ast.unparse(n.test) == 'use_m')
mass_prefix = copy.deepcopy(new_m.body[:2])
new_m.body = new_m.body[2:]
m_weight = ast.unparse(new_m.body.pop(1))
new_i = next(n for n in new_body if isinstance(n, ast.If) and ast.unparse(n.test) == 'use_m and use_f')
i_weight = ast.unparse(new_i.body.pop(2))
old_body[:] = [n for n in old_body if not (isinstance(n, ast.If) and ast.unparse(n.test) == 'self.frequency_only')]
check('fuse_exact_v4_ast_except_post_normalization_m_i_mass',
      removed_arg == 'relation_mass' and removed_default == 'None'
      and m_weight == i_weight == 'projected = projected * weights[:, :, None]'
      and dump(new_fuse) == dump(old_fuse),
      {'source': 'mass_axis_collaboration.py:13', 'mass_applied_after_normalization_to': ['M', 'I'],
       'preserved': ['detached per-relation anchors', 'F projection and amplitude', 'three learned scales', 'three independent gates', 'base prefix', 'descriptor concatenation'],
       'inactive_branch_removed': 'frequency_only is False for the fixed axis_scaled_fullref parent variant'})

variant_assignment = next(n for n in TREES['scaled_axis_collaboration.py'].body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'VARIANTS')
variants = ast.literal_eval(variant_assignment.value)
check('fixed_parent_variant_normalized_structured_fullref',
      variants['axis_scaled_fullref'] == ('axis_collaboration', True, True),
      {'variant': list(variants['axis_scaled_fullref']), 'descriptor': 'Inherited 11*512 = 5632; no new head or descriptor slice.'})

old_runner = TREES['run_experiment.py']
new_runner = TREES['run_mass_experiment.py']
for name in ['write_json', 'configuration', 'evaluate', 'step']:
    check('runner_' + name + '_exact_ast', dump(function(old_runner, name)) == dump(function(new_runner, name)),
          {'source': 'run_mass_experiment.py:' + str(function(new_runner, name).lineno),
           'comparison': 'run_experiment.py:' + str(function(old_runner, name).lineno)})

new_main = copy.deepcopy(function(new_runner, 'main'))
old_main = function(old_runner, 'main')


def variant_parser_stmt(node):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == 'parser.add_argument'
            and node.value.args and ast.literal_eval(node.value.args[0]) == '--variant')


old_variant = next(n for n in old_main.body if variant_parser_stmt(n))
new_variant = next(n for n in new_main.body if variant_parser_stmt(n))
choices = ast.literal_eval(next(k.value for k in new_variant.value.keywords if k.arg == 'choices'))
new_main.body[new_main.body.index(new_variant)] = copy.deepcopy(old_variant)
metadata_keys = ['method_revision', 'relation_routing', 'unchanged_V4_factors']
extra_metadata = [n for n in new_main.body if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Subscript) and ast.unparse(n.targets[0].value) == 'info'
                  and isinstance(n.targets[0].slice, ast.Constant) and n.targets[0].slice.value in metadata_keys]
new_main.body = [n for n in new_main.body if n not in extra_metadata]
check('runner_main_entire_ast_same_after_declared_metadata_and_variant',
      choices == ['axis_mass_fullref'] and len(extra_metadata) == 3 and dump(new_main) == dump(old_main),
      {'preserved': ['split and loader setup', 'seed42 default', 'B64/50 configuration', 'optimizer/scheduler/AMP', 'three-update smoke and strict reload', 'epoch1..50 training', 'per-epoch seed/augmentations/batch-order logging', 'earliest dev mAP checkpoint', 'final dev-only strict reload'],
       'declared_metadata': metadata_keys})

expected_build = ast.parse("def build(args, cfg, classes, cameras):\n    seed_all(args.seed)\n    assert args.variant == 'axis_mass_fullref'\n    model = MassAxisCollaborationDeMo(classes, cfg, cameras).float().cuda()\n    model.contribution_loss_weight = args.contribution_weight\n    return model").body[0]
check('runner_build_same_seed_public_initialization_and_loss_weight',
      dump(function(new_runner, 'build')) == dump(expected_build),
      'Seed reset immediately precedes the identical parent constructor; parent pretraining path and loss weights are unchanged. No checkpoint load or new runtime option is introduced.')
old_imports = [dump(n) for n in old_runner.body if isinstance(n, (ast.Import, ast.ImportFrom))]
new_imports = [dump(n) for n in new_runner.body if isinstance(n, (ast.Import, ast.ImportFrom)) and not (isinstance(n, ast.ImportFrom) and n.module == 'mass_axis_collaboration')]
new_top_level = [n for n in TREES['mass_axis_collaboration.py'].body if not isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef))]
check('no_additional_import_side_effect_or_parameter_allocation',
      old_imports == new_imports and len(new_top_level) == 1 and isinstance(new_top_level[0], ast.Expr)
      and isinstance(new_top_level[0].value, ast.Constant),
      'The additional module contains only a docstring, imports of already loaded parents/torch, and the class declaration; no module-level random draw or model instantiation.')

full_targets = function(TREES['scaled_axis_collaboration.py'], 'full_reference_targets')
target_src = ast.unparse(full_targets)
check('full11_target_stopped_single_reference_and_shared_indices',
      [ast.unparse(n) for n in full_targets.decorator_list] == ['torch.no_grad()']
      and "reference = F.normalize(states['11'].float(), dim=1)" in target_src
      and target_src.count('pos = ') == target_src.count('neg = ') == 1
      and 'reference[pos] - reference[neg]' in target_src
      and 'for key, value in states.items()' in target_src
      and 'self.calibrator.targets = full_reference_targets' in ast.unparse(function(scaled_class, '__init__')),
      {'source': 'scaled_axis_collaboration.py:18',
       'reason': 'Entire target function is no_grad; one current-batch full11 hard-positive/negative selection is reused for all four states. This path is inherited byte unchanged.'})

# Execute only the inherited Python call wiring against spies; no tensor or model code.
controlled_ast = copy.deepcopy(function(axis_class, 'controlled_states'))
controlled_ns = {'torch': SimpleNamespace(zeros_like=lambda _: 'ZERO')}
exec(compile(ast.fix_missing_locations(ast.Module(body=[controlled_ast], type_ignores=[])), '<controlled-call-spy>', 'exec'), controlled_ns)


def controlled(m0, f0, full):
    stub = SimpleNamespace(
        structured=True,
        modality_expert=SimpleNamespace(pool=lambda m, e: ('M', m, e)),
        frequency_expert=SimpleNamespace(pool=lambda f, a, s: ('F', f, a, s)),
        calibrator=lambda b, m, f: ('prediction', (b, m, f)),
        fuse=lambda b, m, f, g, um, uf, e: (b, m, f, g, um, uf, e))
    return controlled_ns['controlled_states'](stub, 'BASE', m0, f0, 'ELIGIBLE', 'AVAILABLE', full)


s0 = controlled('m0', 'f0', 'fullA')
sf = controlled('m0', 'F_CHANGED', 'fullB')
sm = controlled('M_CHANGED', 'f0', 'fullC')
check('closed_state_call_wiring_has_no_opposite_evidence_or_full_route',
      s0['10'] == sf['10'] and s0['01'] == sm['01'] and s0['00'] == sf['00'] == sm['00'] == 'BASE'
      and s0['11'] == 'fullA' and sf['11'] == 'fullB' and sm['11'] == 'fullC',
      {'source': 'axis_collaboration.py:218',
       'execution': 'AST-extracted controlled_states only, with stdlib call spies; no router.conditions, router.route, psi, or neural forward exists on the stub.',
       'unchanged_state10_when': 'F evidence and full11 changed', 'unchanged_state01_when': 'M evidence and full11 changed'})


def softmax(values):
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


class Scores:
    def __init__(self, values):
        self.values = list(values)

    def squeeze(self, dimension):
        assert dimension == -1
        return self

    def masked_fill(self, mask, value):
        return Scores([value if flag else score for score, flag in zip(self.values, mask)])

    def softmax(self, dimension):
        assert dimension == 1
        return Scores(softmax(self.values))

    def __mul__(self, count):
        return [v * count for v in self.values]


class Eligibility:
    def __init__(self, mask):
        self.mask = mask

    def __invert__(self):
        return [not flag for flag in self.mask]

    def sum(self, dimension, keepdim):
        assert dimension == 1 and keepdim
        return sum(self.mask)


mass_code = compile(ast.fix_missing_locations(ast.Module(body=mass_prefix, type_ignores=[])), '<mass-source-prefix>', 'exec')
relations = ast.literal_eval(next(n.value for n in TREES['dual_axis.py'].body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'RELATIONS'))
availability = [a for a in itertools.product([False, True], repeat=3) if any(a)]
masks = [[all(a[index] for index in subset) for subset in relations] for a in availability]
base_a = [-0.6, 0.3, 0.8, -0.2, 0.45, -0.7, 1.1]
calls = []
source_mass_results = []
for mask in masks:
    stub = SimpleNamespace(router=SimpleNamespace(relation_score=lambda m: (calls.append(m), Scores(m))[1]))
    outputs = []
    for supplied_full_mass in [object(), Scores([0.9, 0.01, 0.01, 0.02, 0.01, 0.01, 0.04])]:
        scope = {'self': stub, 'modality': base_a, 'eligible': Eligibility(mask), 'use_f': False,
                 'relation_mass': supplied_full_mass, 'torch': SimpleNamespace(inf=math.inf)}
        exec(mass_code, scope)
        outputs.append(scope['weights'])
    expected = [p * sum(mask) for p in softmax([v if ok else -math.inf for v, ok in zip(base_a, mask)])]
    source_mass_results.append({'eligible_count': sum(mask), 'ignored_full_mass': outputs[0] == outputs[1],
                                'equals_independent_softmax': outputs[0] == expected})
check('m_only_source_branch_recomputes_own_mass_all_seven_masks',
      all(row['ignored_full_mass'] and row['equals_independent_softmax'] for row in source_mass_results)
      and len(calls) == 14,
      {'source': 'mass_axis_collaboration.py:28', 'results': source_mass_results,
       'execution': 'Only the exact mass-prefix AST executes on stdlib score/mask shims; all neural methods and torch execution are absent.'})

full_mass = Scores([0.08, 0.12, 0.05, 0.3, 0.1, 0.2, 0.15])
full_scope = {'self': object(), 'modality': object(), 'eligible': Eligibility([True] * 7), 'use_f': True,
              'relation_mass': full_mass, 'torch': SimpleNamespace(inf=math.inf)}
exec(mass_code, full_scope)
check('full_source_branch_uses_explicit_joint_mass', full_scope['weights'] == full_mass * 7,
      'Full mass prefix executes with self lacking a router: it uses the supplied differentiable joint mass and does not recompute M-only logits.')

# Independent scalar/vector algebra, not an implementation of any neural forward.
def joint(a, mask):
    c = [0.2, -0.35, 0.45]
    psi = [[0.13 * math.sin((s + 1) * (b + 2)) for b in range(3)] for s in range(7)]
    values = [a[s] + c[b] + psi[s][b] if mask[s] else -math.inf for s in range(7) for b in range(3)]
    flat = softmax(values)
    return [flat[s * 3:s * 3 + 3] for s in range(7)]


def norm(values):
    return math.sqrt(sum(v * v for v in values))


def unit(values):
    length = norm(values)
    return [v / length for v in values]


directions = [unit([math.sin(s + 1), math.cos(s + 2), 0.4 + 0.1 * s]) for s in range(7)]
interaction_direction = unit([0.7, -0.8, 0.5])
anchors = [1 + s * 0.15 for s in range(7)]


def residual(a, mask, interaction=False):
    pi = joint(a, mask)
    mass = [sum(row) for row in pi]
    count = sum(mask)
    directions_here = [interaction_direction] * 7 if interaction else directions
    scale_gate = 0.05 * 0.5 if interaction else 0.1 * 0.5
    return [scale_gate * anchors[s] * count * mass[s] * directions_here[s][j]
            for s in range(7) for j in range(3)]


mask_results = []
gradient_results = []
uniform_exact = []
baseline_cancellation_errors = []
for available, mask in zip(availability, masks):
    count = sum(mask)
    pi = joint(base_a, mask)
    mass = [sum(row) for row in pi]
    weights = [count * p for p in mass]
    uniform = [Fraction(count) * 3 * Fraction(1, 3 * count) if ok else Fraction(0) for ok in mask]
    uniform_exact.append(all(w == (1 if ok else 0) for w, ok in zip(uniform, mask)))
    mask_results.append({'available': list(available), 'eligible_count': count,
                         'sum_mass': sum(mass), 'mean_eligible_weight': sum(weights) / count,
                         'invalid_mass_max': max([mass[s] for s in range(7) if not mask[s]] or [0]),
                         'weight_min': min(weights), 'weight_max': max(weights)})
    index = mask.index(True)
    a_changed = base_a.copy()
    a_changed[index] += 0.55
    changed_pi = joint(a_changed, mask)
    pooled = [sum((0.3 + 0.4 * s - 0.2 * b) * row[b] / sum(row) for b in range(3))
              for s, row in enumerate(pi) if mask[s]]
    changed_pooled = [sum((0.3 + 0.4 * s - 0.2 * b) * row[b] / sum(row) for b in range(3))
                      for s, row in enumerate(changed_pi) if mask[s]]
    baseline_cancellation_errors.append(max(abs(x - y) for x, y in zip(pooled, changed_pooled)))
    h = 1e-5
    plus, minus = base_a.copy(), base_a.copy()
    plus[index] += h
    minus[index] -= h
    paths = {}
    for interaction in [False, True]:
        before = residual(base_a, mask, interaction)
        after = residual(a_changed, mask, interaction)
        empirical = [(x - y) / (2 * h) for x, y in zip(residual(plus, mask, interaction), residual(minus, mask, interaction))]
        ds = [interaction_direction] * 7 if interaction else directions
        coefficient = 0.025 if interaction else 0.05
        analytic = [coefficient * anchors[s] * count * mass[s] * ((1 if s == index else 0) - mass[index]) * ds[s][j]
                    for s in range(7) for j in range(3)]
        paths['I' if interaction else 'M'] = {'change_norm': norm([x - y for x, y in zip(after, before)]),
                                            'gradient_norm': norm(analytic),
                                            'finite_difference_max_error': max(abs(x - y) for x, y in zip(empirical, analytic))}
    gradient_results.append({'eligible_count': count, 'paths': paths})

check('normal_and_six_missing_masks_have_correct_mass_and_scale',
      sorted(set(row['eligible_count'] for row in mask_results)) == [1, 3, 7]
      and all(abs(row['sum_mass'] - 1) < 1e-14 and abs(row['mean_eligible_weight'] - 1) < 1e-14
              and row['invalid_mass_max'] == 0 and 0 <= row['weight_min'] <= row['weight_max'] <= row['eligible_count'] for row in mask_results),
      {'algebra': 'w_S = N_eligible * sum_b pi(S,b); mean eligible w=1; invalid rows=0; 0<=w<=N.', 'results': mask_results})
check('uniform_relation_mass_exactly_reduces_to_v4_algebra', all(uniform_exact),
      {'rational_arithmetic_masks': 7, 'identity': 'N * 3 * (1/(3*N)) = 1 for every eligible relation, zero otherwise.',
       'limits': 'Exact rational identity; no claim of bitwise Torch/AMP equivalence before actual M0.'})
check('old_conditional_pooling_cancels_relation_logits', max(baseline_cancellation_errors) < 1e-14,
      {'maximum_float64_cancellation_error': max(baseline_cancellation_errors),
       'scope': 'Fixed arbitrary values and c/psi, only a_S changed; source diagnostic confirmation, not an efficacy attribution.'})
check('m_and_i_post_normalization_mass_has_a_s_derivative',
      all(all(p['finite_difference_max_error'] < 1e-9
              and ((p['gradient_norm'] > 1e-4 and p['change_norm'] > 1e-4) if row['eligible_count'] > 1
                   else p['gradient_norm'] < 1e-14 and p['change_norm'] < 1e-14)
              for p in row['paths'].values()) for row in gradient_results),
      {'derivative': 'd(N*rho_S)/da_T = N*rho_S*(1[S=T]-rho_T), after direction normalization.',
       'results': gradient_results, 'single_relation_case': 'Zero relation-selection derivative is expected when only one valid relation remains.'})

fuse_src = ast.unparse(function(new_class, 'fuse'))
detach_receivers = [ast.unparse(n.func.value) for n in ast.walk(function(new_class, 'fuse')) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'detach']
calibrator_class = cls(TREES['axis_collaboration.py'], 'ContributionCalibrator')
calibrator_forward = ast.unparse(function(calibrator_class, 'forward'))
check('mass_not_detached_and_outer_gates_remain_independent',
      len(detach_receivers) == 2 and all('.norm(' in item for item in detach_receivers)
      and 'return (prediction, (20 * prediction).sigmoid())' in calibrator_forward
      and 'gates[:, :1]' in fuse_src and 'gates[:, 1:2]' in fuse_src and 'gates[:, 2:]' in fuse_src,
      {'detached_in_fuse': detach_receivers,
       'reason': 'Only base norm anchors are stopped. Relation mass remains on the differentiable forward route. M/F/I gates use independent sigmoid coordinates, not an expert softmax; relation mass redistributes M/I internally.'})

check('initial_scale_bound_and_unchanged_frequency_path',
      'self.residual_scale = nn.Parameter(torch.tensor([0.1, 0.05, 0.05]))' in ast.unparse(function(axis_class, '__init__')),
      {'relation_weight_bounds': {'normal': [0, 7], 'one_modality_missing': [0, 3], 'two_modalities_missing': [1, 1]},
       'initial_full_input_per_relation_bound_at_gates_half': 'M+I norm <= (0.1+0.05)*0.5*7*anchor = 0.525*anchor.',
       'initial_full_input_per_relation_bound_at_gate_at_most_one': 'M+I norm <= 1.05*anchor.',
       'qualification': 'These use initial learned scale values, not a bound after unconstrained training. Uniform relation weights retain the V4 scale. No new numerical risk established by source evidence; actual AMP behavior is untested here.'})

failed = [c for c in CHECKS if not c['passed']]
result = {'status': 'BLOCKED' if failed else 'PASS',
          'review_kind': 'SOURCE_AST_STDLIB_ALGEBRA_ONLY',
          'model': 'gpt-6-astra', 'reasoning_effort': 'max', 'fork_turns': 'none',
          'review_independence': 'same-family', 'acceptance_status': 'provisional',
          'observed_at': datetime.now(timezone.utc).isoformat(),
          'blockers': failed, 'checks_count': len(CHECKS), 'checks': CHECKS,
          'checked_source_sha256': actual_hashes,
          'binding_note': 'Only the supplied plan/new-source hashes and existing active frozen-source receipt were checked; no new integrity scheme was introduced.',
          'unexecuted': ['Torch imports or tensor forward', 'GPU/CUDA/SSH', 'optimizer updates', 'new install or signal', 'prior review suites', 'launcher or real GPU contract'],
          'limitations': ['Static constructor equality is not an actual initialized tensor equality receipt.',
                          'Stdlib analytical gradients are not actual Torch autograd or AMP evidence.',
                          'The source fix does not establish a cause of V4 failure or a V5 retrieval gain.'],
          'trace_path': '.aris/traces/experiment-bridge/2026-10-03_axis_mass_v5_model',
          'harness_attempt': 1, 'unexpected_harness_failures': 0,
          'source_or_handoff_modified': False}
write_json(TRACE / '001-deterministic-results.json', result)
print(json.dumps({'status': result['status'], 'checks_count': len(CHECKS), 'failed_checks': [c['name'] for c in failed]}), flush=True)
