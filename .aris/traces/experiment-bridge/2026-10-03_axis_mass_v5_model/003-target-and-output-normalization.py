"""One narrow harness correction plus final normalized-descriptor algebra."""
import ast
import itertools
import json
import math
from pathlib import Path

ROOT = Path(r'C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = Path(__file__).resolve().parent
original = json.loads((TRACE / '001-deterministic-results.json').read_text(encoding='utf-8'))
diagnosis = json.loads((TRACE / '002-target-harness-diagnosis.json').read_text(encoding='utf-8'))
assert [x['name'] for x in original['checks'] if not x['passed']] == ['full11_target_stopped_single_reference_and_shared_indices']
assert diagnosis['iterator_string'] is False
tree = ast.parse((ROOT / 'scaled_axis_collaboration.py').read_text(encoding='utf-8'))
target = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'full_reference_targets')
score = next(n for n in target.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'scores')
generator = score.value.generators[0]
passes = (isinstance(score.value, ast.DictComp) and isinstance(generator.target, ast.Tuple)
          and [n.id for n in generator.target.elts] == ['key', 'value']
          and ast.unparse(generator.iter) == 'states.items()'
          and diagnosis['decorator'] == ['torch.no_grad()'] and diagnosis['single_reference']
          and diagnosis['pos_count'] == diagnosis['neg_count'] == 1
          and diagnosis['direction'] and diagnosis['parent_assignment'])
corrected = {'name': 'full11_target_stopped_single_reference_and_shared_indices', 'passed': passes,
             'evidence': {'source': 'scaled_axis_collaboration.py:18',
                          'reason': 'Entire target function is no_grad; one current-batch full11 hard-positive/negative selection and common direction are reused by the scores comprehension for all four states.',
                          'harness_correction': 'Inspect the comprehension tuple AST instead of assuming ast.unparse omits parentheses. Python 3.10 emits for (key, value) in states.items().',
                          'preserved_first_failure': '001-deterministic-results.json and 001-harness.stdout.log',
                          'primary_diagnosis': '002-target-harness-diagnosis.json',
                          'source_changed': False, 'prior_suite_rerun': False}}


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def normalize(v):
    length = norm(v)
    return [x / length for x in v]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


relations = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
a = [-0.6, 0.3, 0.8, -0.2, 0.45, -0.7, 1.1]
c = [0.2, -0.35, 0.45]
psi = [[0.13 * math.sin((s + 1) * (b + 2)) for b in range(3)] for s in range(7)]
directions = [normalize([math.sin(s + 1), math.cos(s + 2), 0.4 + 0.1 * s]) for s in range(7)]
interaction = normalize([0.7, -0.8, 0.5])
anchors = [1 + s * 0.15 for s in range(7)]
base = [0.3 + 0.17 * math.cos(i * 0.7) for i in range(33)]
base[-3:] = [0, 0, 0]
frequency = [0.016, -0.011, 0.018]


def descriptor(logits, mask):
    entries = [logits[s] + c[b] + psi[s][b] if mask[s] else -math.inf for s in range(7) for b in range(3)]
    peak = max(entries)
    exps = [math.exp(v - peak) for v in entries]
    denominator = sum(exps)
    rho = [sum(exps[3 * s:3 * s + 3]) / denominator for s in range(7)]
    result = base.copy()
    for s in range(7):
        for j in range(3):
            result[9 + 3 * s + j] += (0.05 * directions[s][j] + 0.025 * interaction[j]) * anchors[s] * sum(mask) * rho[s]
    result[-3:] = frequency
    return result, rho


results = []
for available in itertools.product([False, True], repeat=3):
    if not any(available):
        continue
    mask = [all(available[k] for k in group) for group in relations]
    count = sum(mask)
    index = mask.index(True)
    z, rho = descriptor(a, mask)
    y = normalize(z)
    dz = [0.0] * 33
    for s in range(7):
        for j in range(3):
            dz[9 + 3 * s + j] = (0.05 * directions[s][j] + 0.025 * interaction[j]) * anchors[s] * count * rho[s] * ((1 if s == index else 0) - rho[index])
    dy = [(d - value * dot(y, dz)) / norm(z) for d, value in zip(dz, y)]
    h = 1e-5
    plus, minus, changed = a.copy(), a.copy(), a.copy()
    plus[index] += h
    minus[index] -= h
    changed[index] += 0.55
    empirical = [(p - m) / (2 * h) for p, m in zip(normalize(descriptor(plus, mask)[0]), normalize(descriptor(minus, mask)[0]))]
    distance = norm([p - m for p, m in zip(normalize(descriptor(changed, mask)[0]), y)])
    results.append({'eligible_count': count, 'normalized_descriptor_change': distance,
                    'normalized_descriptor_gradient_norm': norm(dy),
                    'finite_difference_max_error': max(abs(p - m) for p, m in zip(dy, empirical))})
normalized_check = {'name': 'later_global_descriptor_normalization_does_not_generically_erase_mass',
                    'passed': all(row['finite_difference_max_error'] < 1e-9
                                  and (row['normalized_descriptor_change'] > 1e-4
                                       and row['normalized_descriptor_gradient_norm'] > 1e-4 if row['eligible_count'] > 1
                                       else row['normalized_descriptor_change'] < 1e-14 and row['normalized_descriptor_gradient_norm'] < 1e-14)
                                  for row in results),
                    'evidence': {'kind': 'Independent stdlib vector algebra with fixed nonzero original prefix and fixed F path; not neural execution.',
                                 'derivative': 'For y=z/||z||, dy/da=(I-y*y^T)*(dz/da)/||z||.',
                                 'results': results,
                                 'limit': 'Shows a valid nonzero post-normalization derivative path, not that actual training gradients or efficacy are nonzero.'}}
output = {'status': 'PASS' if corrected['passed'] and normalized_check['passed'] else 'BLOCKED',
          'corrected_check': corrected, 'additional_check': normalized_check,
          'prior_full_harness_not_rerun': True, 'original_failure_preserved': True}
(TRACE / '003-narrow-check-results.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
print(json.dumps(output, indent=2))
