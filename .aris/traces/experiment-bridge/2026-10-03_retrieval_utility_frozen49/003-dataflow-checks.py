"""Stdlib-only execution of isolated grid and class-control AST fragments."""
import ast
from contextlib import nullcontext
import hashlib
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_retrieval_utility_frozen49'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_retrieval_utility_frozen26_deploy_20261003.py')
checks = []


def tree(name):
    return ast.parse((ROOT / name).read_text(encoding='utf-8'), feature_version=(3, 10))


def function(module, name):
    return next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name)


def isolated(nodes, title):
    return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), title, 'exec')


def assignment(module, name):
    return next(n for n in module.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


class Feature:
    def __init__(self, retained):
        self.retained = retained
        self.rows = tuple((retained, i) for i in range(8))
        self.shape = (8, 5632)

    def __getitem__(self, query):
        return tuple(self.rows[i] for i in query)


MISSING = ast.literal_eval(assignment(tree('missing_evaluation.py'), 'MISSING').value)
main = function(tree('missing_retrieval_utility_development.py'), 'main')
begin = main.body.index(assignment(main, 'features'))
end = next(i for i in range(begin, len(main.body)) if isinstance(main.body[i], ast.Assert) and ast.unparse(main.body[i].test) == 'len(measurements) == 49')
calls = []
distance_calls = []
measurements = {}
queries = (1, 5, 7)
dev = object()
model = object()
cfg = object()
normal = object()
keys = ('RGB', 'NI', 'TI')
labels = {'RGB': 'R', 'NI': 'N', 'TI': 'T'}


def retained(missing):
    return ''.join(labels[k] for k in keys if k not in missing)


def extract_missing(model_arg, records, cfg_arg, seed, missing):
    assert model_arg is model and records is dev and cfg_arg is cfg and seed == 42
    calls.append(tuple(missing))
    return Feature(retained(missing)), {'fixture': True}


def distance(q, g):
    assert tuple(i for _, i in q) == queries
    distance_calls.append((q[0][0], g.retained))
    return (q, g.rows)


def measure(name, distances, missing_query, missing_gallery):
    qname, gname = retained(missing_query), retained(missing_gallery)
    assert name == 'q_' + qname + '_g_' + gname
    assert name not in measurements
    if qname == gname == 'RNT':
        assert distances is normal
    else:
        q, g = distances
        assert q == tuple((qname, i) for i in queries)
        assert g == tuple((gname, i) for i in range(8))
    measurements[name] = [missing_query, missing_gallery]


namespace = dict(clean=Feature('RNT'), MISSING=MISSING, extract_missing=extract_missing,
                 model=model, dev=dev, cfg=cfg, arguments=SimpleNamespace(seed=42),
                 torch=SimpleNamespace(isfinite=lambda value: SimpleNamespace(all=lambda: True)),
                 diagnostics={}, query=queries, saved={'distances': normal}, distance=distance,
                 measure=measure, measurements=measurements)
exec(isolated(main.body[begin:end + 1], 'isolated_49_grid_AST'), namespace)
sets = ('R', 'N', 'T', 'RN', 'RT', 'NT', 'RNT')
assert len(calls) == 6 and set(calls) == set(MISSING.values())
assert set(measurements) == {'q_' + q + '_g_' + g for q, g in itertools.product(sets, repeat=2)}
assert len(distance_calls) == 48 and ('RNT', 'RNT') not in distance_calls
checks.append(dict(name='actual_grid_AST_seven_banks_49_pairs', status='PASS', evidence={
    'clean_bank_already_extracted_before_fragment': 1, 'additional_extractions': len(calls),
    'cartesian_pairs': len(measurements), 'fresh_distance_calls': len(distance_calls),
    'saved_distance_pairs': ['q_RNT_g_RNT'], 'query_sample_order': list(queries),
    'gallery_sample_order': list(range(8)), 'per_sample_descriptors_reused': True,
    'numeric_feature_claim': False}))

# Exercise the actual subclass methods with a parent sentinel, never neural code.
result_object = object()
states_object = object()


class Parent:
    def forward(self, *args, **kwargs):
        self.forward_inputs = (args, kwargs)
        return result_object

    def controlled_states(self, *args, **kwargs):
        self.state_inputs = (args, kwargs)
        return states_object


def forbidden_gain(*args, **kwargs):
    raise AssertionError('gain called in eval')


cls = next(n for n in tree('retrieval_utility_axis.py').body if isinstance(n, ast.ClassDef))
namespace = dict(RelationFrequencyInterfaceDeMo=Parent, joint_retrieval_gain=forbidden_gain)
exec(isolated([cls], 'actual_subclass_AST_sentinel_parent'), namespace)
instance = namespace['RetrievalUtilityDeMo']()
instance.training = False
assert instance.forward('images', cam_label='camera', view_label='scene') is result_object
assert instance.controlled_states('base', key='fixture') is states_object
assert not hasattr(instance, '_gain_states') and not hasattr(instance, 'gain_audit')
checks.append(dict(name='actual_subclass_eval_control_flow_identity', status='PASS', evidence=
    'Compiled only actual class AST with sentinel parent: eval returns the same parent output/state objects and cannot enter gain or save training state. No torch/numpy imports or arithmetic.'))

# Symbols track dependency provenance through the current B controlled_states.
class Symbol:
    def __init__(self, provenance):
        self.provenance = frozenset(provenance)

    def float(self): return self
    def squeeze(self, *a): return self
    def softmax(self, *a): return self
    def sum(self, *a, **kw): return self
    def masked_fill(self, mask, value): return Symbol(self.provenance | mask.provenance)
    def __getitem__(self, index): return self
    def __invert__(self): return self
    def __mul__(self, other): return Symbol(self.provenance | other.provenance)


calibration_calls = []
fuse_calls = []
base, m0, f0, eligible, available, full = [Symbol({key}) for key in ('base', 'm0', 'f0', 'eligible', 'available', 'full')]


def pool_m(value, mask):
    assert value is m0 and mask is eligible
    return Symbol({'m0'})


def pool_f(value, mask, structured):
    assert value is f0 and mask is available and structured is True
    return Symbol({'f0'})


def calibrator(base_arg, m, f):
    assert base_arg is base
    calibration_calls.append((m.provenance, f.provenance))
    return Symbol({'prediction'}), Symbol(m.provenance | f.provenance | {'base'})


def by_relation(value):
    assert value is f0
    return value


def fuse(base_arg, m, f, gates, use_m, use_f, mask, *extras):
    assert base_arg is base and mask is eligible
    fuse_calls.append(dict(m=m.provenance, f=f.provenance, gates=gates.provenance,
                           use_m=use_m, use_f=use_f, extras=[e.provenance for e in extras]))
    return Symbol({'state10' if use_m else 'state01'})


subject = SimpleNamespace(modality_expert=SimpleNamespace(pool=pool_m),
    frequency_expert=SimpleNamespace(pool=pool_f, pool_score=lambda v: v),
    router=SimpleNamespace(by_relation=by_relation, band_score=lambda v: v),
    calibrator=calibrator, structured=True, fuse=fuse)
method = function(tree('relation_frequency_interface.py'), 'controlled_states')
namespace = dict(torch=SimpleNamespace(zeros_like=lambda value: Symbol({'zero'}),
                 autocast=lambda *a, **kw: nullcontext(), inf=float('inf')))
exec(isolated([method], 'actual_B_controlled_states_AST_symbolic'), namespace)
states = namespace['controlled_states'](subject, base, m0, f0, eligible, available, full)
assert states['00'] is base and states['11'] is full
assert calibration_calls == [(frozenset({'m0'}), frozenset({'zero'})), (frozenset({'zero'}), frozenset({'f0'}))]
assert [(c['use_m'], c['use_f']) for c in fuse_calls] == [(True, False), (False, True)]
assert fuse_calls[0]['m'] == {'m0'} and fuse_calls[0]['f'] == {'zero'}
assert fuse_calls[1]['m'] == {'zero'} and fuse_calls[1]['f'] == {'f0'}
assert all(p <= {'f0', 'eligible'} for p in fuse_calls[1]['extras'])
assert not fuse_calls[0]['extras']
mass = function(tree('mass_axis_collaboration.py'), 'fuse')
joint_if = next(n for n in ast.walk(mass) if isinstance(n, ast.If) and ast.unparse(n.test) == 'use_m and use_f')
assert 'self.interaction_projection' in ast.unparse(joint_if)
assert ast.unparse(mass).count('self.interaction_projection') == 1
checks.append(dict(name='current_B_closed_state_dataflow', status='PASS', evidence=
    'Actual B controlled_states AST executed symbolically: state10 uses only m0+zeroF and state01 only zeroM+f0; F relation weights depend only on f0/eligibility. Router conditions/route/joint APIs absent from mock, so such calls would fail. Source confirms interaction projection is inside use_m and use_f. No numeric claim.'))

# Render but never execute remote snippets; all f-string inputs are local metadata.
launch = json.loads((ROOT/'results/preflight/retrieval_utility_launch.json').read_text(encoding='utf-8'))
helper_tree = ast.parse(HELPER.read_text(encoding='utf-8'), feature_version=(3, 10))
root = '/data/gaob/Re-ID/DeMo-DualAxis'
sources = launch['source_sha256']
files = ('missing_retrieval_utility_development.py', 'diagnose_retrieval_utility_axis.py', 'launch_retrieval_utility_frozen_evaluation.py')
new_sources = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
training = launch['output'] + '/development'
output = root + '/runs/axis_collaboration_v10_retrieval_utility_frozen_trial'
context = dict(root=root, sources=sources, files=files, output=output,
               first=training+'/MSVR310_axis_retrieval_utility_fullref_s42',
               all_sources={**sources, **new_sources}, availability={'first_inputs': {}},
               python='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python', training=training)
for key in ('guard', 'start'):
    expression = ast.Expression(assignment(helper_tree, key).value)
    rendered = eval(compile(expression, 'render_only_' + key, 'eval'), context)
    ast.parse(rendered, feature_version=(3, 10))
    (TRACE / ('003-generated-remote-' + key + '.py')).write_text(rendered, encoding='utf-8')
checks.append(dict(name='generated_remote_python310_syntax', status='PASS', evidence=
    'Both helper f-strings rendered from local metadata and parsed under Python3.10 grammar; neither snippet executed as a remote process.'))
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
(TRACE/'003-dataflow-checks.json').write_text(json.dumps(dict(status='PASS', checks=checks,
    no_neural_imports=True, actual_GPU_queries=0, actual_SSH_calls=0, actual_launches=0), indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(status='PASS', dataflow_checks=len(checks), no_neural_imports=True)))
