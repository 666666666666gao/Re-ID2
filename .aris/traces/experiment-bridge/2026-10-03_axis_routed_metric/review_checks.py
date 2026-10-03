"""Source and in-memory launch mocks only. No neural imports or external execution."""
import argparse
import ast
import copy
from datetime import datetime
import difflib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_axis_routed_metric'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_routed_metric26_deploy_20261003.py')
FILES = ('routed_supervision_metric_axis.py', 'run_routed_metric_experiment.py', 'verify_routed_metric_supervision.py', 'launch_routed_metric_trial.py')
SOURCES = {name: (ROOT/name).read_text(encoding='utf-8') for name in FILES}
HASHES = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}
HELPER_HASH = hashlib.sha256(HELPER.read_bytes()).hexdigest()
BASELINE = json.loads((ROOT/'results/preflight/axis_metric_interface_launch.json').read_text())['source_sha256']
RESULTS = []

def check(name, condition, evidence):
    assert condition, name
    RESULTS.append(dict(name=name, status='PASS', evidence=evidence))

def tree(name):
    return ast.parse((ROOT/name).read_text(encoding='utf-8'))

def function(module, name):
    return next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == name)

def dump(node):
    return ast.dump(node, include_attributes=False)

def functions_only(module, names):
    return ast.fix_missing_locations(ast.Module(body=[function(module, name) for name in names], type_ignores=[]))

def no_imports(source):
    t = ast.parse(source)
    t.body = [n for n in t.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    t.body = [n for n in t.body if not (isinstance(n, ast.Expr) and ast.unparse(n).startswith('sys.path.insert('))]
    return ast.fix_missing_locations(t)

for name, source in SOURCES.items():
    compile(source, name, 'exec')
compile(HELPER.read_text(encoding='utf-8'), str(HELPER), 'exec')
check('syntax', True, 'Four reviewed sources and deployment helper compile without importing them.')
plan = json.loads((ROOT/'results/preflight/axis_routed_metric_plan.json').read_text())
check('plan_binding', HASHES == plan['source_sha256'], HASHES)
mismatches = [name for name, sha in BASELINE.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != sha]
check('baseline_source_binding', not mismatches, {'count':len(BASELINE),'mismatches':mismatches})

old = tree('run_metric_mass_experiment.py')
new = tree('run_routed_metric_experiment.py')
for name in ('write_json', 'configuration', 'evaluate', 'step'):
    check('trainer_'+name+'_exact', dump(function(old,name)) == dump(function(new,name)), 'AST identical to MetricA.')
class NormalizeTrainer(ast.NodeTransformer):
    def visit_Name(self,node):
        if node.id == 'RoutedSupervisionMetricDeMo':
            node.id = 'MetricMassAxisCollaborationDeMo'
        return node
    def visit_Constant(self,node):
        if node.value == 'axis_metric_routed_fullref':
            node.value = 'axis_metric_fullref'
        return node
    def visit_Assign(self,node):
        if len(node.targets) == 1 and isinstance(node.targets[0],ast.Subscript):
            t=node.targets[0]
            if isinstance(t.value,ast.Name) and t.value.id=='info' and isinstance(t.slice,ast.Constant) and t.slice.value in ('method_revision','unchanged_V5_factors','unchanged_V7_factors','frequency_auxiliary_supervision'):
                return None
        return self.generic_visit(node)
for name in ('build','main'):
    a = NormalizeTrainer().visit(copy.deepcopy(function(old,name)))
    b = NormalizeTrainer().visit(copy.deepcopy(function(new,name)))
    check('trainer_'+name+'_only_expected_changes',dump(a)==dump(b),'Only selected class/variant and method metadata change; optimizer, native AMP, smoke reload, 50-epoch loop and earliest dev selection retained.')

old_forward = function(tree('mass_axis_collaboration.py'),'forward')
new_forward = function(tree('routed_supervision_metric_axis.py'),'forward')
reduced = copy.deepcopy(new_forward)
added = []
for n in list(reduced.body):
    s=ast.unparse(n)
    if (isinstance(n,ast.With) and 'routed_f = ' in s) or s=='output.extend((routed_score, routed_f))':
        added.append(s)
        reduced.body.remove(n)
check('copied_forward_exact_except_new_auxiliary',len(added)==2 and dump(reduced)==dump(old_forward),added)
constructor = function(tree('routed_supervision_metric_axis.py'),'__init__')
check('constructor_only_loss_weights', [ast.unparse(n) for n in constructor.body] == ['super().__init__(classes, cfg, cameras)', 'self.loss_weights[3] = 0.05', 'self.loss_weights.insert(4, 0.05)'], 'No modules, parameters, buffers or RNG draws added.')
check('new_pair_graph', added[0] == "with torch.autocast('cuda', enabled=False):\n    routed_f = F.normalize(self.frequency_projection(frequency.float()), dim=1)\n    routed_score = 32.0 * F.linear(routed_f, F.normalize(self.frequency_classifier.weight.float(), dim=1))",'Differentiable FP32 path from actual conditioned/joint-routed frequency through existing PF; normalized shared classifier weights, scale32, no detach.')
for direct in (False,True):
    weights=[.25,1.,.1,.1]+([1.] if direct else [1.,1.,1.])
    previous=list(weights)
    weights[3]=.05
    weights.insert(4,.05)
    pairs=['fused','base_moe','independent_M','independent_F','routed_PF']+(['original'] if direct else ['RGB','NI','TI'])
    check('output_weight_alignment_'+str(direct),len(weights)==len(pairs) and weights[5:]==previous[4:] and sum(weights[3:5])==previous[3],dict(pairs=pairs,weights=weights,routed_pair_indices=[8,9]))

diff=''.join(difflib.unified_diff((ROOT/'run_metric_mass_experiment.py').read_text().splitlines(True),SOURCES['run_routed_metric_experiment.py'].splitlines(True),fromfile='run_metric_mass_experiment.py',tofile='run_routed_metric_experiment.py'))
diff+='\n'+''.join(difflib.unified_diff(ast.unparse(old_forward).splitlines(True),ast.unparse(new_forward).splitlines(True),fromfile='inherited-forward',tofile='routed-forward'))
(TRACE/'source_diff.txt').write_text(diff,encoding='utf-8')

class MemoryFS:
    def __init__(self):
        self.files={}
        self.dirs={'/'}
        fs=self
        class MemoryPath:
            def __init__(self,path): self.path=str(path).replace('\\','/').rstrip('/') or '/'
            def __str__(self): return self.path
            def __truediv__(self,tail): return MemoryPath(self.path+'/'+str(tail))
            def exists(self): return self.path in fs.files or self.path in fs.dirs
            def read_bytes(self):
                if self.path not in fs.files: raise FileNotFoundError(self.path)
                return fs.files[self.path]
            def read_text(self,encoding='utf-8'): return self.read_bytes().decode(encoding)
            def write_bytes(self,data): fs.files[self.path]=data; return len(data)
            def mkdir(self,exist_ok=False,parents=False):
                if self.exists() and not exist_ok: raise FileExistsError(self.path)
                fs.dirs.add(self.path)
            def open(self,mode):
                assert mode=='x'
                if self.exists(): raise FileExistsError(self.path)
                fs.files[self.path]=b''
                return io.StringIO()
        self.Path=MemoryPath
    def put(self,path,value):
        if not isinstance(value,bytes): value=json.dumps(value).encode()
        self.files[str(path)]=value

launcher_module=tree('launch_routed_metric_trial.py')
execute_module=tree('launch_axis_scaled.py')
launcher_cases = [
    ('success',None,None,3,True),
    ('tensor_process_failure','tensor','exit',1,False),
    ('tensor_bad_receipt','tensor','bad',1,False),
    ('tensor_missing_receipt','tensor','missing',1,False),
    ('smoke_process_failure','smoke','exit',2,False),
    ('smoke_bad_receipt','smoke','bad',2,False),
    ('smoke_missing_receipt','smoke','missing',2,False),
    ('train_process_failure','train','exit',3,False),
    ('train_bad_completion','train','bad',3,False),
    ('train_wrong_budget','train','epochs',3,False),
    ('train_wrong_interface','train','interface',3,False),
    ('busy_gpu_waits_then_success',None,'busy',3,True),
    ('existing_output',None,'existing',0,False),
]
for case,fail_stage,kind,expected_spawns,complete in launcher_cases:
    fs=MemoryFS(); calls=[]; sleeps=[]; written=[]; queried=[]
    root='/trial'
    if kind=='existing': fs.dirs.add(root)
    def write_json(path,value): fs.put(str(path),value); written.append(str(path))
    def gpu_query(argv,**kwargs):
        queried.append(argv)
        assert argv==['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits']
        return '500' if kind=='busy' and len(queried)==1 else '499'
    def popen(argv,**kwargs):
        stage='tensor' if argv[2]=='verify_routed_metric_supervision.py' else argv[argv.index('--mode')+1]
        calls.append(dict(stage=stage,argv=argv,env=kwargs['env']))
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='1'
        output=argv[argv.index('--output')+1]
        value={'status':'PASS_ROUTED_METRIC_SUPERVISION_TENSOR_CONTRACT'} if stage=='tensor' else {'status':'SMOKE_PASS'} if stage=='smoke' else {'status':'COMPLETE','epochs':50,'retrieval_interface':'metric_weighted_frequency_block'}
        suffix='/smoke.json' if stage=='smoke' else '/result.json'
        if stage==fail_stage and kind=='bad': value['status']='FAILED'
        if stage==fail_stage and kind=='epochs': value['epochs']=49
        if stage==fail_stage and kind=='interface': value['retrieval_interface']='wrong'
        if not (stage==fail_stage and kind=='missing'): fs.put(output+suffix,value)
        return SimpleNamespace(pid=100+len(calls),wait=lambda: 7 if stage==fail_stage and kind=='exit' else 0)
    env=dict(argparse=argparse,json=json,Path=fs.Path,sys=SimpleNamespace(executable='/existing/python'),os=SimpleNamespace(environ={}),subprocess=SimpleNamespace(check_output=gpu_query,Popen=popen,STDOUT=-2),time=SimpleNamespace(time=lambda:1,sleep=lambda s:sleeps.append(s)),write_json=write_json)
    exec(compile(functions_only(execute_module,('idle','execute')),'mock_execute','exec'),env)
    exec(compile(functions_only(launcher_module,('main',)),'mock_launcher','exec'),env)
    previous_argv=sys.argv
    sys.argv=['launch','--data-root','/fixed_data','--pretrained','/public_clip','--output',root]
    error=None
    try: env['main']()
    except (AssertionError,FileNotFoundError,FileExistsError) as exc: error=type(exc).__name__+': '+str(exc)
    finally: sys.argv=previous_argv
    assert len(calls)==expected_spawns,(case,calls,error)
    assert (root+'/controller_result.json' in fs.files)==complete,(case,error)
    assert (error is None)==complete,(case,error)
    assert sleeps==([240] if kind=='busy' else []),(case,sleeps)
    if kind=='exit':
        stage_name='MSVR310_axis_metric_routed_fullref_s42' if fail_stage=='train' else fail_stage
        exit_rows=[json.loads(data) for key,data in fs.files.items() if key.endswith('/'+stage_name+'_exit.json')]
        assert len(exit_rows)==1 and exit_rows[0]['exit_code']==7
    check('launcher_'+case,True,dict(spawns=[c['stage'] for c in calls],wait_seconds=sleeps,terminal_written=complete,exception=error,exit_receipts_preserved=True))

helper_source=HELPER.read_text(encoding='utf-8')
audit_paths={
 'results/preflight/axis_p0_utility_analysis.json':'PASS_ACTUAL_FOUR_CASE_P0_EVIDENCE_AUDIT',
 'results/preflight/axis_metric_msvr_development_analysis.json':'PASS_EVIDENCE_AUDIT',
 'results/preflight/axis_metric_msvr_frozen_analysis.json':'PASS_ACTUAL_FROZEN_EVIDENCE_AUDIT'}
for path,status in audit_paths.items():
    check('prior_'+Path(path).stem,json.loads((ROOT/path).read_text())['status']==status,status)

helper_cases=(
 'success','review_blocked','review_has_blocker','review_source_stale','helper_stale','local_baseline_stale',
 'prior_p0_bad','prior_development_bad','prior_frozen_bad','remote_baseline_stale','remote_development_incomplete',
 'remote_frozen_incomplete','remote_new_source_exists','remote_output_exists','initial_gpu_busy','initial_disk_at_floor',
 'second_gpu_busy','second_disk_at_floor','staged_source_corrupt','scp_failure','existing_remote_log')
for case in helper_cases:
    fs=MemoryFS(); calls=[]; stages=[]; spawns=[]
    project=str(ROOT).replace('\\','/'); remote='/remote/project'; output=remote+'/runs/axis_collaboration_v8_routed_metric_trial'
    review={'status':'PASS','blockers':[],'helper_sha256':HELPER_HASH,'checked_source_sha256':dict(HASHES)}
    if case=='review_blocked': review['status']='BLOCKED'
    if case=='review_has_blocker': review['blockers']=['mock blocker']
    if case=='review_source_stale': review['checked_source_sha256'][FILES[0]]='stale'
    if case=='helper_stale': review['helper_sha256']='stale'
    fs.put(project+'/results/preflight/axis_routed_metric_review.json',review)
    fs.put(str(HELPER).replace('\\','/'),HELPER.read_bytes())
    fs.put(project+'/results/preflight/axis_metric_interface_launch.json',{'source_sha256':BASELINE})
    for name in set(BASELINE)|set(FILES): fs.put(project+'/'+name,(ROOT/name).read_bytes())
    for name in BASELINE: fs.put(remote+'/'+name,(ROOT/name).read_bytes())
    for name in audit_paths: fs.put(project+'/'+name,(ROOT/name).read_bytes())
    for suffix in ('interface','frozen'):
        fs.put(remote+'/runs/axis_collaboration_v7_metric_'+suffix+'_trial/controller_result.json',{'status':'COMPLETE'})
    if case=='local_baseline_stale': fs.put(project+'/'+next(iter(BASELINE)),b'changed')
    if case=='remote_baseline_stale': fs.put(remote+'/'+next(iter(BASELINE)),b'changed')
    for short,path in zip(('p0','development','frozen'),audit_paths):
        if case=='prior_'+short+'_bad': fs.put(project+'/'+path,{'status':'FAILED'})
    if case=='remote_development_incomplete': fs.put(remote+'/runs/axis_collaboration_v7_metric_interface_trial/controller_result.json',{'status':'RUNNING'})
    if case=='remote_frozen_incomplete': fs.put(remote+'/runs/axis_collaboration_v7_metric_frozen_trial/controller_result.json',{'status':'RUNNING'})
    if case=='remote_new_source_exists': fs.put(remote+'/'+FILES[0],b'prior')
    if case=='remote_output_exists': fs.dirs.add(output)
    if case=='existing_remote_log': fs.put(output+'.log',b'prior log')
    def command(argv):
        assert argv[0]=='scp'
        stages.append(argv)
        if case=='scp_failure': raise RuntimeError('mock scp failed')
        src,dest=argv[-2:]
        assert dest.startswith('2026:')
        data=fs.Path(src).read_bytes()
        if case=='staged_source_corrupt' and len(stages)==1: data=b'corrupt staging'
        fs.put(dest.split(':',1)[1],data)
    def remote_python(host,source):
        assert host=='2026'
        calls.append(source)
        boundary='initial' if len(calls)==1 else 'second'
        def gpu_query(argv,**kwargs):
            assert argv==['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits']
            return '500' if case==boundary+'_gpu_busy' else '499'
        def popen(argv,**kwargs):
            assert argv[2]=='launch_routed_metric_trial.py'
            assert kwargs['cwd'].path==remote and kwargs['start_new_session'] is True
            assert argv[argv.index('--output')+1]==output
            spawns.append(dict(argv=argv,env=kwargs['env']))
            return SimpleNamespace(pid=9876)
        stdout=[]
        env=dict(hashlib=hashlib,json=json,Path=fs.Path,subprocess=SimpleNamespace(check_output=gpu_query,Popen=popen,STDOUT=-2),shutil=SimpleNamespace(disk_usage=lambda path:SimpleNamespace(free=3200000000 if case==boundary+'_disk_at_floor' else 3200000001)),os=SimpleNamespace(environ={}),time=SimpleNamespace(time=lambda:1),print=lambda value:stdout.append(value))
        exec(compile(no_imports(source),'mock_remote_'+boundary,'exec'),env)
        return '\n'.join(stdout)
    env=dict(datetime=datetime,hashlib=hashlib,json=json,Path=fs.Path,PROJECT=fs.Path(project),HOSTS={'2026':(remote,'/existing/remote/python')},OPTIONS=[],command=command,remote_python=remote_python,__file__=str(HELPER).replace('\\','/'),print=lambda *args,**kwargs:None)
    error=None
    try: exec(compile(no_imports(helper_source),'mock_helper','exec'),env)
    except (AssertionError,FileNotFoundError,FileExistsError,RuntimeError) as exc: error=type(exc).__name__+': '+str(exc)
    success=case=='success'
    assert len(spawns)==int(success),(case,spawns,error)
    assert (project+'/results/preflight/axis_routed_metric_launch.json' in fs.files)==success,(case,error)
    assert (error is None)==success,(case,error)
    if success:
        assert len(calls)==2 and len(stages)==4
        receipt=json.loads(fs.files[project+'/results/preflight/axis_routed_metric_launch.json'])
        assert receipt['source_sha256']=={**BASELINE,**HASHES} and receipt['gpu']==1
        assert receipt['budget_epochs']==50 and receipt['planned_engineering_updates']==3 and receipt['official_test_uses']==0
    check('helper_'+case,True,dict(mock_remote_calls=len(calls),staged_files=len(stages),spawns=len(spawns),launch_receipt_written=success,exception=error))

result=dict(status='PASS_STDLIB_SOURCE_AND_MOCK_CHECKS',neural_execution=False,ssh_calls=0,gpu_queries=0,package_installations=0,checked_source_sha256=HASHES,helper_sha256=HELPER_HASH,checks=RESULTS)
(TRACE/'checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=result['status'],checks=len(RESULTS),neural_execution=False,ssh_calls=0,gpu_queries=0)))
