"""Narrow source review: only stdlib, AST and in-memory deployment mocks."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import traceback
from types import SimpleNamespace

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = PROJECT / '.aris/traces/experiment-bridge/2026-10-03_axis_mass_v5_workflow'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_v5_mass_deploy_20261003.py')
NAMES = ('mass_axis_collaboration.py', 'run_mass_experiment.py', 'verify_axis_mass.py')
PREFIX = PROJECT / 'results/preflight'
helper_bytes = HELPER.read_bytes()
helper_hash = hashlib.sha256(helper_bytes).hexdigest()
helper_tree = ast.parse(helper_bytes.decode('utf-8'))
sources = {name: (PROJECT / name).read_bytes() for name in NAMES}
hashes = {name: hashlib.sha256(value).hexdigest() for name, value in sources.items()}
prior = json.loads((PREFIX / 'axis_collaboration_v4_missing_development26_repaired_launch.json').read_text(encoding='utf-8'))
analysis = json.loads((PREFIX / 'axis_collaboration_v4_missing_development26_complete_analysis.json').read_text(encoding='utf-8'))
collect = ast.parse((PROJECT / 'collect_results.py').read_text(encoding='utf-8'))
hosts = ast.literal_eval(next(n.value for n in collect.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'HOSTS' for t in n.targets)))
ROOT, PYTHON = hosts['2026']
OUTPUT = ROOT + '/runs/axis_collaboration_v5_mass'
checks, failures = [], []


def require(value, message='assertion failed'):
    assert value, message


def rejects(fn):
    try:
        fn()
    except (AssertionError, FileExistsError):
        return
    raise AssertionError('expected rejection not raised')


def strip(tree):
    return [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]


def execute(nodes, ns):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<reviewed AST>', 'exec'), ns)


def check(name, fn):
    try:
        detail = fn()
        result = dict(name=name, status='PASS', detail=detail)
    except Exception as error:
        result = dict(name=name, status='FAIL', detail=f'{type(error).__name__}: {error}')
        failures.append(traceback.format_exc())
    checks.append(result)
    print(json.dumps(result), flush=True)


class FS:
    def __init__(self):
        self.files, self.dirs, self.writes = {}, set(), []
    def path(self, value):
        return VPath(self, str(value).replace('\\', '/'))
    def put(self, path, value):
        self.files[str(path)] = value.encode() if isinstance(value, str) else value
        self.dirs.add(str(PurePosixPath(str(path)).parent))
    def put_json(self, path, value):
        self.put(str(path), json.dumps(value))


class VPath:
    def __init__(self, fs, value):
        self.fs, self.value = fs, str(PurePosixPath(value))
    def __str__(self): return self.value
    def __truediv__(self, value): return VPath(self.fs, str(PurePosixPath(self.value) / str(value)))
    def read_bytes(self): return self.fs.files[self.value]
    def read_text(self, **kwargs): return self.read_bytes().decode('utf-8')
    def exists(self): return self.value in self.fs.files or self.value in self.fs.dirs
    def mkdir(self, exist_ok=False):
        if self.exists() and not exist_ok: raise FileExistsError(self.value)
        self.fs.dirs.add(self.value)
        self.fs.writes.append(self.value)
    def write_bytes(self, value):
        self.fs.put(self.value, value)
        self.fs.writes.append(self.value)
    def write_text(self, value, **kwargs): self.write_bytes(value.encode('utf-8'))
    def open(self, mode):
        require(mode == 'x')
        if self.exists(): raise FileExistsError(self.value)
        self.write_bytes(b'')
        return io.StringIO()


class Deploy:
    def __init__(self):
        self.local, self.remote = FS(), FS()
        self.events, self.codes, self.spawns = [], [], []
        self.live, self.used = False, {0: 15, 1: 60, 2: 12591, 3: 153}
        self.after_upload = lambda: None
        self.model_review = dict(status='PASS', blockers=[], checked_source_sha256={n: hashes[n] for n in NAMES[:2]})
        self.workflow_review = dict(status='PASS', blockers=[], helper_sha256=helper_hash, checked_source_sha256=hashes.copy())
        self.local.put(HELPER.as_posix(), helper_bytes)
        for name, value in sources.items(): self.local.put((PROJECT / name).as_posix(), value)
        self.local.put_json((PREFIX / 'axis_collaboration_v4_missing_development26_repaired_launch.json').as_posix(), prior)
        self.local.put_json((PREFIX / 'axis_collaboration_v4_missing_development26_complete_analysis.json').as_posix(), analysis)
        for name in prior['source_sha256']: self.remote.put(ROOT + '/' + name, (PROJECT / name).read_bytes())
        self.remote.put_json(prior['output'] + '/controller_result.json', dict(status='COMPLETE', conditions=78, optimizer_updates=0))
    def remote_python(self, host, code):
        require(host == '2026')
        stage = 'guard' if not self.codes else 'launch'
        self.events.append(stage)
        self.codes.append(code)
        output = []
        def ps(argv, **kwargs):
            require(argv == ['ps', '-p', str(prior['pid']), '-o', 'pid='])
            return SimpleNamespace(stdout=str(prior['pid']) if self.live else '')
        def gpu(argv, **kwargs):
            require(argv == ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'])
            return ''.join(f'{i}, {v}\n' for i, v in self.used.items())
        def spawn(argv, **kwargs):
            self.spawns.append((argv, kwargs))
            return SimpleNamespace(pid=12345)
        ns = dict(hashlib=hashlib, json=json, Path=self.remote.path,
                  subprocess=SimpleNamespace(run=ps, check_output=gpu, Popen=spawn, STDOUT=-2), print=output.append)
        execute(strip(ast.parse(code)), ns)
        return output[-1]
    def command(self, argv):
        name = NAMES[len([e for e in self.events if e == 'upload'])]
        require(argv == ['scp', 'MOCK_OPTIONS', (PROJECT / name).as_posix(), '2026:' + ROOT + '/' + name])
        self.events.append('upload')
        self.remote.path(ROOT + '/' + name).write_bytes(sources[name])
        if name == NAMES[-1]: self.after_upload()
        return ''
    def run(self):
        self.local.put_json((PREFIX / 'axis_collaboration_v5_mass_model_review.json').as_posix(), self.model_review)
        self.local.put_json((PREFIX / 'axis_collaboration_v5_mass_workflow_review.json').as_posix(), self.workflow_review)
        ns = dict(datetime=datetime, hashlib=hashlib, json=json, Path=self.local.path,
                  sys=SimpleNamespace(path=[]), PROJECT=self.local.path(PROJECT.as_posix()), HOSTS=hosts,
                  OPTIONS=['MOCK_OPTIONS'], command=self.command, remote_python=self.remote_python,
                  __file__=HELPER.as_posix(), print=lambda *a, **k: None)
        execute(strip(helper_tree), ns)
        self.ns = ns
        return self


def zero_mutation(h):
    require(not h.local.writes and not h.remote.writes and not h.spawns and 'upload' not in h.events)


def preserved_sources():
    require(len(prior['source_sha256']) == 15)
    require(all(hashlib.sha256((PROJECT / n).read_bytes()).hexdigest() == h for n, h in prior['source_sha256'].items()))
    require(analysis['status'] == 'PASS_ACTUAL_78_FROZEN_DEVELOPMENT_CONDITIONS' and analysis['conditions'] == 78 and analysis['optimizer_updates'] == 0)
    require(ROOT == '/data/gaob/Re-ID/DeMo-DualAxis' and PYTHON == '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python')
    return 'All existing 15 sources byte-identical; actual local 78-condition analysis complete; exact physical 2026 warm paths.'


happy = None
def happy_sequence():
    global happy
    happy = Deploy().run()
    require(happy.events == ['guard', 'upload', 'upload', 'upload', 'launch'])
    require(len(happy.spawns) == 1)
    argv, kw = happy.spawns[0]
    require(argv == [PYTHON, '-u', OUTPUT + '/controller.py'] and str(kw['cwd']) == ROOT and kw['start_new_session'])
    require(happy.remote.writes[:3] == [ROOT + '/' + n for n in NAMES])
    require(not set(happy.remote.writes) & {ROOT + '/' + n for n in prior['source_sha256']})
    require(happy.ns['receipt']['budget_epochs'] == 150 and happy.ns['receipt']['jobs'] == {'0':'RGBNT100','1':'RGBNT201','3':'MSVR310'})
    (TRACE / 'generated_scripts.json').write_text(json.dumps(dict(guard=happy.ns['guard'], launch=happy.ns['launch'], controller=happy.ns['controller']), indent=2), encoding='utf-8')
    return 'One read-only guard, exactly three new uploads, source recheck, exclusive new output and one native Python controller; GPU2 remains busy and unused.'


def local_review_gates():
    for kind, field in [('model','status'),('workflow','status'),('model','blockers'),('workflow','blockers'),('workflow','helper_sha256')]:
        h=Deploy(); review=h.model_review if kind=='model' else h.workflow_review
        review[field] = ['blocking'] if field=='blockers' else 'INVALID'
        rejects(h.run); zero_mutation(h); require(not h.codes)
    for kind in ('model','workflow'):
        for name in NAMES[:2] if kind=='model' else NAMES:
            h=Deploy(); review=h.model_review if kind=='model' else h.workflow_review
            review['checked_source_sha256'][name]='changed'
            rejects(h.run); zero_mutation(h); require(not h.codes)
    return 'PASS/blocker/helper and all five checked-source bindings independently reject before any remote call.'


def predecessor_gates():
    h=Deploy(); h.live=True; rejects(h.run); zero_mutation(h)
    for field, bad in [('status','RUNNING'),('conditions',77),('optimizer_updates',1)]:
        h=Deploy(); row=dict(status='COMPLETE',conditions=78,optimizer_updates=0); row[field]=bad
        h.remote.put_json(prior['output']+'/controller_result.json',row)
        rejects(h.run); zero_mutation(h)
    for field, bad in [('status','NOT_COMPLETE'),('conditions',77)]:
        h=Deploy(); row=copy.deepcopy(analysis); row[field]=bad
        h.local.put_json((PREFIX/'axis_collaboration_v4_missing_development26_complete_analysis.json').as_posix(),row)
        rejects(h.run); zero_mutation(h); require(not h.codes)
    return 'Live old PID, incomplete/wrong-budget remote predecessor and incomplete local 78 analysis all reject without mutations.'


def resource_and_exclusive_gates():
    for gpu in (0,1,3):
        h=Deploy(); h.used[gpu]=500; rejects(h.run); zero_mutation(h)
    for path in [OUTPUT]+[ROOT+'/'+n for n in NAMES]:
        h=Deploy(); h.remote.dirs.add(path); rejects(h.run); zero_mutation(h)
    return 'Each selected GPU at threshold500 and every existing destination rejects before uploads; busy GPU2 accepted by happy case.'


def source_guard_rechecks():
    for name in prior['source_sha256']:
        h=Deploy(); h.remote.put(ROOT+'/'+name,b'changed'); rejects(h.run); zero_mutation(h)
    for name in (NAMES[0], 'run_experiment.py'):
        h=Deploy(); h.after_upload=lambda h=h,name=name: h.remote.put(ROOT+'/'+name,b'changed')
        rejects(h.run); require(not h.spawns and OUTPUT not in h.remote.dirs)
    return 'All15 original hashes enforce first guard; original/new tampering after uploads rejects before output creation/spawn.'


class Controller:
    def __init__(self, fault=None):
        self.fs=FS(); self.fs.dirs.add(OUTPUT)
        for name in prior['source_sha256']: self.fs.put(ROOT+'/'+name,(PROJECT/name).read_bytes())
        for name,value in sources.items(): self.fs.put(ROOT+'/'+name,value)
        self.events,self.children,self.rows,self.submissions=[],[],[],[]
        self.fault=fault
        self.used={0:[15],1:[60],3:[153]}
    def idle(self,gpu):
        # Execute the actual unchanged V4 idle function with fake nvidia-smi/time.
        fn=next(n for n in ast.parse((PROJECT/'launch_axis_scaled.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='idle')
        def output(argv,**kw):
            require(argv == ['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'])
            values=self.used[gpu]; value=values.pop(0) if len(values)>1 else values[0]
            return str(value)
        ns=dict(subprocess=SimpleNamespace(check_output=output),time=SimpleNamespace(sleep=lambda s:self.events.append(('sleep',s,gpu))),print=lambda *a,**kw:None)
        execute([fn],ns); ns['idle'](gpu)
        self.events.append(('idle',gpu))
    def write(self,path,value):
        self.fs.path(str(path)).write_text(json.dumps(value))
        self.rows.append((str(path),value)); self.events.append(('write',str(path)))
    def spawn(self,argv,**kwargs):
        require(argv[0]==PYTHON and kwargs['stderr']==-2)
        gpu=int(kwargs['env']['CUDA_VISIBLE_DEVICES']); require(gpu in (0,1,3))
        require(all(kwargs['env'][n]=='4' for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')))
        dataset=argv[argv.index('--dataset')+1]; target=argv[argv.index('--output')+1]
        mode='tensor' if argv[2]=='verify_axis_mass.py' else argv[argv.index('--mode')+1]
        if mode!='tensor':
            require(argv[2]=='run_mass_experiment.py' and argv[argv.index('--seed')+1]=='42' and argv[argv.index('--contribution-weight')+1]=='.05')
        require(argv[argv.index('--data-root')+1]=='/data/gaob/Re-ID/dataset')
        require(argv[argv.index('--pretrained')+1]=='/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt')
        self.events.append(('spawn',dataset,mode,gpu)); self.children.append((dataset,mode,gpu,argv))
        code=0
        if mode=='tensor': result=dict(status='PASS'); path=target
        elif mode=='smoke': result=dict(status='SMOKE_PASS',steps=3,strict_reload_equal=True,gradients={'parameter':True}); path=target+'/smoke.json'
        else: result=dict(status='COMPLETE',epochs=50); path=target+'/result.json'
        if self.fault and self.fault[:2]==(dataset,mode):
            field,value=self.fault[2:]
            if field=='exit_code': code=value
            else: result[field]=value
        def wait():
            self.fs.put_json(path,result)
            return code
        return SimpleNamespace(pid=200+len(self.children),wait=wait)
    def run(self):
        outer=self
        class Pool:
            def __init__(self,max_workers): require(max_workers==3)
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def submit(self,fn,*args):
                outer.submissions.append(args)
                return SimpleNamespace(result=lambda:fn(*args))
        ns=dict(ThreadPoolExecutor=Pool,hashlib=hashlib,json=json,os=SimpleNamespace(environ={}),
                subprocess=SimpleNamespace(Popen=self.spawn,STDOUT=-2),sys=SimpleNamespace(executable=PYTHON,path=[]),
                time=SimpleNamespace(time=lambda:1.),Path=self.fs.path,idle=self.idle,write_json=self.write)
        execute(strip(ast.parse(happy.ns['controller'])),ns)
        self.ns=ns
        return self


def controller_happy():
    c=Controller().run()
    require([(d,m,g) for d,m,g,a in c.children]==[(d,m,3) for d in ('MSVR310','RGBNT201','RGBNT100') for m in ('tensor','smoke')]+[('RGBNT100','train',0),('RGBNT201','train',1),('MSVR310','train',3)])
    gate=next(i for i,e in enumerate(c.events) if e==('write',OUTPUT+'/preflight/controller_result.json'))
    train=next(i for i,e in enumerate(c.events) if e[:1]==('spawn',) and e[2]=='train')
    require(gate<train and c.submissions==[(0,'RGBNT100'),(1,'RGBNT201'),(3,'MSVR310')])
    preflight=json.loads(c.fs.files[OUTPUT+'/preflight/controller_result.json'])
    terminal=json.loads(c.fs.files[OUTPUT+'/controller_result.json'])
    require(preflight['optimizer_updates']==9 and len(preflight['checks'])==6 and terminal['status']=='COMPLETE' and len(terminal['runs'])==3)
    require(len([p for p,v in c.rows if p.endswith('_launch.json')])==9 and len([p for p,v in c.rows if p.endswith('_exit.json')])==9)
    return 'Six ordered exit0+payload gates before dev creation; max_workers3/exact GPU0,1,3 schedule; launch/exit/logs for every child; smoke9 and full150 budgets separate.'


def controller_preflight_failures():
    for dataset in ('MSVR310','RGBNT201','RGBNT100'):
        for mode in ('tensor','smoke'):
            c=Controller((dataset,mode,'exit_code',1)); rejects(c.run)
            require(OUTPUT+'/development' not in c.fs.dirs and all(x[1]!='train' for x in c.children))
            exits=[v for p,v in c.rows if p.endswith('_exit.json')]
            require(exits[-1]['exit_code']==1)
    for mode,field,value in [('tensor','status','FAIL'),('smoke','status','FAIL'),('smoke','steps',2),('smoke','strict_reload_equal',False),('smoke','gradients',{'parameter':False})]:
        c=Controller(('MSVR310',mode,field,value)); rejects(c.run)
        require(OUTPUT+'/development' not in c.fs.dirs and all(x[1]!='train' for x in c.children))
    return 'Every preflight child exit failure and each actual status/steps/reload/gradient gate stops before development; no child is retried.'


def controller_recheck_wait_failure():
    c=Controller(); c.fs.put(ROOT+'/'+NAMES[0],b'changed'); rejects(c.run)
    require(not c.children and OUTPUT+'/preflight' not in c.fs.dirs)
    c=Controller(); c.used[3]=[2410,500,153]; c.run()
    require([e for e in c.events if e[0]=='sleep']==[('sleep',240,3),('sleep',240,3)])
    c=Controller(('RGBNT100','train','exit_code',1)); rejects(c.run)
    require(OUTPUT+'/controller_result.json' not in c.fs.files)
    require(len([x for x in c.children if x[1]=='train' and x[0]=='RGBNT100'])==1)
    return 'Controller verifies all18 source bytes before preflight; actual idle waits240 at >=500; failed train emits exit and prevents COMPLETE without retry.'


def fixed_import_path():
    tree=ast.parse(happy.ns['controller'])
    root_insert=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Expr) and ast.unparse(n).startswith('sys.path.insert('))
    project_imports=[i for i,n in enumerate(tree.body) if isinstance(n,ast.ImportFrom) and n.module in ('launch_axis_scaled','launch_runs')]
    require(root_insert<min(project_imports))
    require(ast.literal_eval(tree.body[root_insert].value.args[1])==ROOT)
    fixture=TRACE/'import_reproduction'; script=fixture/'runs/campaign/controller_fixed.py'
    script.write_text('import sys\nsys.path.insert(0,'+repr(str(fixture.resolve()))+')\nfrom review_project_sibling import VALUE\nprint(VALUE)\n',encoding='utf-8')
    env=dict(os.environ); env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,'-u',str(script.resolve())],cwd=fixture.resolve(),env=env,capture_output=True,text=True)
    require(result.returncode==0 and result.stdout.strip()=='1')
    proof=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,root_insert_before_project_imports=True)
    (TRACE/'controller_import_fixed.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    return 'First file-mode/cwd import failure preserved; minimal explicit project sys.path insertion precedes both project imports and local fixture now exits0.'


def runner_delta():
    old=ast.parse((PROJECT/'run_experiment.py').read_text(encoding='utf-8'))
    new=ast.parse(sources['run_mass_experiment.py'].decode('utf-8'))
    for name in ('write_json','configuration','evaluate','step'):
        get=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        require(ast.dump(get(old))==ast.dump(get(new)),name)
    old_main=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    new_main=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    # The training/smoke/evaluation body after metadata remains exactly V4.
    tail=lambda fn:fn.body[next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Expr) and ast.unparse(n).startswith("write_json(out / 'run.json'")):]
    require(ast.dump(ast.Module(body=tail(old_main),type_ignores=[]))==ast.dump(ast.Module(body=tail(new_main),type_ignores=[])))
    return 'Config B64/50, seeds, optimizer/loss/sampling, smoke3 all-trainable gradients, model+optimizer reload, earliest dev-best and final strict reload exactly original V4 AST.'


check('preserved15_and_native_paths',preserved_sources)
check('deploy_happy_actual_AST',happy_sequence)
check('local_dual_review_and_source_gates',local_review_gates)
check('completed_predecessor_gates',predecessor_gates)
check('resource_and_exclusive_destination_gates',resource_and_exclusive_gates)
check('source_guard_and_launch_recheck',source_guard_rechecks)
check('controller_native_import_fix',fixed_import_path)
check('controller_six_gates_then_three_runs',controller_happy)
check('controller_preflight_failure_stops',controller_preflight_failures)
check('controller_source_wait_train_failure',controller_recheck_wait_failure)
check('runner_unchanged_V4_training_protocol',runner_delta)

(TRACE/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
(TRACE/'harness_errors.txt').write_text('\n'.join(failures),encoding='utf-8')
print(json.dumps(dict(checks_count=len(checks),failures=len(failures),helper_sha256=helper_hash,checked_source_sha256=hashes)),flush=True)
raise SystemExit(bool(failures))
