"""Read-only source review: stdlib fixtures, no SSH/SCP/GPU or third-party imports."""
import argparse
import ast
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
import sys

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE = ROOT / '.aris/traces/experiment-bridge/2026-10-03_missing_development'
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_v4_missing_dev26_deploy_20261003.py')
MODULE = ROOT / 'missing_development.py'
SOURCE = MODULE.read_text(encoding='utf-8')
HELPER_SOURCE = HELPER.read_text(encoding='utf-8')
SHAS = {'module_sha256': hashlib.sha256(MODULE.read_bytes()).hexdigest(),
        'helper_sha256': hashlib.sha256(HELPER.read_bytes()).hexdigest()}
PREVIOUS = json.loads((ROOT/'results/preflight/axis_collaboration_v4_extra_cross26_launch.json').read_text())
HOSTS = {'2025': ('/data2/gb/Re-ID/DeMo-DualAxis', '/data2/gb/Re-ID/conda-envs/tri_reid/bin/python'),
         '2026': ('/data/gaob/Re-ID/DeMo-DualAxis', '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python')}
OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ClearAllForwardings=yes', '-o', 'ConnectTimeout=15']
DATASETS = ('MSVR310', 'RGBNT201', 'RGBNT100')
METRICS = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
MISSING = {'r': ('RGB',), 'n': ('NI',), 't': ('TI',), 'rn': ('RGB', 'NI'), 'rt': ('RGB', 'TI'), 'nt': ('NI', 'TI')}
checks, generated, transfers, remote_calls = [], [], [], []


def check(name, **details):
    checks.append(dict(name=name, status='PASS', **details))


def stripped(source, filename='<generated>'):
    tree = ast.parse(source, filename=filename)
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    return compile(tree, filename, 'exec')


class FS:
    def __init__(self):
        self.files, self.dirs = {}, {'/'}
        fs = self

        class VPath:
            def __init__(self, value):
                self.key = str(PurePosixPath(str(value)))
            def __str__(self): return self.key
            def __truediv__(self, other): return VPath(PurePosixPath(self.key) / str(other))
            @property
            def parent(self): return VPath(PurePosixPath(self.key).parent)
            @property
            def name(self): return PurePosixPath(self.key).name
            @property
            def suffix(self): return PurePosixPath(self.key).suffix
            def exists(self): return self.key in fs.files or self.key in fs.dirs
            def read_bytes(self): return fs.files[self.key]
            def read_text(self, **kwargs): return self.read_bytes().decode(kwargs.get('encoding', 'utf-8'))
            def stat(self): return NS(st_size=len(self.read_bytes()))
            def mkdir(self, exist_ok=False, parents=False):
                if self.exists() and not exist_ok: raise FileExistsError(self.key)
                if not self.parent.exists() and not parents: raise FileNotFoundError(str(self.parent))
                fs.dirs.add(self.key)
                if parents:
                    for p in PurePosixPath(self.key).parents: fs.dirs.add(str(p))
            def write_bytes(self, data):
                assert self.parent.exists(), str(self.parent)
                fs.files[self.key] = bytes(data)
            def write_text(self, data, **kwargs): self.write_bytes(data.encode(kwargs.get('encoding', 'utf-8')))
            def with_suffix(self, suffix): return VPath(PurePosixPath(self.key).with_suffix(suffix))
            def open(self, mode='r', **kwargs):
                if mode == 'x' and self.exists(): raise FileExistsError(self.key)
                p = self
                class Handle(io.StringIO):
                    def close(inner):
                        if any(m in mode for m in ('x', 'w', 'a')): p.write_text(inner.getvalue())
                        super().close()
                return Handle(self.read_text() if mode == 'r' else '')
        self.Path = VPath

    def put(self, path, value):
        path = self.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())

    def clone(self):
        result = FS()
        result.files, result.dirs = self.files.copy(), self.dirs.copy()
        return result


local, fs25, fs26 = FS(), FS(), FS()
filesystems = {'2025': fs25, '2026': fs26}
local.put('/project/results/preflight/axis_collaboration_v4_extra_cross26_launch.json', PREVIOUS)
local.put('/project/results/preflight/axis_collaboration_v4_missing_development_review.json', {'status': 'PASS', 'blockers': [], **SHAS})
local.put('/project/missing_development.py', MODULE.read_bytes())
for filename, digest in PREVIOUS['source_sha256'].items():
    blob = (ROOT / filename).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == digest, filename
    fs26.put(HOSTS['2026'][0] + '/' + filename, blob)
check('all_14_existing_local_sources_equal_previous_launch_digests', count=14)

terminals = {}
for dataset in DATASETS:
    for variant in ('demo', 'axis_scaled_fullref'):
        name = f'{dataset}_{variant}_s42'
        local_dir = ROOT / 'results' / ('full_suite' if variant == 'demo' else 'axis_collaboration_v4_development') / name
        terminal_bytes = (local_dir/'result.json').read_bytes()
        terminal = json.loads(terminal_bytes)
        assert terminal['status'] == 'COMPLETE' and terminal['epochs'] == 50
        assert terminal['arguments']['dataset'] == dataset and terminal['arguments']['variant'] == variant and terminal['arguments']['seed'] == 42
        terminals[name] = terminal
        host = '2025' if variant == 'demo' and dataset != 'RGBNT100' else '2026'
        campaign = 'dynamic_amp_comparison' if variant == 'demo' else 'axis_collaboration_v4_cross26_input'
        base = HOSTS[host][0] + '/runs/' + campaign + '/' + name
        fs = filesystems[host]
        for filename in ('best.pth', 'best_dev_arrays.npz'):
            fs.put(base + '/' + filename, (name + ':' + filename).encode())
        fs.put(base + '/result.json', terminal_bytes)
        if variant == 'demo':
            exit_blob = (local_dir/'exit.json').read_bytes()
            fs.put(base + '/exit.json', exit_blob)
        else:
            exit_blob = (local_dir.parent/(name+'_exit.json')).read_bytes()
            fs.put(str(PurePosixPath(base).parent / (name+'_exit.json')), exit_blob)
        assert json.loads(exit_blob)['exit_code'] == 0
check('six_real_local_terminal_texts_complete50_correct_dataset_variant_seed42_and_exit0',
      counts={name: {'gallery': t['strict_reload']['gallery_count'], 'queries': t['strict_reload']['query_count']} for name,t in terminals.items()})
before26 = fs26.clone()


def remote_exec(source, fs, used=None, popen=None):
    output = io.StringIO()
    def printer(*args, **kwargs):
        print(*args, **{**kwargs, 'file': output})
    def cp(source, target):
        fs.Path(target).write_bytes(fs.Path(source).read_bytes())
        transfers.append(dict(kind='remote_copyfile', host='2026', source=str(source), target=str(target)))
    subprocess = NS(STDOUT=-2,
                    check_output=lambda *args,**kwargs: used or '0, 1000\n1, 1000\n2, 0\n3, 499\n',
                    Popen=popen or (lambda *args,**kwargs: NS(pid=777)))
    namespace = dict(Path=fs.Path, hashlib=hashlib, json=json, subprocess=subprocess,
                     shutil=NS(copyfile=cp), print=printer, os=NS(environ={}), time=time)
    exec(stripped(source), namespace)
    return output.getvalue(), namespace


def remote_python(host, source):
    tree = ast.parse(source)
    if "--query-gpu=index" in source: kind = 'guard'
    elif "exit=root.parent" in source: kind = 'axis_proof'
    elif "terminal=json.loads" in source: kind = 'baseline_proof'
    elif 'shutil.copyfile' in source: kind = 'copy'
    elif "print(\"CREATED\")" in source: kind = 'mkdir'
    elif 'controller.py' in source: kind = 'launch'
    else: raise AssertionError(source)
    generated.append(dict(kind=kind, host=host, source=source, lines=len(source.splitlines())))
    remote_calls.append((host, kind))
    return remote_exec(source, filesystems[host])[0]


def command(argv):
    assert argv[0] == 'scp'
    source, target = argv[-2:]
    shost, spath = source.split(':', 1) if ':' in source else ('local', source)
    thost, tpath = target.split(':', 1)
    sfs = local if shost == 'local' else filesystems[shost]
    filesystems[thost].Path(tpath).write_bytes(sfs.Path(spath).read_bytes())
    transfers.append(dict(kind='scp3' if '-3' in argv else 'scp', source_host=shost, source=spath, target_host=thost, target=tpath, argv=argv))
    return ''


helper_tree = ast.parse(HELPER_SOURCE, filename=str(HELPER))
helper_tree.body = [node for node in helper_tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
                    and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                             and ast.unparse(node.value.func) == 'sys.path.insert')]
outer = dict(datetime=datetime, hashlib=hashlib, json=json, sys=sys, __file__=str(HELPER),
             PROJECT=local.Path('/project'), HOSTS=HOSTS, OPTIONS=OPTIONS, command=command, remote_python=remote_python)
exec(compile(helper_tree, str(HELPER), 'exec'), outer)
controller = outer['controller_code']
ast.parse(controller)
ast.parse(outer['entry'])
check('actual_outer_ast_and_generated_scripts_executed_with_injected_transports',
      remote_script_instances=len(generated), baseline_proofs=3, axis_proofs=3,
      families=['guard', 'baseline_proof', 'axis_proof', 'controller', 'launch'],
      additional_strings=['mkdir(3)', 'same_host_copy(1)', 'entry'])
expected_transfers = []
for ds in DATASETS:
    name = ds+'_demo_s42'
    host = '2026' if ds == 'RGBNT100' else '2025'
    for filename in ('best.pth','best_dev_arrays.npz','result.json','exit.json'):
        source = HOSTS[host][0]+'/runs/dynamic_amp_comparison/'+name+'/'+filename
        target = outer['inputs']+'/'+name+'/'+filename
        expected_transfers.append((host,source,'2026',target))
actual_transfers = [(t.get('source_host', t.get('host')),t['source'],t.get('target_host',t.get('host')),t['target'])
                    for t in transfers if t.get('source_host') != 'local']
assert actual_transfers == expected_transfers
assert [t for t in transfers if t.get('source_host') == 'local'] == [dict(kind='scp',source_host='local',source='/project/missing_development.py',target_host='2026',target=HOSTS['2026'][0]+'/missing_development.py',argv=['scp',*OPTIONS,'/project/missing_development.py','2026:'+HOSTS['2026'][0]+'/missing_development.py'])]
for name, proof in outer['proofs'].items():
    assert len(proof['files']) == 4
    for filename, metadata in proof['files'].items():
        base = outer['inputs']+'/'+name if name.endswith('_demo_s42') else proof['source_path']
        path = str(PurePosixPath(base).parent / filename) if filename.endswith('_exit.json') else base+'/'+filename
        blob = fs26.Path(path).read_bytes()
        assert len(blob) == metadata['bytes'] and hashlib.sha256(blob).hexdigest() == metadata['sha256']
for path, blob in before26.files.items(): assert fs26.files[path] == blob
check('loop_bindings_24_inputs_and_12_actual_copies_exact', proofs=6, input_files=24, scp3_files=8,
      same_host_copies=4, reused_axis_inputs=12, uploaded_module_count=1, original_source_inputs_unchanged=True)


def must_reject(name, callback, expected=(AssertionError,)):
    try:
        callback()
    except expected as error:
        check(name, caught=type(error).__name__)
        return
    raise AssertionError('negative accepted: '+name)


guard = next(item['source'] for item in generated if item['kind']=='guard')
for gpu, usage in ((2,500),(3,500),(2,7000)):
    used = '0, 9000\n1, 9000\n2, '+str(usage if gpu==2 else 0)+'\n3, '+str(usage if gpu==3 else 0)+'\n'
    must_reject(f'guard_gpu{gpu}_{usage}_rejected', lambda u=used: remote_exec(guard,before26.clone(),used=u))
bad = before26.clone(); bad.files[HOSTS['2026'][0]+'/run_experiment.py'] += b'changed'
must_reject('guard_changed_old_source_rejected',lambda: remote_exec(guard,bad))
bad = before26.clone(); bad.Path(outer['output']).mkdir(parents=True)
must_reject('guard_existing_output_rejected',lambda: remote_exec(guard,bad))
proof_sources = [item for item in generated if item['kind'] in ('baseline_proof','axis_proof')]
for index,item in enumerate(proof_sources):
    _, ns = remote_exec(item['source'],filesystems[item['host']].clone())
    base = str(ns['root'])
    expected_name = (DATASETS[index]+'_demo_s42') if index<3 else DATASETS[index-3]+'_axis_scaled_fullref_s42'
    assert PurePosixPath(base).name == expected_name
    assert item['host']==('2025' if index in (0,1) else '2026')
    for field,value in [('status','RUNNING'),('epochs',49)]:
        fs = filesystems[item['host']].clone()
        terminal = json.loads(fs.Path(base+'/result.json').read_text()); terminal[field] = value
        fs.put(base+'/result.json',terminal)
        must_reject('proof_'+expected_name+'_'+field,lambda s=item['source'],f=fs: remote_exec(s,f))
    fs = filesystems[item['host']].clone()
    exit_path = base+'/exit.json' if index<3 else str(PurePosixPath(base).parent/(expected_name+'_exit.json'))
    fs.put(exit_path,{'exit_code':1})
    must_reject('proof_'+expected_name+'_exit1',lambda s=item['source'],f=fs: remote_exec(s,f))
check('all_six_proof_strings_bind_their_own_model_and_host', names=[PurePosixPath(str(remote_exec(x['source'],filesystems[x['host']].clone())[1]['root'])).name for x in proof_sources])
prelaunch = fs26.clone()
for path in list(prelaunch.files):
    if path.startswith(outer['output']+'/'): del prelaunch.files[path]
prelaunch.dirs.discard(outer['output'])
launch = next(item['source'] for item in generated if item['kind']=='launch')
for name in outer['proofs']:
    fs = prelaunch.clone()
    base = outer['inputs']+'/'+name if name.endswith('_demo_s42') else outer['proofs'][name]['source_path']
    path = base+'/best.pth'; original = fs.files[path]
    fs.files[path] = b'X'+original[1:]
    spawned=[]
    must_reject('launch_input_hash_change_'+name,lambda f=fs:remote_exec(launch,f,popen=lambda *a,**k:spawned.append(a)))
    assert not spawned
check('launch_rechecks_all_six_models_before_any_spawn')


def controller_case(label, failing=None, malformed=None, block=None):
    fs = FS(); fs.Path(outer['output']).mkdir(parents=True)
    events, writes, errors = [], [], []
    release, waiting = threading.Event(), threading.Event()
    def write_json(path, data):
        fs.put(str(path), data); writes.append((str(path),copy.deepcopy(data)))
    def idle(gpu): events.append(('idle',gpu))
    class Child:
        pid = 888
        def __init__(self, argv, **kwargs):
            self.argv=argv
            self.gpu=int(kwargs['env']['CUDA_VISIBLE_DEVICES'])
            self.out=fs.Path(argv[argv.index('--output')+1])
            self.stage='smoke' if '--smoke' in argv else 'full'
            self.name=self.out.parent.name
            assert kwargs['env']['OMP_NUM_THREADS']=='4'
            expected_run = outer['inputs']+'/'+self.name if self.gpu==3 else outer['proofs'][self.name]['source_path']
            assert argv[argv.index('--run-dir')+1]==expected_run
            assert argv[argv.index('--data-root')+1]=='/data/gaob/Re-ID/dataset'
            assert argv[argv.index('--pretrained')+1]=='/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt'
            assert argv[2]=='missing_development.py'
            events.append(('start',self.gpu,self.name,self.stage))
        def wait(self):
            key=(self.gpu,self.name,self.stage)
            events.append(('wait',*key))
            if block==key:
                waiting.set(); assert release.wait(10)
            if failing==key: return 1
            record={'status':'PASS' if self.stage=='smoke' else 'COMPLETE','optimizer_updates':0,'normal_feature_max_error':0,'measurements':dict.fromkeys(range(13))}
            if malformed and key==malformed[0]: record[malformed[1]]=malformed[2]
            write_json(self.out/('smoke.json' if self.stage=='smoke' else 'result.json'),record)
            events.append(('return_wait',*key))
            return 0
    env=dict(ThreadPoolExecutor=ThreadPoolExecutor,json=json,os=NS(environ={}),subprocess=NS(Popen=Child,STDOUT=-2),sys=NS(executable='mock-python'),time=time,Path=fs.Path,idle=idle,write_json=write_json)
    def run():
        try: exec(stripped(controller),env)
        except AssertionError as error: errors.append(type(error).__name__)
    if block:
        thread=threading.Thread(target=run); thread.start(); assert waiting.wait(10)
        assert not (fs.Path(outer['output'])/'controller_result.json').exists()
        release.set(); thread.join(10); assert not thread.is_alive()
    else: run()
    final=fs.Path(outer['output'])/'controller_result.json'
    if failing or malformed:
        assert errors and not final.exists()
        bad=failing or malformed[0]
        starts=[e for e in events if e[0]=='start' and e[1]==bad[0]]
        index=next(i for i,e in enumerate(starts) if e[2:]==bad[1:])
        assert len(starts)==index+1
    else:
        assert not errors and json.loads(final.read_text())['conditions']==78
        assert len(json.loads(final.read_text())['runs'])==6
        for gpu,variant in ((2,'axis_scaled_fullref'),(3,'demo')):
            sequence=[(e[2],e[3]) for e in events if e[0]=='start' and e[1]==gpu]
            assert sequence==[(ds+'_'+variant+'_s42',stage) for ds in DATASETS for stage in ('smoke','full')]
    check(label, starts=[e for e in events if e[0]=='start'], aggregate_complete=final.exists(), waited=block is not None)


controller_case('controller_six_models_smoke_then_full_78_conditions')
for gpu,variant in ((2,'axis_scaled_fullref'),(3,'demo')):
    for ds in DATASETS:
        controller_case(f'controller_smoke_failure_stops_worker_{gpu}_{ds}',failing=(gpu,ds+'_'+variant+'_s42','smoke'))
controller_case('controller_full_failure_stops_later_worker_models',failing=(2,'RGBNT201_axis_scaled_fullref_s42','full'))
for field,value in (('status','RUNNING'),('optimizer_updates',1),('normal_feature_max_error',.01)):
    controller_case('controller_exit0_invalid_smoke_'+field,malformed=((3,'MSVR310_demo_s42','smoke'),field,value))
controller_case('controller_incomplete_wait_never_writes_aggregate',block=(3,'RGBNT100_demo_s42','full'))


class Array:
    def __init__(self, values, tag=None, indices=None):
        self.values = copy.deepcopy(values.values if isinstance(values,Array) else list(values))
        self.tag, self.indices = tag, indices
    def __len__(self): return len(self.values)
    def __iter__(self): return iter(self.values)
    @property
    def shape(self): return (len(self.values),len(self.values[0])) if self.values and isinstance(self.values[0],list) else (len(self.values),)
    def __getitem__(self,key):
        if isinstance(key,Array): key=key.values
        if isinstance(key,list):
            values=[v for v,k in zip(self.values,key) if k] if key and isinstance(key[0],bool) else [self.values[k] for k in key]
            return Array(values,self.tag,key)
        value=self.values[key]
        return Array(value,self.tag) if isinstance(key,slice) else value
    def operation(self,other,op):
        if isinstance(other,Array):
            def apply(a,b): return [apply(x,y) for x,y in zip(a,b)] if isinstance(a,list) else op(a,b)
            return Array(apply(self.values,other.values))
        def apply(a): return [apply(x) for x in a] if isinstance(a,list) else op(a,other)
        return Array(apply(self.values))
    def __sub__(self,other): return self.operation(other,lambda a,b:a-b)
    def __add__(self,other): return self.operation(other,lambda a,b:a+b)
    def __truediv__(self,other): return self.operation(other,lambda a,b:a/b)
    def __mul__(self,other): return self.operation(other,lambda a,b:a*b)
    def __rmul__(self,other): return self*other
    def __eq__(self,other): return self.operation(other,lambda a,b:a==b)
    def __and__(self,other): return self.operation(other,lambda a,b:a and b)
    def __invert__(self): return self.operation(None,lambda a,b:not a)
    def numpy(self): return self
    def tolist(self): return copy.deepcopy(self.values)
    def flat(self):
        def walk(v):
            for x in v:
                if isinstance(x,list): yield from walk(x)
                else: yield x
        return list(walk(self.values))
    def max(self): return max(self.flat())
    def all(self): return all(self.flat())


class NP:
    @staticmethod
    def asarray(value): return value if isinstance(value,Array) else Array(value)
    @staticmethod
    def array_equal(a,b):
        return (a.values if isinstance(a,Array) else a)==(b.values if isinstance(b,Array) else b)
    @staticmethod
    def abs(a): return a.operation(None,lambda a,b:abs(a))
    @staticmethod
    def argsort(a,axis=1,kind='stable'): return Array([sorted(range(len(row)),key=lambda i:row[i]) for row in a.values])
    @staticmethod
    def flatnonzero(a): return Array([i for i,x in enumerate(a.values) if x])
    @staticmethod
    def arange(start,end): return Array(range(start,end))
    @staticmethod
    def mean(a):
        values=a.flat() if isinstance(a,Array) else a
        return sum(values)/len(values)


main_ast = ast.Module(body=[n for n in ast.parse(SOURCE).body if isinstance(n,ast.FunctionDef)],type_ignores=[])


def module_case(name, dataset='MSVR310',variant='demo',smoke=False, fault=None):
    fs=FS(); run=fs.Path('/run/'+dataset+'_'+variant+'_s42'); run.mkdir(parents=True)
    terminal=copy.deepcopy(terminals[run.name])
    if fault=='incomplete': terminal['status']='RUNNING'
    if fault=='epoch49': terminal['epochs']=49
    fs.put(str(run/'result.json'),terminal)
    fs.put(str(run/'best.pth'),b'checkpoint'); fs.put(str(run/'best_dev_arrays.npz'),b'arrays')
    exit_path=run/'exit.json' if variant=='demo' else run.parent/(run.name+'_exit.json')
    fs.put(str(exit_path),{'exit_code':1 if fault=='exit1' else 0})
    n=terminal['strict_reload']['gallery_count']; nq=terminal['strict_reload']['query_count']
    dev=[('/installed/'+str(i)+'.jpg',i//3, i%5, i%7) for i in range(n)]
    query=list(range(nq))
    features=[[float(i),float(i)+.25] for i in range(n)]
    saved=dict(features=Array(features),query_indices=Array(query),ids=Array([r[1] for r in dev]),cameras=Array([r[2] for r in dev]),scenes=Array([r[3] for r in dev]),names=Array([str(i)+'.jpg' for i in range(n)]))
    if fault and fault.startswith('saved_'): saved[fault.removeprefix('saved_')].values[0]='BAD'
    args=NS(run_dir=str(run),output='/out',data_root='/installed',pretrained='/pretrained',smoke=smoke)
    events,measured,written=[],[],[]
    state=NS(_version=0)
    class Model:
        training=True
        def load_state_dict(self,value,strict): assert strict and value=='loaded'; events.append('strict_load')
        def eval(self): self.training=False; events.append('eval')
        def modules(self): return [self]
        def state_dict(self): return {'weight':state}
    model=Model()
    def build(arguments,cfg,classes,cameras):
        assert arguments.data_root=='/installed' and arguments.pretrained=='/pretrained' and arguments.seed==42
        events.append('build'); return model
    def extract_missing(model,records,cfg,seed,missing):
        assert not model.training and 'strict_load' in events
        events.append('extract_'+('clean' if not missing else next(k for k,v in MISSING.items() if v==missing)))
        offset=list(MISSING.values()).index(missing)+1 if missing else 0
        values=[[float(i)+offset,float(i)+.25+offset] for i in range(len(records))]
        if fault=='parity' and not missing: values[0][0]+=.1
        if fault=='nan_mask' and missing==MISSING['rn']: values[0][0]=float('nan')
        if fault=='state_mutation' and missing==MISSING['nt']: state._version+=1
        if fault=='input_mutation' and missing==MISSING['nt']: fs.files[str(run/'best.pth')]+=b'changed'
        return Array(values,tag=missing),{'triplets':len(records)}
    def verify_original_mask(*args):
        events.append('original_mask_verification')
        if fault=='original_mask': raise AssertionError('mask mismatch')
        return {'original_forward_bitwise_equal_masks':list(MISSING),'batch':8,'optimizer_updates':0}
    def full_metrics(pair,qids,gids,qe,ge,names,cams,scenes,path):
        q,g=pair
        assert len(q)==nq and len(g)==n
        assert q.indices==query
        assert qids.values==saved['ids'][query].values and gids.values==saved['ids'].values
        key='scenes' if dataset=='MSVR310' else 'cameras'
        assert qe.values==saved[key][query].values and ge.values==saved[key].values
        assert names.values==saved['names'][query].values
        assert cams.values==saved['cameras'][query].values and scenes.values==saved['scenes'][query].values
        name=path.name
        if name=='clean': assert q.tag==g.tag==()
        elif name.startswith('both_missing_'): assert q.tag==g.tag==MISSING[name.removeprefix('both_missing_')]
        else: assert q.tag==MISSING[name.removeprefix('query_missing_')] and g.tag==()
        measured.append(dict(name=name,q_count=len(q),g_count=len(g),q_mask=q.tag,g_mask=g.tag))
        result={**{k:terminal['strict_reload'].get(k,40) for k in METRICS},'CMC_1_to_50':[1]*50,'groups':{},'query_count':nq,'gallery_count':n}
        if fault=='metric_parity' and name=='clean': result['Rank-1']+=1
        return result
    def write_json(path,value): fs.put(str(path),value); written.append((str(path),value))
    class Parser:
        def add_argument(self,*args,**kwargs): pass
        def parse_args(self): return args
    np=NP(); np.load=lambda path:saved
    torch=NS(set_num_threads=lambda n:events.append('threads'),load=lambda *a,**k:'loaded',isfinite=lambda a:Array([math.isfinite(v) for v in a.flat()]),cuda=NS(max_memory_allocated=lambda:0))
    ns=dict(argparse=NS(ArgumentParser=Parser,Namespace=argparse.Namespace),hashlib=hashlib,json=json,Path=fs.Path,time=time,np=np,torch=torch,
            split_records=lambda *a:([],dev,query,1,5),distance=lambda q,g:(q,g),full_metrics=full_metrics,MISSING=MISSING,METRICS=METRICS,
            extract_missing=extract_missing,verify_original_mask=verify_original_mask,build=build,configuration=lambda a:NS(TEST=NS(MISS='nothing')),write_json=write_json,VARIANTS={'axis_scaled_fullref':None},print=lambda *a,**k:None)
    exec(compile(main_ast,str(MODULE),'exec'),ns)
    error=None
    try: ns['main']()
    except AssertionError: error='AssertionError'
    if fault:
        assert error and not written,(name,events,written)
        if fault in ('incomplete','epoch49','exit1') or fault.startswith('saved_'): assert 'build' not in events
    else:
        assert not error and len(written)==1
        record=written[0][1]
        assert record['optimizer_updates']==0 and record['state_tensor_versions_unchanged']
        assert len(measured)==(0 if smoke else 13)
        assert len([e for e in events if e.startswith('extract_')])==7
        if smoke: assert record['triplets']==64
        else: assert len(record['measurements'])==13 and record['status']=='COMPLETE'
        assert events.count('original_mask_verification')==(variant=='demo')
    check(name, fault=fault, output_written=bool(written), measurements=len(measured), extract_calls=sum(e.startswith('extract_') for e in events),
          gallery=n,queries=nq,events=events)


for dataset in DATASETS:
    for variant in ('demo','axis_scaled_fullref'):
        module_case('module_full_'+dataset+'_'+variant,dataset,variant)
        module_case('module_smoke64_'+dataset+'_'+variant,dataset,variant,smoke=True)
for fault in ('incomplete','epoch49','exit1','saved_query_indices','saved_ids','saved_cameras','saved_scenes','saved_names','parity','metric_parity','nan_mask','state_mutation','input_mutation','original_mask'):
    module_case('module_reject_'+fault,fault=fault)
module_case('module_smoke_reject_late_state_mutation',smoke=True,fault='state_mutation')


# Exercise actual mask_images with primitive normalized arrays and original checker
# with a three-key fake model; no neural equality claim is made.
missing_tree=ast.parse((ROOT/'missing_evaluation.py').read_text())
mask_node=next(n for n in missing_tree.body if isinstance(n,ast.FunctionDef) and n.name=='mask_images')
mask_ns={'torch':NS(zeros_like=lambda a:Array([0]*len(a)))}
exec(compile(ast.Module(body=[mask_node],type_ignores=[]),'mask_images.py','exec'),mask_ns)
images={k:Array([.3,-.8,1.2]) for k in ('RGB','NI','TI')}
for code,missing in MISSING.items():
    masked=mask_ns['mask_images'](images,missing)
    assert all(masked[k].values==[0,0,0] if k in missing else masked[k] is images[k] for k in images)
check('actual_mask_images_all_six_sets_normalized_zero_and_unmasked_identity')


# Run actual full_metrics over a fixed 25-gallery example. The independent
# reference implements rank counting directly using Python lists.
full_tree=ast.parse((ROOT/'full_evaluation.py').read_text())
full_node=next(n for n in full_tree.body if isinstance(n,ast.FunctionDef) and n.name=='full_metrics')
def reference(distances,qids,gids,qe,ge):
    rows=[]
    for i,dist in enumerate(distances.values):
        ordered=sorted(range(len(dist)),key=lambda k:dist[k])
        kept=[k for k in ordered if not(gids.values[k]==qids.values[i] and ge.values[k]==qe.values[i])]
        ranks=[rank for rank,k in enumerate(kept,1) if gids.values[k]==qids.values[i]]
        if ranks: rows.append((ranks,len(kept)))
    cmc=[sum(int(ranks[0]<=r) for ranks,_ in rows)/len(rows) for r in range(1,26)]
    map_=sum(sum(j/r for j,r in enumerate(ranks,1))/len(ranks) for ranks,_ in rows)/len(rows)
    return Array(cmc),map_
metric_fs=FS(); metric_fs.Path('/metrics').mkdir()
metric_ns=dict(np=NP(),csv=csv,evaluate_reid=reference,write_json=lambda p,r:metric_fs.put(str(p),r))
exec(compile(ast.Module(body=[full_node],type_ignores=[]),'full_metrics.py','exec'),metric_ns)
gids=Array([1,1,2,2,3]+list(range(20,40))); ge=Array([0,1,0,1,0]+[0]*20)
qids=Array([1,2,99]); qe=Array([0,1,0])
distances=Array([[float(i) for i in range(25)],[float(24-i) for i in range(25)],[float(i) for i in range(25)]])
summary=metric_ns['full_metrics'](distances,qids,gids,qe,ge,Array(['a','b','invalid']),Array([3,4,5]),Array([0,1,2]),metric_fs.Path('/metrics/condition'))
assert summary['query_count']==3 and summary['valid_queries']==2 and summary['invalid_queries']==1 and summary['gallery_count']==25
assert set(METRICS).issubset(summary) and len(summary['CMC_1_to_50'])==25
assert set(summary['groups'])=={'camera','scene','identity'}
rows=list(csv.DictReader(io.StringIO(metric_fs.Path('/metrics/condition.csv').read_text())))
assert len(rows)==3 and rows[2]['valid']=='False'
assert abs(summary['mAP']-100*(1+1/22)/2)<1e-10
assert abs(summary['mINP']-summary['mAP'])<1e-10
assert summary['Rank-1']==50 and summary['Rank-20']==50
check('actual_full_metrics_six_metrics_CMC_groups_and_perquery_csv_reference_fixture',valid_queries=2,invalid_queries=1,gallery=25)


# Source facts are separate from numerical/runtime acceptance.
assert 'official_records' not in SOURCE and 'official_test' not in SOURCE
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','step','train') for n in ast.walk(ast.parse(SOURCE)))
extract=next(n for n in missing_tree.body if isinstance(n,ast.FunctionDef) and n.name=='extract_missing')
assert ast.unparse(extract.decorator_list[0])=='torch.no_grad()'
build_tree=ast.parse((ROOT/'run_experiment.py').read_text())
build_source=ast.get_source_segment((ROOT/'run_experiment.py').read_text(),next(n for n in build_tree.body if isinstance(n,ast.FunctionDef) and n.name=='build'))
assert build_source.count('.float().cuda()')==5
check('static_dev_only_no_training_calls_no_grad_extraction_FP32_build_and_strict_eval')

assert SHAS['module_sha256']==hashlib.sha256(MODULE.read_bytes()).hexdigest()
assert SHAS['helper_sha256']==hashlib.sha256(HELPER.read_bytes()).hexdigest()
for name,digest in PREVIOUS['source_sha256'].items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
artifact={'status':'PASS','checks':checks,'check_count':len(checks),'source_hashes':SHAS,
          'generated':generated+[{'kind':'controller','source':controller,'lines':len(controller.splitlines())},{'kind':'entry','source':outer['entry'],'lines':1}],
          'transfers':transfers,'remote_call_sequence':remote_calls,
          'actual_actions':{'ssh':0,'scp':0,'gpu_queries':0,'remote_process_spawns':0,'optimizer_updates':0,'source_edits':0,'subagents':0},
          'limits':'Stdlib mocks and source checks only. Actual local result/exit JSON used; checkpoint/NPZ payloads are distinct representative bytes. No torch/numpy, neural load or remote/GPU execution. Module orchestration mocks use list-backed arrays and stub extraction; actual inherited full_metrics and mask_images execute with primitive arrays.'}
(TRACE/'validation.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','checks':len(checks),'generated_script_count':len(artifact['generated']),**SHAS}))
