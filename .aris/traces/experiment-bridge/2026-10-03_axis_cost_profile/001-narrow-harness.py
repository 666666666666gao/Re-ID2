"""Execute only reviewed AST with stdlib mocks; no Torch, SSH or subprocesses."""
import ast
from contextlib import redirect_stdout
from datetime import datetime
import gc
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import statistics
from types import SimpleNamespace as NS
import weakref

ROOT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
HERE = Path(__file__).parent
HELPER = Path('C:/Users/gb/.codex_tmp/demo_axis_v5_profile27_deploy_20261003.py')
PROFILE = ROOT/'profile_reid_models.py'
checks = []

def check(name, detail):
    checks.append(dict(name=name, status='PASS', detail=detail))

def code_without_imports(source, filename):
    tree=ast.parse(source, filename)
    tree.body=[node for node in tree.body if not isinstance(node,(ast.Import,ast.ImportFrom))]
    return compile(tree,filename,'exec')

def memory_path(fs, dirs, mutations):
    class P:
        def __init__(self,p): self.p=str(p)
        def __str__(self): return self.p
        def __fspath__(self): return self.p
        def __truediv__(self,p): return P(str(PurePosixPath(self.p)/p))
        @property
        def parent(self): return P(str(PurePosixPath(self.p).parent))
        def exists(self): return self.p in fs or self.p in dirs
        def read_bytes(self): return fs[self.p]
        def read_text(self,encoding=None): return self.read_bytes().decode(encoding or 'utf-8')
        def write_bytes(self,value): fs[self.p]=value; mutations.append(('write',self.p))
        def write_text(self,value,encoding=None): self.write_bytes(value.encode(encoding or 'utf-8'))
        def mkdir(self,exist_ok=False):
            assert exist_ok or not self.exists()
            dirs.add(self.p); mutations.append(('mkdir',self.p))
        def open(self,mode):
            assert mode=='x' and not self.exists()
            self.write_bytes(b'')
            return io.StringIO()
    return P

profile_source=PROFILE.read_text(encoding='utf-8')
helper_source=HELPER.read_text(encoding='utf-8')
source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('profile_reid_models.py','mass_axis_collaboration.py')}
helper_hash=hashlib.sha256(HELPER.read_bytes()).hexdigest()
launch=json.loads((ROOT/'results/preflight/axis_collaboration_v5_mass_launch.json').read_text(encoding='utf-8'))
original={name:sha for name,sha in launch['source_sha256'].items() if name not in ('mass_axis_collaboration.py','run_mass_experiment.py','verify_axis_mass.py','missing_development.py')}
assert len(original)==14
assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in original.items())
check('frozen14_and_current_source_binding','Existing launch receipt gives exactly14 preserved sources; current bytes match all14; helper uses the actual READY state field.')

def profile_mock(mutate_state=False):
    records=[]; rows={}; seeds=[]; live=weakref.WeakSet(); phase={'dataset':None,'model':None,'profile':False,'n':0,'inference':False,'clock':0}
    fs={};dirs=set();mut=[];P=memory_path(fs,dirs,mut)
    class Tensor:
        def __init__(self,shape=(8,),dataset=None): self.shape=shape;self.dataset=dataset
        def cuda(self): records.append(('h2d',phase['dataset']));return self
    class Model:
        def __init__(self,variant):
            assert not live, 'Previous model still retained'
            live.add(self);self.variant=variant;self.training=True;self.version=0
            phase.update(model=variant,n=0);records.append(('create',phase['dataset'],variant))
        def float(self): return self
        def cuda(self): return self
        def eval(self): self.training=False
        def modules(self): return [self]
        def state_dict(self): return {'p':NS(_version=self.version)}
        def parameters(self):
            amount=100 if self.variant=='demo' else 200
            return [NS(numel=lambda:amount,requires_grad=True),NS(numel=lambda:10,requires_grad=False)]
        def __call__(self,images,cam_label,view_label):
            assert not self.training and phase['inference']
            assert all(x.dataset==phase['dataset'] for x in images.values())
            phase['n']+=1; records.append(('forward',phase['dataset'],self.variant,phase['n'],phase['profile']))
            if mutate_state and phase['n']==26: self.version+=1
            return Tensor((8,5120 if self.variant=='demo' else 5632))
    class Context:
        def __enter__(self): phase['inference']=True
        def __exit__(self,*exc): phase['inference']=False
    class Profiler:
        def __init__(self,**kwargs):
            assert kwargs==dict(activities=['cpu','cuda'],record_shapes=True,with_flops=True)
        def __enter__(self):
            assert phase['n']==25;phase['profile']=True;records.append(('profile_enter',));return self
        def __exit__(self,*exc): phase['profile']=False
        def key_averages(self):
            assert phase['n']==26
            return [NS(key='aten::mm',count=2,self_cpu_time_total=4,self_device_time_total=8,flops=800),NS(key='aten::fft_fft2',count=1,self_cpu_time_total=2,self_device_time_total=3,flops=0)]
    class Event:
        def __init__(self,enable_timing): assert enable_timing
        def record(self): records.append(('event',phase['n']))
        def elapsed_time(self,end): return 8.0
    def reset():
        assert phase['n']==5 and not phase['profile'];records.append(('peak_reset',phase['n']))
    def peak(value):
        assert phase['n']==25 and not phase['profile'];records.append(('peak_read',phase['n']));return value
    def empty(): assert not live;records.append(('empty',))
    cuda=NS(Event=Event,synchronize=lambda:records.append(('sync',phase['n'])),reset_peak_memory_stats=reset,memory_allocated=lambda:1024,max_memory_allocated=lambda:peak(4096),max_memory_reserved=lambda:peak(8192),empty_cache=empty,get_device_name=lambda:'MOCK_GPU')
    torch=NS(set_num_threads=lambda n:None,backends=NS(cudnn=NS(benchmark=True)),cuda=cuda,inference_mode=Context,isfinite=lambda x:NS(all=lambda:True),profiler=NS(profile=Profiler,ProfilerActivity=NS(CPU='cpu',CUDA='cuda')),__version__='MOCK_ONLY')
    args=NS(output='/mock/profile',data_root='/mock/data',pretrained='/mock/clip')
    def configuration(a):
        phase['dataset']=a.dataset
        return NS(INPUT=NS(SIZE_TEST=[256,128] if a.dataset=='RGBNT201' else [128,256]))
    def split(root,dataset):
        assert dataset==phase['dataset'];return [],list(range(20)),[],103,8
    def loader(dev,cfg,training,seed):
        assert dev==list(range(8)) and training is False and seed==42
        ds=phase['dataset'];return [({k:Tensor((8,3,*cfg.INPUT.SIZE_TEST),ds) for k in ('RGB','NI','TI')},Tensor(),Tensor(),Tensor(),[f'{ds}_{i}' for i in dev])]
    def clock(): phase['clock']+=.01;return phase['clock']
    parser=NS(add_argument=lambda *a,**k:None,parse_args=lambda:args)
    env=dict(argparse=NS(ArgumentParser=lambda:parser),datetime=datetime,gc=gc,json=json,Path=P,statistics=statistics,time=NS(perf_counter=clock),torch=torch,make_loader=loader,seed_all=seeds.append,split_records=split,MassAxisCollaborationDeMo=lambda *a:Model('axis_mass_fullref'),build=lambda a,*unused:Model(a.variant),configuration=configuration,write_json=lambda p,v:rows.__setitem__(str(p),v),__name__='review_fixture')
    exec(code_without_imports(profile_source,str(PROFILE)),env)
    with redirect_stdout(io.StringIO()):
        env['main']()
    return rows,records,seeds

rows,sequence,seeds=profile_mock()
summary=rows['/mock/profile/summary.json'];data=summary['rows']
assert len(rows)==31 and len(data)==15 and len(seeds)==15
assert len({(r['dataset'],r['variant']) for r in data})==15
assert all(r['batch_triplets']==8 and r['images_per_triplet']==3 for r in data)
assert all(abs(r['wall_batch_ms_median']-10)<1e-8 and r['cuda_event_batch_ms_median']==8 for r in data)
assert all(abs(r['triplets_per_second']-800)<1e-6 and abs(r['images_per_second']-2400)<1e-6 for r in data)
assert all(r['peak_allocated_bytes']==4096 and r['peak_reserved_bytes']==8192 and r['allocated_before_measurement_bytes']==1024 for r in data)
assert all(r['supported_operator_flops_per_triplet_lower_bound']==100 and r['complete_model_FLOPs'] is False and r['operators_without_reported_flops']==['aten::fft_fft2'] for r in data)
assert all(r['optimizer_updates']==0 for r in data) and summary['official_test_uses']==0
assert len([x for x in sequence if x[0]=='forward'])==390
assert len([x for x in sequence if x[0]=='forward' and x[-1]])==15
assert len([x for x in sequence if x[0]=='h2d'])==15
assert len([x for x in sequence if x[0]=='empty'])==15
check('profile15_actual_AST_counts_units_limits','15 distinct cases, one first8 dev batch each dataset,26 forwards/case (5+20+1),31 JSON writes, correct ms-to-throughput and FLOPs/8, all incompleteness/zero-update limits retained.')
check('profile_peak_lifecycle_and_model_release','Mock runtime enforces warmup5 before reset, peak reads after timed20 before profiler, no retained model before next constructor,15 empty-cache calls. CUDA events and synchronization are explicit per trial.')
try:
    profile_mock(mutate_state=True)
except AssertionError:
    check('profile_state_version_guard','A mutated registered tensor version after the26th forward is rejected before writing its condition.')
else:
    raise AssertionError('State-version mutation not rejected')

REMOTE='/data/gb/Re-ID/DeMo-DualAxis'
PYTHON='/data/gb/Re-ID/conda-envs/tri_reid/bin/python'
OUT=REMOTE+'/profile_output/axis_v4_v5_inference_cost_20261003'
review=dict(status='PASS',blockers=[],helper_sha256=helper_hash,checked_source_sha256=source_hashes)
local_root=str(ROOT).replace('\\','/')
local_helper=str(HELPER).replace('\\','/')

def deploy_mock(fault=None):
    local={str(ROOT/name).replace('\\','/'): (ROOT/name).read_bytes() for name in source_hashes}
    local[local_helper]=HELPER.read_bytes()
    local[local_root+'/results/preflight/axis_collaboration_v5_mass_launch.json']=json.dumps(launch).encode()
    local[local_root+'/.aris/compute/ssh2027.json']=b'{"state":"READY"}'
    rr=json.loads(json.dumps(review))
    if fault=='review_status': rr['status']='BLOCKED'
    if fault=='review_blockers': rr['blockers']=['real blocker']
    if fault=='review_helper': rr['helper_sha256']='invalid'
    if fault=='review_source': rr['checked_source_sha256']['profile_reid_models.py']='invalid'
    local[local_root+'/results/preflight/axis_collaboration_v5_profile_review.json']=json.dumps(rr).encode()
    if fault=='env': local[local_root+'/.aris/compute/ssh2027.json']=b'{"state":"BUILDING"}'
    remote={REMOTE+'/'+name:(ROOT/name).read_bytes() for name in original}
    if fault=='old_source': remote[REMOTE+'/run_experiment.py']=b'changed'
    if fault in source_hashes: remote[REMOTE+'/'+fault]=b'existing'
    if fault=='output': remote[OUT]=b'existing'
    local_changes=[];remote_changes=[];uploads=[];calls=[];spawns=[]
    LP=memory_path(local,set(),local_changes);RP=memory_path(remote,{REMOTE},remote_changes)
    def query(argv,text):
        assert argv==['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'] and text
        return '500' if fault=='gpu_guard' or (fault=='gpu_start' and len(calls)==2) else '0'
    def popen(argv,**kw):
        assert argv[:3]==[PYTHON,'-u','-c'] and str(kw['cwd'])==REMOTE and kw['start_new_session']
        assert kw['stdout']=='DEVNULL' and kw['stderr']=='DEVNULL'
        spawns.append((argv,kw));return NS(pid=12345)
    def remote_python(host,code):
        assert host=='2027';calls.append(code)
        if fault=='source_start' and len(calls)==2: remote[REMOTE+'/mass_axis_collaboration.py']=b'changed'
        env=dict(hashlib=hashlib,json=json,shutil=NS(disk_usage=lambda path:NS(free=10**10)),subprocess=NS(check_output=query,Popen=popen,DEVNULL='DEVNULL'),Path=RP,time=NS(time=lambda:1))
        buf=io.StringIO()
        with redirect_stdout(buf): exec(code_without_imports(code,'mock_remote'),env)
        return buf.getvalue()
    def command(argv):
        assert argv[:1]==['scp'] and argv[-1].startswith('2027:'+REMOTE+'/')
        src,dst=argv[-2:];remote[dst.removeprefix('2027:')]=local[src];uploads.append(dst)
    env=dict(datetime=datetime,hashlib=hashlib,json=json,Path=LP,sys=NS(path=[]),PROJECT=LP(local_root),HOSTS={},OPTIONS=['-o','BatchMode=yes'],command=command,remote_python=remote_python,__file__=local_helper)
    error=None
    try:
        with redirect_stdout(io.StringIO()): exec(code_without_imports(helper_source,local_helper),env)
    except AssertionError as e:
        error=str(e)
    return NS(error=error,calls=calls,uploads=uploads,spawns=spawns,local=local,remote=remote,local_changes=local_changes,remote_changes=remote_changes,env=env)

happy=deploy_mock()
assert happy.error is None and len(happy.calls)==2 and len(happy.uploads)==2 and len(happy.spawns)==1
assert set(happy.uploads)=={'2027:'+REMOTE+'/'+name for name in source_hashes}
assert len(happy.env['source_sha'])==16
assert happy.remote_changes==[('mkdir',REMOTE+'/profile_output')]
receipt=json.loads(happy.local[local_root+'/results/preflight/axis_collaboration_v5_profile_launch.json'])
assert receipt['gpu']==0 and receipt['host']=='2027' and receipt['optimizer_updates']==0 and receipt['official_test_uses']==0
check('deploy_happy_actual_AST','Exact physical READY2027 interpreter/project/data paths; one read-only guard, two new file uploads,16-source recheck, output-parent creation and one new-session Python -c launch; GPU0 only.')
for fault in ('review_status','review_blockers','review_helper','review_source','env'):
    result=deploy_mock(fault)
    assert result.error is not None and not result.calls and not result.uploads and not result.spawns and not result.remote_changes
check('local_review_and_READY_gates','Status,blockers,helper/current-source binding and non-READY state each reject before any remote call.')
for fault in ('old_source','profile_reid_models.py','mass_axis_collaboration.py','output','gpu_guard'):
    result=deploy_mock(fault)
    assert result.error is not None and len(result.calls)==1 and not result.uploads and not result.spawns and not result.remote_changes
check('guard_zero_remote_mutation','Old-source mismatch,either preexisting new file,preexisting output,GPU0 memory500 each reject before uploads,remote writes or start.')
for fault in ('source_start','gpu_start'):
    result=deploy_mock(fault)
    assert result.error is not None and len(result.calls)==2 and len(result.uploads)==2 and not result.spawns and not result.remote_changes
check('postupload_source_and_GPU_recheck','Mutated uploaded source and newly occupied GPU0 reject before output mkdir or process start; uploaded files are preserved without retries.')

def wrapper_mock(exit_code):
    fs={};mut=[];P=memory_path(fs,{REMOTE,REMOTE+'/profile_output'},mut);children=[]
    def run(argv,**kw):
        assert argv==[PYTHON,'-u','profile_reid_models.py','--data-root','/data/gb/Re-ID/dataset','--pretrained','/data/gb/Re-ID/pretrained/ViT-B-16.pt','--output',OUT]
        assert kw['cwd']==REMOTE and kw['env']==dict(SENTINEL='kept',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4') and kw['stderr']=='STDOUT'
        kw['stdout'].write('CHILD_EXIT'+str(exit_code));children.append(argv);return NS(returncode=exit_code)
    env=dict(json=json,os=NS(environ={'SENTINEL':'kept'}),subprocess=NS(run=run,STDOUT='STDOUT'),sys=NS(executable=PYTHON),time=NS(time=lambda:2),Path=P)
    error=None
    try: exec(code_without_imports(happy.env['wrapper'],'mock_wrapper'),env)
    except AssertionError as e: error=str(e)
    assert len(children)==1 and OUT+'.log' in fs
    assert json.loads(fs[OUT+'_exit.json'])==dict(exit_code=exit_code,finished=2)
    assert (error is None)==(exit_code==0)
for status in (0,1): wrapper_mock(status)
check('wrapper_log_exit_failure_no_retry','Success and actual nonzero child exit each produce one exclusive log and exit JSON; failed child then raises, with one child invocation and no retry; correct cwd and GPU0-only environment.')

result=dict(status='PASS',checks_count=len(checks),checks=checks,source_sha256=source_hashes,helper_sha256=helper_hash,torch_imports=0,real_remote_calls=0,real_process_spawns=0,gpu_forwards=0,optimizer_updates=0)
(HERE/'001-harness-results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='PASS',checks_count=len(checks))))
