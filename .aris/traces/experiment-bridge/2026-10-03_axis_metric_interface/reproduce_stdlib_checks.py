import argparse, ast, json, math, types
from pathlib import Path
p=Path.cwd()
text=(p/'launch_metric_mass_trial.py').read_text()
node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='main')
code=compile(ast.Module(body=[node],type_ignores=[]),'launch_metric_mass_trial.py','exec')
results=[]
for failure in [None,'tensor_exit','tensor_status','smoke_exit','smoke_status','train_exit','train_status','train_epochs']:
    records={}; calls=[]
    class MemoryPath:
        def __init__(self,s): self.s=str(s)
        def __truediv__(self,s): return MemoryPath(self.s+'/'+str(s))
        def __str__(self): return self.s
        def mkdir(self,**kwargs):
            if not kwargs.get('exist_ok',False): assert self.s not in records
            records[self.s]='DIRECTORY'
        def read_text(self): return json.dumps(records[self.s])
    class Parser:
        def add_argument(self,*args,**kwargs): pass
        def parse_args(self): return types.SimpleNamespace(data_root='DATA',pretrained='PUBLIC_CLIP',output='RUN')
    def execute(argv,output,name,gpu):
        calls.append({'name':name,'argv':argv,'gpu':gpu})
        phase='tensor' if name=='tensor' else ('smoke' if name=='smoke' else 'train')
        if failure==phase+'_exit': raise AssertionError('simulated nonzero child exit')
        target=argv[argv.index('--output')+1]
        if phase=='tensor': records[target+'/result.json']={'status':'FAIL' if failure=='tensor_status' else 'PASS_METRIC_INTERFACE_TENSOR_CONTRACT'}
        elif phase=='smoke': records[target+'/smoke.json']={'status':'FAIL' if failure=='smoke_status' else 'SMOKE_PASS'}
        else: records[target+'/result.json']={'status':'FAIL' if failure=='train_status' else 'COMPLETE','epochs':49 if failure=='train_epochs' else 50,'retrieval_interface':'metric_weighted_frequency_block'}
        return {'name':name,'exit_code':0}
    def write_json(path,value): records[str(path)]=value
    env={'argparse':types.SimpleNamespace(ArgumentParser=Parser),'Path':MemoryPath,'sys':types.SimpleNamespace(executable='PYTHON'),'json':json,'execute':execute,'write_json':write_json}
    exec(code,env)
    stopped=False
    try: env['main']()
    except AssertionError: stopped=True
    expected=3 if failure is None or failure.startswith('train') else (1 if failure.startswith('tensor') else 2)
    assert len(calls)==expected
    assert stopped==(failure is not None)
    assert ('RUN/controller_result.json' in records)==(failure is None)
    assert all(c['gpu']==1 for c in calls)
    if len(calls)==3:
        assert '--mode' in calls[-1]['argv'] and 'train' in calls[-1]['argv']
        assert calls[1]['argv'][calls[1]['argv'].index('--output')+1]!=calls[2]['argv'][calls[2]['argv'].index('--output')+1]
        assert not any('checkpoint' in arg or 'resume' in arg or 'smoke.pth' in arg for arg in calls[-1]['argv'])
    results.append({'case':failure or 'success','children':len(calls),'blocked_as_expected':stopped,'terminal_written':failure is None})
print('LAUNCHER_MOCK',json.dumps(results))
def norm(a): return math.sqrt(sum(x*x for x in a))
def normalize(a): return [x/norm(a) for x in a]
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def descriptor(b,f,w):
    scale=norm(b)*math.sqrt(w/(1-w))
    return normalize(b+[scale*x for x in normalize(f)])
maximum_weight_error=0.;maximum_pair_error=0.
for gq,gg in [(0.5,0.5),(0.2,0.8),(0.001,0.999)]:
    sq=1/(1+math.exp(-math.log(.05/.95)));wq=sq*gq;wg=sq*gg
    bq=[1.,2.,-3.];bg=[-2.,4.,1.];fq=[2.,-1.];fg=[1.,3.]
    q=descriptor(bq,fq,wq);g=descriptor(bg,fg,wg)
    weight_error=abs(sum(x*x for x in q[-2:])-wq)
    expected=math.sqrt((1-wq)*(1-wg))*dot(normalize(bq),normalize(bg))+math.sqrt(wq*wg)*dot(normalize(fq),normalize(fg))
    pair_error=abs(dot(q,g)-expected)
    maximum_weight_error=max(maximum_weight_error,weight_error);maximum_pair_error=max(maximum_pair_error,pair_error)
assert maximum_weight_error<1e-14 and maximum_pair_error<1e-14
print('STDLIB_MATH',json.dumps({'initial_sigmoid':sq,'maximum_weight_error':maximum_weight_error,'maximum_pair_formula_error':maximum_pair_error,'note':'No tensors, NN, AMP or FP32 runtime were executed.'}))
helper=Path('C:/Users/gb/.codex_tmp/demo_axis_metric26_deploy_20261003.py').read_text()
helper_tree=ast.parse(helper)
env={'root':'/review/remote','python':'/review/python','sources':{},'files':('one.py',),'output':'/review/output','all_sources':{}}
for assignment in helper_tree.body:
    if isinstance(assignment,ast.Assign) and isinstance(assignment.targets[0],ast.Name) and assignment.targets[0].id in ('guard','start'):
        name=assignment.targets[0].id
        rendered=eval(compile(ast.Expression(assignment.value),'helper_fstring','eval'),env)
        ast.parse(rendered)
        print('HELPER_EMBEDDED_AST',name,'PASS')
print('LOCAL_ONLY_NO_NEURAL_NO_SUBPROCESS')
