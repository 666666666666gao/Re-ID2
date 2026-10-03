"""Remaining targeted stdlib checks: original mask branch and idle wait."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

ROOT=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
TRACE=ROOT/'.aris/traces/experiment-bridge/2026-10-03_missing_development'
missing_tree=ast.parse((ROOT/'missing_evaluation.py').read_text())
model_tree=ast.parse((ROOT/'modeling/make_model.py').read_text())
main_forward=next(node for node in ast.walk(model_tree) if isinstance(node,ast.FunctionDef) and node.name=='forward' and any(isinstance(n,ast.Attribute) and n.attr=='miss_type' for n in ast.walk(node)))
branch=main_forward.body[0].orelse[:4]
forward=ast.parse('def primitive_forward(self, x, **kwargs):\n return None').body[0]
forward.body=copy.deepcopy(branch)+ast.parse("return tuple((tuple(RGB.values),tuple(NI.values),tuple(TI.values)))").body
ast.fix_missing_locations(forward)

class Value:
    def __init__(self,values): self.values=list(values)
    def __len__(self): return len(self.values)
    def cuda(self,**kwargs): return self

mask=copy.deepcopy(next(node for node in missing_tree.body if isinstance(node,ast.FunctionDef) and node.name=='mask_images'))
verify=copy.deepcopy(next(node for node in missing_tree.body if isinstance(node,ast.FunctionDef) and node.name=='verify_original_mask'))
verify.decorator_list=[]
missing=ast.literal_eval(next(node for node in missing_tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MISSING' for t in node.targets)).value)
images={key:Value([.2,-.3,.7]) for key in ('RGB','NI','TI')}
ns=dict(torch=NS(zeros_like=lambda value:Value([0]*len(value)),equal=lambda a,b:a==b),MISSING=missing,
        make_loader=lambda *args:[(images,None,Value([0,1,2]),Value([0,0,1]),None)])
exec(compile(ast.Module(body=[forward,mask,verify],type_ignores=[]),'mask_primitive_check','exec'),ns)
class Model:
    miss_type='nothing'
    calls=0
    def __call__(self,*args,**kwargs):
        self.calls+=1
        return ns['primitive_forward'](self,*args,**kwargs)
model=Model()
result=ns['verify_original_mask'](model,list(range(64)),None,42)
assert result['original_forward_bitwise_equal_masks']==list(missing) and model.calls==12 and model.miss_type=='nothing'
checks=[dict(name='actual_verify_original_mask_executes_six_comparisons_against_actual_DeMo_mask_branch',status='PASS',comparisons=6,forwards=12,scope='Primitive arrays, source mask branch only; not actual neural forward')]
class WrongModel(Model):
    def __call__(self,*args,**kwargs):
        result=super().__call__(*args,**kwargs)
        return result+(('wrong',),) if self.miss_type=='nt' else result
try:
    ns['verify_original_mask'](WrongModel(),list(range(64)),None,42)
except AssertionError:
    checks.append(dict(name='actual_original_mask_check_rejects_mismatch_on_sixth_mask',status='PASS'))
else: raise AssertionError('Wrong original mask accepted')
idle_node=next(node for node in ast.parse((ROOT/'launch_axis_scaled.py').read_text()).body if isinstance(node,ast.FunctionDef) and node.name=='idle')
usage=iter([2700,500,499]); sleeps=[]; commands=[]
def query(command,**kwargs):
    commands.append(command)
    return str(next(usage))
idle_ns=dict(subprocess=NS(check_output=query),time=NS(sleep=sleeps.append),print=lambda *a,**k:None)
exec(compile(ast.Module(body=[idle_node],type_ignores=[]),'idle_mock','exec'),idle_ns)
idle_ns['idle'](3)
assert sleeps==[240,240] and len(commands)==3 and all(c[c.index('-i')+1]=='3' for c in commands)
checks.append(dict(name='actual_idle_waits_at_500_and_above_returns_only_below500',status='PASS',usage=[2700,500,499],mock_sleep_seconds=sleeps))
(TRACE/'supplement_validation.json').write_text(json.dumps(dict(status='PASS',checks=checks,check_count=len(checks)),indent=2),encoding='utf-8')
print(json.dumps(dict(status='PASS',checks=len(checks))))
