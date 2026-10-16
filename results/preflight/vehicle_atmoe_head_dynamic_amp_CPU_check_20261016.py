import ast,copy,csv,hashlib,json,math,tempfile
from pathlib import Path
P=Path('C:/Users/gb/projects/demo_dual_axis_20261002');PF=P/'results/preflight'
files=['audit_vehicle_atmoe_head_normal.py','resume_vehicle_atmoe_head_baselines.py','results/preflight/vehicle_atmoe_head_dynamic_amp_deploy_20261016.py','results/preflight/observe_vehicle_atmoe_dynamic_amp_start_20261016.py']
for name in files:ast.parse((P/name).read_text('utf-8-sig'),filename=name)
old=ast.parse((P/'launch_vehicle_atmoe_head_baselines.py').read_text('utf-8-sig'))
assignment=next(n for n in old.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CASES' for t in n.targets))
namespace={};exec(compile(ast.Module(body=[assignment],type_ignores=[]),'CASES','exec'),namespace);cases=namespace['CASES']
tree=ast.parse((P/files[1]).read_text());body=next(n.body for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
a=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='evidence' for t in n.targets))
native=compile(ast.Module(body=body[a:a+2],type_ignores=[]),'native_actual_source_slice','exec')
actual=json.loads((PF/'vehicle_atmoe_head_native_failure_actual_20261016.json').read_text())['files']
checks=[]
with tempfile.TemporaryDirectory(prefix='vehicle_amp_cpu_') as temporary:
    root=Path(temporary)
    for name,data in actual.items():
        file=root/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text(json.dumps(data))
    ns=dict(Path=Path,json=json,math=math,old=root,CASES=cases);exec(native,ns)
    assert sum(r['native']['steps'] for r in ns['evidence'])==6 and sum(r['native']['attempts'] for r in ns['evidence'])==7
    checks.append(dict(name='actual_native_total6_updates7_attempts1skip',passed=True))
    file=root/'native/RGBNT100_atmoe32_demo_s42_smoke/smoke.json';original=json.loads(file.read_text())
    mutations=[('wrong_updates',lambda d:d.update(steps=4)),('wrong_skip',lambda d:d.update(amp_skipped_steps=0)),('false_reload',lambda d:d.update(strict_reload_equal=False)),('wrong_head',lambda d:d['arguments'].update(atmoe_heads=8)),('nonfinite_loss',lambda d:d['details'][2].update(loss=float('nan')))]
    for name,edit in mutations:
        data=copy.deepcopy(original);edit(data);file.write_text(json.dumps(data));rejected=False
        try:exec(native,dict(Path=Path,json=json,math=math,old=root,CASES=cases))
        except AssertionError:rejected=True
        assert rejected,name;checks.append(dict(name=name,rejected=True))
    file.write_text(json.dumps(original))
    tree=ast.parse((P/files[0]).read_text());body=next(n.body for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    a=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='orders' for t in n.targets))
    b=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='selected' for t in n.targets))
    count=compile(ast.Module(body=body[a:b],type_ignores=[]),'formal_accounting_source_slice','exec')
    orders=[dict(epoch=i,step=i,optimizer_updated=(i!=3),loss=1.,amp_scale=256.,names=['one','two']) for i in range(1,51)]
    (root/'batch_orders.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in orders))
    metrics=['mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20']
    rows=[dict(epoch=i,steps=i,optimizer_steps=i-int(i>=3),amp_skipped_steps=int(i>=3),epoch_unique_train_records=2,cumulative_unique_train_records=2,loss=1.,amp_scale=256.,**{key:50 for key in metrics}) for i in range(1,51)]
    def table():
        with (root/'epochs.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    table()
    trained=dict(steps=50,optimizer_steps=49,amp_skipped_steps=1,training_coverage=dict(eligible=2,visited=2,unvisited=[]))
    ns=dict(Path=Path,json=json,csv=csv,math=math,run=root,trained=trained,expected_attempts=50,splits={'train':[{'name':'one'},{'name':'two'}]},METRICS=metrics)
    exec(count,ns);assert (ns['attempts'],ns['updates'],ns['skips'])==(50,49,1)
    checks.append(dict(name='formal50_epoch_dynamic_skip_accounting',passed=True))
    for name,edit in [('attempts_as_updates',lambda d:d.update(optimizer_steps=50)),('hidden_skip',lambda d:d.update(amp_skipped_steps=0))]:
        data=copy.deepcopy(trained);edit(data);rejected=False
        try:exec(count,ns|dict(trained=data))
        except AssertionError:rejected=True
        assert rejected,name;checks.append(dict(name=name,rejected=True))
    rows[-1]['optimizer_steps']=50;table();rejected=False
    try:exec(count,ns)
    except AssertionError:rejected=True
    assert rejected;checks.append(dict(name='wrong_final_epoch_cumulative_updates',rejected=True))
result=dict(status='PASS',checks=checks,source_AST_parses=4,new_neural_calls=0,project_imports=0,SSH_calls=0,GPU_queries=0,
    scope='Actual native evidence plus exact AST accounting blocks and focused synthetic50epoch ledger; no installed-GT runtime audit or formal result implied.',
    previous_harness_failure='First local AST-only harness used literal_eval on dict(...) CASES; failed before running checks. Harness now executes only the constant assignment; scientific sources unchanged.',
    sources_sha256={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in files})
(PF/'vehicle_atmoe_head_dynamic_amp_CPU_accounting_20261016.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],checks=len(checks),new_neural_calls=0)))
