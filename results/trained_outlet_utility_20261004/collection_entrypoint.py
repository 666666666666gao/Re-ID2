from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tarfile

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS,command,remote_python

p=PROJECT/'results/preflight'
terminal=json.loads((p/'trained_outlet_utility_observer_terminal.json').read_text(encoding='utf-8'))
launch=json.loads((p/'trained_outlet_utility_2026_launch.json').read_text(encoding='utf-8'))
assert terminal['pid']==launch['pid'] and not terminal['live']
assert terminal['result']['status']=='COMPLETE' and len(terminal['completed'])==6
assert len(terminal['smokes'])==6 and all(v['exit_code']==0 for v in terminal['exits'].values())
plan=json.loads((p/'trained_outlet_utility_plan.json').read_text(encoding='utf-8'))
sources={**plan['old57_sources_unchanged'],**plan['sources']}
remote_root,python=HOSTS['2026'];campaign=launch['output']
code=f'''import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
import hashlib,json,subprocess,sys
from pathlib import Path
r=Path({remote_root!r});c=Path({campaign!r})
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {sources!r}.items())
assert not (c/'independent_cpu_audit.json').exists()
result=subprocess.run([sys.executable,'-u','audit_trained_outlet_utility.py','--root',str(c),'--data-root','/data/gaob/Re-ID/dataset'],cwd=r,text=True,capture_output=True)
(c/'independent_cpu_audit_stdout.log').write_text(result.stdout)
(c/'independent_cpu_audit_stderr.log').write_text(result.stderr)
(c/'independent_cpu_audit_exit.json').write_text(json.dumps(dict(exit_code=result.returncode))+'\\n')
assert result.returncode==0,result.stderr
print(result.stdout,end='')
'''
print('TRAINED_OUTLET_CPU_AUDIT_STARTED_ONCE',flush=True)
print(remote_python('2026',code).strip(),flush=True)
remote_archive=campaign+'_text.tar.gz'
code=f'''import hashlib,json,tarfile
from pathlib import Path
c=Path({campaign!r});archive=Path({remote_archive!r})
assert not archive.exists()
assert not list(c.rglob('*.pth'))
audit=json.loads((c/'independent_cpu_audit.json').read_text())
assert audit['status']=='PASS' and audit['cases']==2352 and audit['perquery_count']==493920
files=[f for f in c.rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.log')]
with tarfile.open(archive,'w:gz') as tar:
 for f in files:tar.add(f,arcname=f.relative_to(c).as_posix(),recursive=False)
print(json.dumps(dict(status='COMPLETE_CPU_AUDITED_TEXT_ONLY',files_sha256={{f.relative_to(c).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),new_weights=0,optimizer_updates=0)))
'''
intake=json.loads(remote_python('2026',code))
root=PROJECT/'results/trained_outlet_utility_20261004'
root.mkdir(exist_ok=False)
archive=Path('C:/Users/gb/.codex_tmp/trained_outlet_utility_20261004_text.tar.gz')
assert not archive.exists()
command(['scp',*OPTIONS,'2026:'+remote_archive,str(archive)])
assert hashlib.sha256(archive.read_bytes()).hexdigest()==intake['archive_sha256']
with tarfile.open(archive,'r:gz') as tar:tar.extractall(root,filter='data')
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in intake['files_sha256'].items())
(root/'intake.json').write_text(json.dumps(intake,indent=2)+'\n',encoding='utf-8')
(root/'collection_entrypoint.py').write_bytes(Path(__file__).read_bytes())
audit=json.loads((root/'independent_cpu_audit.json').read_text(encoding='utf-8'))
metrics=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
stages=plan['stages']
groups=dict(normal=lambda q,g:q==g=='RNT',all49=lambda q,g:True,
 same_availability=lambda q,g:q==g,
 overlap_mismatch=lambda q,g:q!=g and bool(set(q)&set(g)),
 source_disjoint=lambda q,g:not bool(set(q)&set(g)),
 partial_query_full_gallery=lambda q,g:q!='RNT' and g=='RNT',
 both_partial=lambda q,g:q!='RNT' and g!='RNT')
report=dict(status='COMPLETE_FROZEN_IDENTITY_UTILITY_NOT_RETRAINED_METHOD',runs={},
 cases=audit['cases'],condition_query_rows=audit['perquery_count'],optimizer_updates=0,new_weights=0,
 limits='Equal-condition means and repeated query occurrences are descriptive. Aux-to-route comparisons mix conditioning/routing/aggregation. No causal, final-test, multiseed or all-three-dataset superiority claim.')
rows=[]
for key,run in audit['runs'].items():
 pooling,variant=key.split('/')
 folder=root/pooling/('MSVR310_'+variant+'_s42')/'full'
 bank={}
 for availability in ('RNT','R','N','T','RN','RT','NT'):
  with (folder/('scale_'+availability+'.csv')).open(encoding='utf-8') as f:scalars=list(csv.DictReader(f))
  numeric=[n for n in scalars[0] if n not in ('names','availability','row')]
  bank[availability]={n:dict(mean=statistics.mean(float(v[n]) for v in scalars),median=statistics.median(float(v[n]) for v in scalars),min=min(float(v[n]) for v in scalars),max=max(float(v[n]) for v in scalars)) for n in numeric}
 grouped={}
 for group,predicate in groups.items():
  conditions=[c for c in run['conditions'] if predicate(c.split('_')[1],c.split('_')[3])]
  changes={}
  for comparison in ('M_projection','F_projection','M_aux_to_route','F_aux_to_route'):
   values=[run['conditions'][c][comparison] for c in conditions]
   changes[comparison]=dict(delta_pp={m:statistics.mean(v['delta_pp'][m] for v in values) for m in metrics},
    Rank1_harm_condition_query_occurrences=sum(v['Rank1_harm_queries'] for v in values),
    Rank1_rescue_condition_query_occurrences=sum(v['Rank1_rescue_queries'] for v in values))
  grouped[group]=dict(conditions=len(conditions),sixmetrics_equal_condition_mean={s:{m:statistics.mean(run['conditions'][c]['metrics'][s][m] for c in conditions) for m in metrics} for s in stages},comparisons=changes)
 report['runs'][key]=dict(best_epoch=run['selected_epoch'],normal=run['conditions']['q_RNT_g_RNT'],groups=grouped,scalar_banks=bank)
 for condition,value in run['conditions'].items():
  for stage,measured in value['metrics'].items():rows.append(dict(pooling=pooling,variant=variant,condition=condition,stage=stage,**measured))
assert len(rows)==2352
with (root/'all2352_sixmetrics.csv').open('w',encoding='utf-8',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['pooling','variant','condition','stage',*metrics]);writer.writeheader();writer.writerows(rows)
(root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('TRAINED_OUTLET_ANALYSIS_COMPLETE',json.dumps(dict(cases=2352,condition_query_rows=493920,normal={key:{s:value['normal']['metrics'][s]['mAP'] for s in stages} for key,value in report['runs'].items()})),flush=True)
