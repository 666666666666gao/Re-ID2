from datetime import datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tarfile
import time

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS,command,remote_python

p=PROJECT/'results/preflight';start=p/'frequency_relation_m2_completion_wait_started.json'
assert not start.exists()
start.write_text(json.dumps(dict(pid=os.getpid(),status='WAIT_EXISTING_LOCAL_OBSERVER_ONLY',
 started=datetime.now().isoformat(timespec='seconds'),poll_seconds=240,neural_launches=0),indent=2)+'\n',encoding='utf-8')
print('M2_COMPLETION_WAITER_STARTED',os.getpid(),flush=True)
target=p/'frequency_relation_m2_observer_terminal.json'
while not target.exists():time.sleep(240)
terminal=json.loads(target.read_text(encoding='utf-8'))
launch=json.loads((p/'frequency_relation_m2_2026_launch.json').read_text(encoding='utf-8'))
assert terminal['pid']==launch['pid'] and not terminal['live'] and terminal['result']['status']=='COMPLETE'
assert len(terminal['result']['runs'])==3 and all(v['exit_code']==0 for v in terminal['exits'].values())
plan=json.loads((p/'frequency_relation_m2_plan.json').read_text(encoding='utf-8'))
sources={**plan['previous_sources'],**plan['sources']};remote_root,_=HOSTS['2026'];campaign=launch['output']
reference=remote_root+'/runs/'+plan['input_reference']
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
result=subprocess.run([sys.executable,'-u','audit_frequency_relation_trial.py','--root',str(c),'--reference-root',{reference!r},'--data-root','/data/gaob/Re-ID/dataset'],cwd=r,text=True,capture_output=True)
(c/'independent_cpu_audit_stdout.log').write_text(result.stdout)
(c/'independent_cpu_audit_stderr.log').write_text(result.stderr)
(c/'independent_cpu_audit_exit.json').write_text(json.dumps(dict(exit_code=result.returncode))+'\\n')
assert result.returncode==0,result.stderr
print(result.stdout,end='')
'''
print('M2_INDEPENDENT_CPU_AUDIT_STARTED_ONCE',flush=True)
print(remote_python('2026',code).strip(),flush=True)
remote_archive=campaign+'_text.tar.gz'
code=f'''import hashlib,json,tarfile
from pathlib import Path
c=Path({campaign!r});archive=Path({remote_archive!r});assert not archive.exists()
audit=json.loads((c/'independent_cpu_audit.json').read_text())
assert audit['status']=='PASS' and audit['cases']==2058 and audit['perquery_count']==432180
weights=list(c.rglob('*.pth'));assert len(weights)==3 and all(f.name=='best.pth' for f in weights)
files=[f for f in c.rglob('*') if f.is_file() and f.suffix in ('.json','.jsonl','.csv','.log')]
with tarfile.open(archive,'w:gz') as tar:
 for f in files:tar.add(f,arcname=f.relative_to(c).as_posix(),recursive=False)
print(json.dumps(dict(status='COMPLETE_AUDITED_TEXT_ONLY',files_sha256={{f.relative_to(c).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),weights=[dict(relative=f.relative_to(c).as_posix(),bytes=f.stat().st_size) for f in weights])))
'''
intake=json.loads(remote_python('2026',code))
root=PROJECT/'results/frequency_relation_m2_20261004';root.mkdir(exist_ok=False)
archive=Path('C:/Users/gb/.codex_tmp/frequency_relation_m2_20261004_text.tar.gz');assert not archive.exists()
command(['scp',*OPTIONS,'2026:'+remote_archive,str(archive)])
assert hashlib.sha256(archive.read_bytes()).hexdigest()==intake['archive_sha256']
with tarfile.open(archive,'r:gz') as tar:tar.extractall(root,filter='data')
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in intake['files_sha256'].items())
(root/'intake.json').write_text(json.dumps(intake,indent=2)+'\n',encoding='utf-8')
(root/'completion_entrypoint.py').write_bytes(Path(__file__).read_bytes())
audit=json.loads((root/'independent_cpu_audit.json').read_text(encoding='utf-8'))
m1=json.loads((PROJECT/'results/common_outlet_m1_v3_complete/analysis.json').read_text(encoding='utf-8'))
previous_utility=json.loads((PROJECT/'results/trained_outlet_utility_20261004/analysis.json').read_text(encoding='utf-8'))
metrics=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
stages=('deployed','base_common','M_pre','M_post','F_pre','F_post','M_aux','F_aux')
groups=dict(normal=lambda q,g:q==g=='RNT',all49=lambda q,g:True,same_availability=lambda q,g:q==g,
 overlap_mismatch=lambda q,g:q!=g and bool(set(q)&set(g)),source_disjoint=lambda q,g:not bool(set(q)&set(g)),
 partial_query_full_gallery=lambda q,g:q!='RNT' and g=='RNT',both_partial=lambda q,g:q!='RNT' and g!='RNT')
report=dict(status='COMPLETE_M2_SINGLE_FACTOR_NOT_FINAL_METHOD',runs={},fair_comparisons={},
 cases=2058,condition_query_rows=432180,limits='SingleMSVR fit/dev/seed42. Equal-condition means and repeated query occurrences are diagnostics. No official-test or three-dataset/multiseed superiority claim.')
rows=[]
for variant,run in audit['runs'].items():
 name='MSVR310_'+variant+'_s42';trained=json.loads((root/'original_mean/development'/name/'result.json').read_text(encoding='utf-8'))
 grouped={}
 for group,predicate in groups.items():
  conditions=[c for c in run['states']['conditions'] if predicate(c.split('_')[1],c.split('_')[3])]
  state=run['states']['conditions'];utility=run['utility'];prior=m1['runs']['original_mean/'+variant]['groups'][group]
  measured={m:statistics.mean(state[c]['metrics']['11'][m] for c in conditions) for m in metrics}
  grouped[group]=dict(conditions=len(conditions),sixmetrics_equal_condition_mean=measured,
   delta_prior_same_variant_M1_pp={m:measured[m]-prior['sixmetrics_equal_condition_mean'][m] for m in metrics},
   full_vs_base_equal_condition_delta={m:statistics.mean(state[c]['full_vs_base']['delta_pp'][m] for c in conditions) for m in metrics},
   harm_condition_query_occurrences=sum(state[c]['full_vs_base']['Rank1_harm_queries'] for c in conditions),
   rescue_condition_query_occurrences=sum(state[c]['full_vs_base']['Rank1_rescue_queries'] for c in conditions),
   utility_sixmetrics_equal_condition_mean={s:{m:statistics.mean(utility[c]['metrics'][s][m] for c in conditions) for m in metrics} for s in stages})
 report['runs'][variant]=dict(training={k:trained[k] for k in ('epochs','steps','optimizer_steps','amp_skipped_steps','parameters','trainable_parameters','descriptor_dim','peak_memory')},
  best_epoch=trained['best']['epoch'],normal=run['states']['conditions']['q_RNT_g_RNT'],
  normal_utility=run['utility']['q_RNT_g_RNT'],groups=grouped,relation_raw_first=run['relation_raw_first'],relation_raw_last=run['relation_raw_last'],sampling_matches_original_M1=True)
 if variant in ('axis_shared','frequency_shared'):
  report['runs'][variant]['normal_utility_delta_prior_M1_pp']={s:{m:run['utility']['q_RNT_g_RNT']['metrics'][s][m]-previous_utility['runs']['original_mean/'+variant]['normal']['metrics'][s][m] for m in metrics} for s in stages}
 for condition,values in run['states']['conditions'].items():
  for stage,measured in values['metrics'].items():rows.append(dict(variant=variant,condition=condition,protocol='six_states',stage=stage,**measured))
 for condition,values in run['utility'].items():
  for stage,measured in values['metrics'].items():rows.append(dict(variant=variant,condition=condition,protocol='eight_stages',stage=stage,**measured))
for control in ('frequency_shared','twins_shared'):
 report['fair_comparisons'][control]={g:{m:report['runs']['axis_shared']['groups'][g]['sixmetrics_equal_condition_mean'][m]-report['runs'][control]['groups'][g]['sixmetrics_equal_condition_mean'][m] for m in metrics} for g in groups}
assert len(rows)==2058
with (root/'all2058_sixmetrics.csv').open('w',encoding='utf-8',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['variant','condition','protocol','stage',*metrics]);writer.writeheader();writer.writerows(rows)
(root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('M2_ANALYSIS_COMPLETE',json.dumps(dict(cases=2058,normal={k:v['normal']['metrics']['11'] for k,v in report['runs'].items()},fair=report['fair_comparisons'])),flush=True)
