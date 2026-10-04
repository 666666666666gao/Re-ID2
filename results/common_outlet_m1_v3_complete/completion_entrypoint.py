from datetime import datetime
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import sys
import tarfile
import time

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python, sync_handoff

HOSTS['2027'] = ('/data/gb/Re-ID/DeMo-DualAxis', '/data/gb/Re-ID/conda-envs/tri_reid/bin/python')
p = PROJECT / 'results/preflight'
launch = json.loads((p / 'common_outlet_m1_v3_2026_launch.json').read_text(encoding='utf-8'))
started = p / 'common_outlet_m1_completion_wait_started_20261004.json'
assert not started.exists()
started.write_text(json.dumps(dict(status='WAIT_EXISTING_OBSERVER_TERMINAL', local_pid=os.getpid(),
    at=datetime.now().isoformat(timespec='seconds'), controller_pid=launch['pid'], local_poll_seconds=240,
    neural_launches=0, policy='Wait only on the existing local observer file; no duplicate SSH poller, no retry or training restart'),indent=2)+'\n',encoding='utf-8')
print('M1_COMPLETION_WAITER_STARTED',json.dumps(dict(local_pid=os.getpid(),controller_pid=launch['pid'])),flush=True)
terminal_path = p / 'common_outlet_m1_v3_observer_terminal.json'
while not terminal_path.exists():
    time.sleep(240)
terminal = json.loads(terminal_path.read_text(encoding='utf-8'))
assert terminal['pid'] == launch['pid'] and not terminal['live']
assert terminal['result']['status'] == 'COMPLETE' and len(terminal['result']['runs']) == 10
assert all(record['exit_code'] == 0 for record in terminal['exits'].values())
plan = json.loads((p / 'common_outlet_m1_plan.json').read_text(encoding='utf-8'))
source_map = {**plan['previous_sources'],**plan['sources']}
remote_root, _ = HOSTS['2026']
campaign = launch['output']
code = f'''import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
import hashlib,json,subprocess,sys
from pathlib import Path
r=Path({remote_root!r});c=Path({campaign!r})
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {source_map!r}.items())
end=json.loads((c/'controller_result.json').read_text())
assert end['status']=='COMPLETE' and len(end['runs'])==10
result=subprocess.run([sys.executable,'-u','audit_common_outlet_states.py','--root',str(c),'--data-root','/data/gaob/Re-ID/dataset'],cwd=r,check=True,text=True,capture_output=True)
print(result.stdout,end='')
'''
print('M1_ALL10_CPU_AUDIT_STARTED_ONCE',flush=True)
audit_stdout = remote_python('2026',code)
print(audit_stdout.strip(),flush=True)
remote_archive = campaign + '_text_completion.tar.gz'
code = f'''import hashlib,json,tarfile
from pathlib import Path
c=Path({campaign!r});destination=Path({remote_archive!r})
assert not destination.exists()
audit=json.loads((c/'independent_cpu_audit.json').read_text())
assert audit['status']=='PASS' and audit['cases']==2940
files=[f for f in c.rglob('*') if f.is_file() and f.suffix in ('.json','.jsonl','.csv','.log')]
with tarfile.open(destination,'w:gz') as tar:
 for f in files: tar.add(f,arcname=f.relative_to(c).as_posix(),recursive=False)
weights=sorted(c.rglob('*.pth'))
assert len(weights)==10 and all(f.name=='best.pth' for f in weights)
print(json.dumps(dict(status='COMPLETE_AUDITED_TEXT_ONLY',files_sha256={{f.relative_to(c).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},
 weight_inventory=[dict(relative=f.relative_to(c).as_posix(),bytes=f.stat().st_size) for f in weights],archive_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())))
'''
intake = json.loads(remote_python('2026',code))
root = PROJECT / 'results/common_outlet_m1_v3_complete'
root.mkdir(exist_ok=False)
archive = Path('C:/Users/gb/.codex_tmp/common_outlet_m1_v3_complete_text.tar.gz')
assert not archive.exists()
command(['scp',*OPTIONS,'2026:'+remote_archive,str(archive)])
assert hashlib.sha256(archive.read_bytes()).hexdigest() == intake['archive_sha256']
with tarfile.open(archive,'r:gz') as tar:
    tar.extractall(root,filter='data')
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in intake['files_sha256'].items())
audit = json.loads((root/'independent_cpu_audit.json').read_text(encoding='utf-8'))
metrics = ('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
def sets(condition):
    _,q,_,g = condition.split('_')
    return q,g
groups = {
    'normal':lambda q,g:q==g=='RNT',
    'all49':lambda q,g:True,
    'same_availability':lambda q,g:q==g,
    'overlap_mismatch':lambda q,g:q!=g and bool(set(q)&set(g)),
    'source_disjoint':lambda q,g:not(set(q)&set(g)),
    'partial_query_full_gallery':lambda q,g:q!='RNT' and g=='RNT',
    'both_partial':lambda q,g:q!='RNT' and g!='RNT',
}
report = dict(status='COMPLETE_M1_EVIDENCE_NOT_FINAL_METHOD', runs={}, comparisons={}, cases=2940,
    perquery_count=audit['perquery_count'], interpretation='Equal-condition means and repeated condition-query counts are diagnostics. No official test, multiseed or three-dataset conclusion is made.')
rows = []
for key, raw in audit['runs'].items():
    pooling,variant = key.split('/')
    run = root/pooling/'development'/('MSVR310_'+variant+'_s42')
    trained = json.loads((run/'result.json').read_text(encoding='utf-8'))
    orders = [json.loads(line) for line in (run/'batch_orders.jsonl').read_text(encoding='utf-8').splitlines()]
    assert trained['epochs']==50 and len(orders)==trained['steps']
    report['runs'][key] = dict(training={k:trained[k] for k in ('epochs','steps','optimizer_steps','amp_skipped_steps','parameters','trainable_parameters','descriptor_dim','peak_memory')},
        best_epoch=trained['best']['epoch'], normal=raw['conditions']['q_RNT_g_RNT'],groups={})
    for group,predicate in groups.items():
        selected = [c for c in raw['conditions'] if predicate(*sets(c))]
        report['runs'][key]['groups'][group] = dict(conditions=len(selected),
            sixmetrics_equal_condition_mean={m:statistics.mean(raw['conditions'][c]['metrics']['11'][m] for c in selected) for m in metrics},
            full_vs_base_equal_condition_delta={m:statistics.mean(raw['conditions'][c]['full_vs_base']['delta_pp'][m] for c in selected) for m in metrics},
            harm_condition_query_occurrences=sum(raw['conditions'][c]['full_vs_base']['Rank1_harm_queries'] for c in selected),
            rescue_condition_query_occurrences=sum(raw['conditions'][c]['full_vs_base']['Rank1_rescue_queries'] for c in selected))
    for condition,values in raw['conditions'].items():
        for state,values_state in values['metrics'].items():
            rows.append(dict(pooling=pooling,variant=variant,condition=condition,state=state,**values_state))
    reference = root/'original_mean/development/MSVR310_axis_shared_s42/batch_orders.jsonl'
    base_orders = [json.loads(line) for line in reference.read_text(encoding='utf-8').splitlines()]
    assert len(base_orders)==len(orders) and all(all(a[k]==b[k] for k in ('epoch','step','names','partial_set')) for a,b in zip(base_orders,orders))
    report['runs'][key]['batch_sampling_matches_all_controls'] = True
for pooling in ('original_mean','eligible_mean','seed_query'):
    axis = report['runs'][pooling+'/axis_shared']
    for reference in ('frequency_shared','twins_shared'):
        other = report['runs'][pooling+'/'+reference]
        report['comparisons'][pooling+'/'+reference] = {g:{m:axis['groups'][g]['sixmetrics_equal_condition_mean'][m]-other['groups'][g]['sixmetrics_equal_condition_mean'][m] for m in metrics} for g in groups}
summary = root/'analysis.json'
summary.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
with (root/'all2940_sixmetrics.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['pooling','variant','condition','state',*metrics]);writer.writeheader();writer.writerows(rows)
assert len(rows)==2940
intake['local_observer_terminal_sha256']=hashlib.sha256(terminal_path.read_bytes()).hexdigest()
(root/'intake.json').write_text(json.dumps(intake,indent=2)+'\n',encoding='utf-8')
(root/'completion_entrypoint.py').write_bytes(Path(__file__).read_bytes())
now = datetime.now().isoformat(timespec='seconds')
table = '| 出口/模型 | mAP | mINP | R1 | R5 | R10 | R20 | 最佳轮 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for key,run in report['runs'].items():
    table+='| '+key+' | '+' | '.join(f'{run["normal"]["metrics"]["11"][m]:.6f}' for m in metrics)+f' | {run["best_epoch"]} |\n'
comparison = '| 双轴对同出口公平对照 | 完整ΔmAP/R1 | 49均值ΔmAP/R1 | 不相交ΔmAP/R1 |\n|---|---:|---:|---:|\n'
for key,values in report['comparisons'].items():
    comparison+='| '+key+' | '+' | '.join(f'{values[g]["mAP"]:+.6f}/{values[g]["Rank-1"]:+.6f}' for g in ('normal','all49','source_disjoint'))+' |\n'
mechanism = '| 双轴出口 | 完整11−00 mAP/R1 | 完整误伤/救回 | 49均值11−00 mAP/R1 |\n|---|---:|---:|---:|\n'
for pooling in ('original_mean','eligible_mean','seed_query'):
    run=report['runs'][pooling+'/axis_shared'];change=run['normal']['full_vs_base'];mean=run['groups']['all49']['full_vs_base_equal_condition_delta']
    mechanism+=f'| {pooling} | {change["delta_pp"]["mAP"]:+.6f}/{change["delta_pp"]["Rank-1"]:+.6f} | {change["Rank1_harm_queries"]}/{change["Rank1_rescue_queries"]} | {mean["mAP"]:+.6f}/{mean["Rank-1"]:+.6f} |\n'
section=f'''## 最新终态：M1十组完整训练与全量独立核算完成（{now}）

原始平均、合法关系平均、共享查询三种公共出口的双轴/普通频域/普通双专家九组，以及同公共查询DeMo一组，全部完成fresh50。仅2026物理GPU2/3、每卡串行、最多两项并行；温度功率限制取消。十组均按开发mAP最高且最早并列checkpoint严格重载，批次样本名、模态集合、轮数及顺序逐项一致。训练真实/跳步计数全部保留，不从尝试数推断真实更新数。

全490个完整模型条件、2940个六状态条件完成；独立CPU从实际安装数据重建GT，重新排序原始FP32距离，核对六指标、CMC1..50、身份/相机/场景分组、逐查询AP/INP/首末匹配及贡献记录，全量{audit['perquery_count']}条条件—查询记录通过。重复查询不是独立样本数。以下完整/缺失比较同时保留正负结果：

{table}

{comparison}

{mechanism}

全部2940条件六项指标在results/common_outlet_m1_v3_complete/all2940_sixmetrics.csv；原始49条件、六状态、分组、逐查询CSV、训练/launch/exit/log文本及独立核算在同目录，完整组别比较见analysis.json。原始NPZ与十个best.pth留在远端，每实验仅一个best，未复制权重/数据图像到GitHub。该轮只回答公共出口单因素问题，未同时加入关系保持或新跨集合训练。

这份完整证据尚不等于方法定型或原Goal完成：仍须据公平对照、真实11−00/误伤/救回选择后续单因素，明确候选后冻结结构、至少三个配对种子和独立身份确认，再完成统一三个数据集各mAP/Rank-1超过完整DeMo2点、全部缺失条件和真实成本。未进行官方测试选择、没有拼接不同出口最好指标成统一方法。以下旧记录是历史状态。

'''
doc=PROJECT/'docs/实验交接.md';content=doc.read_text(encoding='utf-8');marker='<!-- CURRENT_DEMO_STATUS_START -->\n';assert content.count(marker)==1
doc.write_text(content.replace(marker,marker+section,1),encoding='utf-8');doc_sha=sync_handoff()
plan['status']='COMPLETE_M1_EVIDENCE_PENDING_SINGLE_FACTOR_RESEARCH_DECISION'
plan['actual_neural_runs']=63;plan['actual_50_epoch_runs']=10;plan['actual_50_epoch_completions']=10
plan['actual_neural_runs_scope']='Actual complete history:three CUDA verifier processes(firstfailed/twoPASS),twenty smokes(nineteenPASS/oneoldfailed),ten fresh50 trainings and thirty frozen49/state-smoke/state-full processes. Total63 childstarts, no new NN launch by this CPU completion helper.'
plan['actual_full_model_conditions']=490;plan['actual_six_state_conditions']=2940
plan['actual_independent_CPU_condition_query_rows']=audit['perquery_count']
plan['complete_evidence']='results/common_outlet_m1_v3_complete/analysis.json'
(p/'common_outlet_m1_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
goal_path=p/'research_goal_optimized_20261003.json';goal=json.loads(goal_path.read_text(encoding='utf-8'));goal['updated_at']=now
for milestone in goal['milestones']:
    if milestone['id']=='M1':
        milestone['status']=plan['status'];milestone['actual_preparation']='Ten fresh50,490 complete49conditions,2940sixstates and full independent installed-GT CPU audit actuallycomplete. No final candidate/multiseed/three-dataset claim.'
goal_path.write_text(json.dumps(goal,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
files=[f for f in root.rglob('*') if f.is_file()]
assert all(f.suffix not in ('.pth','.pt','.npz') for f in files)
expected={f.relative_to(root).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
manifest=root/'mirror_manifest.json'
manifest.write_text(json.dumps(expected,indent=2)+'\n',encoding='utf-8')
manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
files.append(manifest)
names=['common_outlet_m1_plan.json','research_goal_optimized_20261003.json']
for host,(r,_) in HOSTS.items():
    command(['scp',*OPTIONS,'-r',str(root),host+':'+r+'/results/'])
    command(['scp',*OPTIONS,*[str(p/n) for n in names],host+':'+r+'/results/preflight/'])
    code='import hashlib,json;from pathlib import Path;r=Path('+repr(r+'/results/'+root.name)+');m=r/"mirror_manifest.json";assert hashlib.sha256(m.read_bytes()).hexdigest()=='+repr(manifest_sha)+';expected=json.loads(m.read_text());assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in expected.items());print(json.dumps(dict(status="PASS",files=len(expected)+1)))'
    checked=json.loads(remote_python(host,code))
    assert checked==dict(status='PASS',files=len(files))
    print('M1_COMPLETE_AUDITED_TEXT_AND_ONE_DOC_EXACT',host,flush=True)
receipt=p/'common_outlet_m1_completion_sync_20261004.json';assert not receipt.exists()
receipt.write_text(json.dumps(dict(at=now,status='COMPLETE_M1_REPORT_FULL_GOAL_UNMET',handoff_sha256=doc_sha,
    copies=['repository','Desktop','2025','2026','2027'],file_sha256=expected,cases=2940,perquery_count=audit['perquery_count']),indent=2)+'\n',encoding='utf-8')
assert not command(['git','diff','--cached','--name-only'],cwd=PROJECT).strip()
command(['git','add','-f','docs/实验交接.md',root.relative_to(PROJECT).as_posix(),*['results/preflight/'+n for n in names],receipt.relative_to(PROJECT).as_posix(),started.relative_to(PROJECT).as_posix()],cwd=PROJECT)
command(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=PROJECT)
command(['git','commit','-m','Complete all M1 pooling controls and independent full-condition retrieval audit'],cwd=PROJECT)
command(['git','push','origin','main'],cwd=PROJECT);head=command(['git','rev-parse','HEAD'],cwd=PROJECT).strip()
assert command(['git','ls-remote','origin','refs/heads/main'],cwd=PROJECT).split()[0]==head
note=f'\nDeMo {now} M1all10fresh50/490fullconditions/2940sixstates/{audit["perquery_count"]}CPUrows actuallyCOMPLETE/PASS. Originalobserverterminalverified,CPUfinalauditor invokedONCE/noTorchNN/no retry. Samebatchsamplesequencesall10exact;only10best.pth retained. Published{head};ONEdoc5SHA{doc_sha};allownedtext3mirrors exact. FullgoalUNMET,M1decisionpending; no3seed/officialtest/3datasetsclaimed. OriginalNNcontrollerterminal; completionnativejobfinished0. Do notrerun completion/deploy/observer.\n'
for name in (f'memory/{datetime.now().date().isoformat()}.md','MEMORY.md'):
    with (Path('C:/Users/gb')/name).open('a',encoding='utf-8') as f:f.write(note)
print('M1_FULL_ROUND_PUBLISHED',json.dumps(dict(head=head,handoff_sha256=doc_sha,cases=2940)),flush=True)
