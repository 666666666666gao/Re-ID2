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

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

p = PROJECT / 'results/preflight'
started = p / 'cross_identity_coordinate_completion_wait_started.json'
assert not started.exists()
started.write_text(json.dumps(dict(pid=os.getpid(), status='WAIT_EXISTING_LOCAL_OBSERVER_ONLY',
    started=datetime.now().isoformat(timespec='seconds'), poll_seconds=240, neural_launches=0), indent=2) + '\n', encoding='utf-8')
print('CROSS_COORDINATE_COMPLETION_WAITER_STARTED', os.getpid(), flush=True)
target = p / 'cross_identity_coordinate_observer_terminal.json'
while not target.exists():
    time.sleep(240)
terminal = json.loads(target.read_text(encoding='utf-8'))
launch = json.loads((p / 'cross_identity_coordinate_2026_launch.json').read_text(encoding='utf-8'))
assert terminal['pid'] == launch['pid'] and not terminal['live']
assert terminal['result']['status'] == 'COMPLETE' and len(terminal['result']['runs']) == 3
assert len(terminal['exits']) == 6 and all(v['exit_code'] == 0 for v in terminal['exits'].values())
plan = json.loads((p / 'cross_identity_coordinate_plan.json').read_text(encoding='utf-8'))
sources = {**plan['previous_sources'], **plan['sources']}
remote_root, _ = HOSTS['2026']
campaign = launch['output']
code = f'''import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import hashlib,json,subprocess,sys
from pathlib import Path
r=Path({remote_root!r});c=Path({campaign!r})
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {sources!r}.items())
assert not (c/'independent_cpu_audit.json').exists()
result=subprocess.run([sys.executable,'-u','audit_cross_identity_coordinates.py','--root',str(c),'--data-root','/data/gaob/Re-ID/dataset'],cwd=r,text=True,capture_output=True)
(c/'independent_cpu_audit_stdout.log').write_text(result.stdout)
(c/'independent_cpu_audit_stderr.log').write_text(result.stderr)
(c/'independent_cpu_audit_exit.json').write_text(json.dumps(dict(exit_code=result.returncode))+'\\n')
assert result.returncode==0,result.stderr
print(result.stdout,end='')
'''
print('CROSS_COORDINATE_INDEPENDENT_CPU_STARTED_ONCE', flush=True)
print(remote_python('2026', code).strip(), flush=True)
remote_archive = campaign + '_text.tar.gz'
code = f'''import hashlib,json,tarfile
from pathlib import Path
c=Path({campaign!r});a=Path({remote_archive!r});assert not a.exists()
audit=json.loads((c/'independent_cpu_audit.json').read_text());assert audit['status']=='PASS' and audit['cases']==1029 and audit['perquery_count']==216090
assert not list(c.rglob('*.pth'))
files=[f for f in c.rglob('*') if f.is_file() and f.suffix in ('.json','.jsonl','.csv','.log')]
with tarfile.open(a,'w:gz') as tar:
 for f in files:tar.add(f,arcname=f.relative_to(c).as_posix(),recursive=False)
print(json.dumps(dict(status='COMPLETE_AUDITED_TEXT_ONLY',files_sha256={{f.relative_to(c).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},archive_sha256=hashlib.sha256(a.read_bytes()).hexdigest(),new_weights=0)))
'''
intake = json.loads(remote_python('2026', code))
root = PROJECT / 'results/cross_identity_coordinates_20261004'
root.mkdir(exist_ok=False)
archive = Path('C:/Users/gb/.codex_tmp/cross_identity_coordinates_20261004_text.tar.gz')
assert not archive.exists()
command(['scp', *OPTIONS, '2026:' + remote_archive, str(archive)])
assert hashlib.sha256(archive.read_bytes()).hexdigest() == intake['archive_sha256']
with tarfile.open(archive, 'r:gz') as tar:
    tar.extractall(root, filter='data')
assert all(hashlib.sha256((root / n).read_bytes()).hexdigest() == sha for n, sha in intake['files_sha256'].items())
(root / 'intake.json').write_text(json.dumps(intake, indent=2) + '\n', encoding='utf-8')
(root / 'completion_entrypoint.py').write_bytes(Path(__file__).read_bytes())
audit = json.loads((root / 'independent_cpu_audit.json').read_text(encoding='utf-8'))
metrics = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')
pairs = ('common_common','F_pre_F_pre','F_post_F_post','F_pre_common','common_F_pre','F_post_common','common_F_post')
groups = dict(normal=lambda q,g: q == g == 'RNT', all49=lambda q,g: True,
    same_availability=lambda q,g:q==g, overlap_mismatch=lambda q,g:q!=g and bool(set(q)&set(g)),
    source_disjoint=lambda q,g:not bool(set(q)&set(g)), partial_query_full_gallery=lambda q,g:q!='RNT' and g=='RNT',
    both_partial=lambda q,g:q!='RNT' and g!='RNT')
report = dict(status='COMPLETE_FROZEN_COMPATIBILITY_DIAGNOSTIC_NOT_FINAL_METHOD', cases=1029,
    condition_query_rows=216090, optimizer_updates=0, new_weights=0, official_test_uses=0, runs={},
    limits='Fixed M2 best/MSVR310/seed42/fit-dev. Cross-coordinate matching is diagnostic; means and condition-query counts are not independent samples. No gate intervention or new training result.')
rows = []
for variant, run in audit['runs'].items():
    grouped = {}
    for group, predicate in groups.items():
        selected = [c for c in run['conditions'] if predicate(c.split('_')[1], c.split('_')[3])]
        mean = {pair:{m:statistics.mean(run['conditions'][c]['metrics'][pair][m] for c in selected) for m in metrics} for pair in pairs}
        grouped[group] = dict(conditions=len(selected), sixmetrics_equal_condition_mean=mean,
            cross_post_minus_within_frequency_pp={pair:{m:mean[pair][m]-mean['F_post_F_post'][m] for m in metrics} for pair in ('F_post_common','common_F_post')},
            cross_post_minus_within_common_pp={pair:{m:mean[pair][m]-mean['common_common'][m] for m in metrics} for pair in ('F_post_common','common_F_post')})
    report['runs'][variant] = dict(selected_epoch=run['selected_epoch'], normal=run['conditions']['q_RNT_g_RNT'], groups=grouped,
        max_matrix_error=run['max_cross_distance_reconstruction_error'], max_sixmetric_error=run['max_sixmetric_recount_error_pp'])
    for condition, values in run['conditions'].items():
        for pair, measured in values['metrics'].items():
            rows.append(dict(variant=variant, condition=condition, pair=pair, **measured))
assert len(rows) == 1029
with (root / 'all1029_sixmetrics.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['variant','condition','pair',*metrics])
    writer.writeheader(); writer.writerows(rows)
(root / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print('CROSS_COORDINATE_ANALYSIS_COMPLETE', json.dumps(dict(cases=1029, normal={v:r['normal']['metrics'] for v,r in report['runs'].items()})), flush=True)
