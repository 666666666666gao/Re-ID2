import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, 'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python

p = PROJECT / 'results/preflight'
launch = json.loads((p / 'common_outlet_m1_v3_2026_launch.json').read_text(encoding='utf-8'))
plan = json.loads((p / 'common_outlet_m1_plan.json').read_text(encoding='utf-8'))
sources = {**plan['previous_sources'], **plan['sources']}
root, _ = HOSTS['2026']
campaign = launch['output']
target = PROJECT / 'results/common_outlet_m1_v3_original_pair'
target.mkdir(exist_ok=False)
code = f'''import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
import hashlib,json,sys,time
from pathlib import Path
r=Path({root!r});sys.path.insert(0,str(r))
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {sources!r}.items())
from audit_shared_identity_states import audit_variant,installed_gt
assert 'torch' not in sys.modules
c=Path({campaign!r});pooling=c/'original_mean'
variants=('axis_shared','frequency_shared')
for v in variants:
 name='MSVR310_'+v+'_s42'
 run=pooling/'development'/name
 terminal=json.loads((run/'result.json').read_text())
 assert terminal['status']=='COMPLETE' and terminal['epochs']==50
 assert terminal['arguments']['pooling']=='original_mean'
 assert json.loads((pooling/'development'/(name+'_exit.json')).read_text())['exit_code']==0
 assert len(list(run.glob('*.pth')))==1 and (run/'best.pth').is_file()
 assert json.loads((pooling/'frozen49'/(name+'_exit.json')).read_text())['exit_code']==0
 assert json.loads((pooling/'controlled_states'/name/'full_exit.json').read_text())['exit_code']==0
out=pooling/'independent_original_pair_cpu_audit.json'
assert not out.exists()
started=time.time()
gt=installed_gt(Path('/data/gaob/Re-ID/dataset'),r/'splits.json')
runs={{v:audit_variant(pooling/'controlled_states',v,gt) for v in variants}}
assert 'torch' not in sys.modules
report=dict(status='PASS',pooling='original_mean',cases=588,perquery_count=588*len(gt['query_indices']),
 source='installed training filenames, fixed dev IDs and independent lexicographic rank of stored FP32 distances',
 runs=runs,torch_imported=False,optimizer_updates=0,official_test_uses=0,seconds=time.time()-started,
 source_sha256={sources!r})
out.write_text(json.dumps(report,indent=2)+'\\n')
print(json.dumps(dict(status=report['status'],cases=report['cases'],perquery_count=report['perquery_count'],seconds=report['seconds'],output=str(out))))
'''
remote = json.loads(remote_python('2026', code))
assert remote['status'] == 'PASS' and remote['cases'] == 588
command(['scp', *OPTIONS, '2026:' + remote['output'], str(target / 'independent_cpu_audit.json')])
for variant in ('axis_shared', 'frequency_shared'):
    run = 'MSVR310_' + variant + '_s42'
    folder = target / variant
    folder.mkdir()
    for rel, label in [('development/' + run + '/result.json', 'training_result.json'),
                       ('development/' + run + '/epochs.csv', 'epochs.csv'),
                       ('development/' + run + '/batch_orders.jsonl', 'batch_orders.jsonl'),
                       ('frozen49/' + run + '/result.json', 'frozen49_result.json'),
                       ('controlled_states/' + run + '/full/result.json', 'six_state_result.json')]:
        command(['scp', *OPTIONS, '2026:' + campaign + '/original_mean/' + rel, str(folder / label)])
(target / 'cpu_entrypoint.py').write_bytes(Path(__file__).read_bytes())
(target / 'intake.json').write_text(json.dumps(dict(status='RAW_CPU_PAIR_AUDIT_PASS',remote=remote,
    files_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in target.glob('*') if f.is_file()},
    pooling='original_mean',complete_campaign_claim=False),indent=2)+'\n',encoding='utf-8')
print('M1_ORIGINAL_PAIR_INDEPENDENT_CPU_PASS',json.dumps(remote),flush=True)
