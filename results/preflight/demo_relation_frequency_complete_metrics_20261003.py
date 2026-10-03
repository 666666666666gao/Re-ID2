import argparse
import hashlib
import json
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, copy_file, remote_python

parser=argparse.ArgumentParser()
parser.add_argument('--run-name',nargs='+',required=True)
args=parser.parse_args()
for name in args.run_name:
 dataset=name.split('_')[0]
 assert dataset in ('RGBNT201','RGBNT100','MSVR310')
 campaign=PROJECT/'results/axis_collaboration_v9_relation_frequency_trial/development'
 folder=campaign/name
 local=json.loads((folder/'result.json').read_text(encoding='utf-8'))
 assert local['status']=='COMPLETE' and local['epochs']==50
 assert json.loads((campaign/(name+'_exit.json')).read_text(encoding='utf-8'))['exit_code']==0
 project=HOSTS['2026'][0]
 run=project+'/runs/axis_collaboration_v9_relation_frequency_trial/development/'+name
 code=f'''import hashlib,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.chdir({project!r})
sys.path.insert(0,{project!r})
import numpy as np
from experiment_data import split_records
from full_evaluation import full_metrics
root=Path({run!r})
terminal=json.loads((root/'result.json').read_text())
assert terminal['status']=='COMPLETE' and terminal['epochs']==50
assert json.loads((root.parent/(root.name+'_exit.json')).read_text())['exit_code']==0
arrays=np.load(root/'best_dev_arrays.npz')
_,dev,query,_,_=split_records(terminal['arguments']['data_root'],{dataset!r})
assert np.array_equal(arrays['query_indices'],query)
assert np.array_equal(arrays['ids'],[r[1] for r in dev])
assert np.array_equal(arrays['cameras'],[r[2] for r in dev]) and np.array_equal(arrays['scenes'],[r[3] for r in dev])
assert np.array_equal(arrays['names'],[Path(r[0] if isinstance(r[0],str) else r[0][0]).name for r in dev])
q=arrays['query_indices']
exclusion=arrays['scenes'] if {dataset!r}=='MSVR310' else arrays['cameras']
output=root/'development_metrics'
output.mkdir(exist_ok=False)
metrics=full_metrics(arrays['distances'],arrays['ids'][q],arrays['ids'],exclusion[q],exclusion,
 arrays['names'][q],arrays['cameras'][q],arrays['scenes'][q],output/'full_metrics')
assert all(abs(metrics[key]-terminal['best'][key])<1e-8 and abs(metrics[key]-terminal['strict_reload'][key])<1e-8 for key in ('mAP','Rank-1','Rank-5','Rank-10'))
receipt={{'dataset':{dataset!r},'run':{name!r},'scope':'CPU metrics from frozen dev-best arrays with CUDA hidden; no GPU forward/update/test',
 'installed_ground_truth_ids_cameras_scenes_names_and_query_order_equal':True,
 'best_dev_arrays_sha256':hashlib.sha256((root/'best_dev_arrays.npz').read_bytes()).hexdigest(),
 'original_and_strict_reload_four_metrics_match':True,'selected_epoch':terminal['best']['epoch']}}
(output/'metric_computation.json').write_text(json.dumps(receipt,indent=2))
files={{path.name:{{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}} for path in output.iterdir() if path.is_file()}}
print(json.dumps({{'metrics':{{key:metrics[key] for key in ('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')}},'files':files}}))
'''
 result=json.loads(remote_python('2026',code))
 for filename,proof in result['files'].items():
  destination=folder/'development_metrics'/filename
  copy_file('2026',run+'/development_metrics/'+filename,destination)
  data=destination.read_bytes()
  assert len(data)==proof['bytes'] and hashlib.sha256(data).hexdigest()==proof['sha256']
 (folder/'development_metrics/intake.json').write_bytes(json.dumps(result['files'],indent=2).encode('utf-8'))
 print('RELATION_FREQUENCY_COMPLETE_FROZEN_DEVELOPMENT_METRICS',name,json.dumps(result['metrics']),flush=True)
