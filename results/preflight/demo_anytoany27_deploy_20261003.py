from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT, HOSTS, OPTIONS, command, remote_python
HOSTS['2027']=('/data/gb/Re-ID/DeMo-DualAxis','/data/gb/Re-ID/conda-envs/tri_reid/bin/python')

review=json.loads((PROJECT/'results/preflight/anytoany49_2027_schedule_review.json').read_text(encoding='utf-8'))
assert review['status']=='PASS' and not review['blockers']
assert review['helper_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
files=('missing_anytoany_development.py','launch_anytoany_frozen27.py')
new_sources={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in files}
assert review['checked_source_sha256']==new_sources
dependencies=('run_experiment.py','experiment_data.py','full_evaluation.py','missing_evaluation.py','scaled_axis_collaboration.py',
    'axis_collaboration.py','residual_dual_axis.py','dual_axis.py','modeling/make_model.py','utils/reid_evaluation.py','splits.json',
    'configs/MSVR310/DeMo.yml','configs/RGBNT201/DeMo.yml','configs/RGBNT100/DeMo.yml')
prior=json.loads((PROJECT/'results/preflight/axis_collaboration_v4_missing_development27_launch.json').read_text(encoding='utf-8'))
proofs={name:proof for name,proof in prior['input_files'].items() if name.endswith('_demo_s42')}
assert len(proofs)==3
launches=[]
for host,variant,training in [('2027','demo','runs/axis_collaboration_v4_missing_development27_input')]:
    root,python=HOSTS[host]
    output=root+'/runs/anytoany49_'+variant+'_20261003'
    sources={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in dependencies}
    if variant=='axis_mass_fullref':
        for name in ('mass_axis_collaboration.py','run_mass_experiment.py'):sources[name]=hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()
    guard=f'''import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path({root!r});training=root/{training!r}
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {sources!r}.items())
assert not Path({output!r}).exists() and all(not (root/name).exists() for name in {files!r})
used={{int(line.split(',')[0]):int(line.split(',')[1]) for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()}}
assert all(used[gpu]<500 for gpu in (1,2));assert shutil.disk_usage(root).free>1000000000
inputs={{}}
for dataset in ('MSVR310','RGBNT201','RGBNT100'):
 name=dataset+'_'+{variant!r}+'_s42';run=training/name
 terminal=json.loads((run/'result.json').read_text());assert terminal['status']=='COMPLETE' and terminal['epochs']==50
 assert terminal['arguments']['dataset']==dataset and terminal['arguments']['variant']=={variant!r}
 exit=run/'exit.json' if {variant!r}=='demo' else run.parent/(name+'_exit.json')
 assert json.loads(exit.read_text())['exit_code']==0
 paths=(run/'best.pth',run/'best_dev_arrays.npz',run/'result.json',exit)
 inputs[name]={{path.name:{{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}} for path in paths}}
 if {variant!r}=='demo':assert inputs[name]=={proofs!r}[name]['files']
print(json.dumps(dict(inputs=inputs,gpus={{str(gpu):used[gpu] for gpu in (1,2)}},disk_free_bytes=shutil.disk_usage(root).free)))
'''
    availability=json.loads(remote_python(host,guard))
    for name in files:command(['scp',*OPTIONS,str(PROJECT/name),host+':'+root+'/'+name])
    all_sources={**sources,**new_sources}
    pretrained='/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt' if host=='2026' else '/data/gb/Re-ID/pretrained/ViT-B-16.pt'
    data='/data/gaob/Re-ID/dataset' if host=='2026' else '/data/gb/Re-ID/dataset'
    start=f'''import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path({root!r});output=Path({output!r});assert not output.exists()
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {all_sources!r}.items())
used={{int(line.split(',')[0]):int(line.split(',')[1]) for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()}}
assert all(used[gpu]<500 for gpu in (1,2))
argv=[{python!r},'-u','launch_anytoany_frozen27.py','--training-root',str(root/{training!r}),'--variant',{variant!r},'--output',str(output),'--data-root',{data!r},'--pretrained',{pretrained!r}]
log=Path(str(output)+'.log')
with log.open('x') as handle:child=subprocess.Popen(argv,cwd=root,start_new_session=True,stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
print(json.dumps(dict(host={host!r},pid=child.pid,output=str(output),log=str(log),source_sha256={all_sources!r},variant={variant!r},started=time.time(),command=argv)))
'''
    receipt=json.loads(remote_python(host,start));receipt.update(availability=availability,observed_at=datetime.now().isoformat(timespec='seconds'))
    target=PROJECT/('results/preflight/anytoany49_'+host+'_launch.json');assert not target.exists()
    target.write_bytes(json.dumps(receipt,indent=2).encode('utf-8'));launches.append(receipt)
    print('ANYTOANY49_FROZEN_STARTED',json.dumps(dict(host=host,pid=receipt['pid'],gpu=[1,2],pairs=147)),flush=True)
