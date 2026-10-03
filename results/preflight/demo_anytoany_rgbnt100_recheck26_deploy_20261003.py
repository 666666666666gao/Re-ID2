from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT,HOSTS,OPTIONS,command,remote_python

review=json.loads((PROJECT/'results/preflight/anytoany49_rgbnt100_recheck_review.json').read_text())
assert review['status']=='PASS' and not review['blockers']
assert review['helper_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
files=('missing_anytoany_development.py','launch_rgbnt100_anytoany_recheck.py')
new_sources={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in files}
assert new_sources==review['checked_source_sha256']
prior=json.loads((PROJECT/'results/preflight/axis_collaboration_v4_missing_development27_launch.json').read_text())
proof=prior['input_files']['RGBNT100_demo_s42']['files']
base=json.loads((PROJECT/'results/preflight/anytoany49_2026_launch.json').read_text())
sources=base['source_sha256'].copy();sources.pop('launch_anytoany_frozen.py')
assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in sources.items())
root,python=HOSTS['2026'];training=root+'/runs/dynamic_amp_comparison'
output=root+'/runs/anytoany49_demo_RGBNT100_samehost_recheck_20261003'
guard=f'''import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path({root!r});run=Path({training!r})/'RGBNT100_demo_s42'
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {sources!r}.items())
assert not Path({output!r}).exists() and not (root/'launch_rgbnt100_anytoany_recheck.py').exists()
for name,proof in {proof!r}.items():
 path=run/name;assert path.stat().st_size==proof['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==proof['sha256']
assert json.loads((run/'exit.json').read_text())['exit_code']==0
terminal=json.loads((run/'result.json').read_text());assert terminal['status']=='COMPLETE' and terminal['epochs']==50 and terminal['arguments']['variant']=='demo'
used=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert used<500
assert shutil.disk_usage(root).free>1000000000
print(json.dumps(dict(gpu0_memory_used_MiB=used,original_input_files={proof!r})))
'''
availability=json.loads(remote_python('2026',guard))
command(['scp',*OPTIONS,str(PROJECT/'launch_rgbnt100_anytoany_recheck.py'),'2026:'+root+'/launch_rgbnt100_anytoany_recheck.py'])
all_sources={**sources,**new_sources}
start=f'''import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path({root!r});output=Path({output!r});assert not output.exists()
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in {all_sources!r}.items())
assert int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<500
argv=[{python!r},'-u','launch_rgbnt100_anytoany_recheck.py','--training-root',{training!r},'--variant','demo','--output',str(output),'--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt']
log=Path(str(output)+'.log')
with log.open('x') as handle:child=subprocess.Popen(argv,cwd=root,start_new_session=True,stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
print(json.dumps(dict(host='2026',pid=child.pid,output=str(output),log=str(log),source_sha256={all_sources!r},variant='demo',started=time.time(),command=argv)))
'''
receipt=json.loads(remote_python('2026',start));receipt.update(availability=availability,observed_at=datetime.now().isoformat(timespec='seconds'))
target=PROJECT/'results/preflight/anytoany49_rgbnt100_recheck_launch.json';assert not target.exists()
target.write_bytes(json.dumps(receipt,indent=2).encode('utf-8'))
print('SAME_HOST_RGBNT100_RECHECK_STARTED',json.dumps(dict(pid=receipt['pid'],gpu=0)),flush=True)
