"""Once-only text collection of all four closed H201 references, then CPU reduction."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tarfile

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command

pf=PROJECT/'results/preflight'
load=lambda path:json.loads(path.read_text(encoding='utf-8'))
actual=load(pf/'r201h_protocol_list_full_actual_session_20261006.json')
assert actual['status']=='ACTUAL_H201_FOUR_FULL50_PROTOCOL_REFERENCE_NORMAL_GT_LOCAL_RAW'
assert actual['exit_code']==0 and actual['successful_updates']==10588 and actual['additional_epochs']==200 and actual['native_updates']==12
assert len(actual['archives'])==4
launch=load(pf/'r201h_protocol_list_launch_source_review_20261006.json')
assert actual['sources_sha256']==launch['sources_sha256']
review=load(pf/'r201h_protocol_list_cpu_closeout_source_review_20261006.json')
assert review['status']=='PASS' and not review['blocking_findings']
assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
for value in actual['archives'].values():
    path=Path(value['local'])
    assert path.stat().st_size==value['file']['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==value['file']['sha256']
local=PROJECT/'results/r201h_protocol_list_20261006'
proof=pf/'r201h_protocol_list_terminal_text_intake_20261006.json'
package=pf/'r201h_protocol_list_closed_text_20261006.tar.gz'
assert not local.exists() and not proof.exists() and not package.exists()
remote='/data/gaob/Re-ID/DeMo-DualAxis/runs/r201h_protocol_list_20261006'
remote_package='/tmp/r201h_protocol_list_closed_text_20261006.tar.gz'
python='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
reuse=review['directly_reused_sources_sha256']
code=f'''import hashlib,json,tarfile
from pathlib import Path
project=Path('/data/gaob/Re-ID/DeMo-DualAxis');root=Path({remote!r});package=Path({remote_package!r})
assert all(hashlib.sha256((project/name).read_bytes()).hexdigest()==sha for name,sha in {reuse!r}.items() if not name.startswith('results/'))
controller=json.loads((root/'controller_result.json').read_text())
assert controller['status']=='COMPLETE' and controller['controls']==4 and controller['additional_epochs']==200
assert controller['successful_updates']==10588 and controller['native_updates']==12 and controller['paired_sampling_exact']
assert controller['normal_archives_local_verified']==4
for row in controller['runs']:
 run=root/'training'/row['name'];data=json.loads((run/'result.json').read_text())
 assert data['status']=='COMPLETE' and data['epochs']==50 and data['steps']==data['optimizer_steps']==2647 and data['amp_skipped_steps']==0
 assert (data['train_records'],data['query_records'],data['gallery_records'])==(3951,836,836)
 assert data['training_heldout_identities']==0 and data['training_coverage']==dict(eligible=3951,visited=3951,unvisited=[])
 assert data['descriptor_dim']==5120 and [p.name for p in run.glob('*.pth')]==['best.pth']
 for phase in ('training','audit'):assert json.loads((root/phase/(row['name']+'_exit.json')).read_text())['exit_code']==0
 audit=json.loads((run/'normal_cpu_audit.json').read_text())
 assert audit['status']=='PASS' and audit['max_metric_error']<1e-8 and audit['CMC50_and_per_query_and_groups']
 assert audit['installed_split_counts']==dict(train=3951,query=836,gallery=836)
 assert json.loads((run/'normal_local_archive.json').read_text())['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
 assert not (run/'best_official_arrays.npz').exists()
assert not package.exists();files={{}}
with tarfile.open(package,'w:gz') as archive:
 for path in sorted(root.rglob('*')):
  if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log','.txt'):
   name=path.relative_to(root).as_posix();content=path.read_bytes()
   files[name]=dict(bytes=len(content),sha256=hashlib.sha256(content).hexdigest())
   archive.add(path,arcname=name,recursive=False)
print(json.dumps(dict(files=files,package=dict(bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()))))'''
inventory=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(python)+' -'],input=code))
command(['scp',*OPTIONS,'2026:'+remote_package,str(package)])
assert package.stat().st_size==inventory['package']['bytes'] and hashlib.sha256(package.read_bytes()).hexdigest()==inventory['package']['sha256']
local.mkdir(exist_ok=False)
with tarfile.open(package,'r:gz') as archive:
    assert set(archive.getnames())==set(inventory['files'])
    archive.extractall(local,filter='data')
assert all((local/name).stat().st_size==value['bytes'] and hashlib.sha256((local/name).read_bytes()).hexdigest()==value['sha256'] for name,value in inventory['files'].items())
command(['ssh',*OPTIONS,'2026',shlex.quote(python)+' -'],input='from pathlib import Path\nPath('+repr(remote_package)+').unlink()\n')
package.unlink()
result=dict(status='ACTUAL_H201_FOUR_FULL50_NORMAL_GT_TEXT_VERIFIED',verified_at=datetime.now().isoformat(timespec='seconds'),
    files=len(inventory['files']),manifest=inventory['files'],local_root=str(local),raw_local_verified=4,new_neural_calls=0,new_optimizer_updates=0)
proof.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print('ACTUAL_H_CLOSED_TEXT_INTAKE '+str(len(inventory['files'])),flush=True)
print(command([sys.executable,'-X','utf8','-B','-S',str(PROJECT/'analyze_r201h_protocol_list.py')]),end='')
