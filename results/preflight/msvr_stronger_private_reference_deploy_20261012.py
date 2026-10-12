"""Whole-official normal reference reload on GPU2; no training admission."""
from pathlib import Path
from datetime import datetime
import base64,hashlib,json,shlex,sys

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
review=load(pf/'msvr_stronger_private_reference_source_review_20261012.json')
assert review['status']=='PASS' and not review['blocking_findings']
assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
target=PROJECT/'results/msvr_stronger_private_reference_20261012';assert not target.exists()
source='evaluate_msvr_stronger_private_reference.py';loader='msvr_stronger_private_teacher.py'
for n in (source,loader):command(['scp',*OPTIONS,str(PROJECT/n),'2026:'+REMOTE+'/'+n])
runtime=review['directly_reused_sources_sha256']|{n:review['sources_sha256'][n] for n in (source,loader)}
arrays=load(pf/'msvr_strong_shared_arrays_present_CPU_actual_20261012.json')['record']
assert arrays['exists'] and arrays['bytes']==37529195
code=f'''import hashlib,json,sys
from pathlib import Path
from gpu_thermal_execute import execute
r=Path({REMOTE!r});root=r/'runs/msvr_stronger_private_reference_20261012';assert not root.exists()
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {runtime!r}.items())
p=Path({arrays['path']!r});assert p.stat().st_size=={arrays['bytes']} and hashlib.sha256(p.read_bytes()).hexdigest()=={arrays['sha256']!r}
root.mkdir();argv=[sys.executable,'-u',{source!r},'--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt','--anchor-run-dir',str(r/'runs/full_official_baselines_20261004/training/MSVR310_demo_s42'),'--output',str(root/'evaluation')]
execute(argv,root,'full_reference',2)
assert p.stat().st_size=={arrays['bytes']} and hashlib.sha256(p.read_bytes()).hexdigest()=={arrays['sha256']!r}
print('ACTUAL_STRONG_PRIVATE_REFERENCE_FULL_QG_COMPLETE_NO_TRAINING',flush=True)'''
command(['ssh',*OPTIONS,'2026','cd '+shlex.quote(REMOTE)+' && '+shlex.quote(PYTHON)+' -u -'],input=code)
read='''import base64,hashlib,json
from pathlib import Path
root=Path('/data/gaob/Re-ID/DeMo-DualAxis/runs/msvr_stronger_private_reference_20261012')
assert json.loads((root/'full_reference_exit.json').read_text())['exit_code']==0 and not list(root.rglob('*.pth'))
text={p.relative_to(root).as_posix():base64.b64encode(p.read_bytes()).decode() for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.log')}
raw={p.relative_to(root).as_posix():dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in root.rglob('*.npz')}
assert len(raw)==2
print(json.dumps(dict(text=text,raw=raw)))'''
collection=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=read));target.mkdir();manifest={};artifacts={}
for n,data in collection['text'].items():
    data=base64.b64decode(data);p=target/n;assert p.resolve().is_relative_to(target.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    manifest[n]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
archive=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/msvr_stronger_private_reference_20261012')
for n,row in collection['raw'].items():
    p=archive/n;assert p.resolve().is_relative_to(archive.resolve()) and not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
    command(['scp',*OPTIONS,'2026:'+row['path'],str(p)])
    assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    clear=f'''import hashlib
from pathlib import Path
p=Path({row['path']!r});assert p.resolve().is_relative_to(Path({REMOTE!r}).resolve()) and p.suffix=='.npz'
assert p.stat().st_size=={row['bytes']} and hashlib.sha256(p.read_bytes()).hexdigest()=={row['sha256']!r};p.unlink()'''
    command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=clear)
    artifacts[n]=dict(**row,local=str(p),matching_remote_copy_removed=True,unique_raw_deleted=False)
result=load(target/'evaluation/result.json')
assert result['status']=='COMPLETE_STRONG_TEACHER_PRIVATE5120_RELOAD_FULL_OFFICIAL_PARITY' and not result['training_admitted'] and result['new_weights']==result['optimizer_updates']==0
assert all(v['full_query']==591 and v['full_gallery']==1055 and v['per_query_equal'] and v['strict_reload_equal'] for v in result['variants'].values())
record=dict(status='ACTUAL_STRONG_TEACHER_PRIVATE_REFERENCE_ALL_OFFICIAL_QUERY_GALLERY_PARITY_NO_TRAINING',at=datetime.now().astimezone().isoformat(timespec='seconds'),source_review='PASS',manifest=manifest,artifacts=artifacts,local_root=str(target),new_weights=0,optimizer_updates=0,training_admitted=False,full_query=591,full_gallery=1055)
(pf/'msvr_stronger_private_reference_actual_20261012.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=record['status'],full_query=591,full_gallery=1055,new_weights=0,training_admitted=False)),flush=True)
