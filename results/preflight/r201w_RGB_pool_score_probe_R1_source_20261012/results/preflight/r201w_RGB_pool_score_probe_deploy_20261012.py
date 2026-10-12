"""Once-only GPU2 W ordinary native3 observation; no source patch or formal launch."""
from pathlib import Path
from datetime import datetime
import base64, hashlib, json, shlex, sys

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
review=load(pf/'r201w_RGB_pool_score_probe_source_review_20261012.json')
assert review['status']=='PASS' and not review['blocking_findings']
assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in review['sources_sha256'].items())
failure=load(pf/'r201w_RGB_native_failure_20261012/collection.json');assert sorted(row['exit_code'] for row in failure['exits'].values())==[0,1]
target=PROJECT/'results/r201w_RGB_pool_score_probe_20261012';assert not target.exists()
source='diagnose_r201w_RGB_pool_score_native.py'
command(['scp',*OPTIONS,str(PROJECT/source),'2026:'+REMOTE+'/'+source])
code=f'''import hashlib,json,sys
from pathlib import Path
from gpu_thermal_execute import execute
r=Path({REMOTE!r});root=r/'runs/r201w_RGB_pool_score_probe_20261012';assert not root.exists()
assert all(hashlib.sha256((r/name).read_bytes()).hexdigest()==sha for name,sha in {review['runtime_sources_sha256']!r}.items())
old=r/'runs/r201w_reduced_auxiliary_20261011/RGBNT100'
assert not (Path('/proc')/str(json.loads((old/'controller_launch.json').read_text())['pid'])).exists()
assert sorted(json.loads(p.read_text())['exit_code'] for p in (old/'native').glob('*_exit.json'))==[0,1]
root.mkdir();argv=[sys.executable,'-u',{source!r},'--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt','--anchor-run-dir',str(r/'runs/full_official_baselines_20261004/training/RGBNT100_demo_s42'),'--output',str(root/'ordinary')]
execute(argv,root,'ordinary_probe',2)
print('ACTUAL_W_RGB_OBSERVATION_COMPLETE_NO_FORMAL',flush=True)'''
command(['ssh',*OPTIONS,'2026','cd '+shlex.quote(REMOTE)+' && '+shlex.quote(PYTHON)+' -u -'],input=code)
read='''import base64,json
from pathlib import Path
root=Path('/data/gaob/Re-ID/DeMo-DualAxis/runs/r201w_RGB_pool_score_probe_20261012')
assert json.loads((root/'ordinary_probe_exit.json').read_text())['exit_code']==0
files={p.relative_to(root).as_posix():base64.b64encode(p.read_bytes()).decode() for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.log')}
assert not list(root.rglob('*.pth'))
print(json.dumps(files))'''
files=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=read));target.mkdir();manifest={}
for name,data in files.items():
    raw=base64.b64decode(data);path=target/name;assert path.resolve().is_relative_to(target.resolve());path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
record=dict(status='ACTUAL_W_RGB_ORDINARY_NATIVE3_OBSERVATION_COMPLETE_NO_FORMAL_ADMISSION',at=datetime.now().astimezone().isoformat(timespec='seconds'),manifest=manifest,review=review['status'],MS_original_84945_untouched=True,formal_training_admitted=False,extra_weights=0)
(pf/'r201w_RGB_pool_score_probe_actual_20261012.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
result=load(target/'ordinary/diagnostic.json');print(json.dumps(dict(status=record['status'],successful_updates=result['successful_updates'],rejected=result['final_native_rejected_parameters'],strict_reload_equal=result['strict_reload_equal'])),flush=True)
