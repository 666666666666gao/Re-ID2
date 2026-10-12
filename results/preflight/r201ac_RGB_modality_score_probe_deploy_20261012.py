"""One GPU0 AC ordinary diagnostic replay; no model patch or formal launch."""
from pathlib import Path
from datetime import datetime
import base64, hashlib, json, shlex, sys

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
review=load(pf/'r201ac_RGB_modality_score_probe_source_review_20261012.json')
assert review['status']=='PASS' and not review['blocking_findings']
assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
closed=load(pf/'r201ac_RGB_hard_identity_original_owner_closed_actual_20261012.json')
assert closed['exit_code']==1 and closed['terminal_consumed'] and closed['original_process_absent'] and closed['remote_children_absent'] and not closed['formal_started']
target=PROJECT/'results/r201ac_RGB_modality_score_probe_20261012';assert not target.exists()
source='diagnose_r201ac_RGB_modality_score_native.py'
runtime=review['directly_reused_sources_sha256']|{source:review['sources_sha256'][source]}
command(['scp',*OPTIONS,str(PROJECT/source),'2026:'+REMOTE+'/'+source])
code=f'''import hashlib,json,sys
from pathlib import Path
from gpu_01_diagnostic_execute import execute
r=Path({REMOTE!r});root=r/'runs/r201ac_RGB_modality_score_probe_20261012';assert not root.exists()
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==sha for n,sha in {runtime!r}.items())
old=r/'runs/r201ac_RGB_hard_identity_20261012/RGBNT100'
assert not (Path('/proc')/str(json.loads((old/'controller_launch.json').read_text())['pid'])).exists()
assert sorted(json.loads(p.read_text())['exit_code'] for p in (old/'native').glob('*_exit.json'))==[0,1]
assert not list((old/'training').iterdir())
root.mkdir();argv=[sys.executable,'-u',{source!r},'--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt','--anchor-run-dir',str(r/'runs/full_official_baselines_20261004/training/RGBNT100_demo_s42'),'--output',str(root/'ordinary')]
execute(argv,root,'ordinary_probe',0)
print('ACTUAL_AC_RGB_M_OBSERVATION_COMPLETE_NO_FORMAL',flush=True)'''
command(['ssh',*OPTIONS,'2026','cd '+shlex.quote(REMOTE)+' && '+shlex.quote(PYTHON)+' -u -'],input=code)
read='''import base64,hashlib,json
from pathlib import Path
root=Path('/data/gaob/Re-ID/DeMo-DualAxis/runs/r201ac_RGB_modality_score_probe_20261012')
assert json.loads((root/'ordinary_probe_exit.json').read_text())['exit_code']==0
files={p.relative_to(root).as_posix():base64.b64encode(p.read_bytes()).decode() for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.log')}
assert not list(root.rglob('*.pth'))
raw=root/'ordinary/actual_M_pool_inputs.npz'
print(json.dumps(dict(files=files,raw=dict(path=str(raw),bytes=raw.stat().st_size,sha256=hashlib.sha256(raw.read_bytes()).hexdigest()))))'''
collection=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=read));target.mkdir();manifest={}
for n,data in collection['files'].items():
    raw=base64.b64decode(data);path=target/n;assert path.resolve().is_relative_to(target.resolve());path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    manifest[n]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
raw=collection['raw'];local=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201ac_RGB_modality_score_probe_20261012/actual_M_pool_inputs.npz')
assert not local.exists();local.parent.mkdir(parents=True,exist_ok=True)
command(['scp',*OPTIONS,'2026:'+raw['path'],str(local)])
assert local.stat().st_size==raw['bytes'] and hashlib.sha256(local.read_bytes()).hexdigest()==raw['sha256']
assert local.stat().st_size==raw['bytes'] and hashlib.sha256(local.read_bytes()).hexdigest()==raw['sha256']
clear=f'''import hashlib,json
from pathlib import Path
p=Path({raw['path']!r});assert p.resolve().is_relative_to(Path({REMOTE!r}).resolve()) and p.name=='actual_M_pool_inputs.npz'
assert p.stat().st_size=={raw['bytes']} and hashlib.sha256(p.read_bytes()).hexdigest()=={raw['sha256']!r}
p.unlink();print(json.dumps(dict(matching_remote_raw_removed=True,unique_raw_deleted=False)))'''
removed=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=clear))
result=load(target/'ordinary/diagnostic.json');attempts=load(target/'ordinary/attempts.json')
axis=load(PROJECT/'results/r201ac_RGB_hard_identity_native_failure_20261012/native/RGBNT100_r201ac_axis_shared_s42_smoke/smoke.json')
assert result['successful_updates']==3 and result['attempts']==3 and result['strict_reload_equal'] and result['training_batch_names']==[d['names'] for d in axis['details']]
assert result['partial_sets']==[[0],[1],[2]]
record=dict(status='ACTUAL_AC_RGB_ORDINARY_NATIVE3_M_OBSERVATION_COMPLETE_NOT_NATIVE_ACCEPTANCE',at=datetime.now().astimezone().isoformat(timespec='seconds'),local_root=str(target),manifest=manifest,raw=dict(**raw,local=str(local),**removed),source_review='PASS',original_39021_closed=True,formal_training_admitted=False,extra_weights=0,successful_updates=3,attempts=3,actual_native_rejected_parameters=result['final_native_rejected_parameters'],strict_reload_equal=result['strict_reload_equal'],same_hard_batch_names_as_actual_axis_native=True)
(pf/'r201ac_RGB_modality_score_probe_actual_20261012.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=record['status'],successful_updates=3,rejected=result['final_native_rejected_parameters'],strict_reload_equal=result['strict_reload_equal'],raw_bytes=raw['bytes'])),flush=True)
