"""Run only after the original two-full50 receiver has closed; zero updates."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS, command

PF = PROJECT/'results/preflight'
REMOTE = '/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON = '/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT = REMOTE+'/runs/r201e_training_signal_probe_20261006'
REVIEW = PF/'r201e_training_signal_probe_source_review_20261006.json'


def main():
    review = json.loads(REVIEW.read_text(encoding='utf-8'))
    assert review['status']=='PASS' and not review['blocking_findings']
    live_sources = json.loads((PF/'r201e_list_objective_fp32check_source_review_20261006.json').read_text(encoding='utf-8'))
    runtime = live_sources['sources_sha256'] | live_sources['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (runtime|review['sources_sha256']).items())
    completed = json.loads((PF/'r201e_list_objective_fp32check_actual_session_20261006.json').read_text(encoding='utf-8'))
    assert completed['status']=='ACTUAL_R201E_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW'
    assert completed['exit_code']==0 and completed['successful_updates']==5294 and len(completed['archives'])==2
    output = PROJECT/'results/r201e_training_signal_probe_20261006'
    proof = PF/'r201e_training_signal_probe_actual_20261006.json'
    assert not output.exists() and not proof.exists()
    code = f'''import hashlib,json,subprocess
from pathlib import Path
project=Path({REMOTE!r});root=Path({ROOT!r})
assert not root.exists()
for pid in (3672643,3673725,3673726):
 assert subprocess.run(['ps','-p',str(pid)],stdout=subprocess.DEVNULL).returncode==1
assert all(hashlib.sha256((project/n).read_bytes()).hexdigest()==sha for n,sha in {runtime!r}.items() if not n.startswith('results/'))
old=json.loads((project/'runs/r201e_list_objective_fp32check_20261006/controller_result.json').read_text())
assert old['status']=='COMPLETE' and old['successful_updates']==5294 and old['normal_archives_local_verified']==2
print('ACTUAL_PRIOR_CONTROLLER_AND_TWO_FULL50_CLOSED')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),end='',flush=True)
    command(['scp',*OPTIONS,str(PROJECT/'probe_list_training_signal.py'),'2026:'+REMOTE+'/probe_list_training_signal.py'])
    probe_sha = review['sources_sha256']['probe_list_training_signal.py']
    code = f'''from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib,json,os
from pathlib import Path
import sys
from gpu_thermal_execute import execute
project=Path({REMOTE!r});root=Path({ROOT!r})
assert hashlib.sha256((project/'probe_list_training_signal.py').read_bytes()).hexdigest()=={probe_sha!r}
root.mkdir(exist_ok=False)
(root/'controller_launch.json').write_text(json.dumps(dict(pid=os.getpid(),started=datetime.now().isoformat(),physical_GPUs=[2,3],maximum_NN=2,optimizer_updates=0,formal_runs=0,source_sha256={probe_sha!r}),indent=2)+'\\n')
def one(variant,gpu):
 argv=[sys.executable,'-u','probe_list_training_signal.py','--variant',variant,'--data-root','/data/gaob/Re-ID/dataset','--pretrained','/data/gaob/Re-ID/Trifusion/pertrained-model/ViT-B-16.pt','--anchor-run-dir',str(project/'runs/full_official_baselines_20261004/training/RGBNT201_demo_s42'),'--output',str(root/(variant+'.json'))]
 finished=execute(argv,root,variant,gpu)
 data=json.loads((root/(variant+'.json')).read_text())
 assert data['status']=='ACTUAL_TRAIN_ONLY_LIST_PRIMARY_CE_FEATURE_GRADIENT_GRID' and data['optimizer_updates']==data['formal_runs']==data['checkpoint_writes']==0
 assert data['neural_forwards']==16 and data['temperatures']==[.01,.02,.05,.1] and data['gradient_scale']==512
 return data
with ThreadPoolExecutor(max_workers=2) as pool:
 jobs=[pool.submit(one,v,g) for v,g in [('frequency_shared',2),('axis_shared',3)]]
 data=[j.result() for j in jobs]
assert [[(r['names'],r['training_labels'],r['camera_labels']) for r in d['rows']] for d in data][0]==[[(r['names'],r['training_labels'],r['camera_labels']) for r in d['rows']] for d in data][1]
assert len({{(d['parameters'],d['trainable_parameters'],d['descriptor_dim']) for d in data}})==1
assert not list(root.rglob('*.pth')) and not list(root.rglob('*.npz'))
result=dict(status='ACTUAL_TWO_TRAIN_ONLY_FEATURE_GRADIENT_PROBES_COMPLETE',neural_forwards=32,optimizer_updates=0,formal_runs=0,paired_training_lists=True,temperature_power_control=False)
(root/'controller_result.json').write_text(json.dumps(result,indent=2)+'\\n')
files={{p.relative_to(root).as_posix():p.read_text() for p in root.rglob('*') if p.is_file()}}
manifest={{n:dict(bytes=len(s.encode()),sha256=hashlib.sha256(s.encode()).hexdigest()) for n,s in files.items()}}
print('R201E_TRAIN_SIGNAL_RESULT '+json.dumps(dict(result=result,files=files,manifest=manifest)))'''
    process = subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026',
        'cd '+shlex.quote(REMOTE)+' && '+shlex.quote(PYTHON)+' -'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,encoding='utf-8')
    print('ACTUAL_ZERO_UPDATE_TRAIN_SIGNAL_CONTROLLER_STARTED',flush=True)
    process.stdin.write(code);process.stdin.close()
    payload = None
    for line in process.stdout:
        if line.startswith('R201E_TRAIN_SIGNAL_RESULT '):
            assert payload is None
            payload = json.loads(line[len('R201E_TRAIN_SIGNAL_RESULT '):])
        else:
            print(line,end='',flush=True)
    assert process.wait()==0 and payload is not None
    assert set(payload['files'])==set(payload['manifest'])
    output.mkdir()
    for name,content in payload['files'].items():
        target = output/name
        assert target.resolve().is_relative_to(output.resolve())
        target.write_bytes(content.encode('utf-8'))
        assert target.stat().st_size==payload['manifest'][name]['bytes']
        assert hashlib.sha256(target.read_bytes()).hexdigest()==payload['manifest'][name]['sha256']
    result = dict(status='ACTUAL_R201E_TRAIN_ONLY_FEATURE_GRADIENT_PROBES_TEXT_VERIFIED',
        verified_at=datetime.now().isoformat(timespec='seconds'),remote_root=ROOT,local_root=str(output),
        neural_forwards=32,optimizer_updates=0,formal_runs=0,raw_transfers=0,
        manifest=payload['manifest'],source_review=str(REVIEW.relative_to(PROJECT)),source_sha256=probe_sha)
    proof.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='manifest'}),flush=True)


if __name__=='__main__':
    main()
