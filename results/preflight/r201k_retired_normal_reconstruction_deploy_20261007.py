"""Restore exactly four required retired K arrays after all normal training closes."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified

REMOTE='/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/r201k_retired_normal_reconstruction_20261007'
LOCAL=PROJECT/'results/r201k_retired_normal_reconstruction_20261007'
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201k_relation_local_pi_20261007')
LANES=(('frequency_shared',2),('axis_shared',3))


def main():
    pf=PROJECT/'results/preflight'
    load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    proof=pf/'r201k_retired_normal_reconstruction_actual_session_20261007.json'
    assert not proof.exists() and not LOCAL.exists()
    review=load(pf/'r201k_retired_normal_reconstruction_source_review_20261007.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    assert set(review['sources_sha256'])=={'reconstruct_r201k_retired_normal.py','results/preflight/r201k_retired_normal_reconstruction_deploy_20261007.py'}
    sources=review['sources_sha256'];reuse=review['directly_reused_sources_sha256']
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (sources|reuse).items())
    normal=load(pf/'r201k_relation_local_pi_actual_session_20261007.json')
    assert normal['status']=='ACTUAL_K_THREE_NORMAL_DATASETS_SIX_FULL50_GT_RAW_COMPLETE' and normal['exit_code']==0
    assert normal['additional_epochs']==300 and normal['successful_updates']==19418 and normal['native_updates']==18
    assert set(normal['datasets'])=={'RGBNT201','MSVR310','RGBNT100'}
    for dataset in normal['datasets']:
        assert load(PROJECT/'results/r201k_relation_local_pi_20261007'/dataset/'normal_analysis/result.json')['status']=='ACTUAL_K_DATASET_TWO_FULL50_NORMAL_CPU_READOUT'
    audit=load(pf/'r201k_raw_retirement_dependency_actual_20261007.json')
    assert audit['status']=='ACTUAL_FOUR_CLOSED_K_NORMAL_RAW_RETIRED_CURRENT_BEST_WEIGHTS_PRESERVED'
    checkpoints={row['name']:row['sha256'] for row in audit['server26_selected_weights']}
    jobs=[]
    for job in audit['jobs']:
        assert not Path(job['normal_local']).exists()
        assert Path(job['normal_local']).resolve().is_relative_to(ARCHIVE.resolve())
        receipt=normal['datasets'][job['dataset']]['archives'][job['name']]
        assert receipt['file']==job['normal_file'] and receipt['local']==job['normal_local']
        jobs.append(dict(job,checkpoint_sha256=checkpoints[job['name']],gpu=dict(LANES)[job['variant']]))
    assert len(jobs)==4 and {j['dataset'] for j in jobs}=={'RGBNT201','MSVR310'}
    required=sum(j['normal_file']['bytes'] for j in jobs)
    assert shutil.disk_usage(ARCHIVE).free>required+1024**3
    neural='reconstruct_r201k_retired_normal.py'
    code=f'''import hashlib,shutil
from pathlib import Path
root=Path({REMOTE!r});assert not Path({ROOT!r}).exists()
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==sha for n,sha in {reuse!r}.items() if not n.startswith('results/'))
assert all(hashlib.sha256((Path(j['run'])/'best.pth').read_bytes()).hexdigest()==j['checkpoint_sha256'] and hashlib.sha256((Path(j['run'])/'result.json').read_bytes()).hexdigest()==j['result_sha256'] for j in {jobs!r})
assert shutil.disk_usage(root).free>{required!r}+1_600_000_000
print('RETIRED_NORMAL_FOUR_UNCHANGED_SELECTED_CHECKPOINTS_READY')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    LOCAL.mkdir(parents=True)
    registry=LOCAL/'jobs.json';registry.write_text(json.dumps(dict(jobs=jobs,new_optimizer_updates=0,all3_normal_closed=True),indent=2)+'\n',encoding='utf-8')
    remote_registry=REMOTE+'/runs/r201k_retired_normal_jobs_20261007.json'
    command(['scp',*OPTIONS,str(PROJECT/neural),'2026:'+REMOTE+'/'+neural])
    command(['scp',*OPTIONS,str(registry),'2026:'+remote_registry])
    code=f'''import hashlib
from pathlib import Path
assert hashlib.sha256(Path({(REMOTE+'/'+neural)!r}).read_bytes()).hexdigest()=={sources[neural]!r}
print('RETIRED_NORMAL_REVIEWED_SOURCE_EXACT')'''
    print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
    state=dict(status='ACTUAL_RETIRED_NORMAL_RECONSTRUCTION_STARTED',started=datetime.now().isoformat(timespec='seconds'),controls=4,new_optimizer_updates=0,completed={})
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    lock=threading.Lock()

    def lane(variant,gpu):
        for job in [j for j in jobs if j['variant']==variant]:
            output=ROOT+'/'+job['dataset']+'/training/'+job['name']
            argv=[PYTHON,'-u',neural,'--jobs',remote_registry,'--name',job['name'],'--output',output]
            execute='cd '+shlex.quote(REMOTE)+' && CUDA_VISIBLE_DEVICES='+str(gpu)+' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 '+shlex.join(argv)
            process=subprocess.Popen(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026',execute],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
            result=None
            with (LOCAL/(job['name']+'.log')).open('x',encoding='utf-8') as log:
                for line in process.stdout:
                    log.write(line);log.flush()
                    if line.startswith('K_NORMAL_RECONSTRUCTED '):
                        assert result is None
                        result=json.loads(line[len('K_NORMAL_RECONSTRUCTED '):])
                    else:
                        with lock:print(line,end='',flush=True)
            assert process.wait()==0 and result['status']=='ACTUAL_RETIRED_K_NORMAL_RECONSTRUCTED_ORIGINAL_SHA_AND_GT_EXACT'
            assert result['name']==job['name'] and result['normal_file']==job['normal_file']
            local=copy_verified(result['normal_path'],result['normal_file'],Path(ROOT),ARCHIVE)
            assert Path(local)==Path(job['normal_local'])
            code=f'''import hashlib
from pathlib import Path
p=Path({result['normal_path']!r});root=Path({ROOT!r})
assert p.resolve().is_relative_to(root.resolve()) and p.name=='best_official_arrays.npz'
assert p.stat().st_size=={job['normal_file']['bytes']!r} and hashlib.sha256(p.read_bytes()).hexdigest()=={job['normal_file']['sha256']!r};p.unlink()
print('ONLY_RECONSTRUCTION_STAGING_RAW_CLEARED_AFTER_LOCAL_ACK')'''
            print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),flush=True)
            with lock:
                state['completed'][job['name']]=dict(result=result,local=local,local_file=job['normal_file'])
                proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
                print('ACTUAL_RECONSTRUCTED_NORMAL_LOCAL_SHA_ACK',job['name'],flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(lane,*pair) for pair in LANES]
        for task in tasks:task.result()
    assert len(state['completed'])==4
    code=f'''import hashlib,json,tarfile
from pathlib import Path
root=Path({ROOT!r});package=Path('/tmp/r201k_retired_normal_closed_text_20261007.tar.gz');assert not package.exists()
files={{}}
with tarfile.open(package,'w:gz') as archive:
 for p in sorted(root.rglob('*')):
  if p.is_file() and p.suffix in ('.json','.csv','.txt','.log'):
   n=p.relative_to(root).as_posix();files[n]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest());archive.add(p,arcname=n,recursive=False)
print(json.dumps(dict(files=files,package=dict(bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()))))'''
    inventory=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    package=pf/'r201k_retired_normal_closed_text_20261007.tar.gz';assert not package.exists()
    command(['scp',*OPTIONS,'2026:/tmp/'+package.name,str(package)])
    assert package.stat().st_size==inventory['package']['bytes'] and hashlib.sha256(package.read_bytes()).hexdigest()==inventory['package']['sha256']
    with tarfile.open(package,'r:gz') as archive:
        assert set(archive.getnames())==set(inventory['files']);archive.extractall(LOCAL,filter='data')
    assert all((LOCAL/n).stat().st_size==r['bytes'] and hashlib.sha256((LOCAL/n).read_bytes()).hexdigest()==r['sha256'] for n,r in inventory['files'].items())
    for job in jobs:
        p=Path(job['normal_local']);assert p.stat().st_size==job['normal_file']['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==job['normal_file']['sha256']
    command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input='from pathlib import Path\nPath('+repr('/tmp/'+package.name)+').unlink()\n')
    package.unlink()
    state.update(status='ACTUAL_FOUR_RETIRED_NORMAL_ARCHIVES_RECONSTRUCTED_ORIGINAL_SHA_GT_LOCAL_ACK_COMPLETE',finished=datetime.now().isoformat(timespec='seconds'),restored_bytes=required,text_manifest=inventory['files'],new_optimizer_updates=0,seed_epoch_reselected=False)
    proof.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(state['status'],flush=True)


if __name__=='__main__':
    main()
