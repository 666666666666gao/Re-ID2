"""Restore selected AX raw references, launch owned GPU23 diagnostic, collect only at completion."""
import argparse,csv,hashlib,json,os,shlex,shutil,subprocess,sys,tarfile
from datetime import datetime
from pathlib import Path
PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
TAG='r201ay_AX_frequency_outlets_20261014';ROOT=REMOTE+'/runs/'+TAG
LOCAL=PROJECT/'results'/TAG;PF=PROJECT/'results/preflight'
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')/TAG
TRAINING=REMOTE+'/runs/r201ax_RGB_frequency_query_normalization_20261014/RGBNT100/training'


def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def remote(code):return json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))


def restored_inputs():
    intake=load(PF/'r201ax_RGB_frequency_query_normalization_20261014_complete_intake.json')
    assert intake['status']=='ACTUAL_AX_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE' and intake['terminal']['all_children_and_controller_absent']
    return {key:dict(local=record['local'],file=dict(bytes=record['bytes'],sha256=record['sha256']),remote=record['remote']) for key,record in intake['archives'].items() if key.endswith(('/best','/raw'))}



def deploy():
    review=load(PF/'r201ay_AX_frequency_outlets_source_review_20261014.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    sources=review['sources_sha256'];reuse=review['directly_reused_sources_sha256']
    assert all(sha(PROJECT/n)==digest for n,digest in (sources|reuse).items())
    assert not LOCAL.exists() and not ARCHIVE.exists() and not (PF/(TAG+'_actual_session.json')).exists()
    assert not load(Path('C:/Users/gb/.codex_tmp/r201l_uniform_k8_continuation_20261007.json'))['live_native_sessions']
    protected=load(PF/'r201ax_RGB_frequency_query_normalization_20261014_actual_session.json')['protected_weights']
    refs=restored_inputs();assert len(refs)==4
    assert shutil.disk_usage(ARCHIVE.parent).free>3_000_000_000
    for record in refs.values():assert Path(record['local']).stat().st_size==record['file']['bytes'] and sha(Path(record['local']))==record['file']['sha256']
    ready=remote(f'''import hashlib,json,shutil,subprocess
from pathlib import Path
r=Path({REMOTE!r});assert not Path({ROOT!r}).exists()
selected={{g:subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip() for g in (2,3)}}
assert not any(selected.values()) and shutil.disk_usage(r).free>3_300_000_000
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {reuse!r}.items() if not n.startswith('results/'))
assert all(not Path(row['remote']).exists() for row in {refs!r}.values())
print(json.dumps(dict(project_free=shutil.disk_usage(r).free,selected_GPU_processes=selected,protected_weights_exact=True)))''')
    for n in sources:
        if not n.startswith('results/'):command(['scp',*OPTIONS,str(PROJECT/n),'2026:'+REMOTE+'/'+n])
    for row in refs.values():command(['scp',*OPTIONS,row['local'],'2026:'+row['remote']])
    admitted=remote(f'''import hashlib,json
from pathlib import Path
r=Path({REMOTE!r})
assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in {(sources|reuse)!r}.items() if not n.startswith('results/'))
assert all(Path(row['remote']).stat().st_size==row['file']['bytes'] and hashlib.sha256(Path(row['remote']).read_bytes()).hexdigest()==row['file']['sha256'] for row in {refs!r}.values())
print(json.dumps(dict(sources_exact=True,restored_normal_arrays_exact=True)))''')
    argv=[PYTHON,'-u','launch_r201ay_AX_frequency_outlets.py','--training-root',TRAINING,'--output',ROOT+'/RGBNT100']
    warm=remote(f"""import importlib,json,sys,torch
sys.path.insert(0,{REMOTE!r})
for n in ('evaluate_r201ay_AX_frequency_outlets','launch_r201ay_AX_frequency_outlets'):
 importlib.import_module(n)
import run_r201ax_RGB_frequency_query_normalization as recipe
assert recipe.REVISION.startswith('AX RGBNT100')
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS',torch_version=torch.__version__,CUDA_initialized=False,models=0,optimizer_updates=0,scope='Actual CPU-only import of deployed diagnostic and AX builder')))
""")
    save(PF/'r201ay_actual_CPU_import_20261014.json',warm)
    assert warm['status']=='PASS' and not warm['CUDA_initialized']
    row=dict(status='ACTUAL_AY_FIXED_AX_STARTING',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        deploy_pid=os.getpid(),capacity=ready,admission=admitted,restored_inputs=refs,
        sources_sha256=sources,reused_sources_sha256=reuse,protected_weights=protected,physical_gpus=[2,3],maximum_NN=2,
        optimizer_updates=0,new_weights=0,estimated_wall_seconds=900,first_observation_after_seconds=900,goal='ACTIVE_UNMET')
    save(PF/(TAG+'_actual_session.json'),row)
    print(json.dumps(dict(status=row['status'],physical_gpus=[2,3],optimizer_updates=0,estimate_seconds=900)),flush=True)
    code=f"""import subprocess
subprocess.run({argv!r},cwd={REMOTE!r},check=True)
"""
    completed=subprocess.run(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','2026',shlex.quote(PYTHON)+' -'],
        input=code,check=True,capture_output=True,text=True,encoding='utf-8')
    print(completed.stdout,end='',flush=True)
    collect()



def collect():
    started=load(PF/(TAG+'_actual_session.json'))
    terminal=remote(f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({ROOT!r});p=r/'RGBNT100';owner=json.loads((p/'controller_launch.json').read_text())['pid'];assert not Path('/proc/'+str(owner)).exists()
data=json.loads((p/'controller_result.json').read_text());assert data['status']=='COMPLETE' and data['cases']==30
assert data['optimizer_updates']==data['new_weights']==0
for phase in ('smoke','full'):
 for f in (p/phase).glob('*_launch.json'):
  launch=json.loads(f.read_text());assert not Path('/proc/'+str(launch['pid'])).exists()
  exit=json.loads(f.with_name(f.name.replace('_launch.json','_exit.json')).read_text());assert exit['exit_code']==0
manifest={{str(f.relative_to(r)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in r.rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.log')}}
pkg=r/'text_intake.tar.gz';assert not pkg.exists()
with tarfile.open(pkg,'w:gz') as archive:
 for n in manifest:archive.add(r/n,arcname=n)
print(json.dumps(dict(controller=data,manifest=manifest,package=dict(bytes=pkg.stat().st_size,sha256=hashlib.sha256(pkg.read_bytes()).hexdigest()),all_children_and_controller_absent=True)))''')
    assert not LOCAL.exists() and not ARCHIVE.exists()
    assert shutil.disk_usage(ARCHIVE.parent).free>sum(row['bytes'] for row in terminal['controller']['raw'].values())+500_000_000
    ARCHIVE.mkdir(parents=True);LOCAL.mkdir(parents=True)
    archives={}
    for name,row in terminal['controller']['raw'].items():
        destination=ARCHIVE/name/'raw.npz';destination.parent.mkdir()
        subprocess.run(['scp',*OPTIONS,'2026:'+row['path'],str(destination)],check=True,capture_output=True,text=True,encoding='utf-8',timeout=900)
        assert destination.stat().st_size==row['bytes'] and sha(destination)==row['sha256']
        assert sha(destination)==row['sha256']
        archives[name]=dict(local=str(destination),remote=row['path'],bytes=row['bytes'],sha256=row['sha256'])
    package=LOCAL/'text_intake.tar.gz';command(['scp',*OPTIONS,'2026:'+ROOT+'/text_intake.tar.gz',str(package)])
    assert package.stat().st_size==terminal['package']['bytes'] and sha(package)==terminal['package']['sha256']
    with tarfile.open(package) as archive:archive.extractall(LOCAL,filter='data')
    assert all((LOCAL/n).stat().st_size==row['bytes'] and sha(LOCAL/n)==row['sha256'] for n,row in terminal['manifest'].items())
    rows=[];comparisons=[]
    for name in archives:
        result=load(LOCAL/'RGBNT100/full'/name/'result.json')
        for stage,value in result['measurements'].items():
            rows.append(dict(variant=result['variant'],selected_epoch=result['selected_epoch'],stage=stage,dimension=result['dimensions'][stage],
                **{m:value[m] for m in ('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')}))
        for comparison,value in result['comparisons'].items():comparisons.append(dict(variant=result['variant'],comparison=comparison,
            **value['delta_pp'],rescue=value['Rank1_rescue_queries'],harm=value['Rank1_harm_queries'],AP_improved=value['AP_improved_queries'],AP_worsened=value['AP_worsened_queries']))
    for filename,values in [('six_metrics.csv',rows),('comparisons.csv',comparisons)]:
        with (LOCAL/filename).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    refs=started['restored_inputs'];protected=started['protected_weights']
    cleared=remote(f'''import hashlib,json,shutil
from pathlib import Path
r=Path({ROOT!r});p=Path({REMOTE!r})
owned={{row['remote']:dict(bytes=row['bytes'],sha256=row['sha256']) for row in {archives!r}.values()}}
owned.update({{row['remote']:row['file'] for row in {refs!r}.values()}})
owned[str(r/'text_intake.tar.gz')]={terminal['package']!r}
assert all(Path(n).resolve().is_relative_to(p.resolve()) and Path(n).stat().st_size==row['bytes'] and hashlib.sha256(Path(n).read_bytes()).hexdigest()==row['sha256'] for n,row in owned.items())
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in {protected!r}.items())
owner=json.loads((r/'RGBNT100/controller_launch.json').read_text())['pid'];assert not Path('/proc/'+str(owner)).exists()
for n in owned:Path(n).unlink()
print(json.dumps(dict(matching_owned_duplicates_cleared=len(owned),bytes=sum(row['bytes'] for row in owned.values()),protected_bests_exact=True,remote_project_free=shutil.disk_usage(p).free)))''')
    package.unlink()
    proof=dict(status='ACTUAL_AY_TWO_FIXED_AX_FULL_NORMAL_30_OUTLETS_AND_SPATIAL_D_ARCHIVED_GT_PARITY_COMPLETE',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        collector_pid=os.getpid(),terminal=terminal,archives=archives,cleared=cleared,D_free=shutil.disk_usage(ARCHIVE).free,
        optimizer_updates=0,new_weights=0,new49=False,missing_evaluated=False,unique_best_deleted=0,goal='ACTIVE_UNMET')
    save(PF/(TAG+'_complete_intake.json'),proof)
    started.update(status=proof['status'],completion=str(PF/(TAG+'_complete_intake.json')),remote_controller_absent=True)
    save(PF/(TAG+'_actual_session.json'),started);print(json.dumps(dict(status=proof['status'],archives=archives,cleared=cleared)),flush=True)


if __name__=='__main__':
    deploy()
