"""Recover the actually failed AF first SCP; no NN replay or source mutation."""
import csv,hashlib,json,os,shutil,subprocess,sys,tarfile
from datetime import datetime
from pathlib import Path
PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT/'results/preflight'))
from r201af_AD_selected_outlets_flow_20261012 import load,save,sha,remote,command,OPTIONS,REMOTE,PYTHON,TAG,ROOT,LOCAL,PF,ARCHIVE,TRAINING


def collect():
    started=load(PF/(TAG+'_actual_session.json'));owner=started['remote_owner']['pid']
    terminal=remote(f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({ROOT!r});p=r/'RGBNT100';assert not Path('/proc/{owner}').exists()
data=json.loads((p/'controller_result.json').read_text());assert data['status']=='COMPLETE' and data['cases']==30
assert data['optimizer_updates']==data['new_weights']==0
for phase in ('smoke','full'):
 for f in (p/phase).glob('*_launch.json'):
  launch=json.loads(f.read_text());assert not Path('/proc/'+str(launch['pid'])).exists()
  exit=json.loads(f.with_name(f.name.replace('_launch.json','_exit.json')).read_text());assert exit['exit_code']==0
manifest={{str(f.relative_to(r)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in r.rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.log')}}
pkg=r/'text_intake.tar.gz';assert pkg.is_file()
print(json.dumps(dict(controller=data,manifest=manifest,package=dict(bytes=pkg.stat().st_size,sha256=hashlib.sha256(pkg.read_bytes()).hexdigest()),all_children_and_controller_absent=True)))''')
    failure=load(PF/'r201af_first_intake_timeout_actual_20261012.json')
    assert failure['exit_code']==1 and failure['terminal_consumed'] and not failure['raw_remote_cleared']
    assert LOCAL.is_dir() and ARCHIVE.is_dir() and not list(LOCAL.rglob('*'))
    assert {str(f):f.stat().st_size for f in ARCHIVE.rglob('*') if f.is_file()}==failure['local_partial_files']
    assert shutil.disk_usage(ARCHIVE.parent).free>sum(row['bytes'] for row in terminal['controller']['raw'].values())+500_000_000
    package=LOCAL/'text_intake.tar.gz';command(['scp',*OPTIONS,'2026:'+ROOT+'/text_intake.tar.gz',str(package)])
    assert package.stat().st_size==terminal['package']['bytes'] and sha(package)==terminal['package']['sha256']
    with tarfile.open(package) as archive:archive.extractall(LOCAL,filter='data')
    assert all((LOCAL/n).stat().st_size==row['bytes'] and sha(LOCAL/n)==row['sha256'] for n,row in terminal['manifest'].items())
    print('AF_COMPLETE_TEXT_INTAKE_VERIFIED_BEFORE_RAW_TRANSFER',flush=True)
    archives={}
    for name,row in terminal['controller']['raw'].items():
        destination=ARCHIVE/name/'raw.npz'
        if name=='RGBNT100_r201ad_hard_frequency_shared_s42':assert destination.parent.is_dir()
        else:destination.parent.mkdir()
        subprocess.run(['scp',*OPTIONS,'2026:'+row['path'],str(destination)],check=True,capture_output=True,text=True,encoding='utf-8',timeout=900)
        assert destination.stat().st_size==row['bytes'] and sha(destination)==row['sha256']
        assert sha(destination)==row['sha256']
        archives[name]=dict(local=str(destination),remote=row['path'],bytes=row['bytes'],sha256=row['sha256'])
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
assert not Path('/proc/{owner}').exists()
for n in owned:Path(n).unlink()
print(json.dumps(dict(matching_owned_duplicates_cleared=len(owned),bytes=sum(row['bytes'] for row in owned.values()),protected_bests_exact=True,remote_project_free=shutil.disk_usage(p).free)))''')
    package.unlink()
    proof=dict(status='ACTUAL_AF_TWO_FIXED_AD_FULL_NORMAL_30_OUTLETS_D_ARCHIVED_GT_PARITY_COMPLETE',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        collector_pid=os.getpid(),terminal=terminal,raw_SCP_timeout_seconds=900,prior_first_intake_failure='results/preflight/r201af_first_intake_timeout_actual_20261012.json',archives=archives,cleared=cleared,D_free=shutil.disk_usage(ARCHIVE).free,
        optimizer_updates=0,new_weights=0,new49=False,missing_evaluated=False,unique_best_deleted=0,goal='ACTIVE_UNMET')
    save(PF/(TAG+'_complete_intake.json'),proof)
    started.update(status=proof['status'],completion=str(PF/(TAG+'_complete_intake.json')),remote_controller_absent=True)
    save(PF/(TAG+'_actual_session.json'),started);print(json.dumps(dict(status=proof['status'],archives=archives,cleared=cleared)),flush=True)


if __name__=='__main__':collect()
