"""Collect one terminal full-normal seed wave before admitting the next wave."""
import argparse
from datetime import datetime
import json,os,shlex,shutil,subprocess,sys,tarfile
from pathlib import Path

from rgbnt100_expert_N5_wave_deploy_20261015 import PROJECT,PF,REMOTE,PYTHON,OPTIONS,load,sha,command


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=(43,44,45,46),required=True)
    args=parser.parse_args();selection_path=PF/'rgbnt100_expert_N5_selected_recipe_lock_20261015.json';selection=load(selection_path)
    assert selection['status']=='ACTUAL_CLOSED_RECIPE_LOCKED_FOR_BOUNDED_EXPERT_N5'
    key=selection['selected_recipe'];tag='rgbnt100_N5_'+key+'_s'+str(args.seed)
    proof_path=PF/(tag+'_complete_intake_20261015.json');assert not proof_path.exists()
    started=load(PF/(tag+'_actual_session_20261015.json'))
    assert started['recipe']==key and started['seed']==args.seed and started['selection_sha256']==sha(selection_path)
    owner=started['remote_owner']['pid'];root=started['remote_root']
    remote=lambda code:json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    terminal=remote(f'''import hashlib,json,tarfile
from pathlib import Path
r=Path({root!r});p=r/'RGBNT100';assert not Path('/proc/{owner}').exists()
d=json.loads((p/'controller_result.json').read_text())
assert d['status']=='COMPLETE' and d['recipe']=={key!r} and d['seed']=={args.seed}
assert d['controls']==2 and d['native_updates']==6 and d['successful_updates']>0
assert d['paired_sampling_exact'] and d['normal_GT_audited'] and d['state_cases']==0 and d['physical_gpus']==[2,3]
children=[]
for phase in ('parity','native','training','audit'):
 for f in (p/phase).glob('*_launch.json'):
  launch=json.loads(f.read_text());assert not Path('/proc/'+str(launch['pid'])).exists()
  ended=json.loads(f.with_name(f.name.replace('_launch.json','_exit.json')).read_text());assert ended['exit_code']==0
  children.append(dict(phase=phase,launch=launch,exit=ended))
assert len(children)=={8 if args.seed==43 else 6}
manifest={{str(f.relative_to(r)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in r.rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.log','.jsonl')}}
pkg=r/'text_intake.tar.gz';assert not pkg.exists()
with tarfile.open(pkg,'w:gz') as tar:
 for name in manifest:tar.add(r/name,arcname=name)
print(json.dumps(dict(controller=d,children=children,all_children_and_controller_absent=True,manifest=manifest,
 package=dict(bytes=pkg.stat().st_size,sha256=hashlib.sha256(pkg.read_bytes()).hexdigest()))))''')
    local=PROJECT/selection['local_family_root']/('s'+str(args.seed))
    archive=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts')/Path(selection['local_family_root']).name/('s'+str(args.seed))
    assert not local.exists() and not archive.exists()
    total=sum(row[field]['bytes'] for row in terminal['controller']['runs'] for field in ('raw','best'))
    assert shutil.disk_usage(archive.parent.parent).free>total+500_000_000
    local.mkdir(parents=True);archive.mkdir(parents=True)
    package=local/'text_intake.tar.gz'
    command(['scp',*OPTIONS,'2026:'+root+'/text_intake.tar.gz',str(package)])
    assert package.stat().st_size==terminal['package']['bytes'] and sha(package)==terminal['package']['sha256']
    with tarfile.open(package) as tar:tar.extractall(local,filter='data')
    assert all((local/name).stat().st_size==row['bytes'] and sha(local/name)==row['sha256'] for name,row in terminal['manifest'].items())
    archives={}
    for run in terminal['controller']['runs']:
        for kind,filename in (('raw','best_official_arrays.npz'),('best','best.pth')):
            row=run[kind];target=archive/run['name']/kind/filename;target.parent.mkdir(parents=True)
            result=subprocess.run(['scp',*OPTIONS,'2026:'+row['path'],str(target)],capture_output=True,text=True,encoding='utf-8',timeout=900)
            if result.returncode:
                print(result.stdout,flush=True);print(result.stderr,file=sys.stderr,flush=True);result.check_returncode()
            assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256']
            archives[run['name']+'/'+kind]=dict(local=str(target),remote=row['path'],bytes=row['bytes'],sha256=row['sha256'])
    subprocess.run([sys.executable,'-X','utf8','-B','-S',str(PROJECT/'analyze_rgbnt100_expert_N5.py'),
        '--selection-file',str(selection_path),'wave','--seed',str(args.seed)],cwd=PROJECT,check=True)
    protected=started['protected_weights']
    cleanup=remote(f'''import hashlib,json,shutil
from pathlib import Path
p=Path({REMOTE!r});r=Path({root!r});assert not Path('/proc/{owner}').exists()
owned={{row['remote']:dict(bytes=row['bytes'],sha256=row['sha256']) for row in {archives!r}.values()}}
owned[str(r/'text_intake.tar.gz')]={terminal['package']!r}
assert all(Path(name).resolve().is_relative_to(r.resolve()) and Path(name).stat().st_size==row['bytes'] and hashlib.sha256(Path(name).read_bytes()).hexdigest()==row['sha256'] for name,row in owned.items())
assert all(hashlib.sha256((p/name).read_bytes()).hexdigest()==value for name,value in {protected!r}.items())
for name in owned:Path(name).unlink()
print(json.dumps(dict(matching_owned_duplicates_cleared=len(owned),bytes=sum(row['bytes'] for row in owned.values()),protected_four_exact=True,project_free=shutil.disk_usage(p).free)))''')
    package.unlink()
    for name,row in archives.items():
        model,kind=name.split('/')
        destination=local/'RGBNT100/training'/model/('local_best_archive.json' if kind=='best' else 'local_archive.json')
        destination.write_text(json.dumps(dict(status='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED',**row),indent=2)+'\n',encoding='utf-8')
    record=dict(status='ACTUAL_N5_WAVE_D_ARCHIVED_COMPLETE',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        recipe=key,seed=args.seed,collector_pid=os.getpid(),terminal=terminal,archives=archives,cleanup=cleanup,
        all_children_and_controller_absent=True,formal_updates=terminal['controller']['successful_updates'],native_updates=6,
        parity_forward_record_executions=terminal['controller']['parity_forward_record_executions'],
        D_free=shutil.disk_usage(archive).free,unique_best_deleted=0,states_deferred=True,missing_evaluated=False,
        limits='This wave is fully archived and remote duplicate files retired. The four-state evaluation is deferred to the final locked seed and matched control. Retire known nonleader local checkpoints by exact size/SHA only after reading the completed prefix under the fixed selection tuple; retain raw normal arrays and every result row before the next wave.',goal='ACTIVE_UNMET')
    proof_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    started.update(status=record['status'],completion=str(proof_path),remote_controller_absent=True)
    (PF/(tag+'_actual_session_20261015.json')).write_text(json.dumps(started,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=record['status'],recipe=key,seed=args.seed,formal_updates=record['formal_updates'],D_free=record['D_free'])),flush=True)


if __name__=='__main__':main()
