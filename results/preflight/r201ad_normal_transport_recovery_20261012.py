"""Recover completed original AD artifacts after actual local ACK-owner loss; no NN."""
from pathlib import Path
from datetime import datetime
import hashlib,json,os,shlex,shutil,subprocess,sys,tarfile
PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified
REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/r201ad_RGB_precision_matched_20261012/RGBNT100'
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201ad_RGB_precision_matched_20261012/post_transport')


def main():
    pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    loss=load(pf/'r201ad_original_local_receiver_loss_actual_20261012.json')
    assert loss['local_process_absent'] and loss['original_local_exit_code'] is None and not loss['formal_restarted']
    review=load(pf/'r201ad_transport_recovery_source_review_20261012.json');assert review['status']=='PASS' and not review['blocking_findings']
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
    local=PROJECT/'results/r201ad_RGB_precision_matched_20261012/RGBNT100'
    proof=pf/'r201ad_recovered_normal_terminal_text_intake_20261012.json'
    package=pf/'r201ad_recovered_normal_closed_text_20261012.tar.gz';remote_package='/tmp/'+package.name
    assert not local.exists() and not proof.exists() and not package.exists() and not ARCHIVE.exists()
    code=f'''from pathlib import Path
from collections import Counter
import hashlib,json,os,signal,time
r=Path({ROOT!r});load=lambda p:json.loads(p.read_text());controller=load(r/'controller_launch.json')
assert controller['pid']==522680 and not (r/'controller_result.json').exists()
native=load(r/'native_acceptance.json');assert native['status']=='PASS' and native['controls']==4 and native['actual_updates']==12
files={{}};jobs=[];orders={{}}
for treatment,variant,gpu,pid in [('random','frequency_shared',0,524171),('random','axis_shared',1,524172),('hard','frequency_shared',2,524173),('hard','axis_shared',3,524174)]:
 n='RGBNT100_r201ad_'+treatment+'_'+variant+'_s42';p=r/'training'/n
 launch=load(r/'training'/(n+'_launch.json'));assert launch['pid']==pid
 assert not Path('/proc/'+str(pid)).exists() and load(r/'training'/(n+'_exit.json'))['exit_code']==0
 d=load(p/'result.json');assert d['status']=='COMPLETE' and d['epochs']==50 and d['steps']==d['optimizer_steps']==6563
 assert d['amp_skipped_steps']==d['training_heldout_identities']==0 and d['descriptor_dim']==5120
 assert (d['train_records'],d['query_records'],d['gallery_records'])==(8675,1715,8575)
 assert d['training_coverage']==dict(eligible=8675,visited=8675,unvisited=[])
 assert [x.name for x in p.glob('*.pth')]==['best.pth']
 audit_launch=load(r/'audit'/(n+'_launch.json'))
 assert not Path('/proc/'+str(audit_launch['pid'])).exists() and load(r/'audit'/(n+'_exit.json'))['exit_code']==0
 audit=load(p/'normal_cpu_audit.json');assert audit['status']=='PASS' and audit['max_metric_error']<1e-8
 assert audit['CMC50_and_per_query_and_groups'] and audit['full_training_coverage']
 raw=p/'best_official_arrays.npz';assert raw.exists() and not (p/'normal_local_archive.json').exists()
 files[n]=dict(path=str(raw),bytes=raw.stat().st_size,sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
 batch=[json.loads(line) for line in (p/'batch_orders.jsonl').read_text().splitlines()]
 assert len(batch)==6563 and all(row['optimizer_updated'] for row in batch)
 orders[(treatment,variant)]=[(x['epoch'],x['step'],x['names'],x['partial_set']) for x in batch]
 jobs.append(dict(name=n,treatment=treatment,variant=variant,gpu=gpu,pid=pid,NN_exit_code=0,audit_pid=audit_launch['pid'],audit_exit_code=0))
assert orders[('random','frequency_shared')]==orders[('random','axis_shared')]
assert orders[('hard','frequency_shared')]==orders[('hard','axis_shared')]
for epoch in range(1,51):
 a=[x for x in orders[('random','axis_shared')] if x[0]==epoch];b=[x for x in orders[('hard','axis_shared')] if x[0]==epoch]
 assert Counter(n for x in a for n in x[2])==Counter(n for x in b for n in x[2])
 assert [(x[0],x[1],x[3]) for x in a]==[(x[0],x[1],x[3]) for x in b]
# Stop only the stale transport owner after every original NN AND audit has closed.
# Its original return code is not inferred or changed to zero.
proc=Path('/proc/522680');stopped=False;argv=[]
if proc.exists():
 argv=(proc/'cmdline').read_bytes().decode().split('\\0')
 assert 'launch_r201ad_RGB_precision_matched.py' in argv and {ROOT!r} in argv
 os.kill(522680,signal.SIGTERM);stopped=True
 for _ in range(30):
  if not proc.exists():break
  time.sleep(1)
 assert not proc.exists()
record=dict(status='ACTUAL_OLD_AD_TRANSPORT_OWNER_CLOSED_AFTER_ALL_NN_AND_AUDITS',original_local_exit_code=None,
 original_terminal_consumed=False,controller_deliberately_terminated=stopped,controller_argv=argv,
 original_controller_complete=False,NN_children_restarted=False,new_neural_calls=0,new_optimizer_updates=0)
(r/'controller_transport_recovery_stop.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(dict(files=files,jobs=jobs,transport=record,all_original_NN_and_audit_children_closed=True)))'''
    ready=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    assert shutil.disk_usage(ARCHIVE.parent).free>sum(row['bytes'] for row in ready['files'].values())+1024**3
    archives={}
    for name,row in ready['files'].items():
        file=dict(bytes=row['bytes'],sha256=row['sha256'])
        destination=copy_verified(row['path'],file,Path(ROOT),ARCHIVE)
        archives[name]=dict(file=file,local=destination)
        print('AD_RECOVERY_ORIGINAL_RAW_LOCAL_FULL_SIZE_SHA',name,flush=True)
    assert all(Path(row['local']).stat().st_size==row['file']['bytes'] and hashlib.sha256(Path(row['local']).read_bytes()).hexdigest()==row['file']['sha256'] for row in archives.values())
    code=f'''from pathlib import Path
import hashlib,json,tarfile
r=Path({ROOT!r});package=Path({remote_package!r});assert not package.exists()
for name,row in {ready['files']!r}.items():
 p=Path(row['path']);assert p.resolve().is_relative_to(r.resolve())
 assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 p.unlink();(p.parent/'normal_recovery_archive.json').write_text(json.dumps(dict(status='NORMAL_LOCAL_SIZE_SHA_VERIFIED_POST_TRANSPORT_REMOTE_CLEARED',file=dict(bytes=row['bytes'],sha256=row['sha256']))))
recovery=dict(status='ACTUAL_AD_ORIGINAL_FOUR_FULL50_GT_D_RAW_RECOVERED',dataset='RGBNT100',controls=4,
 additional_epochs=200,successful_updates=26252,native_updates=12,paired_sampling_exact=True,normal_archives_local_verified=4,
 jobs={ready['jobs']!r},original_controller_complete=False,original_local_exit_code=None,original_terminal_consumed=False,
 all_original_NN_and_audit_children_closed=True,new_neural_calls=0,new_optimizer_updates=0)
(r/'controller_recovery.json').write_text(json.dumps(recovery,indent=2)+'\\n')
manifest={{}}
with tarfile.open(package,'w:gz') as archive:
 for p in sorted(r.rglob('*')):
  if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.log','.txt'):
   name=p.relative_to(r).as_posix();data=p.read_bytes();manifest[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());archive.add(p,arcname=name,recursive=False)
print(json.dumps(dict(manifest=manifest,package=dict(bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()))))'''
    inventory=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    command(['scp',*OPTIONS,'2026:'+remote_package,str(package)])
    assert package.stat().st_size==inventory['package']['bytes'] and hashlib.sha256(package.read_bytes()).hexdigest()==inventory['package']['sha256']
    local.mkdir(parents=True)
    with tarfile.open(package,'r:gz') as archive:
        assert set(archive.getnames())==set(inventory['manifest']);archive.extractall(local,filter='data')
    assert all((local/n).stat().st_size==row['bytes'] and hashlib.sha256((local/n).read_bytes()).hexdigest()==row['sha256'] for n,row in inventory['manifest'].items())
    command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input='from pathlib import Path\nPath('+repr(remote_package)+').unlink()\n')
    package.unlink()
    record=dict(status='ACTUAL_AD_FOUR_FULL50_POST_TRANSPORT_GT_TEXT_VERIFIED',dataset='RGBNT100',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        recovery_receiver_pid=os.getpid(),local_root=str(local),remote_root=ROOT,manifest=inventory['manifest'],archives=archives,
        formal_updates=26252,native_updates=12,all_original_NN_and_audit_children_closed=True,
        original_controller_complete=False,original_local_exit_code=None,original_terminal_consumed=False,new_neural_calls=0,new_optimizer_updates=0)
    proof.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,'-X','utf8','-B','-S',str(PROJECT/'analyze_r201ad_recovered_normal.py'),'--dataset','RGBNT100'],check=True)
    print('ACTUAL_AD_ORIGINAL_FOUR_FULL50_GT_RAW_CPU_RECOVERY_COMPLETE',flush=True)


if __name__=='__main__':main()
