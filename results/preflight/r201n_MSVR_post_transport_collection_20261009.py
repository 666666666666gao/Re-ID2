"""Collect the two original completed N MSVR runs after the observed SSH failure."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified

REMOTE='/data/gaob/Re-ID/DeMo-DualAxis'
PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/r201n_bounded_identity_shift_20261009/MSVR310'
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201n_bounded_identity_shift_20261009')
LOCAL=PROJECT/'results/r201n_bounded_identity_shift_20261009/MSVR310'
METRICS=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')


def main():
    pf=PROJECT/'results/preflight'
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    review=load(pf/'r201n_MSVR_post_transport_collection_source_review_20261009.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
    failure=load(pf/'r201n_transport_interruption_actual_20261009.json')
    observation=load(pf/'r201n_original_MSVR_after_transport_observation_20261009.json')
    assert failure['original_receiver']['exit_code']==1 and not observation['controller_alive']
    assert observation['controller_result'] is None and not observation['RGBNT100_root_exists']
    assert all(not row['process_alive'] and row['status']['status']=='COMPLETE' and row['epochs_count']==50 for row in observation['models'].values())
    native=load(pf/'r201n_bounded_identity_shift_MSVR310_native_actual_20261009.json')
    assert native['actual_native_updates']==6 and native['controlled_final_shift_bound']
    proof=pf/'r201n_MSVR_post_transport_collection_actual_20261009.json'
    intake=pf/'r201n_MSVR_post_transport_text_intake_20261009.json'
    package=pf/'r201n_MSVR_post_transport_text_20261009.tar.gz'
    remote_package='/tmp/'+package.name
    assert not proof.exists() and not intake.exists() and not package.exists() and not LOCAL.exists() and not ARCHIVE.exists()
    assert shutil.disk_usage(ARCHIVE.parent).free>sum(row['files']['best_official_arrays.npz']['bytes'] for row in observation['models'].values())+512*1024**2
    reused={n:s for n,s in review['directly_reused_sources_sha256'].items() if not n.startswith('results/')}
    code=f'''import csv,hashlib,json,tarfile
from pathlib import Path
project=Path({REMOTE!r});root=Path({ROOT!r});package=Path({remote_package!r})
assert all(hashlib.sha256((project/n).read_bytes()).hexdigest()==sha for n,sha in {reused!r}.items())
controller=json.loads((root/'controller_launch.json').read_text())
assert not Path('/proc/'+str(controller['pid'])).exists() and not (root/'controller_result.json').exists()
assert not (root.parent/'RGBNT100').exists() and not package.exists()
jobs=[];orders={{}}
for variant in ('frequency_shared','axis_shared'):
 name='MSVR310_r201n_'+variant+'_s42';run=root/'training'/name
 launch=json.loads((root/'training'/(name+'_launch.json')).read_text())
 assert not Path('/proc/'+str(launch['pid'])).exists()
 assert json.loads((root/'training'/(name+'_exit.json')).read_text())['exit_code']==0
 assert json.loads((root/'audit'/(name+'_exit.json')).read_text())['exit_code']==0
 data=json.loads((run/'result.json').read_text());audit=json.loads((run/'normal_cpu_audit.json').read_text())
 assert data['status']=='COMPLETE' and data['epochs']==50 and data['steps']==data['optimizer_steps']==1000 and data['amp_skipped_steps']==0
 assert (data['train_records'],data['query_records'],data['gallery_records'])==(1032,591,1055)
 assert data['training_coverage']==dict(eligible=1032,visited=1032,unvisited=[])
 assert data['training_heldout_identities']==0 and data['descriptor_dim']==5120
 assert data['arguments']['seed']==42 and data['arguments']['dataset']=='MSVR310' and data['arguments']['variant']==variant
 assert data['method_revision'].startswith('R201N one-factor final unit5120 shift bound')
 assert data['anchor']['final_unit_descriptor_bound']['epsilon']==.10 and data['anchor']['final_unit_descriptor_bound']['parameter_change']==0
 assert '  NUM_INSTANCE: 8' in data['config'].splitlines()
 assert [f.name for f in run.glob('*.pth')]==['best.pth']
 assert audit['status']=='PASS' and audit['max_metric_error']<1e-8 and audit['CMC50_and_per_query_and_groups'] and audit['full_training_coverage']
 assert audit['installed_split_counts']==dict(train=1032,query=591,gallery=1055)
 assert all(abs(data['full_metrics'][k]-data['strict_reload'][k])<1e-8 for k in {METRICS!r})
 epochs=list(csv.DictReader((run/'epochs.csv').open()))
 assert [int(row['epoch']) for row in epochs]==list(range(1,51))
 selected=max(epochs,key=lambda row:float(row['mAP']))
 assert int(selected['epoch'])==data['best']['epoch']
 assert all(abs(float(selected[k])-data['full_metrics'][k])<1e-8 for k in {METRICS!r})
 batches=[json.loads(line) for line in (run/'batch_orders.jsonl').read_text().splitlines()]
 assert len(batches)==1000 and all(row['optimizer_updated'] for row in batches)
 assert all(row['final_descriptor_bound_full']['epsilon']==.10 and row['final_descriptor_bound_full']['bounded_shift_max']<=.10+1e-6 for row in batches)
 orders[variant]=[(r['epoch'],r['step'],r['names'],r['partial_set']) for r in batches]
 smoke=json.loads((root/'native'/(name+'_smoke')/'smoke.json').read_text())
 assert smoke['status']=='SMOKE_PASS' and smoke['steps']==smoke['attempts']==3 and smoke['amp_skipped_steps']==0 and smoke['strict_reload_equal']
 assert json.loads((root/'native'/(name+'_smoke_exit.json')).read_text())['exit_code']==0
 raw=run/'best_official_arrays.npz';best=run/'best.pth'
 assert raw.exists() and not (run/'normal_local_archive.json').exists()
 jobs.append(dict(name=name,variant=variant,run=str(run),selected_epoch=data['best']['epoch'],full_metrics=data['full_metrics'],
  raw=str(raw),file=dict(bytes=raw.stat().st_size,sha256=hashlib.sha256(raw.read_bytes()).hexdigest()),
  best_sha256=hashlib.sha256(best.read_bytes()).hexdigest()))
assert orders['frequency_shared']==orders['axis_shared']
files={{}}
with tarfile.open(package,'w:gz') as archive:
 for path in sorted(root.rglob('*')):
  if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.log','.txt'):
   name=path.relative_to(root).as_posix();body=path.read_bytes();files[name]=dict(bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
   archive.add(path,arcname=name,recursive=False)
print(json.dumps(dict(jobs=jobs,files=files,paired_sampling_exact=True,original_controller_complete=False,
 package=dict(bytes=package.stat().st_size,sha256=hashlib.sha256(package.read_bytes()).hexdigest()))))'''
    inventory=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
    assert len(inventory['jobs'])==2 and inventory['paired_sampling_exact'] and not inventory['original_controller_complete']
    archives={}
    for job in inventory['jobs']:
        path=copy_verified(job['raw'],job['file'],Path(REMOTE+'/runs/r201n_bounded_identity_shift_20261009'),ARCHIVE)
        archives[job['name']]=dict(file=job['file'],local=path,remote=job['raw'],remote_original_retained=True)
    command(['scp',*OPTIONS,'2026:'+remote_package,str(package)])
    assert package.stat().st_size==inventory['package']['bytes'] and hashlib.sha256(package.read_bytes()).hexdigest()==inventory['package']['sha256']
    LOCAL.mkdir(parents=True)
    with tarfile.open(package,'r:gz') as archive:
        assert set(archive.getnames())==set(inventory['files'])
        archive.extractall(LOCAL,filter='data')
    assert all((LOCAL/n).stat().st_size==r['bytes'] and hashlib.sha256((LOCAL/n).read_bytes()).hexdigest()==r['sha256'] for n,r in inventory['files'].items())
    assert not (LOCAL/'controller_result.json').exists()
    stamp=datetime.now().astimezone().isoformat(timespec='seconds')
    for job in inventory['jobs']:
        receipt=dict(status='POST_TRANSPORT_LOCAL_SIZE_SHA_VERIFIED_REMOTE_ORIGINAL_RETAINED',verified_at=stamp,
            original_handshake_completed=False,original_receiver_exit_code=1,**archives[job['name']])
        (LOCAL/'training'/job['name']/'normal_post_transport_archive.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    collected=dict(status='ORIGINAL_N_MSVR_TRAINING_GT_AND_RAW_COLLECTED_AFTER_TRANSPORT',verified_at=stamp,
        dataset='MSVR310',controls=2,original_receiver_exit_code=1,original_controller_complete=False,
        successful_updates=2000,additional_epochs=100,native_updates=6,paired_sampling_exact=True,verified_local_raw_files=2,
        archives=archives,jobs=inventory['jobs'],new_neural_calls=0,new_optimizer_updates=0,RGBNT100_started=False,
        limits='Collect original completed NN runs only; no fabricated controller, original ACK/clear or receiver success. Remote unique raw retained.')
    (LOCAL/'post_transport_collection.json').write_text(json.dumps(collected,indent=2)+'\n',encoding='utf-8')
    intake.write_text(json.dumps(dict(status='ACTUAL_N_ORIGINAL_MSVR_TWO_FULL50_GT_TEXT_VERIFIED_AFTER_TRANSPORT',dataset='MSVR310',verified_at=stamp,
        manifest=inventory['files'],files=len(inventory['files']),original_controller_complete=False,original_receiver_exit_code=1,raw_local_verified=2,
        new_neural_calls=0,new_optimizer_updates=0),indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,'-X','utf8','-B','-S',str(PROJECT/'analyze_r201n_MSVR_post_transport_normal.py'),'--dataset','MSVR310'],check=True)
    assert load(LOCAL/'normal_analysis/result.json')['status']=='ACTUAL_N_DATASET_TWO_FULL50_NORMAL_CPU_READOUT'
    proof.write_text(json.dumps(dict(status='ACTUAL_N_ORIGINAL_MSVR_POST_TRANSPORT_GT_RAW_TEXT_CPU_COMPLETE',finished=datetime.now().astimezone().isoformat(timespec='seconds'),
        collection=collected,manifest=inventory['files'],source_review=str(pf/'r201n_MSVR_post_transport_collection_source_review_20261009.json'),
        new_neural_calls=0,new_optimizer_updates=0,original_receiver_exit_code=1,original_controller_complete=False,remote_raw_retained=True),indent=2)+'\n',encoding='utf-8')
    command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input='from pathlib import Path\np=Path('+repr(remote_package)+');assert p.parent==Path("/tmp");p.unlink()\n')
    assert package.resolve().parent==pf.resolve();package.unlink()
    print(json.dumps(dict(status='ACTUAL_N_ORIGINAL_MSVR_POST_TRANSPORT_GT_RAW_TEXT_CPU_COMPLETE',text_files=len(inventory['files']),normal_raw_copied=2,remote_raw_retained=True,new_training_starts=0,new_neural_calls=0)),flush=True)


if __name__=='__main__':main()
