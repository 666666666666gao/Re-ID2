"""Finish the diagnosed closed RGB100 RAW delivery; no training or model forward."""
from datetime import datetime
import hashlib,json,shlex,subprocess,sys
from pathlib import Path

PROJECT=Path('C:/Users/gb/projects/demo_dual_axis_20261002');sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
from results.preflight.full_official_anchor_stream_bridge_20261005 import copy_verified

REMOTE='/data/gaob/Re-ID/DeMo-DualAxis';PYTHON='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
ROOT=REMOTE+'/runs/r201l_uniform_k8_20261007'
ARCHIVE=Path('D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/r201l_uniform_k8_20261007')

def main():
 pf=PROJECT/'results/preflight';load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
 recovery_path=pf/'r201l_RGBNT100_closed_archive_recovery_actual_20261008.json';assert not recovery_path.exists()
 review=load(pf/'r201l_RGBNT100_closed_archive_recovery_review_20261008.json');assert review['status']=='PASS' and not review['blocking_findings']
 assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==review['source_sha256']
 prior_review=load(pf/'r201l_uniform_K8_source_review_20261007.json');assert prior_review['status']=='PASS'
 sources=prior_review['sources_sha256']|prior_review['directly_reused_sources_sha256']
 assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==s for n,s in sources.items())
 inspection=load(pf/'r201l_RGBNT100_closed_stream_inspection_actual_20261008.json')
 assert inspection['status']=='ACTUAL_CLOSED_RGB100_CONTROLLER_ARCHIVE_METADATA_INSPECTION' and inspection['controller_result']['status']=='ABSENT'
 byname={r['name']:r for r in inspection['records']};assert len(byname)==2
 ordinary='RGBNT100_r201l_frequency_shared_s42';axis='RGBNT100_r201l_axis_shared_s42'
 old=byname[ordinary]['files']['training/'+ordinary+'/normal_local_archive.json'];assert old['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED' and not byname[ordinary]['raw']['exists']
 old_info=old['file'];axis_raw=byname[axis]['raw'];assert axis_raw['exists']
 new_info={k:axis_raw[k] for k in ('bytes','sha256')}
 ordinary_local=ARCHIVE/'RGBNT100/training'/ordinary/'best_official_arrays.npz'
 assert ordinary_local.stat().st_size==old_info['bytes'] and hashlib.sha256(ordinary_local.read_bytes()).hexdigest()==old_info['sha256']
 actual=pf/'r201l_uniform_k8_RGBNT100_actual_session_20261007.json';assert not actual.exists()
 queue_path=pf/'r201l_uniform_k8_actual_session_20261007.json';queue_bytes=queue_path.read_bytes();queue=json.loads(queue_bytes)
 assert queue['status']=='ACTUAL_L_DATASET_NATIVE6_PASSED_FULL50_RUNNING' and queue['receiver_pid']==5824 and queue['native_updates_closed']==12
 original_snapshot=pf/'r201l_original_interrupted_queue_before_recovery_20261008.json';assert not original_snapshot.exists();original_snapshot.write_bytes(queue_bytes)
 protected={n:s for n,s in sources.items() if not n.startswith('results/')}
 code=f'''import csv,hashlib,json
from pathlib import Path
project=Path({REMOTE!r});root=Path({(ROOT+'/RGBNT100')!r})
assert not Path('/proc/83696').exists() and not Path('/proc/84802').exists() and not Path('/proc/84803').exists()
assert all(hashlib.sha256((project/n).read_bytes()).hexdigest()==s for n,s in {protected!r}.items())
assert not (root/'controller_result.json').exists()
native=json.loads((root/'native_acceptance.json').read_text());assert native['status']=='PASS' and native['actual_updates']==6
runs=[];orders=[];counts=[]
for variant,gpu in [('frequency_shared',2),('axis_shared',3)]:
 name='RGBNT100_r201l_'+variant+'_s42';run=root/'training'/name;data=json.loads((run/'result.json').read_text())
 assert data['status']=='COMPLETE' and data['epochs']==50 and data['steps']==data['optimizer_steps']==6563 and data['amp_skipped_steps']==0
 assert (data['train_records'],data['query_records'],data['gallery_records'])==(8675,1715,8575)
 assert data['training_heldout_identities']==0 and data['descriptor_dim']==5120 and data['arguments']['seed']==42
 assert data['training_coverage']==dict(eligible=8675,visited=8675,unvisited=[]) and '  NUM_INSTANCE: 8' in data['config'].splitlines()
 assert data['arguments']['output']==str(run) and [p.name for p in run.glob('*.pth')]==['best.pth']
 epochs=list(csv.DictReader((run/'epochs.csv').open()));assert [int(r['epoch']) for r in epochs]==list(range(1,51))
 selected=max(epochs,key=lambda r:float(r['mAP']));assert int(selected['epoch'])==data['best']['epoch']
 assert all(abs(float(selected[k])-data['full_metrics'][k])<1e-8 for k in ('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20'))
 audit=json.loads((run/'normal_cpu_audit.json').read_text());assert audit['status']=='PASS' and audit['max_metric_error']<1e-8 and audit['CMC50_and_per_query_and_groups']
 assert audit['installed_split_counts']==dict(train=8675,query=1715,gallery=8575)
 exits={{phase:json.loads((root/phase/(name+'_exit.json')).read_text()) for phase in ('training','audit')}}
 assert all(r['exit_code']==0 and r['gpu']==gpu and not r['forced_own_group_cleanup'] for r in exits.values())
 runs.append(dict(name=name,variant=variant,gpu=gpu,train=exits['training'],audit=exits['audit']))
 batches=[json.loads(line) for line in (run/'batch_orders.jsonl').read_text().splitlines()];assert len(batches)==6563 and all(r['optimizer_updated'] for r in batches)
 orders.append([(r['epoch'],r['step'],r['names'],r['partial_set']) for r in batches]);counts.append((data['parameters'],data['trainable_parameters']))
assert orders[0]==orders[1] and counts[0]==counts[1]
old=root/'training'/{ordinary!r}/'normal_local_archive.json';assert json.loads(old.read_text())['file']=={old_info!r}
assert not (root/'training'/{ordinary!r}/'best_official_arrays.npz').exists()
raw=root/'training'/{axis!r}/'best_official_arrays.npz';assert raw.stat().st_size=={new_info['bytes']!r} and hashlib.sha256(raw.read_bytes()).hexdigest()=={new_info['sha256']!r}
assert not (raw.parent/'normal_local_archive.json').exists()
print(json.dumps(dict(status='ACTUAL_EXISTING_RGB100_TWO_FULL50_GT_RAW_READY_FOR_ARCHIVE_ONLY_RECOVERY',runs=runs,new_neural_calls=0,new_optimizer_updates=0)))'''
 ready=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code))
 assert ready['status']=='ACTUAL_EXISTING_RGB100_TWO_FULL50_GT_RAW_READY_FOR_ARCHIVE_ONLY_RECOVERY'
 axis_local=Path(copy_verified(axis_raw['path'],new_info,Path(ROOT),ARCHIVE))
 assert axis_local.resolve().is_relative_to(ARCHIVE.resolve()) and axis_local.stat().st_size==new_info['bytes'] and hashlib.sha256(axis_local.read_bytes()).hexdigest()==new_info['sha256']
 recovered_at=datetime.now().isoformat(timespec='seconds')
 controller=dict(status='COMPLETE',dataset='RGBNT100',controls=2,runs=ready['runs'],additional_epochs=100,successful_updates=13126,native_updates=6,paired_sampling_exact=True,normal_archives_local_verified=2,missing_evaluation='Deferred until all normal stage is closed',recovery=dict(administrative_only=True,recorded_at=recovered_at,original_controller_result_was_absent=True,original_controller_exit_code=None,reason='Original local receiver was absent; ordinary RAW already ACK-cleared, axis RAW remained. Training and GT audit original exits are0; interruption cause unverified.',new_neural_calls=0,new_optimizer_updates=0))
 code=f'''import hashlib,json
from pathlib import Path
project=Path({REMOTE!r});root=Path({(ROOT+'/RGBNT100')!r});raw=Path({axis_raw['path']!r})
assert raw.resolve().is_relative_to(root.resolve()) and raw.stat().st_size=={new_info['bytes']!r} and hashlib.sha256(raw.read_bytes()).hexdigest()=={new_info['sha256']!r}
assert all(hashlib.sha256((project/n).read_bytes()).hexdigest()==s for n,s in {protected!r}.items())
assert not (root/'controller_result.json').exists() and not (raw.parent/'normal_local_archive.json').exists()
raw.unlink();(raw.parent/'normal_local_archive.json').write_text(json.dumps(dict(status='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED',file={new_info!r},recovery_administrative_only=True),indent=2)+'\\n')
(root/'controller_result.json').write_text(json.dumps({controller!r},indent=2)+'\\n')
assert not raw.exists() and json.loads((root/'controller_result.json').read_text())=={controller!r}
print('ACTUAL_EXISTING_RGB100_LOCAL_SHA_ACK_AND_ADMINISTRATIVE_CONTROLLER_TERMINAL')'''
 print(command(['ssh',*OPTIONS,'2026',shlex.quote(PYTHON)+' -'],input=code),end='',flush=True)
 archives={ordinary:dict(file=old_info,local=str(ordinary_local)),axis:dict(file=new_info,local=str(axis_local))}
 value=dict(status='ACTUAL_L_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW',finished=recovered_at,exit_code=0,dataset='RGBNT100',remote_root=ROOT+'/RGBNT100',archives=archives,native_updates=6,successful_updates=13126,additional_epochs=100,sources_sha256=prior_review['sources_sha256'],delivery_recovery=controller['recovery'],exit_code_provenance='Existing training/audit exits0 plus this archive recovery; original stream controller exit not attested.')
 actual.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
 subprocess.run([sys.executable,'-X','utf8','-B','-S',str(pf/'r201l_dataset_normal_complete_intake_20261007.py'),'--dataset','RGBNT100'],check=True)
 subprocess.run([sys.executable,'-X','utf8','-B','-S',str(PROJECT/'analyze_r201l_dataset_normal.py'),'--dataset','RGBNT100'],check=True)
 outcomes={d:load(pf/('r201l_uniform_k8_'+d+'_actual_session_20261007.json')) for d in ('MSVR310','RGBNT100')}
 assert all(r['status']=='ACTUAL_L_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW' and r['exit_code']==0 for r in outcomes.values())
 assert sum(r['successful_updates'] for r in outcomes.values())==15126
 prior=load(PROJECT/'results/r201k_relation_local_pi_20261007/RGBNT201/normal_analysis/result.json')
 terminal=dict(status='ACTUAL_L_TWO_CHANGED_NORMAL_DATASETS_FOUR_FULL50_GT_RAW_COMPLETE_201_K8_REUSED',finished=datetime.now().isoformat(timespec='seconds'),exit_code=0,datasets=outcomes,native_updates=12,successful_updates=15126,additional_epochs=200,sources_sha256=prior_review['sources_sha256'],RGBNT201_reused_K8=prior,missing_evaluated=False,delivery_recovery=controller['recovery'],limits='Existing four formal50/native12/GT/RAW/text/CPU completed without retraining; original interrupted queue preserved, archival recovery only. Goal remains ACTIVE_UNMET.')
 queue_path.write_text(json.dumps(terminal,indent=2)+'\n',encoding='utf-8')
 recovery_path.write_text(json.dumps(dict(status='ACTUAL_EXISTING_RGB100_ARCHIVE_ONLY_RECOVERY_AND_THREE_NORMAL_TERMINAL',finished=terminal['finished'],controller=controller,archives=archives,new_neural_calls=0,new_optimizer_updates=0,goal='ACTIVE_UNMET'),indent=2)+'\n',encoding='utf-8')
 print('ACTUAL_EXISTING_L_THREE_NORMAL_TERMINAL_RECOVERED_WITHOUT_NN_RESTART',flush=True)

if __name__=='__main__':main()
