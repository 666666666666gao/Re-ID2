from datetime import datetime
import csv, hashlib, json, math
from pathlib import Path

project=Path(__file__).resolve().parents[2]
output=project/'results/preflight/r201l_MSVR_closed_primary_activity_actual_20261008.json'
assert not output.exists()
proof=json.loads((project/'results/r201l_uniform_k8_20261007/MSVR310/normal_analysis/result.json').read_text(encoding='utf-8-sig'))
assert proof['status']=='ACTUAL_L_DATASET_TWO_FULL50_NORMAL_CPU_READOUT'
readouts=[];sources={}
for family,total,instances in [('K',705,4),('L',1000,8)]:
 for variant in ('frequency_shared','axis_shared'):
  stage='r201k_relation_local_pi_20261007' if family=='K' else 'r201l_uniform_k8_20261007'
  folder=project/'results'/stage/'MSVR310/training'/('MSVR310_r201'+family.lower()+'_'+variant+'_s42')
  batch_path=folder/'batch_orders.jsonl';epoch_path=folder/'epochs.csv';result_path=folder/'result.json'
  contents={p:p.read_bytes() for p in (batch_path,epoch_path,result_path)}
  result=json.loads(contents[result_path]);batches=[json.loads(line) for line in contents[batch_path].decode('utf-8-sig').splitlines()]
  epochs=list(csv.DictReader(contents[epoch_path].decode('utf-8-sig').splitlines()))
  assert result['status']=='COMPLETE' and result['epochs']==50 and result['steps']==result['optimizer_steps']==total and result['amp_skipped_steps']==0
  assert len(batches)==total and [int(r['epoch']) for r in epochs]==list(range(1,51))
  assert all(r['optimizer_updated'] and r['primary_full_metric']=='unit5120_margin03_batch_hard_triplet' and r['primary_margin']==.3 and r['primary_metric_coefficient']==1 and r['auxiliary_original_soft_triplet'] and r['partial_loss_effective']==0 for r in batches)
  assert all(math.isfinite(r['primary_triplet_loss']) and 0<=r['primary_margin_violations']<=64 for r in batches)
  for path,body in contents.items():sources[path.relative_to(project).as_posix()]=dict(bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
  for label,first,last in [('all',1,50),('first5',1,5),('last10',41,50)]:
   rows=[r for r in batches if first<=r['epoch']<=last];assert rows
   readouts.append(dict(model=family+'_'+variant,NUM_INSTANCE=instances,window=label,first_epoch=first,last_epoch=last,batches=len(rows),primary_loss_zero_batches=sum(r['primary_triplet_loss']==0 for r in rows),primary_loss_zero_fraction=sum(r['primary_triplet_loss']==0 for r in rows)/len(rows),primary_loss_mean=math.fsum(r['primary_triplet_loss'] for r in rows)/len(rows),primary_violation_mean=math.fsum(r['primary_margin_violations'] for r in rows)/len(rows),total_loss_mean=math.fsum(r['loss'] for r in rows)/len(rows),epoch_last_mAP=float(epochs[last-1]['mAP']),epoch_last_Rank1=float(epochs[last-1]['Rank-1'])))
value=dict(status='ACTUAL_CLOSED_MSVR_K_L_PRIMARY_ACTIVITY_CPU_READOUT',recorded_at=datetime.now().isoformat(timespec='seconds'),dataset='MSVR310',expert_seed=42,formal_models=4,total_actual_batch_rows=3410,windows=readouts,sources=sources,new_neural_calls=0,new_optimizer_updates=0,goal='ACTIVE_UNMET',limits='Only already closed official-data K/L batch telemetry; no SSH, GPU, new model forward or parameters. K P16K4 versus L P8K8 changes positive multiplicity, negative identity pool and 705/1000 updates per arm; paired ordinary/axis within family match, K/L orders and budgets differ. Active primary loss does not prove helpful gradients or causal diagnosis; these windows are repeated training batches, not independent test samples. Negative all-six/selected/final-epoch evidence remains in official normal analysis; no benchmark-based new choice or missing inference.')
assert sum(r['batches'] for r in readouts if r['window']=='all')==value['total_actual_batch_rows']
output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=value['status'],new_neural_calls=0,last10=[r for r in readouts if r['window']=='last10']),ensure_ascii=False))
