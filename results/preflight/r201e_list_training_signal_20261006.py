"""Read the first20 completed real batches by epoch; CPU only, no NN launch."""
from datetime import datetime
import json
from pathlib import Path
import shlex
import sys

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS, command

ROOT = '/data/gaob/Re-ID/DeMo-DualAxis/runs/r201e_list_objective_fp32check_20261006'
output = PROJECT/'results/r201e_training_signal_20261006/result.json'
assert not output.exists()
code = '''import hashlib,json,math
from pathlib import Path
root=Path(__ROOT__);models={};orders={}
for variant in ('frequency_shared','axis_shared'):
 folder=root/'training'/('RGBNT201_r201e_'+variant+'_s42')
 rows=[json.loads(line) for line in (folder/'batch_orders.jsonl').read_text().splitlines()]
 rows=[row for row in rows if row['epoch']<=20]
 assert len(rows)==1058 and all(row['optimizer_updated'] for row in rows)
 assert all(row['temperature']==.01 and row['metric_coefficient']==1 and row['added_losses']==0 for row in rows)
 assert all(row['primary_full_metric']=='label_masked_smooth_AP_unit5120' for row in rows)
 orders[variant]=[(r['epoch'],r['step'],r['names'],r['partial_set']) for r in rows]
 by_epoch=[]
 for epoch in range(1,21):
  items=[r for r in rows if r['epoch']==epoch];values=[r['smooth_ap_loss'] for r in items]
  assert values and all(math.isfinite(v) and -1e-6<=v<=1.000001 for v in values)
  means={key:math.fsum(r[key] for r in items)/len(items) for key in ('loss','full_loss','partial_ce','cross_triplet','smooth_ap_loss')}
  assert abs(means['loss']-means['full_loss']-.25*means['partial_ce']-.5*means['cross_triplet'])<1e-5
  by_epoch.append(dict(epoch=epoch,batches=len(items),displayed_AP_loss_exact_zero=sum(v==0 for v in values),
    AP_loss_min=min(values),AP_loss_max=max(values),**means))
 ap=[r['smooth_ap_loss'] for r in rows];unique={n for r in rows for n in r['names']}
 models[variant]=dict(actual_training_batches=1058,actual_optimizer_updates=1058,
   visited_training_files=len(unique),epochs=by_epoch,
   AP_loss_mean=math.fsum(ap)/len(ap),AP_loss_max=max(ap),
   displayed_AP_loss_exact_zero_batches=sum(v==0 for v in ap),
   AP_loss_below_1e_minus6_batches=sum(v<1e-6 for v in ap),
   primary_full_metric='label_masked_smooth_AP_unit5120',temperature=.01,
   sampling_sha256=hashlib.sha256(json.dumps(orders[variant],ensure_ascii=False).encode()).hexdigest())
assert orders['frequency_shared']==orders['axis_shared']
print(json.dumps(dict(status='ACTUAL_R201E_FIRST20_TRAINING_LIST_SIGNAL_CPU_REDUCTION',models=models,
  actual_completed_epoch_rows=40,actual_logged_batch_rows=2116,paired_order_exact=True,
  new_neural_calls=0,new_optimizer_updates=0,
  limits='Observed training loss only, not validation gain. A displayed FP32 loss0 does not imply zero derivative; no formal parameter-gradient dominance measured. Entire training protocol unchanged.')))
'''.replace('__ROOT__',repr(ROOT))
data = json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python')+' -'],input=code))
data['observed_at'] = datetime.now().isoformat(timespec='seconds')
output.parent.mkdir(exist_ok=False)
output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=data['status'],observed_at=data['observed_at'],
    models={k:{field:v[field] for field in ('AP_loss_mean','AP_loss_max','displayed_AP_loss_exact_zero_batches','AP_loss_below_1e_minus6_batches','visited_training_files')} for k,v in data['models'].items()})),flush=True)
