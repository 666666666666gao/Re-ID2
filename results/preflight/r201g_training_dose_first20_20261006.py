"""Read completed first20 training rows; all final full-data checks remain required."""
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import shlex
import sys

PROJECT = Path('C:/Users/gb/projects/demo_dual_axis_20261002')
sys.path.insert(0, str(PROJECT))
from collect_results import OPTIONS, command

pf = PROJECT / 'results/preflight'
live = json.loads((pf / 'r201g_normal_priority_timed_latest_20261006.json').read_text(encoding='utf-8'))
assert all(row['epoch'] >= 20 for row in live['runs'])
output = PROJECT / 'results/r201g_training_dose_first20_20261006/result.json'
assert not output.exists()
references = {}
for variant in ('frequency_shared', 'axis_shared'):
    path = PROJECT / ('results/rgbnt201_identity_outlet_r201c_20261005/training/RGBNT201_identity_' + variant + '_narrow_s42/batch_orders.jsonl')
    with path.open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in itertools.islice(handle, 1058)]
    assert len(rows) == 1058 and rows[-1]['epoch'] == 20
    order = [(row['epoch'], row['step'], row['names'], row['partial_set']) for row in rows]
    references[variant] = hashlib.sha256(json.dumps(order, ensure_ascii=False).encode()).hexdigest()
root = '/data/gaob/Re-ID/DeMo-DualAxis/runs/r201g_normal_priority_20261006'
code = f'''import hashlib,itertools,json,math
from pathlib import Path
root=Path({root!r});models={{}};orders={{}}
for variant in ('frequency_shared','axis_shared'):
 folder=root/'training'/('RGBNT201_r201g_'+variant+'_s42')
 with (folder/'batch_orders.jsonl').open() as handle:
  rows=[json.loads(line) for line in itertools.islice(handle,1058)]
 assert len(rows)==1058 and rows[-1]['epoch']==20 and all(r['optimizer_updated'] for r in rows)
 assert all(r['primary_full_metric']=='unit5120_soft_triplet' and r['loss']==r['full_loss'] and
  r['partial_loss_effective']==r['partial_CE_weight']==r['partial_triplet_weight']==0 for r in rows)
 orders[variant]=[(r['epoch'],r['step'],r['names'],r['partial_set']) for r in rows]
 digest=hashlib.sha256(json.dumps(orders[variant],ensure_ascii=False).encode()).hexdigest()
 assert digest=={references!r}[variant]
 by_epoch=[]
 for epoch in range(1,21):
  selected=[r for r in rows if r['epoch']==epoch]
  assert selected
  means={{key:math.fsum(r[key] for r in selected)/len(selected) for key in ('loss','full_loss','partial_ce','cross_triplet','partial_loss_effective')}}
  assert all(math.isfinite(value) for value in means.values()) and means['loss']==means['full_loss'] and means['partial_loss_effective']==0
  by_epoch.append(dict(epoch=epoch,batches=len(selected),**means))
 models[variant]=dict(actual_batches=1058,actual_optimizer_updates=1058,epochs=by_epoch,
  visited_training_files=len({{name for r in rows for name in r['names']}}),sampling_sha256=digest,
  partial_losses_diagnostic_only=True,partial_loss_effective_zero_batches=1058)
assert orders['frequency_shared']==orders['axis_shared']
print(json.dumps(dict(status='ACTUAL_R201G_FIRST20_COMPLETED_OBJECTIVE_SAMPLING_CPU_CHECK',
 models=models,actual_completed_epoch_rows=40,actual_logged_batch_rows=2116,
 paired_order_exact=True,paired_closed_unit_order_exact=True,new_neural_calls=0,new_optimizer_updates=0,
 limits='Completed20-epoch training prefix only; not a terminal model/GT/missing/multiseed result or a new gradient measurement. Raw partial losses are computed but excluded from the optimizer objective.')))
'''
data = json.loads(command(['ssh', *OPTIONS, '2026',
    shlex.quote('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python') + ' -'], input=code))
data['observed_at'] = datetime.now().isoformat(timespec='seconds')
output.parent.mkdir(exist_ok=False)
output.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: data[key] for key in ('status', 'observed_at', 'actual_logged_batch_rows', 'new_neural_calls')}))
