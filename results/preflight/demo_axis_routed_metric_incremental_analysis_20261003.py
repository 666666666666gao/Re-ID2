import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,'C:/Users/gb/projects/demo_dual_axis_20261002')
from collect_results import PROJECT

def read(path):return json.loads((PROJECT/path).read_text(encoding='utf-8'))

name='MSVR310_axis_metric_routed_fullref_s42'
root='results/axis_collaboration_v8_routed_metric_trial/development/'+name
result=read(root+'/result.json')
assert result['status']=='COMPLETE' and result['epochs']==50
assert read('results/axis_collaboration_v8_routed_metric_trial/development/'+name+'_exit.json')['exit_code']==0
assert result['arguments']['variant']=='axis_metric_routed_fullref' and result['retrieval_interface']=='metric_weighted_frequency_block'
assert result['arguments']['seed']==42 and result['arguments']['contribution_weight']==.05
with (PROJECT/root/'epochs.csv').open(encoding='utf-8',newline='') as handle:epochs=list(csv.DictReader(handle))
assert [int(row['epoch']) for row in epochs]==list(range(1,51))
assert all(math.isfinite(float(row[key])) for row in epochs for key in ('loss','mAP','Rank-1','Rank-5','Rank-10','seconds'))
maximum=max(float(row['mAP']) for row in epochs)
earliest=next(int(row['epoch']) for row in epochs if float(row['mAP'])==maximum)
assert result['best']['epoch']==earliest and abs(result['best']['mAP']-maximum)<1e-9
metrics=read(root+'/development_metrics/full_metrics.json')
assert all(abs(metrics[key]-result['best'][key])<1e-8 and abs(metrics[key]-result['strict_reload'][key])<1e-8 for key in ('mAP','Rank-1','Rank-5','Rank-10'))
intake=read(root+'/intake.json')
assert intake['exit_code']==0
for filename,proof in intake['files'].items():
    path=PROJECT/'results/axis_collaboration_v8_routed_metric_trial'/filename
    assert path.stat().st_size==proof['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==proof['sha256']
v5root='results/axis_collaboration_v5_mass/development/MSVR310_axis_mass_fullref_s42'
v5=read(v5root+'/result.json')
assert result['parameters']==v5['parameters'] and result['trainable_parameters']==v5['trainable_parameters'] and result['descriptor_dim']==v5['descriptor_dim']==5632
with (PROJECT/root/'batch_orders.jsonl').open(encoding='utf-8') as handle:new=[json.loads(row) for row in handle]
with (PROJECT/v5root/'batch_orders.jsonl').open(encoding='utf-8') as handle:old=[json.loads(row) for row in handle]
assert len(new)==len(old)==result['steps']==453
assert all((left['epoch'],left['step'],left['names'])==(right['epoch'],right['step'],right['names']) for left,right in zip(new,old))
updates=sum(bool(row['optimizer_updated']) for row in new)
assert updates==result['optimizer_steps'] and result['steps']-updates==result['amp_skipped_steps']
# Read the already audited six-metric tables, independent of printed summaries.
v5metrics=read('results/axis_collaboration_v5_mass/development/MSVR310_axis_mass_fullref_s42/development_metrics/full_metrics.json')
metricA=read('results/axis_collaboration_v7_metric_interface_trial/development/MSVR310_axis_metric_fullref_s42/development_metrics/full_metrics.json')
with (PROJECT/'results/axis_collaboration_v5_mass_frozen_development/all3_missing_metrics_78.csv').open(encoding='utf-8',newline='') as handle:baseline_rows=list(csv.DictReader(handle))
demo=next(row for row in baseline_rows if row['dataset']=='MSVR310' and row['model']=='DeMo' and row['condition']=='clean')
keys=('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')
deltas={label:{key:metrics[key]-float(base[key]) for key in keys} for label,base in [('DeMo',demo),('V5',v5metrics),('MetricA',metricA)]}
report=dict(status='PASS_EVIDENCE_AUDIT',observed_at=datetime.now().isoformat(timespec='seconds'),dataset='MSVR310',run=name,
    metrics=metrics,deltas=deltas,selected_epoch=earliest,clean50_epochs=50,batch_attempts=453,optimizer_updates=updates,
    amp_skips=result['amp_skipped_steps'],all50_finite=True,earliest_best_verified=True,strict_four_metrics_verified=True,
    sampler_epoch_step_names_identical_V5=True,parameters=result['parameters'],trainable_parameters=result['trainable_parameters'],descriptor_dim=5632,
    pass_two_point_gate=deltas['DeMo']['mAP']>=2 and deltas['DeMo']['Rank-1']>=2,
    limits='Single-seed actual-route frequency auxiliary supervision on MetricA geometry, F AUX total .1 split .05 independent/.05 actual routed normalized PF with reused cosine classifier scale32. Ordinary controls not retrained with this objective. Frozen/missing diagnosis separate, no official test or cross-dataset stability claim.')
output=PROJECT/'results/preflight/axis_routed_metric_msvr_development_analysis.json';assert not output.exists()
output.write_bytes(json.dumps(report,indent=2).encode('utf-8'))
print('ROUTED_METRIC_MSVR_AUDITED',json.dumps(dict(metrics={key:metrics[key] for key in keys},deltas=deltas,gate=report['pass_two_point_gate'])),flush=True)
