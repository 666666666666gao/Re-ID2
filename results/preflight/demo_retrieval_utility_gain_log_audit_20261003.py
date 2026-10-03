import json
import math
from pathlib import Path

project=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
root=project/'results/axis_collaboration_v10_retrieval_utility_trial/development/MSVR310_axis_retrieval_utility_fullref_s42'
result=json.loads((root/'result.json').read_text())
with (root/'batch_orders.jsonl').open(encoding='utf-8') as handle:rows=[json.loads(line) for line in handle]
assert result['status']=='COMPLETE' and result['epochs']==50 and len(rows)==453
assert result['retrieval_gain']['weight']==.25 and result['retrieval_gain']['margin']==.02 and result['retrieval_gain']['temperature']==.1
for row in rows:
 gain=row['retrieval_gain'];count=len(row['names'])
 assert gain['margin']==.02 and gain['temperature']==.1 and gain['stopped_reference'] and gain['stopped_targets'] and gain['direct_base_gradient_removed']
 assert all(math.isfinite(gain[key]) for key in ('loss','weight_mean','active_fraction','base_margin_mean','full_margin_mean','stopped_target_mean'))
 assert gain['loss']>=0 and 0<gain['weight_mean']<=1 and 0<=gain['active_fraction']<=1
 assert len(gain['positive_indices'])==len(gain['negative_indices'])==count
 assert all(0<=pos<count and pos!=i and 0<=neg<count and pos!=neg and neg!=i for i,(pos,neg) in enumerate(zip(gain['positive_indices'],gain['negative_indices'])))
summary=dict(training_batches=453,all_gain_statistics_finite=True,recorded_hyperparameters_exact=True,indices_valid_no_self_or_positive_negative_overlap=True,
 actual_stop_gradient_tensor_witness='preflight/tensor/result.json; base gain grad exact0 and residual finite nonzero',
 mean_gain_loss=sum(row['retrieval_gain']['loss'] for row in rows)/len(rows),
 mean_active_fraction=sum(row['retrieval_gain']['active_fraction'] for row in rows)/len(rows),
 mean_hardness_weight=sum(row['retrieval_gain']['weight_mean'] for row in rows)/len(rows),
 first_gain=rows[0]['retrieval_gain'],last_gain=rows[-1]['retrieval_gain'])
path=project/'results/preflight/retrieval_utility_msvr_development_analysis.json'
audit=json.loads(path.read_text());assert audit['status']=='PASS_EVIDENCE_AUDIT'
audit['retrieval_gain_log_checks']=summary
audit['limits']='Single-seed unchanged interface B with added stopped-reference joint retrieval margin gain (weight.25, margin.02, hardness temperature.1). Direct base-coordinate gain gradient removed; experts still update sharedCLIP. Original auxiliary losses and gates unchanged. Output5632 padded with512zero tail, active5120. No matched retrained control, official test or stable cross-dataset claim. Original frozen analysis filename was copied from B; this explicit gain-log audit corrects its descriptive limit without changing any metric.'
path.write_bytes(json.dumps(audit,indent=2).encode('utf-8'))
print('ACTUAL_GAIN_LOGS_AUDITED',json.dumps({key:value for key,value in summary.items() if key not in ('first_gain','last_gain')}))
