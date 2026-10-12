"""Locate pre-adaptation divergence in already-completed five-epoch batch logs."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import hashlib,json,shlex,sys

PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from collect_results import OPTIONS,command
PF=PROJECT/'results/preflight'


def main():
    phase=json.loads((PF/'r201ak_first_phase_actual_20261013.json').read_text())
    assert phase['controller_alive'] and all(v['latest_batch']['stage_epoch']>=6 for v in phase['workers'].values())
    old={};inputs={}
    for variant in ('frequency_shared','axis_shared'):
        f=PROJECT/'results/r201ag_RGB_coordinate_metric_20261012/RGBNT100/training'/('RGBNT100_r201ag_'+variant+'_s42')/'batch_orders.jsonl'
        values=[]
        with f.open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line)
                if row['epoch']>5:break
                values.append(row)
        old[variant]=values;inputs[f.relative_to(PROJECT).as_posix()]=hashlib.sha256(f.read_bytes()).hexdigest()
    fields={variant:[k for k in values[0] if k!='freeze_identity_encoder'] for variant,values in old.items()}
    code=f'''import json
from pathlib import Path
r=Path('/data/gaob/Re-ID/DeMo-DualAxis/runs/r201ak_RGB_lastblock_staged_20261013/RGBNT100/training');fields={fields!r};data={{}}
for variant,keys in fields.items():
 values=[]
 with (r/('RGBNT100_r201ak_'+variant+'_s42')/'batch_orders.jsonl').open() as stream:
  for line in stream:
   row=json.loads(line)
   if row['epoch']>5:break
   assert row['adaptation_phase']=='outlets' and row['last_visual_block_effective_LR_factor']==0
   assert row['original_core_eval_modes_fixed'] and row['original_core_buffers_retained']
   values.append({{key:row[key] for key in keys}})
 data[variant]=values
print(json.dumps(data))'''
    actual=json.loads(command(['ssh',*OPTIONS,'2026',shlex.quote('/data/gaob/Re-ID/conda-envs/tri_reid/bin/python')+' -'],input=code))
    records=[]
    for variant,before in old.items():
        after=actual[variant];assert len(before)==len(after)
        assert all(tuple(a[k] for k in ('epoch','step','names','partial_set'))==tuple(b[k] for k in ('epoch','step','names','partial_set')) for a,b in zip(before,after))
        counts=Counter();first={};exact=0
        for a,b in zip(before,after):
            changed=[key for key in fields[variant] if a[key]!=b[key]]
            if not changed:exact+=1
            for key in changed:
                counts[key]+=1
                if key not in first:first[key]=dict(epoch=a['epoch'],step=a['step'],AG=a[key],AK=b[key])
        records.append(dict(variant=variant,completed_batches=len(before),exact_logged_batches=exact,
            record_names_partial_sets_epoch_steps_exact=True,changed_field_counts=dict(counts),first_field_difference=first,
            maximum_absolute_full_loss_difference=max(abs(a['full_loss']-b['full_loss']) for a,b in zip(before,after))))
    output=PF/'r201ak_first5_batch_parity_actual_20261013.json';assert not output.exists()
    result=dict(status='ACTUAL_FIRST5_COMPLETED_BATCH_AG_AK_DIFFERENCE_LOCATED',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        records=records,inputs_sha256=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_neural_calls=0,new_optimizer_updates=0,CPU_remote_completed_log_intakes=1,
        limits='Only retained complete pre-adaptation batch logs. Same filenames/partial sets do not prove identical input tensors; recorded numeric divergence does not establish a CUDA nondeterminism or model-mode root cause. No source hot edit, replay, or training restart; subsequent full runs and paired-seed controls remain necessary.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],records=[{k:v[k] for k in ('variant','completed_batches','exact_logged_batches','maximum_absolute_full_loss_difference','record_names_partial_sets_epoch_steps_exact')}|dict(first_loss=v['first_field_difference'].get('full_loss'),first_positive=v['first_field_difference'].get('positive_indices'),first_negative=v['first_field_difference'].get('negative_indices')) for v in records]),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
