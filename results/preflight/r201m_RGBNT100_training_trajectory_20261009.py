"""CPU-only temporal readout of closed M RGBNT100 training; no neural replay."""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

p=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
root=p/'results/r201m_primary_margin06_20261008/RGBNT100'
pf=p/'results/preflight'
proof=pf/'r201m_RGBNT100_training_trajectory_actual_20261009.json'
table=root/'normal_analysis/training_trajectory.csv'
assert not proof.exists() and not table.exists()
load=lambda path:json.loads(path.read_text(encoding='utf-8'))
actual=load(pf/'r201m_primary_margin06_RGBNT100_actual_session_20261008.json')
assert actual['exit_code']==0 and actual['successful_updates']==13126 and actual['native_updates']==6
normal=load(root/'normal_analysis/result.json')
assert normal['status']=='ACTUAL_M_DATASET_TWO_FULL50_NORMAL_CPU_READOUT'
with (root/'normal_analysis/six_metrics.csv').open(encoding='utf-8',newline='') as handle:
    references=[r for r in csv.DictReader(handle) if r['model']=='original_DeMo']
assert len(references)==1
baseline=float(references[0]['mAP'])
outlet=p/'relation_local_identity_outlet.py'
assert 'weights = relation_mass.float() * eligible.sum(1, keepdim=True)' in outlet.read_text(encoding='utf-8')
means=lambda rows,key:math.fsum(float(r[key]) for r in rows)/len(rows)
rows_out=[];models={};input_hashes={str(path.relative_to(p).as_posix()):hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (root/'normal_analysis/six_metrics.csv',root/'normal_analysis/result.json',outlet,p/'dual_axis.py')}
for variant in ('frequency_shared','axis_shared'):
    folder=root/'training'/('RGBNT100_r201m_'+variant+'_s42')
    result=load(folder/'result.json')
    batches=[json.loads(line) for line in (folder/'batch_orders.jsonl').read_text(encoding='utf-8').splitlines()]
    with (folder/'epochs.csv').open(encoding='utf-8',newline='') as handle:epochs=list(csv.DictReader(handle))
    assert result['epochs']==50 and result['steps']==result['optimizer_steps']==len(batches)==6563
    assert result['amp_skipped_steps']==0 and len(epochs)==50
    assert all(b['optimizer_updated'] and b['partial_loss_effective']==0 and b['primary_margin']==.6 for b in batches)
    for name in ('result.json','epochs.csv','batch_orders.jsonl'):
        path=folder/name;input_hashes[str(path.relative_to(p).as_posix())]=hashlib.sha256(path.read_bytes()).hexdigest()
    series=[]
    for epoch,row in enumerate(epochs,1):
        assert int(row['epoch'])==epoch
        group=[b for b in batches if b['epoch']==epoch];assert len(group)==int(row['steps'])-(int(epochs[epoch-2]['steps']) if epoch>1 else 0)
        gate=[math.fsum(b['full_gate_mean'][i] for b in group)/len(group) for i in range(3)]
        scales=[math.fsum(b['residual_scales'][i] for b in group)/len(group) for i in range(3)]
        relation=[math.fsum(b['route_relation_mass_mean'][i] for b in group)/len(group) for i in range(7)]
        band=[math.fsum(b['route_band_mass_mean'][i] for b in group)/len(group) for i in range(3)]
        full=means(group,'full_loss');hinge=means(group,'primary_triplet_loss')
        record=dict(variant=variant,epoch=epoch,mAP=float(row['mAP']),Rank1=float(row['Rank-1']),
            mAP_delta_original_DeMo_pp=float(row['mAP'])-baseline,full_loss_mean=full,
            final_hinge06_loss_mean=hinge,other_full_loss_including_CE_aux_contribution_mean=full-hinge,
            hinge06_fraction_of_full_loss=hinge/full,hinge06_active_anchor_fraction=means(group,'primary_margin_violations')/64,
            route_entropy_mean=means(group,'route_entropy_mean'),route_nonindependence_mean=means(group,'route_nonindependence_mean'),
            max_epoch_mean_relation_mass=max(relation),max_epoch_mean_band_mass=max(band),
            max_epoch_mean_relation_outlet_multiplier=7*max(relation),
            relation_argmax='R N T RN RT NT RNT'.split()[relation.index(max(relation))],
            band_argmax='L M H'.split()[band.index(max(band))],
            gate_M=gate[0],gate_F=gate[1],gate_I=gate[2],scale_M=scales[0],scale_F=scales[1],scale_I=scales[2])
        assert all(math.isfinite(v) for v in record.values() if isinstance(v,float))
        rows_out.append(record);series.append(record)
    find=lambda predicate:next((r for r in series if predicate(r)),None)
    models[variant]=dict(selected_best_epoch=result['best']['epoch'],
        selected_best_epoch_telemetry=series[result['best']['epoch']-1],end=series[-1],
        first_all_three_mean_gates_ge_095=find(lambda r:min(r['gate_M'],r['gate_F'],r['gate_I'])>=.95),
        first_mean_route_entropy_below_1nat=find(lambda r:r['route_entropy_mean']<1),
        first_epoch_mean_relation_mass_ge_09=find(lambda r:r['max_epoch_mean_relation_mass']>=.9),
        first_mAP_below_original_by_1pp=find(lambda r:r['mAP_delta_original_DeMo_pp']<=-1),
        first_mAP_below_original_by_5pp=find(lambda r:r['mAP_delta_original_DeMo_pp']<=-5),
        relation_argmax_epochs={name:[r['epoch'] for r in series if r['relation_argmax']==name] for name in sorted({r['relation_argmax'] for r in series})},
        band_argmax_epochs={name:[r['epoch'] for r in series if r['band_argmax']==name] for name in sorted({r['band_argmax'] for r in series})})
with table.open('x',encoding='utf-8',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows_out[0]));writer.writeheader();writer.writerows(rows_out)
record=dict(status='ACTUAL_CLOSED_RGBNT100_CPU_PER_EPOCH_TELEMETRY_RETRIEVAL_TRAJECTORY',
    observed_at=datetime.now().astimezone().isoformat(timespec='seconds'),dataset='RGBNT100',models=models,
    epoch_rows=100,training_batch_rows=13126,training_updates_already_complete=13126,
    new_neural_calls=0,new_optimizer_updates=0,input_sha256=input_hashes,original_DeMo_mAP=baseline,
    table=str(table.relative_to(p).as_posix()),table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
    limits='Read only closed normal full50 records. Epoch retrieval is benchmark GT at epoch end; detached telemetry is averaged over pre-update training batches and does not measure the exact epoch-end embedding. Full-view has seven eligible relations: outlet multiplier=7*relation mass, so the maximum mean multiplier is exact for logged full-view mean marginal; it is not final correction norm, and products of separate gate/scale/mass means are not measured correction magnitudes. Thresholds .95 gate, 1nat entropy and .9 relation mass are descriptive labels, not preregistered success criteria. Temporal co-occurrence is not causal evidence. The full-loss remainder includes final CE and all auxiliary/base/contribution losses; no individual CE or late gradients were logged. Argmax refers to the mean route marginal, not every sample. Only selected-best binary was saved; late neural states cannot be reloaded from this CSV.')
proof.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==sha for n,sha in input_hashes.items())
print(json.dumps(dict(status=record['status'],epoch_rows=100,training_batch_rows=13126,table=record['table'],new_neural_calls=0,new_optimizer_updates=0)))
