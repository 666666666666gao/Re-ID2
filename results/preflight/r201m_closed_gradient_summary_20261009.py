"""Summarize the 16 completed train-only gradient observations; no model replay."""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

project=Path('C:/Users/gb/projects/demo_dual_axis_20261002')
pf=project/'results/preflight'
source=pf/'r201m_selected_task_gradients_actual_20261009.json'
data=json.loads(source.read_text(encoding='utf-8'))
assert data['status']=='ACTUAL_FOUR_SELECTED_M_TRAIN_ONLY_TASK_GRADIENT_DIAGNOSTICS_COMPLETE'
assert data['controls']==4 and data['batches']==data['neural_forwards']==16
assert data['optimizer_updates']==data['checkpoint_writes']==data['official_query_gallery_neural_uses']==0
assert data['paired_training_names_labels'] and not data['new49']
proof=pf/'r201m_closed_selected_gradient_summary_actual_20261009.json'
table=project/'results/r201m_selected_task_gradients_20261009/task_gradient_role_summary.csv'
assert not proof.exists() and not table.exists()
digest=hashlib.sha256(source.read_bytes()).hexdigest()
mean=lambda values:math.fsum(values)/len(values)
nonempty_mean=lambda values:mean([v for v in values if v is not None]) if any(v is not None for v in values) else None
models={};records=[]
for item in data['rows']:
    job,result=item['job'],item['result'];rows=result['rows']
    assert result['parameters_unchanged'] and result['frozen_identity_unchanged'] and len(rows)==4
    assert result['optimizer_updates']==result['checkpoint_writes']==result['official_query_gallery_neural_uses']==0
    reports={}
    for role in rows[0]['deploy_parameter_roles']:
        values=[row['deploy_parameter_roles'][role] for row in rows]
        ratios=[value['gradient_l2']['M_F_aux']/value['gradient_l2']['fused_primary']
                if value['gradient_l2']['fused_primary']>0 else None for value in values]
        reg_ratios=[value['gradient_l2']['fused_primary']/value['gradient_l2']['contribution']
                    if value['gradient_l2']['contribution']>0 else None for value in values]
        cosines=[value['fused_primary_vs_M_F_aux']['cosine'] for value in values]
        reg_cosines=[value['fused_primary_vs_contribution']['cosine'] for value in values]
        record=dict(dataset=job['dataset'],variant=job['variant'],selected_epoch=job['selected_epoch'],role=role,
            observations=4,primary_CE_gradient_L2_mean=mean([v['gradient_l2']['primary_CE'] for v in values]),
            primary_hinge_gradient_L2_mean=mean([v['gradient_l2']['primary_hinge'] for v in values]),
            fused_primary_gradient_L2_mean=mean([v['gradient_l2']['fused_primary'] for v in values]),
            M_F_aux_gradient_L2_mean=mean([v['gradient_l2']['M_F_aux'] for v in values]),
            contribution_gradient_L2_mean=mean([v['gradient_l2']['contribution'] for v in values]),
            aux_to_fused_primary_ratio_mean=nonempty_mean(ratios),
            primary_to_contribution_ratio_mean=nonempty_mean(reg_ratios),
            fused_primary_vs_aux_cosine_mean=nonempty_mean(cosines),
            primary_aux_negative_cosine_observations=sum(v is not None and v<0 for v in cosines),
            primary_aux_defined_cosine_observations=sum(v is not None for v in cosines),
            fused_primary_vs_contribution_cosine_mean=nonempty_mean(reg_cosines))
        assert all(math.isfinite(v) for v in record.values() if isinstance(v,float))
        records.append(record)
        reports[role]=dict(summary=record,aux_to_primary_ratios=ratios,primary_aux_cosines=cosines,
            primary_to_contribution_ratios=reg_ratios,primary_contribution_cosines=reg_cosines)
    models[job['name']]=dict(roles=reports,normalized_shift_mean=mean([r['normalized_descriptor_shift_l2_mean'] for r in rows]),
        normalized_angle_degrees_mean=mean([r['normalized_descriptor_angle_degrees_mean'] for r in rows]),
        scale_primary_gradients=[r['residual_scale_gradients']['fused_primary'] for r in rows],
        negative_primary_scale_gradient_components=sum(v<0 for r in rows for v in r['residual_scale_gradients']['fused_primary']))
with table.open('x',encoding='utf-8',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
record=dict(status='ACTUAL_16_SELECTED_TRAIN_ONLY_GRADIENT_ROLE_READOUT_CPU_COMPLETE',
    observed_at=datetime.now().astimezone().isoformat(timespec='seconds'),models=models,
    source_sha256=digest,source=str(source.relative_to(project).as_posix()),table=str(table.relative_to(project).as_posix()),
    table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),role_rows=len(records),new_neural_calls=0,new_optimizer_updates=0,
    limits='Weighted actual CE.25+hinge.6 and existing M/F auxiliary.1 at each selected best, four same-order training batches/control. Not late gradients, entire two-forward step, global conflict, causal failure, or a new retrieval improvement. None cosine means a disconnected/zero gradient, not zero-angle agreement. Negative scale gradients locally encourage increases under gradient descent; Adam history and later gradients were not measured.')
proof.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=record['status'],controls=len(models),role_rows=len(records),new_neural_calls=0,new_optimizer_updates=0)))
