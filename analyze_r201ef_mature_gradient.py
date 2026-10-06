"""Reduce actual mature, train-only parameter gradients without neural calls."""
import csv
from datetime import datetime
import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parent


def main():
    actual = json.loads((PROJECT / 'results/preflight/r201ef_mature_gradient_actual_20261006.json').read_text(encoding='utf-8'))
    assert actual['status'] == 'ACTUAL_E_F_MATURE_TRAIN_ONLY_DEPLOY_GRADIENT_PAIR_COMPLETE'
    assert actual['neural_forwards'] == 16 and actual['optimizer_updates'] == actual['checkpoint_writes'] == 0
    rows, summaries, scale_rows = [], {}, []
    for model, observed in actual['rows'].items():
        assert observed['batches'] == 4 and observed['parameters_unchanged'] and observed['frozen_identity_unchanged']
        summaries[model] = dict(selected_epoch=observed['selected_epoch'], temperature=observed['temperature'],
            observed_training_identities=observed['observed_training_identities'],
            learned_scales=observed['learned_residual_scales_at_load'], roles={})
        roles = list(observed['rows'][0]['deploy_parameter_roles'])
        for role in roles:
            values = [row['deploy_parameter_roles'][role] for row in observed['rows']]
            means = {key: math.fsum(row['gradient_l2'][key] for row in values) / 4 for key in values[0]['gradient_l2']}
            ce, full = means['primary_CE'], means['full_total']
            comparisons = {}
            for key in ('AP_vs_CE', 'full_vs_modality_aux', 'full_vs_frequency_aux', 'full_vs_partial'):
                defined = [row[key]['cosine'] for row in values if row[key]['cosine'] is not None]
                comparisons[key] = dict(defined=len(defined), negative=sum(value < 0 for value in defined),
                    mean=None if not defined else math.fsum(defined) / len(defined))
            summaries[model]['roles'][role] = dict(mean_component_l2=means,
                ratio_of_mean_AP_to_weighted_CE=None if ce == 0 else means['primary_AP'] / ce,
                ratio_of_mean_partial_to_full=None if full == 0 else means['partial_total'] / full,
                comparisons=comparisons)
            for batch, row in enumerate(values):
                rows.append(dict(model=model, selected_epoch=observed['selected_epoch'], temperature=observed['temperature'],
                    batch=observed['rows'][batch]['batch'], role=role, **row['gradient_l2'],
                    **{key + '_cosine': row[key]['cosine'] for key in comparisons},
                    **{key + '_unused_tensors': value for key, value in row['unused_parameter_tensors'].items()}))
        for row in observed['rows']:
            for component, vector in row['residual_scale_gradients'].items():
                scale_rows.append(dict(model=model, batch=row['batch'], component=component,
                    M=None if vector is None else vector[0], F=None if vector is None else vector[1],
                    I=None if vector is None else vector[2], connected=vector is not None))
    output = PROJECT / 'results/r201ef_mature_gradient_20261006'
    output.mkdir(exist_ok=False)
    for name, data in (('parameter_roles_by_batch', rows), ('residual_scale_gradients_by_batch', scale_rows)):
        with (output / (name + '.csv')).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    result = dict(status='ACTUAL_MATURE_TRAIN_PARAMETER_GRADIENT_CPU_REDUCTION_COMPLETE',
        completed_at=datetime.now().isoformat(timespec='seconds'), models=summaries, role_batch_rows=len(rows),
        residual_component_batch_rows=len(scale_rows), new_neural_calls=0, new_optimizer_updates=0,
        limits='Four paired train-only batches per differently matured selected E/F checkpoint, fixed scaled derivatives, not Adam updates. '
               'Ratios divide mean component L2 norms; not mean per-batch ratios or whole-model/whole-training gradient conclusions. '
               'Zero gradients and undefined cosines reflect actual disconnected paths. No query/gallery training, retrieval gain or causal-temperature claim.')
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'completed_at', 'role_batch_rows', 'new_neural_calls')}))


if __name__ == '__main__':
    main()
