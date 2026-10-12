"""Saved AD/AG early gradient and selected-route diagnostics, without new NN."""
import csv
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / 'results/rgbnt100_AD_AG_route_gradient_CPU_20261012'


def main():
    assert not OUTPUT.exists()
    rows, routes, inputs, orders = [], [], {}, {}
    for tag, prefix in (('r201ad_RGB_precision_matched_20261012', 'r201ad_hard'),
                        ('r201ag_RGB_coordinate_metric_20261012', 'r201ag')):
        for variant in ('frequency_shared', 'axis_shared'):
            name = 'RGBNT100_' + prefix + '_' + variant + '_s42'
            root = PROJECT / 'results' / tag / 'RGBNT100'
            smoke_file = root / 'native' / (name + '_smoke') / 'smoke.json'
            trained_file = root / 'training' / name / 'result.json'
            smoke, trained = [json.loads(f.read_text()) for f in (smoke_file, trained_file)]
            assert smoke['status'] == 'SMOKE_PASS' and smoke['steps'] == smoke['attempts'] == 3
            assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50
            assert trained['steps'] == trained['optimizer_steps'] == 6563
            orders[name] = [(d['names'], d['partial_set']) for d in smoke['details']]
            for step, detail in enumerate(smoke['details'], 1):
                for expert in ('M', 'F'):
                    gradient = detail['primary_vs_aux_' + expert]
                    assert gradient['parameter'] == expert + '.output.1.weight'
                    assert gradient['auxiliary_weight'] == .001
                    assert gradient['primary_norm'] > 0 and gradient['weighted_auxiliary_norm'] > 0
                    assert -1 <= gradient['cosine'] <= 1
                    rows.append(dict(model=name, step=step, expert=expert, **gradient,
                        auxiliary_primary_norm_ratio=gradient['weighted_auxiliary_norm'] / gradient['primary_norm']))
            runtime = trained['runtime']
            route = runtime['mean_route']
            assert len(route) == 7 and all(len(row) == 3 for row in route)
            assert abs(sum(map(sum, route)) - 1) < 1e-5
            routes.append(dict(model=name, selected_epoch=trained['best']['epoch'],
                mAP=trained['full_metrics']['mAP'], Rank_1=trained['full_metrics']['Rank-1'],
                mean_route=route, band_mass=[sum(row[b] for row in route) for b in range(3)],
                relation_mass=list(map(sum, route)),
                mean_route_entropy=runtime['mean_route_entropy'],
                mean_per_sample_joint_minus_marginal_product_L1=runtime['mean_joint_minus_marginal_product_L1'],
                mean_outer_gates=runtime['mean_outer_gates']))
            for file in (smoke_file, trained_file):
                inputs[file.relative_to(PROJECT).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    assert all(order == next(iter(orders.values())) for order in orders.values())
    OUTPUT.mkdir()
    for name, values in (('native_gradient_comparisons.csv', rows), ('selected_route_diagnostics.csv', routes)):
        with (OUTPUT / name).open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    result = dict(status='ACTUAL_AD_AG_RGB_ROUTE_EARLY_GRADIENT_CPU_DIAGNOSTICS_COMPLETE',
        models=4, matched_native_batches_per_model=3, gradient_comparisons=24,
        native_gradient_rows=rows, selected_routes=routes, sources_sha256=inputs,
        new_neural_calls=0, new_optimizer_updates=0, new_weights=0,
        limits='Reuses actual native3 early-step gradients at one output weight per expert, not training-wide '
               'M-versus-F gradient conflict or shared-backbone conflict. Norm ratio is not gradient direction '
               'or harmfulness. Route L1 is the saved mean of per-sample non-independence, not computed from '
               'mean-route alone; it differs from training KL telemetry. Best epochs differ, so selected-route '
               'contrast is not an epoch-matched causal analysis. Greater non-independence does not prove '
               'retrieval utility. No NN rerun, clipping intervention, gradient arbitration or seed selection.')
    (OUTPUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'models', 'gradient_comparisons', 'new_neural_calls')}))


if __name__ == '__main__':
    main()
