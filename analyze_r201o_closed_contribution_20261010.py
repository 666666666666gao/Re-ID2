"""Reduce the closed O MSVR logs only; no model imports or remote calls."""
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT / 'results/r201o_same_state_contribution_20261010/MSVR310'
METRICS = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')


def average(values):
    return math.fsum(values) / len(values)


def pearson(xs, ys):
    mx, my = average(xs), average(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    denominator = math.sqrt(math.fsum(x*x for x in dx) * math.fsum(y*y for y in dy))
    assert denominator > 0
    return math.fsum(x*y for x, y in zip(dx, dy)) / denominator


def main():
    receipt = PROJECT / 'results/preflight/r201o_same_state_contribution_MSVR310_actual_session_20261010.json'
    completed = json.loads(receipt.read_text(encoding='utf-8'))
    assert completed['status'] == 'ACTUAL_O_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW'
    assert completed['successful_updates'] == 2000 and completed['native_updates'] == 6
    inputs = [receipt]
    epochs, summaries = [], {}
    for variant in ('frequency_shared', 'axis_shared'):
        folder = ROOT / 'training' / ('MSVR310_r201o_' + variant + '_s42')
        files = [folder/name for name in ('result.json', 'epochs.csv', 'batch_orders.jsonl')]
        inputs.extend(files)
        result = json.loads(files[0].read_text(encoding='utf-8'))
        assert result['status'] == 'COMPLETE' and result['epochs'] == 50
        assert result['optimizer_steps'] == 1000 and result['amp_skipped_steps'] == 0
        with files[1].open(encoding='utf-8', newline='') as stream:
            official = {int(row['epoch']): row for row in csv.DictReader(stream)}
        batches = [json.loads(line) for line in files[2].read_text(encoding='utf-8').splitlines()]
        assert len(batches) == 1000 and set(official) == set(range(1, 51))
        assert all(row['optimizer_updated'] and row['contribution_zero_loss_raw'] > 0 for row in batches)
        assert all(row['contribution_reference'] == 'same_state_gallery_fixed00_indices' for row in batches)
        assert all(row['reference_requires_grad'] is False and row['primary_margin'] == .6 for row in batches)
        model = 'O_' + variant
        rows = []
        for epoch in range(1, 51):
            members = [row for row in batches if row['epoch'] == epoch]
            assert len(members) == 20
            mean = lambda key: average([row[key] for row in members])
            prediction_loss, zero_loss = mean('contribution_loss_raw'), mean('contribution_zero_loss_raw')
            row = dict(dataset='MSVR310', model=model, epoch=epoch, batches=len(members),
                       prediction_smoothL1=prediction_loss, zero_smoothL1=zero_loss,
                       prediction_over_zero_loss=prediction_loss/zero_loss,
                       batch_losses_worse_than_zero=sum(r['contribution_loss_raw'] > r['contribution_zero_loss_raw'] for r in members),
                       route_entropy_nats=mean('route_entropy_mean'),
                       route_nonindependence_nats=mean('route_nonindependence_mean'),
                       primary_margin06_violations=mean('primary_margin_violations'),
                       same_feature_margin03_violations=mean('primary03_same_feature_violations'),
                       primary_triplet_loss=mean('primary_triplet_loss'),
                       bound_clipped_fraction=average([r['final_descriptor_bound_full']['clipped_fraction'] for r in members]),
                       candidate_shift_mean=average([r['final_descriptor_bound_full']['candidate_shift_mean'] for r in members]),
                       bounded_shift_mean=average([r['final_descriptor_bound_full']['bounded_shift_mean'] for r in members]),
                       **{key: float(official[epoch][key]) for key in METRICS})
            for index, component in enumerate(('M', 'F', 'I')):
                for field, prefix in (('contribution_target_mean', 'target'), ('contribution_prediction_mean', 'prediction'),
                                      ('full_gate_mean', 'gate'), ('residual_scales', 'scale')):
                    row[prefix + '_' + component] = average([r[field][index] for r in members])
            rows.append(row)
        selected = rows[result['best']['epoch'] - 1]
        assert all(abs(selected[k] - result['best'][k]) < 1e-8 for k in METRICS)
        assert result['best']['epoch'] == 3
        summary = dict(batches=1000, epochs=50, selected_epoch=result['best']['epoch'],
                       all_batches_prediction_worse_than_zero=sum(r['contribution_loss_raw'] > r['contribution_zero_loss_raw'] for r in batches),
                       minimum_zero_loss=min(r['contribution_zero_loss_raw'] for r in batches),
                       selected=selected, first=rows[0], last=rows[-1],
                       all_epoch_mean_clip_fraction=average([r['bound_clipped_fraction'] for r in rows]),
                       all_epoch_mean_margin06_violations=average([r['primary_margin06_violations'] for r in rows]),
                       all_epoch_mean_same_feature03_violations=average([r['same_feature_margin03_violations'] for r in rows]),
                       final_minus_best={k: rows[-1][k] - selected[k] for k in METRICS},
                       batch_mean_prediction_target_pearson={
                           component: pearson([r['contribution_prediction_mean'][i] for r in batches],
                                              [r['contribution_target_mean'][i] for r in batches])
                           for i, component in enumerate(('M', 'F', 'I'))})
        summaries[model] = summary
        epochs.extend(rows)
    inputs.extend(PROJECT/name for name in ('run_r201o_same_state_contribution.py', 'axis_collaboration.py',
                                           'rgbnt201_identity_outlet.py', 'run_r201m_primary_margin06.py'))
    hashes = {f.relative_to(PROJECT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs}
    output = ROOT / 'closed_contribution_diagnostic'
    output.mkdir(exist_ok=False)
    with (output/'epochs.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(epochs[0]))
        writer.writeheader()
        writer.writerows(epochs)
    report = dict(status='ACTUAL_CLOSED_O_MSVR_CONTRIBUTION_BOUND_ROUTE_CPU_REDUCTION_COMPLETE',
                  at=datetime.now().astimezone().isoformat(timespec='seconds'), dataset='MSVR310',
                  models=summaries, source_and_input_sha256=hashes, epoch_rows=len(epochs), batch_rows=2000,
                  remote_calls=0, model_imports=0, neural_calls=0, optimizer_updates=0,
                  observations=[
                      'Both variants have larger empirical contribution regression loss than zero prediction in every one of the 1000 observed training batches.',
                      'Same-state gallery scoring alone did not establish a calibrated predictor or normal +1 gain in this completed MSVR experiment.',
                      'The score predicting contribution also controls sigmoid20 gates and receives main retrieval gradients; the source confirms this coupling.',
                      'Batch-average target/prediction vectors, gate means, route diagnostics and cap activity are descriptive training measurements; they do not isolate a causal failure.'
                  ],
                  limits=[
                      'One completed training seed and one dataset. Epochs and batches are dependent observations, not repeated independent experiments.',
                      'Pearson uses batch means, not per-sample contributions or per-query AP. Gate/target means cannot prove individual harmful fusion.',
                      'The recorded margin03 loss/violation counts use the existing margin06 features; no margin03 retraining is represented.',
                      'No fresh gradient conflict measurement, missing-modality inference, head intervention, or current RGBNT100 status query.',
                      'Existing M3 separated-head negative results remain relevant; these observations do not prove that separating heads will improve performance.'
                  ])
    (output/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], batch_rows=report['batch_rows'], epoch_rows=report['epoch_rows'],
                          selected_loss_ratios={k: v['selected']['prediction_over_zero_loss'] for k, v in summaries.items()})))


if __name__ == '__main__':
    main()
