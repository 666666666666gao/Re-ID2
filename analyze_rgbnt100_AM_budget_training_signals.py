"""Reduce completed AM training telemetry without new neural inference."""
import csv
import json
import math
from pathlib import Path

P = Path(__file__).resolve().parent
TAG = 'r201am_RGB_correction_budget_20261013'
METRICS = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')


def main():
    rows, summaries = [], []
    for variant in ('frequency_shared', 'axis_shared'):
        root = P / 'results' / TAG / 'RGBNT100/training' / ('RGBNT100_r201am_' + variant + '_s42')
        result = json.loads((root / 'result.json').read_text(encoding='utf-8'))
        assert result['status'] == 'COMPLETE' and result['epochs'] == 50
        epochs = list(csv.DictReader((root / 'epochs.csv').open(encoding='utf-8', newline='')))
        assert [int(r['epoch']) for r in epochs] == list(range(1, 51))
        batches = {epoch: [] for epoch in range(1, 51)}
        count = zero = beats_zero = 0
        with (root / 'batch_orders.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                r = json.loads(line)
                count += 1
                assert r['step'] == count and r['optimizer_updated']
                assert r['partial_loss_effective'] == 0 and r['correction_budget'] == .20
                assert r['freeze_identity_encoder'] and r['routing_temperature'] == 2.
                assert r['primary_margin'] == .6 and r['sampling_batch'] == 64
                zero += r['primary_triplet_loss'] == 0
                beats_zero += r['contribution_loss_raw'] < r['contribution_zero_loss_raw']
                batches[r['epoch']].append(r)
        assert count == result['optimizer_steps'] == 6563
        series = []
        for ep in epochs:
            epoch = int(ep['epoch'])
            batch = batches[epoch]
            mean = lambda key: math.fsum(r[key] for r in batch) / len(batch)
            bound = lambda key: math.fsum(r['final_descriptor_bound_full'][key] for r in batch) / len(batch)
            assert abs(mean('loss') - float(ep['loss'])) < 1e-8
            row = dict(model='AM_' + variant, epoch=epoch, batches=len(batch), loss=mean('loss'),
                primary_triplet_loss=mean('primary_triplet_loss'), main_CE_effective=mean('main_CE_effective'),
                primary_margin_violation_fraction=mean('primary_margin_violations') / 64,
                zero_primary_triplet_batch_fraction=sum(r['primary_triplet_loss'] == 0 for r in batch) / len(batch),
                contribution_beats_zero_batch_fraction=sum(r['contribution_loss_raw'] < r['contribution_zero_loss_raw'] for r in batch) / len(batch),
                full_training_route_entropy_mean=mean('route_entropy_mean'),
                full_training_route_joint_product_KL_mean=mean('route_nonindependence_mean'),
                bound_clipped_fraction=bound('clipped_fraction'), candidate_shift_mean=bound('candidate_shift_mean'),
                bounded_shift_mean=bound('bounded_shift_mean'),
                bounded_shift_max=max(r['final_descriptor_bound_full']['bounded_shift_max'] for r in batch),
                **{key: float(ep[key]) for key in METRICS})
            assert row['bounded_shift_max'] <= .20 + 1e-6
            rows.append(row)
            series.append(row)
        selected = result['best']['epoch']
        assert max(epochs, key=lambda r: float(r['mAP']))['epoch'] == str(selected)
        summaries.append(dict(model='AM_' + variant, selected_epoch=selected, actual_batches=count,
            first=series[0], selected=series[selected-1], last=series[-1],
            zero_primary_triplet_batches=zero, contribution_beats_zero_batches=beats_zero))
    out = P / 'results/rgbnt100_AM_budget_training_signals_CPU_20261013'
    out.mkdir(exist_ok=False)
    with (out / 'full_training_signals.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    proof = dict(status='ACTUAL_AM_FULL_TRAINING_TELEMETRY_CPU_COMPLETE', models=2,
        epoch_rows=len(rows), batch_rows=13126, summaries=summaries, new_neural_calls=0, new_optimizer_updates=0,
        trace_scope='First full-view primary-loss callback before partial forward and Adam; bound report from the actual full11 descriptor.',
        limits='Training batch telemetry, not per-epoch full official Q/G routes. Loss ratios are not gradient ratios. Contribution comparison uses the stopped-gradient batch target, not calibrated AP utility. Single seed42; early-best/last differences do not establish a unique causal failure. Matched AL/AM changes only the final correction budget .10 to .20.')
    (out / 'result.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=proof['status'], models=2, epoch_rows=len(rows), batch_rows=13126)))


if __name__ == '__main__':
    main()
