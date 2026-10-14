"""Summarize the completed BC reference gate's saved updates, without new NN calls."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from analyze_identity_coordinate_three_normal import METRICS, table

PROJECT = Path(__file__).resolve().parent
TAG = 'r201bc_RGB_reference_interaction_20261015'


def main():
    intake_path = PROJECT / 'results/preflight' / (TAG + '_complete_intake.json')
    intake = json.loads(intake_path.read_text(encoding='utf-8'))
    assert intake['status'] == 'ACTUAL_BC_TWO_FULL50_NORMAL_GT_EIGHT_STATES_D_ARCHIVED_COMPLETE'
    assert intake['formal_updates'] == 13126 and intake['native_updates'] == 6
    root = PROJECT / 'results' / TAG / 'RGBNT100'
    output = root / 'reference_gate_CPU'
    assert not output.exists()
    curves = []
    summaries = []
    inputs = [intake_path]
    for variant in ('frequency_shared', 'axis_shared'):
        folder = root / 'training' / ('RGBNT100_r201bc_' + variant + '_s42')
        batch_path = folder / 'batch_orders.jsonl'
        epoch_path = folder / 'epochs.csv'
        result_path = folder / 'result.json'
        inputs.extend([batch_path, epoch_path, result_path])
        result = json.loads(result_path.read_text(encoding='utf-8'))
        assert result['status'] == 'COMPLETE' and result['steps'] == result['optimizer_steps'] == 6563
        assert result['epochs'] == 50 and result['amp_skipped_steps'] == 0
        batches = [json.loads(line) for line in batch_path.read_text(encoding='utf-8').splitlines()]
        assert len(batches) == 6563 and all(b['optimizer_updated'] for b in batches)
        records = [b['identity_reference_interaction'] for b in batches]
        assert all(r['parameters_added'] == 32768 and r['base_reference_stop_gradient'] and r['final_descriptor_dim'] == 5120 for r in records)
        assert all(math.isfinite(r[k]) and r[k] >= 0 for r in records
                   for k in ('weight_norm_before', 'weight_norm_after', 'unscaled_task_gradient_norm'))
        assert records[0]['weight_norm_before'] == 0
        with epoch_path.open(encoding='utf-8', newline='') as stream:
            epochs = list(csv.DictReader(stream))
        assert [int(e['epoch']) for e in epochs] == list(range(1, 51))
        selected = max(epochs, key=lambda e: float(e['mAP']))
        assert int(selected['epoch']) == result['best']['epoch']
        for epoch in epochs:
            number = int(epoch['epoch'])
            members = [b for b in batches if b['epoch'] == number]
            gates = [b['identity_reference_interaction'] for b in members]
            gradients = [r['unscaled_task_gradient_norm'] for r in gates]
            curves.append(dict(variant=variant,epoch=number,updates=len(members),
                selected_checkpoint=number == result['best']['epoch'],
                weight_norm_first=gates[0]['weight_norm_before'],weight_norm_last=gates[-1]['weight_norm_after'],
                objective_gradient_mean=math.fsum(gradients)/len(gradients),
                objective_gradient_median=statistics.median(gradients),
                objective_gradient_min=min(gradients),objective_gradient_max=max(gradients),
                zero_gradient_updates=sum(g == 0 for g in gradients),
                mean_loss=math.fsum(b['loss'] for b in members)/len(members),
                **{k:float(epoch[k]) for k in METRICS}))
        summaries.append(dict(variant=variant,updates=len(batches),selected_epoch=result['best']['epoch'],
            selected_six={k:float(selected[k]) for k in METRICS},last_six={k:float(epochs[-1][k]) for k in METRICS},
            initial_weight_norm=records[0]['weight_norm_before'],final_weight_norm=records[-1]['weight_norm_after'],
            maximum_weight_norm=max(r['weight_norm_after'] for r in records),
            zero_gradient_updates=sum(r['unscaled_task_gradient_norm'] == 0 for r in records),
            maximum_between_update_norm_discontinuity=max(abs(a['weight_norm_after']-b['weight_norm_before'])
                                                         for a,b in zip(records, records[1:]))))
    output.mkdir()
    table(output/'epoch_curves.csv', curves)
    record = dict(status='ACTUAL_BC_REFERENCE_GATE_SAVED_UPDATE_CPU_COMPLETE',formal_updates=13126,
        epoch_rows=len(curves),summaries=summaries,
        input_sha256={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_neural_calls=0,new_optimizer_updates=0,new_weights=0,
        limits='Recorded unscaled gradient norms cover the parent training objective, not a separately measured retrieval-only gradient. A changing gate weight and nonzero gradient do not prove useful modulation, calibrated contributions or a causal gain. Saved records do not contain per-query gate values, saturation or spatial evidence. Norm continuity and curves are descriptive; no checkpoint, seed, gate strength or hyperparameter is selected by this analysis.')
    (output/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=record['status'],summaries=summaries)),flush=True)


if __name__ == '__main__':
    main()
