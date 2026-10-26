"""Full normal RGBNT100 readout of the completed B128-reference AV pair, no NN."""
import csv
import json
import math
from pathlib import Path

from analyze_identity_coordinate_three_normal import METRICS, load, query_rows, six, table

P = Path(__file__).resolve().parent
TAG = 'rgbnt100_B128_reference_experts_normfix_20261026'
META = ('query_index', 'name', 'identity', 'camera', 'scene', 'valid', 'relevant_gallery', 'kept_gallery')


def compare(candidate, reference, reads):
    new, ns = reads[candidate]
    old, os = reads[reference]
    assert len(new) == len(old) == 1715
    assert all(tuple(a[k] for k in META) == tuple(b[k] for k in META) for a, b in zip(new, old))
    delta = {k: ns[k] - os[k] for k in METRICS}
    rescued = sum(int(a['Rank-1']) > int(b['Rank-1']) for a, b in zip(new, old))
    harmed = sum(int(a['Rank-1']) < int(b['Rank-1']) for a, b in zip(new, old))
    assert abs(delta['Rank-1'] - 100 * (rescued - harmed) / 1715) < 1e-8
    result = dict(model=candidate, reference=reference, **delta, rank1_rescued=rescued, rank1_harmed=harmed,
        AP_improved=sum(float(a['AP']) - float(b['AP']) > 1e-12 for a, b in zip(new, old)),
        AP_declined=sum(float(a['AP']) - float(b['AP']) < -1e-12 for a, b in zip(new, old)))
    paired = [dict(model=candidate, reference=reference, **{k: a[k] for k in META[:5]},
        delta_AP_pp=100 * (float(a['AP']) - float(b['AP'])),
        delta_INP_pp=100 * (float(a['INP']) - float(b['INP'])),
        delta_rank1=int(a['Rank-1']) - int(b['Rank-1'])) for a, b in zip(new, old)]
    return result, paired


def main():
    root = P / 'results' / TAG / 'experiments'
    controller = load(root / 'controller_result.json')
    assert controller['status'] == 'COMPLETE' and controller['controls'] == 2
    assert controller['native_updates'] == 6 and controller['successful_updates'] == 13126
    assert controller['additional_epochs'] == 100 and controller['states_cases'] == 8
    assert controller['paired_sampling_exact'] and controller['reference00_full_parity']
    assert controller['physical_gpus'] == [2, 3] and controller['maximum_NN'] == 2
    folders = dict(
        original_DeMo=P / 'results/full_official_baselines_20261004/training/RGBNT100_demo_s42',
        B128_head8_teacher=P / 'results/rgbnt100_paper_batch_baselines_20261016/experiments/training/RGBNT100_B128_atmoe8_demo_s42',
        AV_ordinary=P / 'results/r201av_RGB_frequency_value_decay_20261014/RGBNT100/training/RGBNT100_r201av_frequency_shared_s42',
        AV_dual=P / 'results/r201av_RGB_frequency_value_decay_20261014/RGBNT100/training/RGBNT100_r201av_axis_shared_s42',
        new_ordinary=root / 'training/RGBNT100_B128ref_AV_frequency_shared_s42',
        new_dual=root / 'training/RGBNT100_B128ref_AV_axis_shared_s42')
    normal, groups, cmc, curves, reads, costs, orders = [], [], [], [], {}, {}, {}
    for model, folder in folders.items():
        run = load(folder / 'result.json')
        assert run['status'] == 'COMPLETE' and run['epochs'] == 50 and run['descriptor_dim'] == 5120
        assert (run['train_records'], run['query_records'], run['gallery_records']) == (8675, 1715, 8575)
        assert run['training_heldout_identities'] == 0 and run['amp_skipped_steps'] == 0
        assert run['steps'] == run['optimizer_steps'] > 0
        assert run['training_coverage'] == dict(eligible=8675, visited=8675, unvisited=[])
        rows, summary = query_rows(folder, 'RGBNT100')
        assert all(abs(summary[k] - run['full_metrics'][k]) < 1e-8 for k in METRICS)
        epochs = list(csv.DictReader((folder / 'epochs.csv').open(encoding='utf-8', newline='')))
        assert [int(v['epoch']) for v in epochs] == list(range(1, 51))
        selected = max(epochs, key=lambda v: float(v['mAP']))
        assert int(selected['epoch']) == run['best']['epoch']
        assert all(abs(float(selected[k]) - summary[k]) < 1e-8 for k in METRICS)
        normal.append(dict(model=model, selected_epoch=run['best']['epoch'], **{k: summary[k] for k in METRICS}))
        groups.extend(dict(model=model, grouping=axis, **group) for axis, values in summary['groups'].items() for group in values)
        cmc.extend(dict(model=model, rank=rank, accuracy=value) for rank, value in enumerate(summary['CMC_1_to_50'], 1))
        curves.extend(dict(model=model, epoch=int(v['epoch']), loss=float(v['loss']), steps=int(v['steps']),
            optimizer_steps=int(v['optimizer_steps']), amp_skipped_steps=int(v['amp_skipped_steps']),
            **{k: float(v[k]) for k in METRICS}) for v in epochs)
        reads[model] = rows, summary
        costs[model] = dict(parameters=run['parameters'], trainable_parameters=run['trainable_parameters'],
            descriptor_dim=5120, stage_epochs=50, pipeline_epochs=100 if model.startswith(('AV_', 'new_')) else 50,
            attempts=run['steps'], updates=run['optimizer_steps'], skips=run['amp_skipped_steps'],
            peak_memory=run['peak_memory'], full_QG_runtime=run['runtime'], selected_epoch=run['best']['epoch'],
            last={k: float(epochs[-1][k]) for k in METRICS})
        if model.startswith('new_'):
            assert (run['parameters'], run['trainable_parameters']) == (99585158, 1887366)
            assert run['steps'] == 6563 and load(folder / 'normal_cpu_audit.json')['status'] == 'PASS'
            batches = [json.loads(v) for v in (folder / 'batch_orders.jsonl').read_text().splitlines()]
            assert len(batches) == 6563 and all(v['optimizer_updated'] and v['loss'] == v['full_loss'] and v['partial_loss_effective'] == 0 for v in batches)
            assert all(v['sampling_batch'] == 64 and v['sampling_identities'] == v['sampling_instances'] == 8 for v in batches)
            orders[model] = [(v['epoch'], v['step'], v['names'], v['partial_set']) for v in batches]
            for epoch in epochs:
                members = [v for v in batches if v['epoch'] == int(epoch['epoch'])]
                assert members and abs(math.fsum(v['loss'] for v in members) / len(members) - float(epoch['loss'])) < 1e-8
    assert orders['new_ordinary'] == orders['new_dual']
    assert costs['new_ordinary']['parameters'] == costs['new_dual']['parameters']
    comparisons, paired = [], []
    for candidate, reference in [('new_dual', 'new_ordinary'), ('new_dual', 'B128_head8_teacher'),
        ('new_ordinary', 'B128_head8_teacher'), ('new_dual', 'original_DeMo'), ('new_ordinary', 'original_DeMo'),
        ('new_dual', 'AV_dual'), ('new_ordinary', 'AV_ordinary')]:
        result, changes = compare(candidate, reference, reads)
        comparisons.append(result)
        paired.extend(changes)
    state_metrics, state_comparisons, state_cmc = [], [], []
    for variant, model in [('frequency_shared', 'new_ordinary'), ('axis_shared', 'new_dual')]:
        folder = root / 'states' / ('RGBNT100_B128ref_AV_' + variant + '_s42')
        record = load(folder / 'result.json')
        assert record['status'] == 'COMPLETE' and record['state_cases'] == 4
        assert record['full11_features_distance_equal_selected_normal'] and record['state_tensor_versions_unchanged']
        for state in ('00', '10', '01', '11'):
            rows = list(csv.DictReader((folder / ('state_' + state + '.csv')).open(encoding='utf-8', newline='')))
            assert len(rows) == 1715 and all(v['valid'] == 'True' for v in rows)
            assert [int(v['query_index']) for v in rows] == list(range(1715))
            summary = record['measurements'][state]
            assert all(abs(six(rows)[k] - summary[k]) < 1e-8 for k in METRICS)
            assert all(tuple(a[k] for k in META) == tuple(b[k] for k in META) for a, b in zip(rows, reads[model][0]))
            reads[model + '/' + state] = rows, summary
            state_metrics.append(dict(model=model, state=state, **{k: summary[k] for k in METRICS}))
            for rank in range(1, 51):
                value = 100 * sum(int(v['first_match']) <= rank for v in rows) / 1715
                assert abs(value - summary['CMC_1_to_50'][rank - 1]) < 1e-8
                state_cmc.append(dict(model=model, state=state, rank=rank, accuracy=value))
        assert reads[model + '/00'][0] == reads['B128_head8_teacher'][0]
        assert reads[model + '/11'][0] == reads[model][0]
        for state in ('00', '10', '01'):
            value, _ = compare(model + '/11', model + '/' + state, reads)
            state_comparisons.append(value)
    paper = {model: dict(mAP=summary['mAP'], Rank_1=summary['Rank-1'],
        paper_mAP=86.2, paper_Rank_1=97.6, target_mAP=87.2, target_Rank_1=98.6,
        delta_mAP_vs_paper=summary['mAP'] - 86.2, delta_Rank_1_vs_paper=summary['Rank-1'] - 97.6,
        met=summary['mAP'] >= 87.2 and summary['Rank-1'] >= 98.6,
        role='new_dual_axis' if model == 'new_dual' else 'ordinary_control' if model == 'new_ordinary' else 'reference')
        for model, (_, summary) in reads.items() if '/' not in model}
    out = root / 'normal_analysis'
    out.mkdir(exist_ok=False)
    for name, rows in [('six_metrics', normal), ('group_metrics', groups), ('cmc', cmc), ('training_curves', curves),
        ('comparisons', comparisons), ('paired_query_changes', paired), ('state_metrics', state_metrics),
        ('state_comparisons', state_comparisons), ('state_cmc', state_cmc)]:
        table(out / (name + '.csv'), rows)
    result = dict(status='ACTUAL_RGBNT100_B128_REFERENCE_EXPERT_PAIR_FULL_NORMAL_CPU_READOUT', models=6,
        new_full50_models=2, new_formal_updates=13126, native_updates_separate=6, state_cases=8,
        paired_order_exact=True, official_full_scope=dict(train=8675, classes=50, query=1715, gallery=8575),
        training=costs, paper_targets=paper, comparisons=comparisons, state_comparisons=state_comparisons,
        new_NN=0, optimizer_updates=0, missing_evaluated=False, new49=False,
        limits='CSV reductions of installed-GT-audited exports, not a new neural or independent raw-GT audit. '
        'New ordinary/dual pair share the selected B128/head8 teacher and its train-only bank, seed42, AV architecture/loss and extra50. '
        'Historical AV comparison changes teacher and its deterministic neighbor arrangement together. '
        'The isolated reference fix removes duplicate00 normalization only; no scoring/gate relaxation or model change. '
        'Paper acceptance is87.2/98.6, not a delta from own reproduction; baseline credit and expert increment are separate. '
        'All six metrics per row are one earliest normal-mAP-best checkpoint; benchmark participated in epoch/configuration selection. '
        'One seed, complete official train/Q/G, no missing/49 or seed search; frozen states are not retrained ablations.')
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    print(json.dumps(dict(status=result['status'], paper_targets=paper, comparisons=comparisons), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
