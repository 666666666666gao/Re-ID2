"""Completed B128 identity baselines: full metrics, curves, paired changes and paper targets."""
import csv
import json
import math
from pathlib import Path

from analyze_identity_coordinate_three_normal import METRICS, load, query_rows, table

P = Path(__file__).resolve().parent
TAG = 'rgbnt100_paper_batch_baselines_20261016'


def main():
    root = P/'results'/TAG/'experiments'
    controller = load(root/'controller_result.json')
    assert controller['status'] == 'COMPLETE' and controller['cases'] == 2
    assert controller['formal_attempts'] == 6270 and controller['formal_epochs'] == 100
    assert controller['paired_sampling_exact'] and controller['native_updates'] == 6
    runs = dict(
        original_DeMo=P/'results/full_official_baselines_20261004/training/RGBNT100_demo_s42',
        AV_axis=P/'results/r201av_RGB_frequency_value_decay_20261014/RGBNT100/training/RGBNT100_r201av_axis_shared_s42',
        B64_head32=P/'results/vehicle_atmoe_head_dynamic_amp_20261016/RGBNT100/training/RGBNT100_atmoe32_demo_s42')
    runs.update({'B128_head'+str(h):root/'training'/('RGBNT100_B128_atmoe'+str(h)+'_demo_s42') for h in (8,32)})
    metrics, groups, cmc, curves, costs, reads, orders = [], [], [], [], {}, {}, {}
    for label, folder in runs.items():
        run = load(folder/'result.json')
        assert run['status'] == 'COMPLETE' and run['epochs'] == 50 and run['descriptor_dim'] == 5120
        assert (run['train_records'],run['query_records'],run['gallery_records']) == (8675,1715,8575)
        assert run['training_heldout_identities'] == 0 and run['training_coverage'] == dict(eligible=8675,visited=8675,unvisited=[])
        assert 0 < run['optimizer_steps'] <= run['steps'] and run['amp_skipped_steps'] == run['steps']-run['optimizer_steps']
        rows, summary = query_rows(folder, 'RGBNT100')
        assert all(abs(summary[key]-run['full_metrics'][key]) < 1e-8 for key in METRICS)
        epochs = list(csv.DictReader((folder/'epochs.csv').open(encoding='utf-8',newline='')))
        assert [int(row['epoch']) for row in epochs] == list(range(1,51))
        best = max(epochs,key=lambda row:float(row['mAP']))
        assert int(best['epoch']) == run['best']['epoch']
        assert all(abs(float(best[key])-summary[key]) < 1e-8 for key in METRICS)
        metrics.append(dict(model=label,selected_epoch=run['best']['epoch'],**{key:summary[key] for key in METRICS}))
        groups.extend(dict(model=label,grouping=axis,**value) for axis,values in summary['groups'].items() for value in values)
        cmc.extend(dict(model=label,rank=i,accuracy=value) for i,value in enumerate(summary['CMC_1_to_50'],1))
        reads[label] = rows, summary
        costs[label] = dict(parameters=run['parameters'],trainable_parameters=run['trainable_parameters'],
            descriptor_dim=5120,stage_epochs=50,pipeline_epochs=100 if label=='AV_axis' else 50,
            attempts=run['steps'],updates=run['optimizer_steps'],skips=run['amp_skipped_steps'],
            peak_memory=run['peak_memory'],full_QG_runtime=run['runtime'],
            selected_epoch=run['best']['epoch'],last={key:float(epochs[-1][key]) for key in METRICS})
        curves.extend(dict(model=label,**row) for row in epochs)
        if not label.startswith('B128_'):
            continue
        head = int(label.removeprefix('B128_head'))
        assert run['arguments']['atmoe_heads'] == head and run['arguments']['training_batch'] == 128 and run['steps'] == 3135
        assert run['parameters'] == {8:97952256,32:97780224}[head]
        checked = load(folder/'normal_cpu_audit.json')
        assert checked['status'] == 'PASS' and checked['batch_size'] == 128 and checked['P'] == 8 and checked['K'] == 16
        batches = [json.loads(line) for line in (folder/'batch_orders.jsonl').read_text().splitlines()]
        assert len(batches) == 3135 and all(len(row['names']) == 128 and isinstance(row['optimizer_updated'],bool) for row in batches)
        assert sum(row['optimizer_updated'] for row in batches) == run['optimizer_steps']
        orders[label] = [(row['epoch'],row['step'],row['names']) for row in batches]
        attempted = updated = 0
        for epoch in epochs:
            members = [row for row in batches if row['epoch'] == int(epoch['epoch'])]
            assert members and abs(math.fsum(row['loss'] for row in members)/len(members)-float(epoch['loss'])) < 1e-8
            attempted += len(members)
            updated += sum(row['optimizer_updated'] for row in members)
            assert (int(epoch['steps']),int(epoch['optimizer_steps']),int(epoch['amp_skipped_steps'])) == (attempted,updated,attempted-updated)
    assert orders['B128_head8'] == orders['B128_head32']
    assert sum(costs[label]['updates'] for label in orders) == controller['formal_optimizer_updates']
    assert sum(costs[label]['skips'] for label in orders) == controller['formal_amp_skips']
    comparisons, paired = [], []
    for candidate, reference in [('B128_head8','original_DeMo'),('B128_head32','B64_head32'),
        ('B128_head32','B128_head8'),('B128_head8','AV_axis'),('B128_head32','AV_axis'),('B128_head32','original_DeMo')]:
        new, new_summary = reads[candidate]
        old, old_summary = reads[reference]
        fields = ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')
        assert all(tuple(a[key] for key in fields) == tuple(b[key] for key in fields) for a,b in zip(new,old))
        rescued = sum(int(a['Rank-1']) > int(b['Rank-1']) for a,b in zip(new,old))
        harmed = sum(int(a['Rank-1']) < int(b['Rank-1']) for a,b in zip(new,old))
        delta = {key:new_summary[key]-old_summary[key] for key in METRICS}
        assert abs(delta['Rank-1']-100*(rescued-harmed)/1715) < 1e-8
        comparisons.append(dict(model=candidate,reference=reference,**delta,rank1_rescued=rescued,rank1_harmed=harmed,
            AP_improved=sum(float(a['AP'])-float(b['AP']) > 1e-12 for a,b in zip(new,old)),
            AP_declined=sum(float(a['AP'])-float(b['AP']) < -1e-12 for a,b in zip(new,old))))
        paired.extend(dict(model=candidate,reference=reference,**{key:a[key] for key in fields[:5]},
            delta_AP_pp=100*(float(a['AP'])-float(b['AP'])),delta_INP_pp=100*(float(a['INP'])-float(b['INP'])),
            delta_rank1=int(a['Rank-1'])-int(b['Rank-1'])) for a,b in zip(new,old))
    paper = {label:dict(mAP=summary['mAP'],Rank_1=summary['Rank-1'],target_mAP=87.2,target_Rank_1=98.6,
        met=summary['mAP']>=87.2 and summary['Rank-1']>=98.6,role='identity_baseline' if label.startswith('B128_') else 'historical_reference')
        for label,(_,summary) in reads.items()}
    # Select a whole run, never combine metrics or checkpoints. This is a baseline selection only.
    choice = max(orders,key=lambda label:(paper[label]['met'],paper[label]['mAP'],-int(label.removeprefix('B128_head'))))
    out = root/'normal_analysis'
    out.mkdir(exist_ok=False)
    for name, rows in [('six_metrics',metrics),('group_metrics',groups),('cmc',cmc),('training_curves',curves),
                       ('comparisons',comparisons),('paired_query_changes',paired)]:
        table(out/(name+'.csv'),rows)
    record = dict(status='ACTUAL_RGBNT100_B128_PAIR_FULL_NORMAL_CPU_READOUT',models=5,new_full50_models=2,
        formal_attempts=6270,formal_updates=controller['formal_optimizer_updates'],amp_skips=controller['formal_amp_skips'],
        training=costs,paper_targets=paper,selected_baseline_configuration=choice,
        selection_rule='Among two completed new baselines: both paper+1 targets first, then mAP, then smaller head count; all six from the existing earliest mAP-best checkpoint of one run.',
        comparisons=comparisons,paired_order_exact=True,new_NN=0,optimizer_updates=0,missing_evaluated=False,
        limits='CSV reductions of installed-GT-audited exports; not a new neural or independent GT rerun. '
            'Only the B128 head8/32 pair is freshly matched for this head-setting comparison. '
            'Historical B64 comparisons change P and attempted updates; AV is original50 plus expert50. '
            'Baseline performance is not dual-axis contribution. Official benchmark selected epochs/configurations; single seed42. '
            'Repeated query comparisons are not independent new samples. Raw archives/selected weights retained for fresh result audit.')
    (out/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(dict(status=record['status'],paper_targets=paper,selected_baseline=choice),ensure_ascii=False))


if __name__ == '__main__':
    main()
