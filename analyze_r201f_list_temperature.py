"""Reduce closed RGBNT201 Smooth-AP controls against both metric controls."""
import csv
from datetime import datetime
import json
import math
from pathlib import Path

from analyze_identity_coordinate_three_normal import METRICS, load, query_rows, table

PROJECT = Path(__file__).resolve().parent


def main():
    root = PROJECT / 'results/r201f_list_temperature_20261006'
    controller = load(root / 'controller_result.json')
    assert controller['status'] == 'COMPLETE' and controller['successful_updates'] == 5294
    assert controller['additional_epochs'] == 100 and controller['paired_sampling_exact']
    folders = {'original_DeMo': PROJECT / 'results/full_official_baselines_20261004/training/RGBNT201_demo_s42'}
    for variant in ('frequency_shared', 'axis_shared'):
        folders['unit_'+variant] = PROJECT / ('results/rgbnt201_identity_outlet_r201c_20261005/training/RGBNT201_identity_'+variant+'_narrow_s42')
        folders['raw_'+variant] = PROJECT / ('results/r201d_full_triplet_20261006/training/RGBNT201_r201d_'+variant+'_s42')
        folders['list_'+variant] = root / 'training' / ('RGBNT201_r201f_'+variant+'_s42')
        folders['list01_'+variant] = PROJECT / ('results/r201e_list_objective_fp32check_20261006/training/RGBNT201_r201e_'+variant+'_s42')
    reads, orders, training, normal, groups, curves = {}, {}, {}, [], [], []
    for model, folder in folders.items():
        data = load(folder / 'result.json')
        assert data['status'] == 'COMPLETE' and data['epochs'] == 50
        assert data['steps'] == data['optimizer_steps'] == 2647 and data['amp_skipped_steps'] == 0
        assert (data['train_records'], data['query_records'], data['gallery_records']) == (3951,836,836)
        assert data['training_heldout_identities'] == 0 and data['descriptor_dim'] == 5120
        rows, summary = query_rows(folder, 'RGBNT201')
        with (folder / 'epochs.csv').open(encoding='utf-8', newline='') as handle:
            epochs = list(csv.DictReader(handle))
        assert [int(row['epoch']) for row in epochs] == list(range(1,51))
        selected = max(epochs, key=lambda row: float(row['mAP']))
        assert int(selected['epoch']) == data['best']['epoch']
        assert all(abs(float(selected[key])-summary[key]) < 1e-8 and abs(data['full_metrics'][key]-summary[key]) < 1e-8 for key in METRICS)
        training[model] = dict(selected_epoch=data['best']['epoch'],parameters=data['parameters'],
            trainable_parameters=data['trainable_parameters'],descriptor_dim=5120,
            total_budget_epochs=50 if model=='original_DeMo' else 100,
            peak_memory=data['peak_memory'],runtime=data['runtime'],
            final_epoch={key:float(epochs[-1][key]) for key in METRICS})
        normal.append(dict(model=model,selected_epoch=data['best']['epoch'],**{key:summary[key] for key in METRICS}))
        for axis, values in summary['groups'].items():
            groups.extend(dict(model=model,grouping=axis,**value) for value in values)
        reads[model] = rows, summary
        if model.startswith('list_'):
            audit = load(folder / 'normal_cpu_audit.json')
            assert audit['status']=='PASS' and audit['max_metric_error'] < 1e-8 and audit['full_training_coverage']
            assert load(folder / 'normal_local_archive.json')['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
            batches = [json.loads(line) for line in (folder/'batch_orders.jsonl').read_text(encoding='utf-8').splitlines()]
            assert len(batches)==2647 and all(row['optimizer_updated'] for row in batches)
            assert all(row['primary_full_metric']=='label_masked_smooth_AP_unit5120' and row['temperature']==.05 and row['metric_coefficient']==1 and row['added_losses']==0 for row in batches)
            assert all(math.isfinite(row['smooth_ap_loss']) and -1e-6<=row['smooth_ap_loss']<=1.000001 for row in batches)
            orders[model] = [(row['epoch'],row['step'],row['names'],row['partial_set']) for row in batches]
            for kind in ('unit_','raw_','list01_'):
                counterpart = folders[kind+model.removeprefix('list_')]
                old_orders = [json.loads(line) for line in (counterpart/'batch_orders.jsonl').read_text(encoding='utf-8').splitlines()]
                assert orders[model]==[(row['epoch'],row['step'],row['names'],row['partial_set']) for row in old_orders]
            for epoch in epochs:
                batch = [row for row in batches if row['epoch']==int(epoch['epoch'])]
                means = {key:math.fsum(row[key] for row in batch)/len(batch) for key in ('loss','full_loss','partial_ce','cross_triplet','smooth_ap_loss')}
                assert abs(means['loss']-float(epoch['loss']))<1e-8
                assert abs(means['loss']-means['full_loss']-.25*means['partial_ce']-.5*means['cross_triplet'])<1e-5
                curves.append(dict(model=model,epoch=int(epoch['epoch']),batches=len(batch),**means,
                    AP_loss_exact_zero_batches=sum(row['smooth_ap_loss']==0 for row in batch),
                    AP_loss_max=max(row['smooth_ap_loss'] for row in batch),
                    **{key:float(epoch[key]) for key in METRICS}))
    assert orders['list_frequency_shared']==orders['list_axis_shared']
    assert len({(training[m]['parameters'],training[m]['trainable_parameters']) for m in training if m!='original_DeMo'})==1
    comparisons, pairs = [], []
    for improved, reference in (('list_axis_shared','original_DeMo'),('list_frequency_shared','original_DeMo'),
        ('list_axis_shared','list_frequency_shared'),('list_axis_shared','unit_axis_shared'),
        ('list_frequency_shared','unit_frequency_shared'),('list_axis_shared','raw_axis_shared'),
        ('list_frequency_shared','raw_frequency_shared'),
        ('list_axis_shared','list01_axis_shared'),('list_frequency_shared','list01_frequency_shared')):
        a, am = reads[improved]
        b, bm = reads[reference]
        assert len(a)==len(b)==836
        assert all(tuple(x[key] for key in ('name','identity','camera','scene'))==tuple(y[key] for key in ('name','identity','camera','scene')) for x,y in zip(a,b))
        delta = {key:am[key]-bm[key] for key in METRICS}
        comparisons.append(dict(improved=improved,reference=reference,**delta,
            both_plus2=delta['mAP']>=2 and delta['Rank-1']>=2,
            rank1_rescued=sum(int(x['Rank-1'])==1 and int(y['Rank-1'])==0 for x,y in zip(a,b)),
            rank1_harmed=sum(int(x['Rank-1'])==0 and int(y['Rank-1'])==1 for x,y in zip(a,b)),
            AP_improved=sum(float(x['AP'])>float(y['AP']) for x,y in zip(a,b)),
            AP_worsened=sum(float(x['AP'])<float(y['AP']) for x,y in zip(a,b))))
        pairs.extend(dict(improved=improved,reference=reference,
            **{key:x[key] for key in ('query_index','name','identity','camera','scene')},
            delta_AP_pp=100*(float(x['AP'])-float(y['AP'])),delta_INP_pp=100*(float(x['INP'])-float(y['INP'])),
            delta_rank1=int(x['Rank-1'])-int(y['Rank-1'])) for x,y in zip(a,b))
    assert len(normal)==len(comparisons)==9 and len(pairs)==7524 and len(curves)==100
    output = root/'normal_analysis'
    output.mkdir(exist_ok=False)
    for name, values in (('six_metrics',normal),('group_metrics',groups),('comparisons',comparisons),
                         ('paired_query_changes',pairs),('training_loss_curves',curves)):
        table(output/(name+'.csv'),values)
    result = dict(status='ACTUAL_R201F_TWO_FULL50_NORMAL_CPU_ANALYSIS_COMPLETE',
        completed_at=datetime.now().isoformat(timespec='seconds'),comparisons=comparisons,training=training,
        selected_models=9,paired_query_rows=7524,new_epoch_rows=100,new_batch_rows=5294,
        paired_names_partial_sets_and_epochs_with_closed_unit_raw_and_AP01_training=True,
        six_metrics_CMC50_identity_camera_scene_reduced=True,new_neural_calls=0,new_optimizer_updates=0,
        limits='One seed; official benchmark selects earliest mAP-best. Original50 versus extra50 experts total100; unit/raw/list experts have matched extra50 budgets. CPU reduction validates installed-GT receipts, not a new neural or independent GT run. Zero displayed AP loss does not prove zero derivative. No new missing or three-dataset confirmation.')
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],comparisons=comparisons)),flush=True)


if __name__=='__main__':
    main()
