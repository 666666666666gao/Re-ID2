"""Reduce one unified G recipe on all three complete official normal datasets."""
import csv
from datetime import datetime
import json
import math
from pathlib import Path

from analyze_identity_coordinate_three_normal import COUNTS, METRICS, load, query_rows, table

PROJECT = Path(__file__).resolve().parent


def main():
    other = PROJECT / 'results/r201g_other_two_normal_20261006'
    actual = load(PROJECT / 'results/preflight/r201g_other_two_normal_actual_session_20261006.json')
    assert actual['status'] == 'ACTUAL_UNCHANGED_G_OTHER_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW'
    assert actual['exit_code'] == 0 and actual['successful_updates'] == 14124 and actual['additional_epochs'] == 200
    assert load(PROJECT / 'results/preflight/r201g_other_two_normal_terminal_text_intake_20261006.json')['status'] == 'ACTUAL_G_OTHER_TWO_NORMAL_FOUR50_TERMINAL_TEXT_VERIFIED'
    assert load(PROJECT / 'results/r201g_normal_priority_20261006/normal_analysis/result.json')['status'] == 'ACTUAL_R201G_TWO_FULL50_NORMAL_CPU_ANALYSIS_COMPLETE'
    old = PROJECT / 'results/identity_coordinate_three_normal_completed_20261005'
    assert load(old / 'normal_analysis/result.json')['status'] == 'ACTUAL_THREE_NORMAL_DATASETS_CPU_ANALYSIS_COMPLETE'
    normal, groups, comparisons, pairs, curves, training = [], [], [], [], [], {}
    for dataset, counts in COUNTS.items():
        train, query, gallery, steps = counts
        root = PROJECT / 'results/r201g_normal_priority_20261006' if dataset == 'RGBNT201' else other / dataset
        controller = load(root / 'controller_result.json')
        assert controller['status'] == 'COMPLETE' and controller['dataset'] == dataset
        assert controller['controls'] == 2 and controller['successful_updates'] == 2 * steps and controller['additional_epochs'] == 100
        assert controller['paired_sampling_exact'] and controller['normal_archives_local_verified'] == 2
        folders = {'original_DeMo': PROJECT / 'results/full_official_baselines_20261004/training' / (dataset + '_demo_s42')}
        for variant in ('frequency_shared','axis_shared'):
            folders['unit_'+variant] = old / 'training' / (dataset + '_identity_' + variant + '_narrow_s42')
            folders['normal_'+variant] = root / 'training' / (dataset + '_r201g_' + variant + '_s42')
        reads, orders = {}, {}
        for model, folder in folders.items():
            data = load(folder / 'result.json')
            assert data['status'] == 'COMPLETE' and data['epochs'] == 50
            assert data['steps'] == data['optimizer_steps'] == steps and data['amp_skipped_steps'] == 0
            assert (data['train_records'],data['query_records'],data['gallery_records']) == (train,query,gallery)
            assert data['training_heldout_identities'] == 0 and data['descriptor_dim'] == 5120
            assert data['training_coverage'] == dict(eligible=train,visited=train,unvisited=[])
            rows, summary = query_rows(folder,dataset)
            with (folder / 'epochs.csv').open(encoding='utf-8',newline='') as handle:
                epochs = list(csv.DictReader(handle))
            assert [int(row['epoch']) for row in epochs] == list(range(1,51))
            selected = max(epochs,key=lambda row:float(row['mAP']))
            assert int(selected['epoch']) == data['best']['epoch']
            assert all(abs(float(selected[key])-summary[key]) < 1e-8 and abs(data['full_metrics'][key]-summary[key]) < 1e-8 for key in METRICS)
            training[dataset+'/'+model] = dict(selected_epoch=data['best']['epoch'],parameters=data['parameters'],trainable_parameters=data['trainable_parameters'],
                descriptor_dim=5120,total_training_budget_epochs=50 if model=='original_DeMo' else 100,
                stage_updates=steps,peak_memory=data['peak_memory'],runtime=data['runtime'],final_epoch={key:float(epochs[-1][key]) for key in METRICS})
            normal.append(dict(dataset=dataset,model=model,selected_epoch=data['best']['epoch'],**{key:summary[key] for key in METRICS}))
            for axis, values in summary['groups'].items():
                groups.extend(dict(dataset=dataset,model=model,grouping=axis,**value) for value in values)
            reads[model] = rows, summary
            if model != 'original_DeMo':
                batches = [json.loads(line) for line in (folder/'batch_orders.jsonl').read_text(encoding='utf-8').splitlines()]
                assert len(batches) == steps and all(row['optimizer_updated'] for row in batches)
                orders[model] = [(row['epoch'],row['step'],row['names'],row['partial_set']) for row in batches]
                if model.startswith('normal_'):
                    assert all(row['loss']==row['full_loss'] and row['partial_CE_weight']==row['partial_triplet_weight']==row['partial_loss_effective']==0 and row['primary_full_metric']=='unit5120_soft_triplet' for row in batches)
                    audit = load(folder/'normal_cpu_audit.json')
                    assert audit['status']=='PASS' and audit['max_metric_error']<1e-8 and audit['full_training_coverage']
                    assert load(folder/'normal_local_archive.json')['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
                    for epoch in epochs:
                        selected_batches = [row for row in batches if row['epoch']==int(epoch['epoch'])]
                        means = {key:math.fsum(row[key] for row in selected_batches)/len(selected_batches) for key in ('loss','full_loss','partial_ce','cross_triplet','partial_loss_effective')}
                        assert means['loss']==means['full_loss'] and means['partial_loss_effective']==0
                        assert abs(means['loss']-float(epoch['loss']))<1e-8
                        curves.append(dict(dataset=dataset,model=model,epoch=int(epoch['epoch']),batches=len(selected_batches),**means,**{key:float(epoch[key]) for key in METRICS}))
        assert all(order==orders['normal_axis_shared'] for order in orders.values())
        assert len({(training[dataset+'/'+model]['parameters'],training[dataset+'/'+model]['trainable_parameters']) for model in folders if model!='original_DeMo'}) == 1
        for improved, reference in (('normal_axis_shared','original_DeMo'),('normal_frequency_shared','original_DeMo'),('normal_axis_shared','normal_frequency_shared'),('normal_axis_shared','unit_axis_shared'),('normal_frequency_shared','unit_frequency_shared')):
            arows, a = reads[improved]
            brows, b = reads[reference]
            assert all(all(x[key]==y[key] for key in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')) for x,y in zip(arows,brows))
            deltas = {key:a[key]-b[key] for key in METRICS}
            comparisons.append(dict(dataset=dataset,improved=improved,reference=reference,**deltas,
                both_plus2=deltas['mAP']>=2 and deltas['Rank-1']>=2,
                rank1_rescued=sum(int(x['Rank-1'])==1 and int(y['Rank-1'])==0 for x,y in zip(arows,brows)),
                rank1_harmed=sum(int(x['Rank-1'])==0 and int(y['Rank-1'])==1 for x,y in zip(arows,brows)),
                AP_improved=sum(float(x['AP'])>float(y['AP']) for x,y in zip(arows,brows)),
                AP_worsened=sum(float(x['AP'])<float(y['AP']) for x,y in zip(arows,brows))))
            pairs.extend(dict(dataset=dataset,improved=improved,reference=reference,**{key:x[key] for key in ('query_index','name','identity','camera','scene')},
                delta_AP_pp=100*(float(x['AP'])-float(y['AP'])),delta_INP_pp=100*(float(x['INP'])-float(y['INP'])),delta_rank1=int(x['Rank-1'])-int(y['Rank-1'])) for x,y in zip(arows,brows))
    assert len(normal)==len(comparisons)==15 and len(pairs)==15710 and len(curves)==300
    output = other / 'three_dataset_normal_analysis'
    output.mkdir(exist_ok=False)
    for name, rows in (('six_metrics',normal),('group_metrics',groups),('comparisons',comparisons),('paired_query_changes',pairs),('training_curves',curves)):
        table(output/(name+'.csv'),rows)
    result = dict(status='ACTUAL_UNIFIED_G_THREE_OFFICIAL_NORMAL_DATASETS_CPU_READOUT',completed_at=datetime.now().isoformat(timespec='seconds'),
        model_results=15,comparisons=comparisons,paired_query_rows=15710,new_epoch_rows=300,training=training,seed=42,
        new_stage_formal_updates=19418,new_neural_calls=0,new_optimizer_updates=0,
        all_three_original_both_plus2=all(row['both_plus2'] for row in comparisons if row['improved']=='normal_axis_shared' and row['reference']=='original_DeMo'),
        all_three_fair_positive=all(row['mAP']>0 and row['Rank-1']>0 for row in comparisons if row['improved']=='normal_axis_shared' and row['reference']=='normal_frequency_shared'),
        missing_success_proven=False,multiseed_success_proven=False,
        limits='All same unified G recipe with frozen originalDeMo reference per dataset, original50+additional50/newAdam versus original50. Highest officialnormalmAP/earliesttie one checkpoint/all six; benchmark selected, not untouched test. Old unit pair has matched sampling/parameter/dim and nonzero partialloss. frequency_shared is historical mixed-source ordinary-expert control, not a literal new single frequency branch. NormalCPU verifies prior installedGT exports/CMC50/groups; no missing, calibrated contribution, psi retraining or paired-seed acceptance.')
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('status','completed_at','all_three_original_both_plus2','all_three_fair_positive','paired_query_rows')}))


if __name__ == '__main__':
    main()
