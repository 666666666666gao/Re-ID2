"""Read complete cutoff-only normal results against the same-seed global AV controls."""
import csv
import json
import math
from pathlib import Path

from analyze_identity_coordinate_three_normal import METRICS, load, query_rows, table

P = Path(__file__).resolve().parent
TAG = 'rgbnt100_band_ranges_AV_20261016'
CONTROLS = ('frequency_shared', 'axis_shared')
IDENTIFIERS = ('query_index', 'name', 'identity', 'camera', 'scene', 'valid', 'relevant_gallery', 'kept_gallery')


def compare(after, before, after_name, before_name):
    arows, a = after
    brows, b = before
    assert len(arows) == len(brows) == 1715
    assert all(tuple(x[k] for k in IDENTIFIERS) == tuple(y[k] for k in IDENTIFIERS) for x,y in zip(arows,brows))
    rescue = sum(int(x['Rank-1']) == 1 and int(y['Rank-1']) == 0 for x,y in zip(arows,brows))
    harm = sum(int(x['Rank-1']) == 0 and int(y['Rank-1']) == 1 for x,y in zip(arows,brows))
    delta = {k:a[k]-b[k] for k in METRICS}
    assert abs(delta['Rank-1']-100*(rescue-harm)/1715) < 1e-8
    return dict(improved=after_name,reference=before_name,**delta,rank1_rescued=rescue,rank1_harmed=harm,
                AP_improved=sum(float(x['AP'])-float(y['AP']) > 1e-12 for x,y in zip(arows,brows)),
                AP_worsened=sum(float(y['AP'])-float(x['AP']) > 1e-12 for x,y in zip(arows,brows)),
                both_plus1=delta['mAP'] >= 1 and delta['Rank-1'] >= 1)


def main():
    root = P/'results'/TAG/'RGBNT100'
    controller = load(root/'controller_result.json')
    assert controller['status'] == 'COMPLETE' and (controller['band_low_cutoff'],controller['band_high_cutoff'],controller['frequency_bin_counts']) == (.25,.375,[25,30,73])
    assert controller['state_cases'] == 8 and controller['paired_sampling_exact']
    folders = {'original_DeMo':P/'results/full_official_baselines_20261004/training/RGBNT100_demo_s42'}
    for variant in CONTROLS:
        folders['AV_'+variant] = P/'results/r201av_RGB_frequency_value_decay_20261014/RGBNT100/training'/('RGBNT100_r201av_'+variant+'_s42')
        folders['band_ranges_'+variant] = root/'training'/('RGBNT100_ranges_AV_'+variant+'_s42')
    reads, orders, costs = {}, {}, {}
    metrics, groups, curves, cmc = [], [], [], []
    for model, folder in folders.items():
        data = load(folder/'result.json')
        rows, summary = query_rows(folder,'RGBNT100')
        assert data['status'] == 'COMPLETE' and data['epochs'] == 50 and data['amp_skipped_steps'] == 0
        assert (data['train_records'],data['query_records'],data['gallery_records']) == (8675,1715,8575)
        assert data['training_heldout_identities'] == 0 and data['training_coverage'] == dict(eligible=8675,visited=8675,unvisited=[])
        assert data['descriptor_dim'] == 5120
        epochs = list(csv.DictReader((folder/'epochs.csv').open(encoding='utf-8',newline='')))
        assert [int(r['epoch']) for r in epochs] == list(range(1,51))
        selected = max(epochs,key=lambda r:float(r['mAP']))
        assert int(selected['epoch']) == data['best']['epoch']
        assert all(abs(float(selected[k])-summary[k]) < 1e-8 and abs(data['full_metrics'][k]-summary[k]) < 1e-8 for k in METRICS)
        metrics.append(dict(model=model,selected_epoch=data['best']['epoch'],**{k:summary[k] for k in METRICS}))
        cmc.extend(dict(model=model,rank=rank,value=value) for rank,value in enumerate(summary['CMC_1_to_50'],1))
        groups.extend(dict(model=model,grouping=axis,**value) for axis,values in summary['groups'].items() for value in values)
        reads[model] = rows, summary
        costs[model] = dict(parameters=data['parameters'],trainable_parameters=data['trainable_parameters'],descriptor_dim=5120,
                            total_epochs=50 if model == 'original_DeMo' else 100,runtime=data['runtime'],peak_memory=data['peak_memory'],
                            selected_epoch=data['best']['epoch'],final_epoch={k:float(epochs[-1][k]) for k in METRICS})
        if model != 'original_DeMo':
            batches = [json.loads(line) for line in (folder/'batch_orders.jsonl').read_text('utf-8').splitlines()]
            assert data['steps'] == data['optimizer_steps'] == len(batches) == 6563
            assert all(r['optimizer_updated'] and r['loss'] == r['full_loss'] and r['partial_loss_effective'] == 0 for r in batches)
            orders[model] = [(r['epoch'],r['step'],r['names'],r['partial_set']) for r in batches]
            assert load(folder/'normal_cpu_audit.json')['status'] == 'PASS'
        if model.startswith('band_ranges_'):
            assert (data['parameters'],data['trainable_parameters']) == (99585158,1887366)
            ranges = data['anchor']['frequency_band_ranges']
            assert (data['arguments']['band_low_cutoff'],data['arguments']['band_high_cutoff']) == (.25,.375)
            assert (ranges['low'],ranges['high'],ranges['bin_counts']) == (.25,.375,[25,30,73])
            assert ranges['parameters_added'] == 0 and ranges['global_M_F_readers_unchanged']
            for epoch in epochs:
                members = [r for r in batches if r['epoch'] == int(epoch['epoch'])]
                loss = math.fsum(r['loss'] for r in members)/len(members)
                assert abs(loss-float(epoch['loss'])) < 1e-8
                curves.append(dict(model=model,epoch=int(epoch['epoch']),loss=loss,batches=len(members),**{k:float(epoch[k]) for k in METRICS}))
    assert all(value == orders['AV_frequency_shared'] for value in orders.values())
    assert controller['successful_updates'] == 13126
    pairs = [('band_ranges_axis_shared','original_DeMo'),('band_ranges_frequency_shared','original_DeMo'),
             ('band_ranges_axis_shared','band_ranges_frequency_shared'),('band_ranges_axis_shared','AV_axis_shared'),
             ('band_ranges_frequency_shared','AV_frequency_shared')]
    comparisons = [compare(reads[a],reads[b],a,b) for a,b in pairs]
    query_changes = [dict(improved=a,reference=b,**{k:x[k] for k in IDENTIFIERS},
                          delta_AP_pp=100*(float(x['AP'])-float(y['AP'])),delta_INP_pp=100*(float(x['INP'])-float(y['INP'])),
                          delta_rank1=int(x['Rank-1'])-int(y['Rank-1'])) for a,b in pairs for x,y in zip(reads[a][0],reads[b][0])]
    state_metrics, state_changes = [], []
    for variant in CONTROLS:
        name = 'RGBNT100_ranges_AV_'+variant+'_s42'
        folder = root/'states'/name
        data = load(folder/'result.json')
        assert data['status'] == 'COMPLETE' and data['state_cases'] == 4
        assert (data['band_low_cutoff'],data['band_high_cutoff'],data['frequency_bin_counts']) == (.25,.375,[25,30,73])
        assert data['full11_features_distance_equal_selected_normal'] and data['state_tensor_versions_unchanged']
        states = {}
        for state in ('00','10','01','11'):
            rows = list(csv.DictReader((folder/('state_'+state+'.csv')).open(encoding='utf-8',newline='')))
            assert len(rows) == 1715
            states[state] = rows,data['measurements'][state]
            state_metrics.append(dict(model=variant,state=state,**{k:data['measurements'][state][k] for k in METRICS}))
        zero = compare(states['00'],reads['original_DeMo'],variant+'_00','original_DeMo')
        assert all(abs(zero[k]) < 1e-8 for k in METRICS) and zero['rank1_rescued'] == zero['rank1_harmed'] == zero['AP_improved'] == zero['AP_worsened'] == 0
        normal = compare(states['11'],reads['band_ranges_'+variant],variant+'_11','selected_normal')
        assert all(abs(normal[k]) < 1e-8 for k in METRICS) and normal['AP_improved'] == normal['AP_worsened'] == 0
        for before in ('00','10','01'):
            state_changes.append(compare(states['11'],states[before],variant+'_11',variant+'_'+before))
    output = root/'normal_analysis'
    output.mkdir(exist_ok=False)
    for name,values in [('six_metrics',metrics),('group_metrics',groups),('comparisons',comparisons),('paired_query_changes',query_changes),
                        ('training_curves',curves),('CMC_1_to_50',cmc),('state_metrics',state_metrics),('state_comparisons',state_changes)]:
        table(output/(name+'.csv'),values)
    result = dict(status='ACTUAL_BAND_RANGE_TWO_FULL50_NORMAL_EIGHT_STATES_CPU_READOUT',comparisons=comparisons,state_comparisons=state_changes,
                  training=costs,model_results=5,state_cases=8,paired_query_rows=len(query_changes),new_epoch_rows=len(curves),
                  new_stage_formal_updates=13126,original_AV42_record_order_exact=True,new_neural_calls=0,new_optimizer_updates=0,
                  limits='Fixed teacher42, fresh expert42/Adam and extra50. Only existing FFT cutoff masks change to .25/.375 for both M/F; ordinary receives the same masks. Original global readers are retained. Same-seed original global AV42 is the structural control, not best-of5 ordinary45. Frequency-position counts do not establish energy, identity value or a causal failure. No parameters added does not mean equal latency. Frozen states do not replace retrained ablations. Full official benchmark mAP participates in earliest-tie epoch selection. Normal only; no missing, new49 or multiple cutoff seeds.')
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(dict(status=result['status'],comparisons=comparisons),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
