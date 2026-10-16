"""Compare completed full-normal CSVs; no model import or new inference."""
import csv
import json
from pathlib import Path

P = Path(__file__).resolve().parent
RUNS = dict(
    original_DeMo=P/'results/full_official_baselines_20261004/training/RGBNT100_demo_s42',
    AV_axis=P/'results/r201av_RGB_frequency_value_decay_20261014/RGBNT100/training/RGBNT100_r201av_axis_shared_s42',
    head32_B64=P/'results/vehicle_atmoe_head_dynamic_amp_20261016/RGBNT100/training/RGBNT100_atmoe32_demo_s42')
METRICS = ('mAP', 'mINP', 'Rank-1', 'Rank-5', 'Rank-10', 'Rank-20')


def main():
    selected, queries = {}, {}
    for name, run in RUNS.items():
        info = json.loads((run/'result.json').read_text('utf-8'))
        assert info['status'] == 'COMPLETE' and info['epochs'] == 50
        assert info['query_records'] == 1715 and info['gallery_records'] == 8575
        selected[name] = info['full_metrics']
        with (run/'best_per_query.csv').open(encoding='utf-8', newline='') as file:
            queries[name] = list(csv.DictReader(file))
        assert len(queries[name]) == 1715 and all(row['valid'] == 'True' for row in queries[name])
        fields = dict(mAP='AP', mINP='INP') | {key:key for key in METRICS[2:]}
        assert all(abs(100*sum(float(row[field]) for row in queries[name])/1715-selected[name][key]) < 1e-8
                   for key, field in fields.items())
    compared = []
    for base in ('original_DeMo', 'AV_axis'):
        old, new = queries[base], queries['head32_B64']
        assert all(tuple(a[key] for key in ('query_index','name','identity','camera','scene')) ==
                   tuple(b[key] for key in ('query_index','name','identity','camera','scene')) for a,b in zip(old,new))
        rescued = sum(float(a['Rank-1']) == 0 and float(b['Rank-1']) == 1 for a,b in zip(old,new))
        harmed = sum(float(a['Rank-1']) == 1 and float(b['Rank-1']) == 0 for a,b in zip(old,new))
        row = dict(candidate='head32_B64', reference=base, rank1_rescued=rescued, rank1_harmed=harmed,
            rank1_net=rescued-harmed, AP_improved=sum(float(b['AP'])-float(a['AP']) > 1e-12 for a,b in zip(old,new)),
            AP_declined=sum(float(b['AP'])-float(a['AP']) < -1e-12 for a,b in zip(old,new)))
        row.update({key+'_delta_pp':selected['head32_B64'][key]-selected[base][key] for key in METRICS})
        assert abs(row['Rank-1_delta_pp']-(rescued-harmed)*100/1715) < 1e-8
        compared.append(row)
    out = P/'results/vehicle_atmoe_head_dynamic_amp_20261016/RGBNT100/normal_comparison'
    out.mkdir(exist_ok=False)
    with (out/'comparisons.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(compared[0]));writer.writeheader();writer.writerows(compared)
    record = dict(status='COMPLETED_CSV_ONLY_FULL1715_PAIRED_DIAGNOSTIC', selected_metrics=selected,
        comparisons=compared, new_NN=0, optimizer_updates=0, missing_evaluated=False,
        scope='All query identifiers/order and six-metric reductions checked against each selected full run. '
            'Historical independent trainings/configurations; not a freshly retrained one-factor or expert-utility ablation. '
            'No query-label selection is used at deployment. Paper targets remain87.2/98.6.')
    (out/'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    print(json.dumps(compared, ensure_ascii=False))


if __name__ == '__main__':
    main()
