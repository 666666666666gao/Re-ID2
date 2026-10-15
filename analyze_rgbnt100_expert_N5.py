"""CPU-only complete-wave/five-seed readout with one predeclared seed-selection rule."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from analyze_identity_coordinate_three_normal import METRICS, load, query_rows, table

PROJECT=Path(__file__).resolve().parent
VARIANTS=('frequency_shared','axis_shared')


def read_run(folder,selection,seed,variant):
    d=load(folder/'result.json');rows,summary=query_rows(folder,'RGBNT100')
    assert d['status']=='COMPLETE' and d['epochs']==50 and d['steps']==d['optimizer_steps']>0
    assert d['amp_skipped_steps']==d['training_heldout_identities']==0
    assert d['training_coverage']==dict(eligible=8675,visited=8675,unvisited=[])
    assert (d['parameters'],d['trainable_parameters'],d['descriptor_dim'])==(selection['parameters'],selection['trainable_parameters'],5120)
    assert d['arguments']['seed']==seed and d['arguments']['variant']==variant
    if seed!=42:
        assert d['arguments']['expert_recipe']==selection['selected_recipe'] and d['anchor']['anchor_seed']==42 and d['anchor']['expert_seed']==seed
    assert load(folder/'normal_cpu_audit.json')['status']=='PASS'
    with (folder/'epochs.csv').open(encoding='utf-8',newline='') as stream:epochs=list(csv.DictReader(stream))
    assert [int(e['epoch']) for e in epochs]==list(range(1,51))
    chosen=max(epochs,key=lambda e:float(e['mAP']))
    assert int(chosen['epoch'])==d['best']['epoch']
    assert all(abs(float(chosen[k])-summary[k])<1e-8 and abs(d['full_metrics'][k]-summary[k])<1e-8 for k in METRICS)
    values=[json.loads(line) for line in (folder/'batch_orders.jsonl').read_text().splitlines()]
    assert len(values)==d['steps'] and all(v['optimizer_updated'] and v['loss']==v['full_loss'] and v['partial_loss_effective']==0 for v in values)
    order=[(v['epoch'],v['step'],v['names'],v['partial_set']) for v in values]
    metric=dict(recipe=selection['selected_recipe'],seed=seed,variant=variant,selected_epoch=d['best']['epoch'],successful_updates=d['steps'],
        **{k:summary[k] for k in METRICS},run_dir=str(folder.relative_to(PROJECT)).replace('\\','/'))
    training=dict(seed=seed,variant=variant,selected_epoch=d['best']['epoch'],last_six={k:float(epochs[-1][k]) for k in METRICS},
        teacher_seed=42,teacher_epochs=50,expert_epochs=50,runtime=d['runtime'],peak_memory=d['peak_memory'])
    return metric,rows,order,training


def compare(after,before):
    a,arows=after;b,brows=before
    fields=('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')
    assert len(arows)==len(brows)==1715
    assert all(tuple(x[k] for k in fields)==tuple(y[k] for k in fields) for x,y in zip(arows,brows))
    rescued=sum(int(x['Rank-1'])>int(y['Rank-1']) for x,y in zip(arows,brows))
    harmed=sum(int(x['Rank-1'])<int(y['Rank-1']) for x,y in zip(arows,brows))
    delta={k:a[k]-b[k] for k in METRICS}
    assert abs(delta['Rank-1']-100*(rescued-harmed)/1715)<1e-8
    return dict(improved=a['variant'],improved_seed=a['seed'],reference=b['variant'],reference_seed=b['seed'],**delta,
        rank1_rescued=rescued,rank1_harmed=harmed,
        AP_improved=sum(float(x['AP'])-float(y['AP'])>1e-12 for x,y in zip(arows,brows)),
        AP_worsened=sum(float(y['AP'])-float(x['AP'])>1e-12 for x,y in zip(arows,brows)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--selection-file',required=True)
    modes=parser.add_subparsers(dest='mode',required=True)
    wave=modes.add_parser('wave');wave.add_argument('--seed',type=int,choices=(43,44,45,46),required=True)
    modes.add_parser('five')
    args=parser.parse_args();selection_path=Path(args.selection_file);selection=load(selection_path)
    assert selection['status']=='ACTUAL_CLOSED_RECIPE_LOCKED_FOR_BOUNDED_EXPERT_N5' and selection['expert_seeds']==[42,43,44,45,46]
    key=selection['selected_recipe'];family=PROJECT/selection['local_family_root']
    seeds=[args.seed] if args.mode=='wave' else [42,43,44,45,46]
    metrics=[];comparisons=[];training=[];reads={};original_rows,original_summary=query_rows(PROJECT/'results/full_official_baselines_20261004/training/RGBNT100_demo_s42','RGBNT100')
    original=dict(seed=42,variant='original_DeMo',**{k:original_summary[k] for k in METRICS})
    assert selection['targets']=={k:original[k]+1 for k in ('mAP','Rank-1')}
    for seed in seeds:
        wave_root=family/('s'+str(seed))/'RGBNT100'
        if seed!=42:
            controller=load(wave_root/'controller_result.json')
            assert controller['status']=='COMPLETE' and controller['seed']==seed and controller['recipe']==key and controller['native_updates']==6
            assert controller['state_cases']==0 and controller['normal_GT_audited'] and controller['paired_sampling_exact']
        orders=[]
        for variant in VARIANTS:
            folder=PROJECT/selection['seed42_local_runs'][variant] if seed==42 else wave_root/'training'/('RGBNT100_n5_'+key+'_'+variant+'_s'+str(seed))
            m,rows,order,t=read_run(folder,selection,seed,variant)
            m['both_plus1']=all(m[k]>=selection['targets'][k] for k in ('mAP','Rank-1'))
            metrics.append(m);training.append(t);orders.append(order);reads[(seed,variant)]=(m,rows)
            comparisons.append(compare((m,rows),(original,original_rows)))
        assert orders[0]==orders[1]
        if seed!=42:assert controller['successful_updates']==2*len(orders[0])
        comparisons.append(compare(reads[(seed,'axis_shared')],reads[(seed,'frequency_shared')]))
    if args.mode=='wave':
        root=family/('s'+str(args.seed))/'RGBNT100'
        out=root/'normal_analysis';out.mkdir(exist_ok=False)
        result=dict(status='ACTUAL_N5_ONE_COMPLETE_NORMAL_WAVE_CPU_READOUT',recipe=key,seed=args.seed,model_results=2,
            metrics=metrics,comparisons=comparisons,training=training,new_neural_calls=0,new_optimizer_updates=0,
            limits='One full50 matched pair, conditional expert seed on teacher42. BenchmarkmAP earliesttie; all six from the same checkpoint. No four states, missing, selected N5 main run or five-seed claim.')
    else:
        assert len(metrics)==10
        for seed in (43,44,45,46):
            intake=load(PROJECT/'results/preflight'/('rgbnt100_N5_'+key+'_s'+str(seed)+'_complete_intake_20261015.json'))
            assert intake['status']=='ACTUAL_N5_WAVE_D_ARCHIVED_COMPLETE' and intake['all_children_and_controller_absent']
        def choice(row):return (row['both_plus1'],row['mAP'],-row['seed'])
        best={variant:max((row for row in metrics if row['variant']==variant),key=choice) for variant in VARIANTS}
        unfiltered={variant:max((row for row in metrics if row['variant']==variant),key=lambda row:(row['mAP'],-row['seed'])) for variant in VARIANTS}
        paired_seed=best['axis_shared']['seed']
        paired_ordinary=reads[(paired_seed,'frequency_shared')][0]
        statistics_rows=[]
        for variant in VARIANTS:
            subset=[row for row in metrics if row['variant']==variant]
            for metric in METRICS:
                values=[row[metric] for row in subset]
                statistics_rows.append(dict(variant=variant,metric=metric,N=5,mean=statistics.mean(values),sample_SD=statistics.stdev(values)))
        out=family/'five_normal_analysis';out.mkdir(exist_ok=False)
        table(out/'supplementary_mean_SD.csv',statistics_rows)
        result=dict(status='ALL_FIVE_COMPLETE_SELECTED',recipe=key,teacher_seed=42,expert_seeds=[42,43,44,45,46],seed42_reused=True,
            metrics=metrics,main_best_single=best,unfiltered_maximum_mAP_single=unfiltered,
            paired_ordinary_for_selected_axis=paired_ordinary,main_seed_selection_rule=selection['main_seed_selection_rule'],
            comparisons=comparisons,training=training,main_axis_both_plus1=best['axis_shared']['both_plus1'],
            selected_pair_changes=compare(reads[(paired_seed,'axis_shared')],reads[(paired_seed,'frequency_shared')]),
            state_cases=0,missing_evaluated=False,new_neural_calls=0,new_optimizer_updates=0,
            selection_file_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
            limits='Conditional expert-stage five seeds on one fixed benchmark-selected teacher42, not five full pipelines or an untouched test. Per-seed checkpoint is mAP-best/earliesttie; main seed uses the predeclared target-priority/mAP/seed tuple equally for ordinary/axis, and the unfiltered max-mAP run is separately disclosed. All attempted runs retained. Selected frozen-state inference and final fresh result audit remain separate obligations; no missing or method-completion claim.')
    table(out/'six_metrics.csv',metrics);table(out/'comparisons.csv',comparisons)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],recipe=key,model_results=len(metrics))),flush=True)


if __name__=='__main__':main()
