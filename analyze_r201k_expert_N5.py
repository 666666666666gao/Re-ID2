"""Full-normal RGBNT201 expert-stage repeats: all seeds, paired effects and separate best."""
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics

from analyze_identity_coordinate_three_normal import METRICS,load,query_rows,table

PROJECT=Path(__file__).resolve().parent
ROOT=PROJECT/'results/r201k_expert_N5_20261007'


def main():
    assert load(ROOT/'controller_result.json')['status']=='COMPLETE'
    baseline=PROJECT/'results/full_official_baselines_20261004/training/RGBNT201_demo_s42'
    original_rows,original=query_rows(baseline,'RGBNT201')
    values,groups,curves,pairs,differences,training=[],[],[],[],[],{}
    reads={}
    for seed in (42,43,44,45,46):
        base=PROJECT/'results/r201k_relation_local_pi_20261007/RGBNT201' if seed==42 else ROOT/('s'+str(seed))
        controller=load(base/'controller_result.json')
        assert controller['status']=='COMPLETE' and controller['controls']==2
        assert controller['additional_epochs']==100 and controller['native_updates']==6
        assert controller['paired_sampling_exact'] and controller['normal_archives_local_verified']==2
        orders={}
        for variant in ('frequency_shared','axis_shared'):
            folder=base/'training'/('RGBNT201_r201k_'+variant+'_s'+str(seed))
            data=load(folder/'result.json')
            assert data['status']=='COMPLETE' and data['epochs']==50
            assert data['arguments']['seed']==seed and data['arguments']['dataset']=='RGBNT201'
            assert data['method_revision'].startswith('R201K single PI-outlet factor')
            assert data['anchor']['method']==data['method_revision']
            assert data['steps']==data['optimizer_steps'] and data['amp_skipped_steps']==0
            assert (data['train_records'],data['query_records'],data['gallery_records'])==(3951,836,836)
            assert data['training_coverage']==dict(eligible=3951,visited=3951,unvisited=[])
            assert data['training_heldout_identities']==0 and data['descriptor_dim']==5120
            assert data['anchor']['original_run'].endswith('/RGBNT201_demo_s42')
            if seed!=42:
                assert data['anchor']['anchor_seed']==42 and data['anchor']['expert_seed']==seed
            assert '  NUM_INSTANCE: 8' in data['config'].splitlines()
            audit=load(folder/'normal_cpu_audit.json')
            assert audit['status']=='PASS' and audit['max_metric_error']<1e-8
            assert audit['full_training_coverage'] and audit['CMC50_and_per_query_and_groups']
            assert load(folder/'normal_local_archive.json')['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
            rows,summary=query_rows(folder,'RGBNT201')
            epochs=list(csv.DictReader((folder/'epochs.csv').open(encoding='utf-8',newline='')))
            assert [int(e['epoch']) for e in epochs]==list(range(1,51))
            chosen=max(epochs,key=lambda e:float(e['mAP']))
            assert int(chosen['epoch'])==data['best']['epoch']
            assert all(abs(float(chosen[k])-summary[k])<1e-8 and abs(data['full_metrics'][k]-summary[k])<1e-8 for k in METRICS)
            batches=[json.loads(line) for line in (folder/'batch_orders.jsonl').read_text(encoding='utf-8').splitlines()]
            assert len(batches)==data['optimizer_steps'] and all(b['optimizer_updated'] for b in batches)
            assert all(b['loss']==b['full_loss'] and b['partial_loss_effective']==b['partial_CE_weight']==b['partial_triplet_weight']==0
                and b['primary_margin']==.3 and b['auxiliary_original_soft_triplet']
                and b['PI_outlet']=='shared_relation_local' for b in batches)
            orders[variant]=[(b['epoch'],b['step'],b['names'],b['partial_set']) for b in batches]
            model=variant+'_s'+str(seed)
            reads[model]=(rows,summary)
            values.append(dict(seed=seed,model=variant,selected_epoch=data['best']['epoch'],**{k:summary[k] for k in METRICS}))
            groups.extend(dict(seed=seed,model=variant,grouping=axis,**group) for axis,gs in summary['groups'].items() for group in gs)
            training[model]=dict(steps=data['steps'],parameters=data['parameters'],trainable_parameters=data['trainable_parameters'],runtime=data['runtime'],peak_memory=data['peak_memory'],final_epoch={k:float(epochs[-1][k]) for k in METRICS})
            curves.extend(dict(seed=seed,model=variant,**e) for e in epochs)
        assert orders['frequency_shared']==orders['axis_shared']
        for reference,(reference_rows,reference_values) in (
            ('frequency_shared',reads['frequency_shared_s'+str(seed)]),('original_DeMo42',(original_rows,original))):
            ours,ours_values=reads['axis_shared_s'+str(seed)]
            assert all(all(a[k]==b[k] for k in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')) for a,b in zip(ours,reference_rows))
            delta={k:ours_values[k]-reference_values[k] for k in METRICS}
            rescued=sum(int(a['Rank-1'])==1 and int(b['Rank-1'])==0 for a,b in zip(ours,reference_rows))
            harmed=sum(int(a['Rank-1'])==0 and int(b['Rank-1'])==1 for a,b in zip(ours,reference_rows))
            assert abs(delta['Rank-1']-100*(rescued-harmed)/836)<1e-8
            differences.append(dict(seed=seed,reference=reference,**delta,rank1_rescued=rescued,rank1_harmed=harmed,both_plus2=delta['mAP']>=2 and delta['Rank-1']>=2))
            pairs.extend(dict(seed=seed,reference=reference,**{k:a[k] for k in ('query_index','name','identity','camera','scene')},delta_AP_pp=100*(float(a['AP'])-float(b['AP'])),delta_INP_pp=100*(float(a['INP'])-float(b['INP'])),delta_rank1=int(a['Rank-1'])-int(b['Rank-1'])) for a,b in zip(ours,reference_rows))
    assert len({(r['parameters'],r['trainable_parameters']) for r in training.values()})==1
    summaries=[]
    for label,rows in [(variant,[v for v in values if v['model']==variant]) for variant in ('frequency_shared','axis_shared')]+[(ref,[d for d in differences if d['reference']==ref]) for ref in ('frequency_shared','original_DeMo42')]:
        kind='value' if rows[0].get('model') else 'paired_axis_delta'
        summaries.append(dict(kind=kind,model_or_reference=label,N=5,**{suffix+k:fn([r[k] for r in rows]) for k in METRICS for suffix,fn in (('mean_',statistics.mean),('sample_sd_',statistics.stdev))}))
    best=[max([v for v in values if v['model']==variant],key=lambda v:v['mAP']) for variant in ('frequency_shared','axis_shared')]
    assert len(values)==10 and len(curves)==500 and len(pairs)==8360
    output=ROOT/'normal_seed_analysis';output.mkdir(exist_ok=False)
    for name,rows in (('per_seed_six',values),('mean_sample_sd',summaries),('paired_deltas',differences),('best_of_five',best),('group_metrics',groups),('training_curves',curves),('paired_query_changes',pairs)):
        table(output/(name+'.csv'),rows)
    result=dict(status='ACTUAL_K_RGBNT201_FIVE_EXPERT_SEEDS_NORMAL_CPU_COMPLETE',completed_at=datetime.now().isoformat(timespec='seconds'),seeds=[42,43,44,45,46],fixed_anchor_seed=42,new_models=8,new_epochs=400,seed42_reused=True,training=training,summary=summaries,best_of_five=best,paired_deltas=differences,new_neural_calls=0,
        limits='Conditional on one fixed original DeMo42 anchor; whole-pipeline repeatability untested. Official benchmark selects epochs and Best-of5, not untouched validation. Best-of5 and mean are distinct. No missing49/causal attribution/three-dataset success inferred.')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','seeds','fixed_anchor_seed','best_of_five')}),flush=True)


if __name__=='__main__':main()
