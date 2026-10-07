"""Interim K42/K43 normal readout. Does not complete or select the fixed N5 queue."""
import csv
from datetime import datetime
import hashlib,json,math
from pathlib import Path
from analyze_identity_coordinate_three_normal import METRICS,load,query_rows,table

PROJECT=Path(__file__).resolve().parent
ROOT=PROJECT/'results/r201k_expert_N5_20261007'
OUTPUT=ROOT/'first_two_seed_interim'


def main():
    assert not OUTPUT.exists()
    receipt=load(PROJECT/'results/preflight/r201k_N5_expert_s43_actual_20261007.json')
    assert receipt['status']=='COMPLETE' and receipt['expert_seed']==43 and receipt['updates']==5292
    assert receipt['additional_epochs']==100 and len(receipt['archives'])==2
    for record in receipt['archives'].values():
        path=Path(record['local']);assert path.stat().st_size==record['file']['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['file']['sha256']
    baseline=PROJECT/'results/full_official_baselines_20261004/training/RGBNT201_demo_s42'
    baseline_rows,baseline_summary=query_rows(baseline,'RGBNT201')
    values=[dict(model='original_DeMo_s42',expert_seed=42,selected_epoch=load(baseline/'result.json')['best']['epoch'],**{k:baseline_summary[k] for k in METRICS})]
    differences=[];changes=[];curves=[];groups=[];training=[];inputs={}
    for seed in (42,43):
        base=PROJECT/'results/r201k_relation_local_pi_20261007/RGBNT201' if seed==42 else ROOT/'s43'
        controller=load(base/'controller_result.json')
        assert controller['status']=='COMPLETE' and controller['controls']==2 and controller['additional_epochs']==100
        assert controller['native_updates']==6 and controller['paired_sampling_exact'] and controller['normal_archives_local_verified']==2
        reads={};orders={};params=set();steps=0
        for variant in ('frequency_shared','axis_shared'):
            folder=base/'training'/('RGBNT201_r201k_'+variant+'_s'+str(seed));data=load(folder/'result.json')
            assert data['status']=='COMPLETE' and data['epochs']==50 and data['arguments']['seed']==seed
            assert data['steps']==data['optimizer_steps'] and data['amp_skipped_steps']==0
            assert (data['train_records'],data['query_records'],data['gallery_records'])==(3951,836,836)
            assert data['training_coverage']==dict(eligible=3951,visited=3951,unvisited=[]) and data['training_heldout_identities']==0
            assert data['descriptor_dim']==5120 and data['anchor']['original_run'].endswith('/RGBNT201_demo_s42')
            if seed==43:assert data['anchor']['anchor_seed']==42 and data['anchor']['expert_seed']==43
            audit=load(folder/'normal_cpu_audit.json');assert audit['status']=='PASS' and audit['max_metric_error']<1e-8
            assert audit['full_training_coverage'] and audit['CMC50_and_per_query_and_groups']
            assert load(folder/'normal_local_archive.json')['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
            rows,summary=query_rows(folder,'RGBNT201')
            epochs=list(csv.DictReader((folder/'epochs.csv').open(encoding='utf-8',newline='')))
            assert [int(e['epoch']) for e in epochs]==list(range(1,51))
            best=max(epochs,key=lambda e:float(e['mAP']));assert int(best['epoch'])==data['best']['epoch']
            assert all(abs(float(best[k])-summary[k])<1e-8 and abs(data['full_metrics'][k]-summary[k])<1e-8 for k in METRICS)
            order=[]
            with (folder/'batch_orders.jsonl').open(encoding='utf-8') as handle:
                for line in handle:
                    batch=json.loads(line);assert batch['optimizer_updated'] and len(batch['names'])==64
                    order.append((batch['epoch'],batch['step'],batch['names'],batch['partial_set']))
            assert len(order)==data['steps'];orders[variant]=order;steps+=data['steps'];params.add((data['parameters'],data['trainable_parameters']))
            name='K_'+variant+'_s'+str(seed);reads[variant]=(rows,summary,name)
            values.append(dict(model=name,expert_seed=seed,selected_epoch=data['best']['epoch'],**{k:summary[k] for k in METRICS}))
            curves.extend(dict(model=name,expert_seed=seed,epoch=int(e['epoch']),**{k:float(e[k]) for k in ('loss',*METRICS)}) for e in epochs)
            training.append(dict(model=name,steps=data['steps'],parameters=data['parameters'],trainable_parameters=data['trainable_parameters'],
                full_training_coverage=True,normal_GT_pass=True,started=data['started'],finished=data['finished'],
                epoch50={k:float(epochs[-1][k]) for k in METRICS},mAP_peak_to_epoch50=summary['mAP']-float(epochs[-1]['mAP'])))
            for axis,items in summary['groups'].items():groups.extend(dict(model=name,grouping=axis,**r) for r in items)
            for filename in ('result.json','epochs.csv','batch_orders.jsonl','normal_cpu_audit.json','normal_local_archive.json','best_per_query.json','best_per_query.csv','official_split_manifest.json'):
                path=folder/filename;inputs[path.relative_to(PROJECT).as_posix()]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        assert len(params)==1 and orders['frequency_shared']==orders['axis_shared'] and steps==controller['successful_updates']
        for improved,reference in (('axis_shared','original_DeMo'),('axis_shared','frequency_shared'),('frequency_shared','original_DeMo')):
            new_rows,new,name=reads[improved]
            old_rows,old,old_name=(baseline_rows,baseline_summary,'original_DeMo_s42') if reference=='original_DeMo' else reads[reference]
            rescued=harmed=better=worse=0
            for a,b in zip(new_rows,old_rows):
                assert all(a[k]==b[k] for k in ('query_index','name','identity','camera','scene'))
                rescue=int(a['Rank-1'])>int(b['Rank-1']);harm=int(a['Rank-1'])<int(b['Rank-1'])
                rescued+=rescue;harmed+=harm;delta=100*(float(a['AP'])-float(b['AP']));better+=delta>1e-10;worse+=delta < -1e-10
                changes.append(dict(expert_seed=seed,improved=name,reference=old_name,query_index=a['query_index'],name=a['name'],identity=a['identity'],camera=a['camera'],scene=a['scene'],AP_delta_pp=delta,rescued=rescue,harmed=harm))
            differences.append(dict(expert_seed=seed,improved=name,reference=old_name,**{k:new[k]-old[k] for k in METRICS},rescued=rescued,harmed=harmed,AP_better=better,AP_worse=worse,both_plus2=(new['mAP']-old['mAP']>=2 and new['Rank-1']-old['Rank-1']>=2)))
    assert len(values)==5 and len(curves)==200 and len(differences)==6 and len(changes)==5016
    OUTPUT.mkdir()
    for name,rows in (('six_metrics.csv',values),('paired_deltas.csv',differences),('paired_query_changes.csv',changes),('training_curves.csv',curves),('group_metrics.csv',groups)):
        table(OUTPUT/name,rows)
    result=dict(status='ACTUAL_K_FIRST_TWO_EXPERT_SEEDS_NORMAL_CPU_INTERIM_COMPLETE_N5_PENDING',completed_at=datetime.now().isoformat(timespec='seconds'),
        completed_expert_seeds=[42,43],declared_expert_seeds=[42,43,44,45,46],fixed_anchor_seed=42,reused_models=2,new_complete_models=2,new_complete_epochs=100,new_complete_updates=5292,
        curve_rows=200,paired_query_rows=5016,all_six_CMC50_and_query_groups=True,inputs=inputs,training=training,comparisons=differences,new_neural_calls=0,new_optimizer_updates=0,
        limits='Interim only two completed expert seeds, fixed DeMo42 teacher; not completed N5, full-pipeline repeats, best-of5 deployment selection or final variance. All curves and negative metrics preserved, maximum normal benchmark mAP earliest epoch tie selected; benchmark participates in selection. No additional missing NN for seed43, no measured seed43 00/10/01 state inference; inherited freeze invariants do not substitute for final winner verification. Existing K42 all3/all49 results unchanged; no whole-goal completion.')
    (OUTPUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','completed_at','completed_expert_seeds','new_complete_updates','comparisons')},ensure_ascii=False))


if __name__=='__main__':main()
