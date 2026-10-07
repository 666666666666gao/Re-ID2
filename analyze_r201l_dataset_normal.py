"""CPU reduction of uniform-K8 L versus original DeMo and unchanged-graph K."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from analyze_identity_coordinate_three_normal import COUNTS,METRICS,load,query_rows,table

PROJECT=Path(__file__).resolve().parent
SAMPLER_K={'RGBNT201':8,'MSVR310':4,'RGBNT100':16}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=tuple(COUNTS),required=True)
    dataset=parser.parse_args().dataset;pf=PROJECT/'results/preflight'
    review=load(pf/'r201l_uniform_K8_source_review_20261007.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    assert {'results/preflight/r201l_dataset_normal_complete_intake_20261007.py','analyze_r201l_dataset_normal.py'} <= set(review['sources_sha256'])
    assert all(hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==sha for name,sha in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
    intake=load(pf/('r201l_'+dataset+'_normal_terminal_text_intake_20261007.json'))
    assert intake['status']=='ACTUAL_L_DATASET_TWO_FULL50_NORMAL_GT_TEXT_VERIFIED' and intake['dataset']==dataset
    root=PROJECT/'results/r201l_uniform_k8_20261007'/dataset
    controller=load(root/'controller_result.json');train,count,gallery,old_steps=COUNTS[dataset]
    steps=load(pf/'r201l_uniform_K8_sampler_cpu_replay_20261007.json')['datasets'][dataset]['8']['total_steps']
    assert controller['status']=='COMPLETE' and controller['dataset']==dataset and controller['controls']==2
    assert controller['successful_updates']==2*steps and controller['additional_epochs']==100 and controller['native_updates']==6
    assert controller['paired_sampling_exact'] and controller['normal_archives_local_verified']==2
    groot=PROJECT/'results/r201k_relation_local_pi_20261007'/dataset
    folders={'original_DeMo':PROJECT/'results/full_official_baselines_20261004/training'/(dataset+'_demo_s42')}
    for variant in ('frequency_shared','axis_shared'):
        folders['K_'+variant]=groot/'training'/(dataset+'_r201k_'+variant+'_s42')
        folders['L_'+variant]=root/'training'/(dataset+'_r201l_'+variant+'_s42')
    metrics,groups,comparisons,pairs,curves,training,reads,orders=[],[],[],[],[],{},{},{}
    for model,folder in folders.items():
        data=load(folder/'result.json')
        assert data['status']=='COMPLETE' and data['epochs']==50 and data['steps']==data['optimizer_steps']==(steps if model.startswith('L_') else old_steps) and data['amp_skipped_steps']==0
        assert (data['train_records'],data['query_records'],data['gallery_records'])==(train,count,gallery)
        assert data['training_coverage']==dict(eligible=train,visited=train,unvisited=[])
        assert data['training_heldout_identities']==0 and data['descriptor_dim']==5120
        assert '  NUM_INSTANCE: '+str(8 if model.startswith('L_') else SAMPLER_K[dataset]) in data['config'].splitlines()
        rows,summary=query_rows(folder,dataset)
        epochs=list(csv.DictReader((folder/'epochs.csv').open(encoding='utf-8',newline='')))
        assert [int(row['epoch']) for row in epochs]==list(range(1,51))
        selected=max(epochs,key=lambda row:float(row['mAP']))
        assert int(selected['epoch'])==data['best']['epoch']
        assert all(abs(float(selected[key])-summary[key])<1e-8 and abs(data['full_metrics'][key]-summary[key])<1e-8 for key in METRICS)
        metrics.append(dict(dataset=dataset,model=model,selected_epoch=data['best']['epoch'],**{key:summary[key] for key in METRICS}))
        groups.extend(dict(dataset=dataset,model=model,grouping=axis,**value) for axis,values in summary['groups'].items() for value in values)
        training[model]=dict(selected_epoch=data['best']['epoch'],parameters=data['parameters'],trainable_parameters=data['trainable_parameters'],descriptor_dim=5120,
            budget_epochs=50 if model=='original_DeMo' else 100,final_epoch={key:float(epochs[-1][key]) for key in METRICS},runtime=data['runtime'],peak_memory=data['peak_memory'])
        reads[model]=rows,summary
        if model!='original_DeMo':
            batches=[json.loads(line) for line in (folder/'batch_orders.jsonl').read_text(encoding='utf-8').splitlines()]
            assert len(batches)==(steps if model.startswith('L_') else old_steps) and all(row['optimizer_updated'] for row in batches)
            orders[model]=[(row['epoch'],row['step'],row['names'],row['partial_set']) for row in batches]
            assert all(row['loss']==row['full_loss'] and row['partial_CE_weight']==row['partial_triplet_weight']==row['partial_loss_effective']==0 for row in batches)
            audit=load(folder/'normal_cpu_audit.json')
            assert audit['status']=='PASS' and audit['max_metric_error']<1e-8 and audit['CMC50_and_per_query_and_groups'] and audit['full_training_coverage']
            assert load(folder/'normal_local_archive.json')['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
            if model.startswith('L_'):
                identities={r['name']:r['identity'] for r in load(folder/'official_split_manifest.json')['train']}
                assert all(row['sampling_identities']==row['sampling_instances']==8 and row['sampling_batch']==64
                    and len(Counter(identities[n] for n in row['names']))==8
                    and set(Counter(identities[n] for n in row['names']).values())=={8} for row in batches)
                assert all(row['primary_full_metric']=='unit5120_margin03_batch_hard_triplet' and row['primary_margin']==.3 and row['primary_metric_coefficient']==1. and row['auxiliary_original_soft_triplet'] and row['PI_outlet']=='shared_relation_local' and 0<=row['primary_margin_violations']<=64 for row in batches)
                for epoch in epochs:
                    members=[row for row in batches if row['epoch']==int(epoch['epoch'])]
                    means={key:math.fsum(row[key] for row in members)/len(members) for key in ('loss','full_loss','primary_triplet_loss','primary_margin_violations','primary_positive_distance_mean','primary_negative_distance_mean','partial_ce','cross_triplet','partial_loss_effective')}
                    assert abs(means['loss']-float(epoch['loss']))<1e-8 and means['loss']==means['full_loss'] and means['partial_loss_effective']==0
                    curves.append(dict(dataset=dataset,model=model,epoch=int(epoch['epoch']),batches=len(members),**means,**{key:float(epoch[key]) for key in METRICS}))
            else:
                assert all(row['primary_full_metric']=='unit5120_margin03_batch_hard_triplet' and row['auxiliary_original_soft_triplet'] for row in batches)
    assert orders['L_axis_shared']==orders['L_frequency_shared'] and orders['K_axis_shared']==orders['K_frequency_shared']
    assert orders['L_axis_shared']!=orders['K_axis_shared']
    assert len({(v['parameters'],v['trainable_parameters']) for model,v in training.items() if model!='original_DeMo'})==1
    for improved,reference in (('L_axis_shared','original_DeMo'),('L_frequency_shared','original_DeMo'),('L_axis_shared','L_frequency_shared'),('L_axis_shared','K_axis_shared'),('L_frequency_shared','K_frequency_shared'),('L_axis_shared','K_frequency_shared')):
        arows,a=reads[improved];brows,b=reads[reference]
        assert all(all(x[key]==y[key] for key in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')) for x,y in zip(arows,brows))
        delta={key:a[key]-b[key] for key in METRICS}
        rescue=sum(int(x['Rank-1'])==1 and int(y['Rank-1'])==0 for x,y in zip(arows,brows));harm=sum(int(x['Rank-1'])==0 and int(y['Rank-1'])==1 for x,y in zip(arows,brows))
        assert abs(delta['Rank-1']-100*(rescue-harm)/count)<1e-8
        comparisons.append(dict(dataset=dataset,improved=improved,reference=reference,**delta,both_plus2=delta['mAP']>=2 and delta['Rank-1']>=2,
            rank1_rescued=rescue,rank1_harmed=harm,AP_improved=sum(float(x['AP'])>float(y['AP']) for x,y in zip(arows,brows)),AP_worsened=sum(float(x['AP'])<float(y['AP']) for x,y in zip(arows,brows))))
        pairs.extend(dict(dataset=dataset,improved=improved,reference=reference,**{key:x[key] for key in ('query_index','name','identity','camera','scene')},delta_AP_pp=100*(float(x['AP'])-float(y['AP'])),delta_INP_pp=100*(float(x['INP'])-float(y['INP'])),delta_rank1=int(x['Rank-1'])-int(y['Rank-1'])) for x,y in zip(arows,brows))
    assert len(metrics)==5 and len(comparisons)==6 and len(pairs)==6*count and len(curves)==100
    output=root/'normal_analysis';output.mkdir(exist_ok=False)
    for name,rows in (('six_metrics',metrics),('group_metrics',groups),('comparisons',comparisons),('paired_query_changes',pairs),('training_curves',curves)):table(output/(name+'.csv'),rows)
    result=dict(status='ACTUAL_L_DATASET_TWO_FULL50_NORMAL_CPU_READOUT',completed_at=datetime.now().isoformat(timespec='seconds'),dataset=dataset,model_results=5,comparisons=comparisons,paired_query_rows=len(pairs),new_epoch_rows=100,new_stage_formal_updates=2*steps,training=training,
        original_both_plus2=next(r['both_plus2'] for r in comparisons if r['improved']=='L_axis_shared' and r['reference']=='original_DeMo'),official_GT_six_CMC50_perquery_groups=True,paired_L_batch_orders_and_partial_sets=True,parent_K_orders_differ=True,new_neural_calls=0,new_optimizer_updates=0,
        limits='One weak dataset/seed42; L only NUM_INSTANCE8 versus K per-dataset4/16, both positive multiplicity and negative identity pool and updates differ. All50 epochs, fullofficial/benchmark-selected earliestmAP/same6/all49later. Frozen same originalDeMo42 teacher, architecture/loss/params/5120 unchanged; ordinary/axis L matched actualupdate/order budget. Original50 versus two-stage100 disclosed, historical frequency name is mixed-source ordinary. CPU checks actual GT audits; no guaranteed +2/causality/novelty/all3/missing/wholepipeline repeats.')
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','dataset','model_results','paired_query_rows','original_both_plus2')}),flush=True)


if __name__=='__main__':main()
