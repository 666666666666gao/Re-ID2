"""Read the closed per-dataset G pair, including its retained unit controls."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

from analyze_identity_coordinate_three_normal import COUNTS,METRICS,load,query_rows,table

PROJECT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=('MSVR310','RGBNT100'),required=True)
    dataset=parser.parse_args().dataset
    intake=load(PROJECT/'results/preflight'/('r201g_'+dataset+'_normal_terminal_text_intake_20261006.json'))
    assert intake['status']=='ACTUAL_G_DATASET_TWO_FULL50_NORMAL_GT_TEXT_VERIFIED' and intake['dataset']==dataset
    root=PROJECT/'results/r201g_other_two_normal_20261006'/dataset
    train,count,gallery,steps=COUNTS[dataset]
    folders={'original_DeMo':PROJECT/'results/full_official_baselines_20261004/training'/(dataset+'_demo_s42')}
    for variant in ('frequency_shared','axis_shared'):
        folders['unit_'+variant]=PROJECT/'results/identity_coordinate_three_normal_completed_20261005/training'/(dataset+'_identity_'+variant+'_narrow_s42')
        folders['normal_'+variant]=root/'training'/(dataset+'_r201g_'+variant+'_s42')
    metrics,groups,comparisons,pairs,training,reads,orders=[],[],[],[],{},{},{}
    for model,folder in folders.items():
        data=load(folder/'result.json')
        assert data['status']=='COMPLETE' and data['epochs']==50 and data['steps']==data['optimizer_steps']==steps and data['amp_skipped_steps']==0
        assert (data['train_records'],data['query_records'],data['gallery_records'])==(train,count,gallery)
        assert data['training_coverage']==dict(eligible=train,visited=train,unvisited=[])
        assert data['training_heldout_identities']==0 and data['descriptor_dim']==5120
        rows,summary=query_rows(folder,dataset)
        epochs=list(csv.DictReader((folder/'epochs.csv').open(encoding='utf-8')))
        assert [int(row['epoch']) for row in epochs]==list(range(1,51))
        selected=max(epochs,key=lambda row:float(row['mAP']))
        assert int(selected['epoch'])==data['best']['epoch'] and all(abs(float(selected[key])-summary[key])<1e-8 and abs(data['full_metrics'][key]-summary[key])<1e-8 for key in METRICS)
        metrics.append(dict(dataset=dataset,model=model,selected_epoch=data['best']['epoch'],**{key:summary[key] for key in METRICS}))
        groups.extend(dict(dataset=dataset,model=model,grouping=grouping,**value) for grouping,values in summary['groups'].items() for value in values)
        training[model]=dict(selected_epoch=data['best']['epoch'],parameters=data['parameters'],trainable_parameters=data['trainable_parameters'],descriptor_dim=5120,
            budget_epochs=50 if model=='original_DeMo' else 100,final_epoch={key:float(epochs[-1][key]) for key in METRICS},runtime=data['runtime'],peak_memory=data['peak_memory'])
        reads[model]=(rows,summary)
        if model!='original_DeMo':
            batches=[json.loads(line) for line in (folder/'batch_orders.jsonl').read_text(encoding='utf-8').splitlines()]
            assert len(batches)==steps and all(row['optimizer_updated'] for row in batches)
            orders[model]=[(row['epoch'],row['step'],row['names'],row['partial_set']) for row in batches]
            if model.startswith('normal_'):
                assert all(row['loss']==row['full_loss'] and row['partial_CE_weight']==row['partial_triplet_weight']==row['partial_loss_effective']==0 for row in batches)
                audit=load(folder/'normal_cpu_audit.json')
                assert audit['status']=='PASS' and audit['max_metric_error']<1e-8 and audit['CMC50_and_per_query_and_groups'] and audit['full_training_coverage']
                assert load(folder/'normal_local_archive.json')['status']=='NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
    assert all(value==orders['normal_axis_shared'] for value in orders.values())
    assert len({(value['parameters'],value['trainable_parameters']) for model,value in training.items() if model!='original_DeMo'})==1
    for improved,reference in (('normal_axis_shared','original_DeMo'),('normal_frequency_shared','original_DeMo'),('normal_axis_shared','normal_frequency_shared'),('normal_axis_shared','unit_axis_shared'),('normal_frequency_shared','unit_frequency_shared')):
        arows,a=reads[improved];brows,b=reads[reference]
        assert all(all(x[key]==y[key] for key in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')) for x,y in zip(arows,brows))
        delta={key:a[key]-b[key] for key in METRICS}
        comparisons.append(dict(dataset=dataset,improved=improved,reference=reference,**delta,both_plus2=delta['mAP']>=2 and delta['Rank-1']>=2,
            rank1_rescued=sum(int(x['Rank-1'])==1 and int(y['Rank-1'])==0 for x,y in zip(arows,brows)),rank1_harmed=sum(int(x['Rank-1'])==0 and int(y['Rank-1'])==1 for x,y in zip(arows,brows))))
        pairs.extend(dict(dataset=dataset,improved=improved,reference=reference,**{key:x[key] for key in ('query_index','name','identity','camera','scene')},
            delta_AP_pp=100*(float(x['AP'])-float(y['AP'])),delta_INP_pp=100*(float(x['INP'])-float(y['INP'])),delta_rank1=int(x['Rank-1'])-int(y['Rank-1'])) for x,y in zip(arows,brows))
    assert len(metrics)==len(comparisons)==5 and len(pairs)==5*count
    output=root/'normal_analysis';output.mkdir(exist_ok=False)
    for name,rows in (('six_metrics',metrics),('group_metrics',groups),('comparisons',comparisons),('paired_query_changes',pairs)):
        table(output/(name+'.csv'),rows)
    result=dict(status='ACTUAL_G_DATASET_TWO_FULL50_NORMAL_CPU_READOUT',completed_at=datetime.now().isoformat(timespec='seconds'),dataset=dataset,model_results=5,
        comparisons=comparisons,paired_query_rows=len(pairs),training=training,official_GT_six_CMC50_perquery_groups=True,new_neural_calls=0,new_optimizer_updates=0,
        limits='One dataset/seed42, officialbenchmark mAP-best/earliesttie. Original50 versus original50+new50, matched normal and unit expert budgets/params/dims/sampling. Not all3, missing49, multiseed or retraining ablations. NormalmAP-best may trade off Rank1. Historical frequency_shared is mixed-source ordinary-expert control.')
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],dataset=dataset,comparisons=comparisons,paired_query_rows=len(pairs))))


if __name__=='__main__':main()
