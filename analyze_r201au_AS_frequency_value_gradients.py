"""Read paired closed selected-AS training-gradient observations on CPU."""
import csv,json,statistics
from pathlib import Path

ROOT=Path('results/r201au_AS_frequency_value_gradients_20261014')


def main():
    controller=json.loads((ROOT/'controller_result.json').read_text());assert controller['status']=='COMPLETE' and controller['paired_names_labels']
    rows=[];proofs=[]
    for variant in ('frequency_shared','axis_shared'):
        name='RGBNT100_r201as_'+variant+'_s42';d=json.loads((ROOT/'full'/(name+'.json')).read_text())
        assert d['status']=='COMPLETE' and d['batches']==4 and d['optimizer_updates']==d['checkpoint_writes']==d['official_query_gallery_neural_uses']==0
        assert d['parameters_unchanged'] and d['frozen_identity_unchanged'];proofs.append(d)
        for batch in d['rows']:
            for target,v in batch['targets'].items():
                row=dict(model='AS_'+variant,selected_epoch=d['selected_epoch'],batch=batch['batch'],target=target,
                    selected_weight_norm=v['selected_weight_norm'],fresh_build_seed42_weight_norm=v['fresh_build_seed42_weight_norm'],L2_gradient_norm=v['L2_gradient_norm'],
                    L2_to_full_task_norm_ratio=v['L2_to_full_task_norm_ratio'],task_vs_L2_cosine=v['full_task_vs_L2']['cosine'],primary_vs_F_aux_cosine=v['primary_vs_F_aux']['cosine'],
                    full_task_gradient_nonzero_elements=v['full_task_gradient_nonzero_elements'],total_elements=v['total_elements'])
                row.update({k+'_gradient_norm':value for k,value in v['component_gradient_norm'].items()});rows.append(row)
    assert [(x['names'],x['training_labels']) for x in proofs[0]['rows']]==[(x['names'],x['training_labels']) for x in proofs[1]['rows']]
    out=ROOT/'CPU_analysis';out.mkdir(exist_ok=False)
    with (out/'selected_F_value_task_L2_gradients.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries=[]
    for variant in ('frequency_shared','axis_shared'):
        for target in proofs[0]['targets']:
            selected=[r for r in rows if r['model']=='AS_'+variant and r['target']==target]
            summaries.append(dict(model='AS_'+variant,target=target,batches=4,
                mean_full_task_gradient_norm=statistics.mean(r['full_total_gradient_norm'] for r in selected),
                L2_gradient_norm=selected[0]['L2_gradient_norm'],selected_weight_norm=selected[0]['selected_weight_norm'],fresh_build_seed42_weight_norm=selected[0]['fresh_build_seed42_weight_norm'],
                L2_larger_than_full_task_batches=sum(r['L2_gradient_norm']>r['full_total_gradient_norm'] for r in selected),
                full_task_shrinks_weight_batches=sum(r['task_vs_L2_cosine'] is not None and r['task_vs_L2_cosine']>0 for r in selected)))
    result=dict(status='ACTUAL_AU_AS_SELECTED_TRAIN_ONLY_F_VALUE_GRADIENTS_CPU_COMPLETE',paired_batches=4,summary=summaries,
        optimizer_updates=0,checkpoint_writes=0,new_retrieval_results=0,
        limits='Local full-objective gradient observations, not full training or a causal Adam replay. Positive task/L2 cosine means gradient has a radial shrink component under plain gradient descent, not guaranteed shrink under Adam moments. Compare local data pressure and current regularization magnitude; do not claim persistent conflict, erased identity evidence or gains.')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
