"""Saved full-normal MSVR310 evidence: public/private complement and AE retention."""
from collections import Counter
import csv,json
from pathlib import Path
from analyze_identity_coordinate_three_normal import METRICS,load,query_rows,six,table

PROJECT=Path(__file__).resolve().parent


def main():
    blocks=PROJECT/'results/msvr_reference_blocks_p0_20261009'
    old=load(blocks/'result.json')
    assert old['status']=='COMPLETE_CPU_MSVR310_SELECTED_BASELINE_BLOCK_DIAGNOSIS'
    assert old['full_split_counts']==dict(train=1032,query=591,gallery=1055)
    assert old['neural_inference_calls']==old['optimizer_updates']==0
    ae=PROJECT/'results/r201ae_MS_strong_staged_20261012/MSVR310'
    manifest=load(PROJECT/'results/full_official_baselines_20261004/training/MSVR310_demo_s42/official_split_manifest.json')
    expected=[(r['name'],str(r['identity']),str(r['camera']),str(r['scene'])) for r in manifest['query']]
    assert len(expected)==591
    reads={};summaries={};metrics=[];inputs=[]
    for name in ('original_DeMo','shared_private_only','shared_public_only','shared_fused'):
        path=blocks/(name+'.csv')
        rows=list(csv.DictReader(path.open(encoding='utf-8',newline='')))
        summary=load(blocks/(name+'.json'))
        assert len(rows)==591 and all(r['valid']=='True' for r in rows)
        assert [(r['name'],r['identity'],r['camera'],r['scene']) for r in rows]==expected
        assert all(abs(six(rows)[k]-summary[k])<1e-8 and abs(summary[k]-old['measurements'][name][k])<1e-8 for k in METRICS)
        reads[name]=rows;summaries[name]=summary;inputs.append(str(path.relative_to(PROJECT)))
        dimension={'original_DeMo':5120,'shared_private_only':5120,'shared_public_only':512,'shared_fused':5632}[name]
        epoch=old['checkpoints']['demo' if name=='original_DeMo' else 'demo_shared']
        metrics.append(dict(model=name,dimension=dimension,selected_epoch=epoch,**{k:summary[k] for k in METRICS}))
    for control in ('axis_frozen','axis_staged','frequency_staged','demo_staged'):
        folder=ae/'training'/('MSVR310_r201ae_'+control+'_s42')
        result=load(folder/'result.json');rows,summary=query_rows(folder,'MSVR310')
        assert result['status']=='COMPLETE' and result['epochs']==50 and result['steps']==result['optimizer_steps']==705
        assert result['training_heldout_identities']==0 and result['amp_skipped_steps']==0
        assert load(folder/'normal_cpu_audit.json')['status']=='PASS'
        assert [(r['name'],r['identity'],r['camera'],r['scene']) for r in rows]==expected
        assert all(abs(summary[k]-result['full_metrics'][k])<1e-8 for k in METRICS)
        name='AE_'+control;reads[name]=rows;summaries[name]=summary
        inputs.append(str((folder/'best_per_query.csv').relative_to(PROJECT)))
        metrics.append(dict(model=name,dimension=5120,selected_epoch=result['best']['epoch'],**{k:summary[k] for k in METRICS}))
    correct=lambda name:{i for i,r in enumerate(reads[name]) if int(r['Rank-1'])==1}
    original,private,public,fused=(correct(n) for n in ('original_DeMo','shared_private_only','shared_public_only','shared_fused'))
    universe=set(range(591))
    cohorts=dict(public_block_rescues=fused-private,public_block_harms=private-fused,
        fused_correct_both_blocks_wrong=fused-private-public,
        private_public_both_correct=(private&public),
        strong_fused_remaining_errors=universe-fused,
        strong_fused_rescues_original=fused-original,strong_fused_harms_original=original-fused)
    def cohort_stats(indices):
        return dict(queries=len(indices),identity_counts=dict(Counter(expected[i][1] for i in sorted(indices))),
            camera_counts=dict(Counter(expected[i][2] for i in sorted(indices))),scene_counts=dict(Counter(expected[i][3] for i in sorted(indices))),
            model_correct={name:len(indices&correct(name)) for name in reads},
            mean_AP_pp={name:100*sum(float(rows[i]['AP']) for i in indices)/len(indices) for name,rows in reads.items()} if indices else {})
    comparisons=[]
    for name in ('shared_fused','AE_axis_frozen','AE_axis_staged','AE_frequency_staged','AE_demo_staged'):
        for reference in ('original_DeMo','shared_private_only','shared_fused'):
            if name==reference:continue
            rescue=correct(name)-correct(reference);harm=correct(reference)-correct(name)
            delta={k:summaries[name][k]-summaries[reference][k] for k in METRICS}
            assert abs(delta['Rank-1']-100*(len(rescue)-len(harm))/591)<1e-8
            comparisons.append(dict(model=name,reference=reference,**delta,rank1_rescued=len(rescue),rank1_harmed=len(harm)))
    flags=[]
    for i,(name,identity,camera,scene) in enumerate(expected):
        row=dict(query_index=i,name=name,identity=identity,camera=camera,scene=scene)
        row.update({label+'_correct':int(i in correct(label)) for label in reads})
        row.update({label+'_AP':float(values[i]['AP']) for label,values in reads.items()})
        row.update({label:int(i in indices) for label,indices in cohorts.items()});flags.append(row)
    assert len(metrics)==8 and len(flags)==591
    out=PROJECT/'results/msvr_strong_full_retention_cohorts_CPU_20261012';out.mkdir(exist_ok=False)
    for name,values in [('six_metrics',metrics),('comparisons',comparisons),('query_cohort_flags',flags)]:table(out/(name+'.csv'),values)
    result=dict(status='ACTUAL_CLOSED_MSVR_FULL_STRONG_BLOCK_AE_QUERY_COHORTS_CPU_COMPLETE',dataset='MSVR310',
        train_records=1032,query_records=591,gallery_records=1055,model_results=8,inputs=inputs,metrics=metrics,
        comparisons=comparisons,cohorts={name:cohort_stats(indices) for name,indices in cohorts.items()},
        new_neural_calls=0,new_optimizer_updates=0,new_weights=0,new49=False,missing_evaluated=False,
        limits='Saved normal GT reports only, no new installed-GT recount or neural inference. Block comparison uses same fixed strong teacher37; AE selected additional50 private5120 controls differ in budget/heads. Cohort membership uses true labels only offline and is not a deployable selector or a new model result. Benchmark used for checkpoint selection; no seed/generalization guarantee or automatic claim that retaining public block will yield expert gain.')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],cohorts=result['cohorts']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
