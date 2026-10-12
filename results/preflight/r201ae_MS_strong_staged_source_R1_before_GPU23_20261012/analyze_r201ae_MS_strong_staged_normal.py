"""Reduce AE normal results, baseline attribution, stage curves and expert net gain."""
from pathlib import Path
from datetime import datetime
from collections import Counter
import csv,hashlib,json,math
from analyze_identity_coordinate_three_normal import METRICS,load,query_rows,table,six

PROJECT=Path(__file__).resolve().parent
CONTROLS=('axis_frozen','axis_staged','frequency_staged','demo_staged')


def read_rows(prefix):
    summary=load(prefix.with_suffix('.json'))
    rows=list(csv.DictReader(prefix.with_suffix('.csv').open(encoding='utf-8',newline='')))
    assert len(rows)==591 and summary['query_count']==summary['valid_queries']==591
    assert all(abs(six(rows)[key]-summary[key])<1e-8 for key in METRICS)
    assert len(summary['CMC_1_to_50'])==50 and summary['gallery_count']==1055
    return rows,summary


def main():
    pf=PROJECT/'results/preflight';review=load(pf/'r201ae_MS_strong_staged_source_review_20261012.json')
    assert review['status']=='PASS' and not review['blocking_findings']
    assert all(hashlib.sha256((PROJECT/n).read_bytes()).hexdigest()==sha for n,sha in (review['sources_sha256']|review['directly_reused_sources_sha256']).items())
    intake=load(pf/'r201ae_MS_strong_staged_terminal_text_intake_20261012.json');assert intake['status']=='ACTUAL_AE_FOUR_FULL50_NORMAL_STATES_TEXT_VERIFIED'
    root=Path(intake['local_root']);controller=load(root/'controller_result.json')
    assert controller['status']=='COMPLETE' and controller['controls']==4 and controller['successful_updates']==2820
    folders={name:PROJECT/'results/full_official_baselines_20261004/training'/('MSVR310_'+variant+'_s42')
             for name,variant in (('original_DeMo','demo'),('strong_full5632','demo_shared'))}
    folders.update({name:root/'training'/('MSVR310_r201ae_'+name+'_s42') for name in CONTROLS})
    metrics,groups,curves,training,orders,reads=[],[],[],{}, {},{}
    for name,folder in folders.items():
        d=load(folder/'result.json');rows,summary=query_rows(folder,'MSVR310')
        assert d['status']=='COMPLETE' and d['epochs']==50 and d['amp_skipped_steps']==d['training_heldout_identities']==0
        assert d['training_coverage']==dict(eligible=1032,visited=1032,unvisited=[])
        assert all(abs(d['full_metrics'][key]-summary[key])<1e-8 for key in METRICS)
        epochs=list(csv.DictReader((folder/'epochs.csv').open(encoding='utf-8',newline='')))
        assert [int(row['epoch']) for row in epochs]==list(range(1,51))
        selected=max(epochs,key=lambda row:float(row['mAP']));assert int(selected['epoch'])==d['best']['epoch']
        assert all(abs(float(selected[key])-summary[key])<1e-8 for key in METRICS)
        metrics.append(dict(model=name,descriptor_dim=d['descriptor_dim'],selected_epoch=d['best']['epoch'],**{key:summary[key] for key in METRICS}))
        reads[name]=rows,summary
        groups.extend(dict(model=name,grouping=axis,**group) for axis,values in summary['groups'].items() for group in values)
        training[name]=dict(parameters=d['parameters'],initial_trainable_parameters=d['trainable_parameters'],descriptor_dim=d['descriptor_dim'],
            budget_epochs=100 if name in CONTROLS else 50,selected_epoch=d['best']['epoch'],last={key:float(epochs[-1][key]) for key in METRICS},runtime=d['runtime'],peak_memory=d['peak_memory'])
        if name not in CONTROLS:continue
        assert d['steps']==d['optimizer_steps']==705 and d['descriptor_dim']==5120
        assert load(folder/'normal_cpu_audit.json')['status']=='PASS'
        assert load(folder/'normal_local_archive.json')['status']=='LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
        batch=[json.loads(line) for line in (folder/'batch_orders.jsonl').read_text().splitlines()]
        assert len(batch)==705 and all(row['optimizer_updated'] for row in batch)
        identities={r['name']:r['identity'] for r in load(folder/'official_split_manifest.json')['train']}
        for row in batch:
            assert len(row['names'])==64 and len(Counter(identities[n] for n in row['names']))==16
            assert set(Counter(identities[n] for n in row['names']).values())=={4}
            expected='relations' if name!='axis_frozen' and row['epoch']>5 else 'outlets'
            assert row['adaptation_phase']==expected and row['generalFusion_effective_LR_factor']==(.1 if expected=='relations' else 0.)
            assert row['loss']==row['full_loss'] and row['partial_loss_effective']==row['partial_CE_weight']==row['partial_triplet_weight']==0
        orders[name]=[(r['epoch'],r['step'],r['names'],r['partial_set']) for r in batch]
        for epoch in epochs:
            members=[r for r in batch if r['epoch']==int(epoch['epoch'])]
            assert abs(math.fsum(r['loss'] for r in members)/len(members)-float(epoch['loss']))<1e-8
            curves.append(dict(model=name,epoch=int(epoch['epoch']),phase=members[0]['adaptation_phase'],batches=len(members),loss=float(epoch['loss']),
                primary_triplet_loss=math.fsum(r['primary_triplet_loss'] for r in members)/len(members),**{key:float(epoch[key]) for key in METRICS}))
        training[name]['phase_trainable_parameters']=d['trainable_parameters_by_phase']
    assert all(orders[name]==orders['axis_frozen'] for name in CONTROLS)
    assert training['axis_staged']['parameters']==training['frequency_staged']['parameters']
    assert training['axis_staged']['phase_trainable_parameters']==training['frequency_staged']['phase_trainable_parameters']
    prefix=PROJECT/'results/msvr_stronger_private_reference_20261012/evaluation/cached_private_reference'
    reads['strong_private5120']=read_rows(prefix)
    metrics.append(dict(model='strong_private5120',descriptor_dim=5120,selected_epoch=37,**{key:reads['strong_private5120'][1][key] for key in METRICS}))
    state_metrics=[]
    for name in CONTROLS:
        directory=root/'states'/('MSVR310_r201ae_'+name+'_s42');d=load(directory/'result.json')
        assert d['status']=='COMPLETE' and d['full11_six_metrics_equal_selected_normal']
        assert load(directory/'states_local_archive.json')['status']=='LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
        for state in ('00','10','01','11'):
            pair=read_rows(directory/('state_'+state));reads[name+'/'+state]=pair
            assert all(abs(pair[1][key]-d['measurements'][state][key])<1e-8 for key in METRICS)
            state_metrics.append(dict(model=name,state=state,**{key:pair[1][key] for key in METRICS}))
        assert all(abs(reads[name+'/11'][1][key]-reads[name][1][key])<1e-8 for key in METRICS)
    assert all(reads['axis_frozen/00'][0][i]==reads['strong_private5120'][0][i] for i in range(591))
    comparisons=[];pairs=[]
    targets=[(name,reference) for name in CONTROLS for reference in ('original_DeMo','strong_private5120','strong_full5632')]
    targets += [('axis_staged',reference) for reference in ('axis_frozen','frequency_staged','demo_staged')]
    targets += [(name+'/'+state,name+'/00') for name in CONTROLS for state in ('10','01','11')]
    for improved,reference in targets:
        arows,a=reads[improved];brows,b=reads[reference]
        assert all(all(x[key]==y[key] for key in ('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')) for x,y in zip(arows,brows))
        delta={key:a[key]-b[key] for key in METRICS}
        rescue=sum(int(x['Rank-1'])==1 and int(y['Rank-1'])==0 for x,y in zip(arows,brows))
        harm=sum(int(x['Rank-1'])==0 and int(y['Rank-1'])==1 for x,y in zip(arows,brows))
        assert abs(delta['Rank-1']-100*(rescue-harm)/591)<1e-8
        comparisons.append(dict(improved=improved,reference=reference,**delta,both_plus1=delta['mAP']>=1 and delta['Rank-1']>=1,
            rank1_rescued=rescue,rank1_harmed=harm,AP_improved=sum(float(x['AP'])>float(y['AP']) for x,y in zip(arows,brows)),AP_worsened=sum(float(x['AP'])<float(y['AP']) for x,y in zip(arows,brows))))
        pairs.extend(dict(improved=improved,reference=reference,query_index=x['query_index'],name=x['name'],identity=x['identity'],camera=x['camera'],scene=x['scene'],
            delta_AP_pp=100*(float(x['AP'])-float(y['AP'])),delta_INP_pp=100*(float(x['INP'])-float(y['INP'])),delta_rank1=int(x['Rank-1'])-int(y['Rank-1'])) for x,y in zip(arows,brows))
    output=root/'normal_analysis';output.mkdir(exist_ok=False)
    for name,rows in (('six_metrics',metrics),('group_metrics',groups),('training_curves',curves),('state_metrics',state_metrics),('comparisons',comparisons),('paired_query_changes',pairs)):table(output/(name+'.csv'),rows)
    assert len(metrics)==7 and len(curves)==200 and len(state_metrics)==16 and len(comparisons)==27 and len(pairs)==27*591
    result=dict(status='ACTUAL_AE_FOUR_FULL50_NORMAL_GT_RAW_STATES_CPU_READOUT',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        training=training,comparisons=comparisons,paired_queries=len(pairs),new_epochs=200,formal_updates=2820,new_neural_calls=0,new_optimizer_updates=0,
        limits='Private5120 teacher initialization, full5632 existing baseline, relation adaptation and experts separately reported. '
            'Staged ordinary/axis equal parameters; noexpert fewer parameters; total teacher50+extra50. '
            'Benchmark selected, not untouched test. All repeated state queries are dependent. No missing/49/201/seed claims.')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(result['status'],flush=True)


if __name__=='__main__':main()
