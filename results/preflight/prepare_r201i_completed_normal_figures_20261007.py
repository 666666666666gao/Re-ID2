"""Use only the two closed official datasets; no neural inference or live-source edit."""
import csv,hashlib,json,math
from datetime import datetime
from pathlib import Path
from analyze_identity_coordinate_three_normal import METRICS,load,table

PROJECT=Path(__file__).resolve().parents[2]
OUT=PROJECT/'results/r201i_completed_normal_figures_20261007'


def main():
    assert not OUT.exists();OUT.mkdir()
    curves,endpoints,routes,costs=[],[],[],[]
    data={'datasets':{},'metrics':list(METRICS),'seed':42,'anchor_seed':42}
    sources={}
    for dataset in ('RGBNT201','MSVR310'):
        analysis=PROJECT/'results/r201i_primary_margin_20261006'/dataset/'normal_analysis'
        assert load(analysis/'result.json')['status']=='ACTUAL_I_DATASET_TWO_FULL50_NORMAL_CPU_READOUT'
        folder={'original_DeMo':PROJECT/'results/full_official_baselines_20261004/training'/(dataset+'_demo_s42')}
        gbase=PROJECT/'results/r201g_normal_priority_20261006' if dataset=='RGBNT201' else PROJECT/'results/r201g_other_two_normal_20261006'/dataset
        for variant in ('frequency_shared','axis_shared'):
            folder['G_'+variant]=gbase/'training'/(dataset+'_r201g_'+variant+'_s42')
            folder['I_'+variant]=PROJECT/'results/r201i_primary_margin_20261006'/dataset/'training'/(dataset+'_r201i_'+variant+'_s42')
        dataset_data={}
        for model,path in folder.items():
            result=load(path/'result.json')
            assert result['status']=='COMPLETE' and result['epochs']==50 and result['amp_skipped_steps']==0
            assert result['training_coverage']['eligible']==result['training_coverage']['visited']
            assert not result['training_coverage']['unvisited'] and result['training_heldout_identities']==0
            epochs=list(csv.DictReader((path/'epochs.csv').open(encoding='utf-8',newline='')))
            assert [int(e['epoch']) for e in epochs]==list(range(1,51))
            selected=max(epochs,key=lambda e:float(e['mAP']))
            assert int(selected['epoch'])==result['best']['epoch']
            assert all(abs(float(selected[k])-result['full_metrics'][k])<1e-8 for k in METRICS)
            for source in (path/'result.json',path/'epochs.csv'):
                sources[source.relative_to(PROJECT).as_posix()]=hashlib.sha256(source.read_bytes()).hexdigest()
            epoch_rows=[dict(epoch=int(e['epoch']),loss=float(e['loss']),**{k:float(e[k]) for k in METRICS}) for e in epochs]
            dataset_data[model]=dict(epochs=epoch_rows,selected_epoch=result['best']['epoch'],selected_metrics=result['full_metrics'])
            if model!='original_DeMo':
                curves.extend(dict(dataset=dataset,model=model,**e) for e in epoch_rows)
                endpoints.append(dict(dataset=dataset,model=model,selected_epoch=result['best']['epoch'],
                    **{prefix+k:value for k in METRICS for prefix,value in (('selected_',float(selected[k])),('last_',float(epochs[-1][k])),('last_minus_selected_',float(epochs[-1][k])-float(selected[k])))}))
                runtime=result['runtime'];route=runtime['mean_route']
                assert len(route)==7 and all(len(row)==3 and all(math.isfinite(x) and x>=0 for x in row) for row in route)
                band=[math.fsum(row[b] for row in route) for b in range(3)]
                assert abs(math.fsum(band)-1)<1e-5
                routes.append(dict(dataset=dataset,model=model,selected_epoch=result['best']['epoch'],mAP=result['full_metrics']['mAP'],Rank1=result['full_metrics']['Rank-1'],low_percent=100*band[0],middle_percent=100*band[1],high_percent=100*band[2],joint_minus_marginal_product_L1_sample_mean=runtime['mean_joint_minus_marginal_product_L1'],route_entropy_sample_mean=runtime['mean_route_entropy'],gate_M=runtime['mean_outer_gates'][0],gate_F=runtime['mean_outer_gates'][1],gate_I=runtime['mean_outer_gates'][2],routed_query_gallery_occurrences=runtime['triplets']))
                dataset_data[model]['mean_route']=route
            runtime=result['runtime']
            costs.append(dict(dataset=dataset,model=model,selected_epoch=result['best']['epoch'],parameters=result['parameters'],trainable_parameters=result['trainable_parameters'],descriptor_dim=result['descriptor_dim'],total_training_epochs=50 if model=='original_DeMo' else 100,recorded_extract_seconds=runtime['end_to_end_seconds'],routed_triplets=runtime['triplets'],recorded_triplets_per_second=runtime['triplets']/runtime['end_to_end_seconds'],process_allocator_peak_MiB=result['peak_memory']/1024**2))
        extra=list(csv.DictReader((analysis/'training_curves.csv').open(encoding='utf-8',newline='')))
        assert len(extra)==100
        for model in ('I_axis_shared','I_frequency_shared'):
            rows=[r for r in extra if r['model']==model]
            assert [int(r['epoch']) for r in rows]==list(range(1,51))
            dataset_data[model]['primary']=[dict(epoch=int(r['epoch']),loss=float(r['primary_triplet_loss']),violations=float(r['primary_margin_violations'])) for r in rows]
        sources[(analysis/'training_curves.csv').relative_to(PROJECT).as_posix()]=hashlib.sha256((analysis/'training_curves.csv').read_bytes()).hexdigest()
        data['datasets'][dataset]=dataset_data
    assert len(curves)==400 and len(endpoints)==len(routes)==8 and len(costs)==10
    for name,rows in (('closed_400_epoch_curves',curves),('selected_and_final_six',endpoints),('selected_route_readout',routes),('recorded_costs',costs)):
        table(OUT/(name+'.csv'),rows)
    (OUT/'inputs.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    result=dict(status='ACTUAL_CLOSED_TWO_DATASET_SCALAR_ROUTE_AND_COST_REDUCTION_CPU_ONLY',prepared_at=datetime.now().isoformat(timespec='seconds'),datasets=['RGBNT201','MSVR310'],epoch_rows=400,selected_final_rows=8,route_rows=8,cost_rows=10,input_sources_sha256=sources,new_neural_calls=0,new_optimizer_updates=0,
        limits='Only completed normal seed42; RGBNT100 and extra seeds/missing excluded. Dots use normal mAP-best, all six from same epoch. Routes average actual query+gallery occurrences, not independent repeats; low routed band mass does not prove other bands unused because cross-band/condition updates precede routing. Joint L1 is per-sample mean deviation from marginal product, not mutual information or causal utility. Recorded extract includes decode/first batch; shared-server timing not isolated performance. Process allocator peak covers training/evaluation/reload, not isolated inference peak or fullFLOPs. Original50 vs frozen teacher50+expert50 budget disclosed.')
    (OUT/'data_reduction.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'epoch_rows':400,'route_readout':routes},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
