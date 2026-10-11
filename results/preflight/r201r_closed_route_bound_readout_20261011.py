"""CPU-only completed R MSVR50 route, gate and bound readout."""
from collections import defaultdict
from datetime import datetime
import csv,hashlib,json,math,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def main():
    tags={'R':'r201r_primary_ce_20261011'}
    sources={};panels=[];selected={};orders={}
    review=json.loads((ROOT/'results/preflight/r201r_primary_ce_source_review_20261011.json').read_text())
    for name in ('axis_collaboration.py','run_r201p_auxiliary_balance.py','run_r201q_task_gate.py','dual_axis.py','run_r201r_primary_ce.py'):
        digest=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        assert digest==(review['sources_sha256']|review['directly_reused_sources_sha256'])[name]
        sources[name]=digest
    summary={}
    for family,tag in tags.items():
        folder=ROOT/'results'/tag/'MSVR310';result=folder/'normal_analysis/result.json'
        data=json.loads(result.read_text())
        assert data['status']=='ACTUAL_'+family+'_DATASET_TWO_FULL50_NORMAL_CPU_READOUT' and data['official_GT_six_CMC50_perquery_groups']
        sources[result.relative_to(ROOT).as_posix()]=hashlib.sha256(result.read_bytes()).hexdigest()
        for variant in ('frequency_shared','axis_shared'):
            run=folder/'training'/('MSVR310_r201'+family.lower()+'_'+variant+'_s42')
            trained_path=run/'result.json';batch_path=run/'batch_orders.jsonl';epochs_path=run/'epochs.csv'
            for path in (trained_path,batch_path,epochs_path):sources[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
            trained=json.loads(trained_path.read_text());assert trained['status']=='COMPLETE' and trained['epochs']==50 and trained['steps']==trained['optimizer_steps']==1000 and trained['amp_skipped_steps']==0
            batches=[json.loads(s) for s in batch_path.read_text().splitlines()];assert len(batches)==1000 and all(b['optimizer_updated'] for b in batches)
            with epochs_path.open(encoding='utf-8-sig',newline='') as f:epochs=list(csv.DictReader(f))
            assert [int(r['epoch']) for r in epochs]==list(range(1,51))
            model=family+'_'+variant;orders[model]=[(b['epoch'],b['step'],b['names'],b['partial_set']) for b in batches]
            best=max(epochs,key=lambda r:float(r['mAP']));assert int(best['epoch'])==trained['best']['epoch'];selected[model]=int(best['epoch'])
            group=defaultdict(list)
            for b in batches:
                assert b['final_full_CE_weight']==.05 and b['auxiliary_config_ID_LOSS_WEIGHT']==.25 and b['xent_observed_calls']==2 and b['zero_weight_partial_CE_not_rescaled']
                assert b['primary_margin']==.6 and b['M_F_auxiliary_weight']==.01 and b['partial_loss_effective']==0
                if family=='R':assert b['contribution_regression_weight']==b['contribution_regression_effective']==0.
                assert len(b['full_gate_mean'])==len(b['residual_scales'])==len(b['route_band_mass_mean'])==3 and len(b['route_relation_mass_mean'])==7
                assert abs(sum(b['route_band_mass_mean'])-1)<1e-5 and abs(sum(b['route_relation_mass_mean'])-1)<1e-5
                assert 0<=b['route_entropy_mean']<=math.log(21)+1e-5 and b['route_nonindependence_mean']>=-1e-6
                assert all(0<=v<=1 for v in b['full_gate_mean'])
                bound=b['final_descriptor_bound_full'];assert bound['epsilon']==.1 and bound['bounded_shift_max']<=.100001
                group[b['epoch']].append(b)
            means=lambda seq:math.fsum(seq)/len(seq)
            model_panels=[]
            for ep in epochs:
                number=int(ep['epoch']);bs=group[number];assert len(bs)==20
                row=dict(model=model,epoch=number,selected=number==selected[model],batch_count=20,
                    loss=means([b['loss'] for b in bs]),primary_hinge_loss=means([b['primary_triplet_loss'] for b in bs]),
                    primary06_active_fraction=means([b['primary_margin_violations']/64 for b in bs]),
                    detached03_active_fraction=means([b['primary03_same_feature_violations']/64 for b in bs]),
                    route_entropy=means([b['route_entropy_mean'] for b in bs]),route_entropy_fraction_of_log21=means([b['route_entropy_mean']/math.log(21) for b in bs]),
                    route_nonindependence_KL=means([b['route_nonindependence_mean'] for b in bs]),
                    clipped_fraction=means([b['final_descriptor_bound_full']['clipped_fraction'] for b in bs]),
                    candidate_shift_mean=means([b['final_descriptor_bound_full']['candidate_shift_mean'] for b in bs]),
                    bounded_shift_mean=means([b['final_descriptor_bound_full']['bounded_shift_mean'] for b in bs]),
                    final_full_CE_weight=.05,final_full_CE_effective=means([b['final_full_CE_effective'] for b in bs]),
                    contribution_raw_loss=means([b['contribution_loss_raw'] for b in bs]),
                    contribution_weighted_loss=means([(.05 if family=='P' else 0.)*b['contribution_loss_raw'] for b in bs]),
                    **{k:float(ep[k]) for k in ('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')})
                assert abs(row['loss']-float(ep['loss']))<1e-8
                for i,role in enumerate(('M','F','I')):
                    row['gate_'+role]=means([b['full_gate_mean'][i] for b in bs]);row['scale_'+role]=means([b['residual_scales'][i] for b in bs])
                for i,band in enumerate(('L','M','H')):row['band_mass_'+band]=means([b['route_band_mass_mean'][i] for b in bs])
                for i,relation in enumerate(('R','N','T','RN','RT','NT','RNT')):row['relation_mass_'+relation]=means([b['route_relation_mass_mean'][i] for b in bs])
                model_panels.append(row);panels.append(row)
            chosen=next(r for r in model_panels if r['selected']);last=model_panels[-1]
            summary[model]=dict(selected_epoch=selected[model],selected=chosen,last=last,
                last_minus_selected_mAP=last['mAP']-chosen['mAP'],last_minus_selected_R1=last['Rank-1']-chosen['Rank-1'],
                loss_last_vs_first_percent=100*(last['loss']/model_panels[0]['loss']-1),
                all50_primary06_active_fraction=statistics.mean(r['primary06_active_fraction'] for r in model_panels),
                all50_detached03_active_fraction=statistics.mean(r['detached03_active_fraction'] for r in model_panels))
    assert orders['R_axis_shared']==orders['R_frequency_shared']
    assert len(panels)==100
    output=ROOT/'results/r201r_closed_diagnostics_20261011';assert not output.exists();output.mkdir()
    table=output/'route_gate_bound_full50.csv'
    with table.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(panels[0]));writer.writeheader();writer.writerows(panels)
    record=dict(status='ACTUAL_COMPLETED_R_PAIRED100_EPOCH_ROUTE_GATE_BOUND_CPU_READOUT',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        sources_sha256=sources,raw_data_tables=[n for n in sources if n.endswith('batch_orders.jsonl')],derived_table=table.relative_to(ROOT).as_posix(),
        models=summary,rows=100,paired_full50_batch_orders_exact=True,new_neural_calls=0,new_optimizer_updates=0,new_checkpoint_writes=0,
        formulas='Arithmetic means over20 observed training batches/epoch. R records contribution coefficient/effective0 explicitly and finalfull CE.05 separately. Nonindependence is saved route KL versus product marginals; entropy normalized by log21. Means do not measure per-query gate quality.',
        next_step='Await fresh review/actual R full normal quartet before changing another factor. Use selected and late curves plus GT/query rescue-harm and actual controlled states to select the next intervention.',
        limits='Closed R seed42 MSVR only; no current RGBNT100 data read. Training-batch aggregate route statistics are not retrieval contribution, causal attribution, calibrated usefulness or independent repeated trials. Detached .3 readings are not a trained .3 model. Retain current K201 best and full weak-dataset+paper-six goal.')
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in sources.items())
    (ROOT/'results/preflight/r201r_closed_route_bound_readout_actual_20261011.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=record['status'],rows=100,selected={k:{n:v['selected'][n] for n in ('epoch','mAP','Rank-1','route_entropy_fraction_of_log21','route_nonindependence_KL','gate_M','gate_F','gate_I','clipped_fraction','bounded_shift_mean')} for k,v in summary.items()},last={k:{n:v['last'][n] for n in ('mAP','Rank-1','gate_M','gate_F','gate_I','clipped_fraction')} for k,v in summary.items()}),ensure_ascii=False))

if __name__=='__main__':main()
