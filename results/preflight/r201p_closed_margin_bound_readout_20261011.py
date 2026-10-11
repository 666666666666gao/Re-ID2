"""Read completed P MSVR curves only; no imports of NN/SSH/training code."""
from datetime import datetime
import csv,hashlib,json,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def main():
    folder=ROOT/'results/r201p_auxiliary_balance_20261010/MSVR310/normal_analysis'
    inputs=[folder/'training_curves.csv',folder/'six_metrics.csv',folder/'result.json',
            ROOT/'results/preflight/r201p_auxiliary_balance_MSVR310_actual_session_20261010.json']
    fingerprints={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    proof=json.loads(inputs[3].read_text(encoding='utf-8'))
    assert proof['status']=='ACTUAL_P_TWO_FULL50_NORMAL_GT_AND_LOCAL_RAW' and proof['exit_code']==0 and proof['successful_updates']==2000
    result=json.loads(inputs[2].read_text(encoding='utf-8'))
    assert result['status']=='ACTUAL_P_DATASET_TWO_FULL50_NORMAL_CPU_READOUT' and result['official_GT_six_CMC50_perquery_groups']
    with inputs[0].open(encoding='utf-8-sig',newline='') as f: raw=list(csv.DictReader(f))
    with inputs[1].open(encoding='utf-8-sig',newline='') as f: metrics=list(csv.DictReader(f))
    assert len(raw)==100
    rows=[];models={}
    fields=['loss','primary_margin_violations','primary03_same_feature_violations','bound_clipped_fraction',
            'bound_candidate_shift_mean','bound_bounded_shift_mean','bounded_shift_max',
            'contribution_loss_raw','contribution_zero_loss_raw','mAP','Rank-1']
    for name in ('P_frequency_shared','P_axis_shared'):
        group=[r for r in raw if r['model']==name]
        assert [int(r['epoch']) for r in group]==list(range(1,51)) and all(int(r['batches'])==20 for r in group)
        best=max(group,key=lambda r:float(r['mAP']))
        published=next(r for r in metrics if r['model']==name)
        assert int(best['epoch'])==int(published['selected_epoch'])
        assert all(float(best[k])==float(published[k]) for k in ('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20'))
        assert all(0<=float(r['bound_clipped_fraction'])<=1 and float(r['bounded_shift_max'])<=.100001 for r in group)
        assert all(0<=float(r['primary_margin_violations'])<=64. for r in group)
        for tag,r in [('first',group[0]),('selected_mAP_best',best),('epoch10',group[9]),('epoch25',group[24]),('last',group[-1])]:
            row=dict(model=name,position=tag,epoch=int(r['epoch']),**{k:float(r[k]) for k in fields})
            row['primary06_active_fraction']=row['primary_margin_violations']/64
            row['detached_primary03_active_fraction']=row['primary03_same_feature_violations']/64
            row['regression_raw_over_zero_reference']=row['contribution_loss_raw']/row['contribution_zero_loss_raw']
            rows.append(row)
        models[name]=dict(selected_epoch=int(best['epoch']),selected_clip_fraction=float(best['bound_clipped_fraction']),
            selected_shift_mean=float(best['bound_bounded_shift_mean']),selected_mAP=float(best['mAP']),selected_Rank1=float(best['Rank-1']),
            last_mAP=float(group[-1]['mAP']),last_Rank1=float(group[-1]['Rank-1']),
            last_minus_selected_mAP=float(group[-1]['mAP'])-float(best['mAP']),
            last_clip_fraction=float(group[-1]['bound_clipped_fraction']),last_shift_mean=float(group[-1]['bound_bounded_shift_mean']),
            all50_primary06_active_fraction=statistics.mean(float(r['primary_margin_violations'])/64 for r in group),
            primary06_active_fraction_min=min(float(r['primary_margin_violations'])/64 for r in group),
            primary06_all64_epoch_count=sum(float(r['primary_margin_violations'])==64 for r in group),
            detached_primary03_active_fraction_mean=statistics.mean(float(r['primary03_same_feature_violations'])/64 for r in group),
            detached_primary03_active_fraction_min=min(float(r['primary03_same_feature_violations'])/64 for r in group),
            detached_primary03_active_fraction_max=max(float(r['primary03_same_feature_violations'])/64 for r in group),
            last_vs_first_loss_percent=100*(float(group[-1]['loss'])/float(group[0]['loss'])-1))
    out=folder/'closed_margin_bound_readout.csv';assert not out.exists()
    with out.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    findings=[
        dict(id=1,observation='At selected epoch2 both controls have clipped_fraction0 and mean final shift~0.031; no boundary clipping at the selected best.',
             interpretation='The0.10 cap is not an observed clipping explanation for the early near-zero retrieval gain.',
             implication='Do not relax the cap based on the selected score alone.',next_step='Await single-factor Q result and diagnose the direction/objective of its actual correction if still weak.'),
        dict(id=2,observation='Late50 both controls clip>93% of train samples while mAP is below selectedbest and loss decreases~42%.',
             interpretation='Continued optimization increases constrained correction, but does not sustain benchmark retrieval gain. This does not isolate a cause.',
             implication='A larger correction allowance is not established as beneficial; retain best/last/all50 and harm/rescue.',next_step='Compare completed Q same-budget curves before choosing a new intervention.'),
        dict(id=3,observation='The trained hinge margin0.6 is active for~99.95% of anchors over all50 epoch averages for both controls; detached margin0.3 probe is active for only part of the same batches.',
             interpretation='The two margins would apply different active-pair sets. Detached counterfactual loss is not a trained margin0.3 result.',
             implication='The main loss stays active, so small improvement is not explained by zero hinge gradient. This does not prove the margin causes poor generalization.',next_step='Audit previous margin trials; if Q fails, separately measure CE versus metric gradients on selected deploy paths before repeating or tuning margins.'),
        dict(id=4,observation='Current-batch contribution raw regression exceeds its recorded zero-predictor reference at the sampled checkpoints.',
             interpretation='This reinforces a limited regression/gate diagnostic; the values are losses rather than validation MAE.',
             implication='Do not claim calibrated usefulness prediction.',next_step='Use Q only to test removing regression influence; require real retrieval and missing metrics before any benefit claim.')]
    record=dict(status='ACTUAL_CLOSED_P_MSVR_CPU_MARGIN_BOUND_READOUT',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        scope='Completed MSVR P only; no current Q/RGB epoch or live-model observation',inputs_sha256=fingerprints,
        original_curve_rows=100,derived_checkpoint_rows=len(rows),models=models,findings=findings,
        raw_data_table='results/r201p_auxiliary_balance_20261010/MSVR310/normal_analysis/training_curves.csv',
        derived_table=str(out.relative_to(ROOT)).replace('\\','/'),new_neural_calls=0,new_optimizer_updates=0,
        limits='One expertseed42/reference42 pair, benchmark-selected. Epoch/batch anchors are repeated training observations, not independent samples; no multi-seed significance or global/causal conflict attribution. Margin0.3 is a detached same-feature probe, not a retrain.')
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==s for n,s in fingerprints.items())
    dest=ROOT/'results/preflight/r201p_closed_margin_bound_readout_actual_20261011.json';assert not dest.exists()
    dest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=record['status'],rows=len(rows),models=models,new_neural_calls=0),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
