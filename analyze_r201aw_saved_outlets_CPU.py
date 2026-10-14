"""Summarize actual fixed AV outlet and spatial records; stdlib only, no NN."""
from pathlib import Path
import csv,json,statistics
P=Path('C:/Users/gb/projects/demo_dual_axis_20261002');TAG='r201aw_AV_frequency_outlets_20261014';ROOT=P/'results'/TAG
load=lambda p:json.loads(p.read_text('utf-8-sig'))

def rows(p):
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n','utf-8')

def main():
    proof=load(P/'results/preflight'/(TAG+'_complete_intake.json'))
    assert proof['status']=='ACTUAL_AW_TWO_FIXED_AV_FULL_NORMAL_30_OUTLETS_AND_SPATIAL_D_ARCHIVED_GT_PARITY_COMPLETE'
    out=ROOT/'CPU_analysis';out.mkdir(exist_ok=False)
    alloutlets=[];allspatial=[];cohorts={}
    for variant in ('frequency_shared','axis_shared'):
        name='RGBNT100_r201av_'+variant+'_s42';root=ROOT/'RGBNT100/full'/name;result=load(root/'result.json')
        assert result['status']=='COMPLETE' and result['optimizer_updates']==result['new_weights']==0
        assert result['normal_features_and_distance_exact'] and result['spatial_records']==10290 and result['attention_fields']==275
        stage_rows={stage:rows(root/(stage+'.csv')) for stage in result['stages']}
        baseline=stage_rows['base00'];normal=stage_rows['normal11'];assert len(baseline)==len(normal)==1715
        wrong={i for i,r in enumerate(baseline) if float(r['Rank-1'])==0};assert len(wrong)==58
        saved=rows(root/'spatial_attention_records.csv');assert len(saved)==10290
        assert [r['name'] for r in saved[:1715]]==[r['name'] for r in baseline]
        union=set();cohorts[variant]={}
        for stage,data in stage_rows.items():
            assert len(data)==1715 and [(r['name'],r['identity'],r['camera'],r['scene']) for r in data]==[(r['name'],r['identity'],r['camera'],r['scene']) for r in baseline]
            rescue=[i for i in wrong if float(data[i]['Rank-1'])==1]
            harm=[i for i in range(1715) if float(baseline[i]['Rank-1'])==1 and float(data[i]['Rank-1'])==0]
            if stage not in ('base00','normal11'):union.update(rescue)
            cohorts[variant][stage]=dict(rescue_original_error_indices=sorted(rescue),harm_original_correct_indices=harm)
            value=result['measurements'][stage]
            alloutlets.append(dict(variant=variant,selected_epoch=result['selected_epoch'],stage=stage,dimension=result['dimensions'][stage],**{k:value[k] for k in ('mAP','mINP','Rank-1','Rank-5','Rank-10','Rank-20')},original58_correct=len(rescue),original_correct_harmed=len(harm)))
        cohorts[variant]['internal_outlet_rescue_union']=sorted(union)
        for side in ('query','gallery'):
            data=[r for r in saved if r['side']==side];assert len(data)==(1715 if side=='query' else 8575)
            for band in range(3):
                for stage in ('input','after_LN','after_linear','after_GELU'):
                    values=[float(r[f'F_source{s}_band{band}_{stage}_spatial_variance']) for r in data for s in range(3)]
                    allspatial.append(dict(variant=variant,side=side,band=band,measure=stage+'_spatial_variance',mean=statistics.mean(values),minimum=min(values),maximum=max(values)))
                for measure in ('head_mean_probability_maximum_lift','per_head_maximum_lift','logit_centered_spatial_std'):
                    values=[float(r[f'F_{s}_band{band}_{measure}']) for r in data for s in range(3)]
                    allspatial.append(dict(variant=variant,side=side,band=band,measure=measure,mean=statistics.mean(values),minimum=min(values),maximum=max(values)))
    for name,data in [('outlet_identity_metrics.csv',alloutlets),('F_spatial_processing.csv',allspatial)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    save(out/'result.json',dict(status='ACTUAL_AW_SAVED_OUTLET_IDENTITY_AND_SPATIAL_CPU_SUMMARY_COMPLETE',model_runs=2,outlets=30,spatial_summary_rows=len(allspatial),original_errors=58,cohorts=cohorts,new_NN=0,new_optimizer_updates=0,limits='All labelled per-stage rescue unions are offline diagnostic over fixed homogeneous Q/G outlet distances, never deployment routing or new method scores. Distinct pooled512, relation3584 and deployment5120 groups retain labels. Residual independent retrieval is not additive final utility; actual source/band variances and attention are feature diagnostics, not foreground/causal identity evidence. Ordinary F slots read all input sources. Summaries average3 source/slot values across actual per-side records, no sample filtering. Fixed selected normal11 score unchanged.'))
    print(json.dumps(dict(status='ACTUAL_AW_CPU_SUMMARY_COMPLETE',model_runs=2,outlets=30,spatial_summary_rows=len(allspatial),new_NN=0)))

if __name__=='__main__':main()
