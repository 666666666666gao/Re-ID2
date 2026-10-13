"""Summarize completed full-Q/G attention measurements without model execution."""
from pathlib import Path
import csv,hashlib,json,statistics

ROOT=Path('results/r201ap_AL_spatial_attention_20261013')
SOURCE=ROOT/'full/AL_axis_shared'
OUT=ROOT/'analysis'


def main():
    result=json.loads((SOURCE/'result.json').read_text())
    assert result['status']=='COMPLETE' and result['records']==10290
    assert (result['query_records'],result['gallery_records'],result['forward_batches'])==(1715,8575,161)
    assert result['optimizer_updates']==result['checkpoint_writes']==0
    assert result['parameters_unchanged'] and result['frozen_identity_unchanged']
    assert result['output_features_exact_saved_selected_cache'] and result['attention_fields']==107
    with (SOURCE/'spatial_attention_records.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==10290 and [int(r['index']) for r in rows]==list(range(10290))
    groups={'all_query':rows[:1715],'gallery':rows[1715:]}
    for role in ('correct_preserved','remaining_original_error','rescued','newly_harmed'):
        groups[role]=[r for r in rows[:1715] if r['role']==role]
    assert {k:len(groups[k]) for k in groups if k not in ('all_query','gallery')}==dict(
        correct_preserved=1650,remaining_original_error=53,rescued=5,newly_harmed=7)
    maps=json.loads((SOURCE/'teacher_case_maps.json').read_text())
    assert len(maps['maps'])==4 and all(set(v)=={'0','1','2'} for v in maps['maps'].values())
    selections={'teacher_mean_sources':[f'teacher_{m}_head_mean_probability' for m in range(3)]}
    for stage in range(2):
        for band in range(3):selections[f'M{stage}_band{band}']=[f'M_{stage}_band{band}_head_mean_probability']
    for band in range(3):
        selections[f'F_mean_sources_band{band}']=[f'F_{m}_band{band}_head_mean_probability' for m in range(3)]
    summaries=[]
    for group,members in groups.items():
        for name,prefixes in selections.items():
            row=dict(group=group,records=len(members),measurement=name)
            for metric in ('normalized_entropy','effective_support_fraction','maximum_lift'):
                values=[statistics.mean(float(r[p+'_'+metric]) for p in prefixes) for r in members]
                row.update({metric+'_mean':statistics.mean(values),metric+'_median':statistics.median(values)})
            summaries.append(row)
    OUT.mkdir(exist_ok=False)
    with (OUT/'head_mean_spatial_comparison.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    facts=dict(status='ACTUAL_AP_FULL_QG_SPATIAL_MEASUREMENT_ANALYSIS',full_queries=1715,full_gallery=8575,
        metadata_and_row_counts_verified=True,original_model_output_exact=True,optimizer_updates=0,new_neural_calls=0,
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in SOURCE.iterdir() if f.is_file()},
        estimator='Average raw probabilities over heads FIRST; condition patches within each source. Teacher excludes CLS and records its removed mass separately. M summaries average equal sources and relations; F summaries average sources. Per-head entropy remains separately labelled in original full CSV and is not directly compared to teacher.',
        summaries=summaries,preserved_full_six=result['full_six_metrics_equal_selected'],
        limits='One mature AL epoch20, full official Q/G. Teacher and experts attend different representations, dimensions and key sets, so equal entropy convention does not make their features identical or prove spatial identity utility. Five rescue/seven harm cohorts and four predeclared AG cases are small diagnostics, not representative causal estimates, foreground truth, oracle routing or additional model improvements. No training objective chosen from per-query test labels.')
    (OUT/'result.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:facts[k] for k in ('status','full_queries','full_gallery','optimizer_updates','new_neural_calls')}),flush=True)


if __name__=='__main__':main()
