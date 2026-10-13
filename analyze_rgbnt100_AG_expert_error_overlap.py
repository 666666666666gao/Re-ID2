"""Cross-expert overlap on the 58 audited original errors; diagnostic only, no NN."""
import csv,json
from pathlib import Path

P=Path(__file__).resolve().parent
SOURCE=P/'results/rgbnt100_AG_outlet_query_survival_CPU_20261012'
OUT=P/'results/rgbnt100_AG_expert_error_overlap_CPU_20261013'

def main():
    proof=json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))
    assert proof['status']=='ACTUAL_AI_CURRENT_AG_RGBNT100_OUTLET_QUERY_SURVIVAL_CPU_COMPLETE'
    assert proof['export_csv_rows_checked']==51450 and proof['original_error_rows']==116
    with (SOURCE/'original_error_flags.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==116
    summaries=[];flags=[]
    for variant in ('frequency_shared','axis_shared'):
        selected=[r for r in rows if r['variant']==variant]
        assert len(selected)==len({r['query_index'] for r in selected})==58
        assert all(r['base00_correct']=='0' for r in selected)
        original=P/'results/r201ai_AG_selected_outlets_20261012/RGBNT100/full'/('RGBNT100_r201ag_'+variant+'_s42')
        for stage in ('base00','normal11',*[e+'_'+s for e in ('M','F') for s in ('raw_pool','routed_pool','unit_direction','projected_auxiliary')]):
            with (original/(stage+'.csv')).open(encoding='utf-8',newline='') as f:full=list(csv.DictReader(f))
            assert len(full)==1715
            assert all(tuple(full[int(r['query_index'])][k] for k in ('name','identity','camera','scene'))==
                tuple(r[k] for k in ('name','identity','camera','scene')) and
                full[int(r['query_index'])]['Rank-1']==r[stage+'_correct'] for r in selected)
            if stage=='base00':assert {r['query_index'] for r in full if r['Rank-1']=='0'}=={r['query_index'] for r in selected}
        for suffix in ('raw_pool','routed_pool','unit_direction','projected_auxiliary'):
            m='M_'+suffix+'_correct';f='F_'+suffix+'_correct'
            M={r['query_index'] for r in selected if r[m]=='1'}
            F={r['query_index'] for r in selected if r[f]=='1'}
            fused={r['query_index'] for r in selected if r['normal11_correct']=='1'}
            summaries.append(dict(variant=variant,stage=suffix,original_errors=58,
                M_correct=len(M),F_correct=len(F),both_correct=len(M&F),M_only=len(M-F),F_only=len(F-M),
                either_correct=len(M|F),neither_correct=58-len(M|F),
                final_rescued=len(fused),either_correct_final_wrong=len((M|F)-fused),
                neither_correct_final_rescued=len(fused-(M|F))))
            flags.extend(dict(variant=variant,stage=suffix,**{k:r[k] for k in ('query_index','name','identity','camera','scene')},
                M_correct=int(r[m]),F_correct=int(r[f]),normal11_correct=int(r['normal11_correct'])) for r in selected)
    OUT.mkdir(exist_ok=False)
    for name,data in [('overlap',summaries),('query_flags',flags)]:
        with (OUT/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    result=dict(status='ACTUAL_CLOSED_AG_EXPERT_ERROR_OVERLAP_CPU_COMPLETE',source=str(SOURCE.relative_to(P)),
        summaries=summaries,query_rows=116,stage_query_rows=len(flags),direct_original_CSV_rows_checked=34300,
        selected_flags_original_metadata_and_predictions_exact=True,new_neural_calls=0,new_optimizer_updates=0,
        limits='Fixed AG selected checkpoints and original58 first-match errors percontrol, reusing the previously full1715-Q installed-GT audited outlet tables. '
        'Correct-set overlap is not vector/gradient conflict, causality, achievable fusion accuracy, a label-based deployment selector, '
        'or AL performance. Standalone endpoints differ in dimension/normalization/function. Complete harm and all-query AP statistics remain in the existing full-cohort audit.')
    (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
