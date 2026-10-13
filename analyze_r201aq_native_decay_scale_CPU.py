"""CPU arithmetic on saved first native updates; no NN, SSH or live training reads."""
from pathlib import Path
import csv,hashlib,json,math

P=Path(__file__).resolve().parent
SOURCE=P/'results/preflight/r201aq_RGB_attention_decay_20261013_actual_session.json'
OUTPUT=P/'results/r201aq_native_decay_scale_CPU_20261013'
TARGETS=('modality_expert.attention.in_proj_weight',
    'frequency_expert.attention.0.in_proj_weight','frequency_expert.attention.1.in_proj_weight',
    'frequency_expert.attention.2.in_proj_weight','frequency_expert.cross_band.in_proj_weight')


def main():
    data=SOURCE.read_bytes();record=json.loads(data.decode('utf-8-sig'))
    admission=record['native_formal_admission']
    assert admission['native_acceptance']['status']=='PASS' and admission['native_acceptance']['actual_updates']==6
    natives=admission['native'];assert [n['variant'] for n in natives]==['frequency_shared','axis_shared']
    initial=[n['attention_decay_policy']['weight_norm_before'] for n in natives]
    assert initial[0]==initial[1]
    rows=[];summaries=[]
    for native in natives:
        p=native['attention_decay_policy']
        assert tuple(p['targets'])==TARGETS and p['old']==1e-4 and p['new']==0 and p['packed_includes_V']
        assert all(len(p[key])==15 for key in ('weight_norm_before','weight_norm_after','unscaled_task_gradient_norm','counterfactual_old_L2_gradient_norm'))
        for i,name in enumerate(TARGETS):
            for j,part in enumerate(('Q','K','V')):
                k=3*i+j;before=p['weight_norm_before'][k];after=p['weight_norm_after'][k]
                g=p['unscaled_task_gradient_norm'][k];l2=p['counterfactual_old_L2_gradient_norm'][k]
                assert all(math.isfinite(v) and v>0 for v in (before,after,g,l2))
                assert l2==1e-4*before
                rows.append(dict(variant=native['variant'],native_update=1,parameter=name,part=part,
                    weight_norm_before=before,weight_norm_after=after,unscaled_task_gradient_norm=g,
                    counterfactual_old_L2_gradient_norm=l2,counterfactual_L2_to_task_norm_ratio=l2/g,
                    actual_weight_norm_change=after-before))
        own=[r for r in rows if r['variant']==native['variant']]
        for parts in (('Q','K'),('V',)):
            values=[r['counterfactual_L2_to_task_norm_ratio'] for r in own if r['part'] in parts]
            summaries.append(dict(variant=native['variant'],parts=list(parts),rows=len(values),
                minimum_L2_to_task_norm_ratio=min(values),maximum_L2_to_task_norm_ratio=max(values)))
    assert len(rows)==30
    OUTPUT.mkdir(exist_ok=False)
    with (OUTPUT/'first_native_update_parameter_scales.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    proof=dict(status='ACTUAL_SAVED_FIRST_NATIVE_UPDATE_CPU_REDUCTION_COMPLETE',
        source=dict(path=str(SOURCE.relative_to(P)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=30,controls=2,
        native_updates_represented=2,native_updates_admitted_total=6,initial_QKV_norms_paired_exact=True,
        model_forwards=0,optimizer_updates=0,live_training_queries=0,summaries=summaries,
        limits='Only update1 per control from saved actual native admission, not all6 native updates or formal50. Old L2 is counterfactual 1e-4 times pre-update parameter norm, not an old-optimizer replay or Adam effective-update comparison. Norm ratios omit gradient direction/momentum and do not prove long-run cause, selectivity, benefit or test gains. Packed intervention includes V. Current AQ gradients are real task gradients after unscale, decay0; no new gradients measured by this CPU reduction. No empirical uncertainty/seed independence claim.')
    (OUTPUT/'result.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(proof,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
