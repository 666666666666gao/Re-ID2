"""Inspect saved normal error cohorts; labels never enter training or routing."""
from pathlib import Path
import csv,json,statistics

P=Path(__file__).resolve().parent
OUTPUT=P/'results/rgbnt100_original_error_structure_CPU_20261014'


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    assert not OUTPUT.exists()
    sources={}
    for tag,stem in [('AS','r201as_RGB_query_normalization_20261014'),('AV','r201av_RGB_frequency_value_decay_20261014')]:
        for variant in ('frequency_shared','axis_shared'):
            root=P/'results'/stem/'RGBNT100/states'/('RGBNT100_r201'+tag.lower()+'_'+variant+'_s42')
            sources[tag+'_'+variant]=root/'state_11.csv'
    base_path=P/'results/r201av_RGB_frequency_value_decay_20261014/RGBNT100/states/RGBNT100_r201av_axis_shared_s42/state_00.csv'
    base=read(base_path);models={key:read(path) for key,path in sources.items()}
    assert len(base)==1715 and [int(r['query_index']) for r in base]==list(range(1715))
    metadata=('query_index','name','identity','camera','scene')
    assert all([tuple(r[k] for k in metadata) for r in rows]==[tuple(r[k] for k in metadata) for r in base] for rows in models.values())
    assert all(r['valid']=='True' for rows in [base,*models.values()] for r in rows)
    wrong={i for i,r in enumerate(base) if int(r['Rank-1'])==0};assert len(wrong)==58
    detail=[];summaries={};groups=[]
    for model,rows in models.items():
        rescued={i for i in wrong if int(rows[i]['Rank-1'])==1}
        harmed={i for i in range(1715) if i not in wrong and int(rows[i]['Rank-1'])==0}
        summaries[model]=dict(rescued_queries=len(rescued),harmed_queries=len(harmed),net_queries=len(rescued)-len(harmed),
            rescue_identities=sorted({int(base[i]['identity']) for i in rescued}),
            harm_identities=sorted({int(base[i]['identity']) for i in harmed}),
            rescue_cameras=sorted({int(base[i]['camera']) for i in rescued}),
            harm_cameras=sorted({int(base[i]['camera']) for i in harmed}))
        for i in sorted(wrong|harmed):
            detail.append(dict(model=model,**{k:base[i][k] for k in metadata},
                role=('rescued' if i in rescued else 'remaining_original_error') if i in wrong else 'newly_harmed',
                base_AP=float(base[i]['AP']),model_AP=float(rows[i]['AP']),AP_delta_pp=100*(float(rows[i]['AP'])-float(base[i]['AP'])),
                base_first_match=int(base[i]['first_match']),model_first_match=int(rows[i]['first_match'])))
        for axis in ('camera','identity'):
            for value in sorted({int(r[axis]) for r in base}):
                cohort=[i for i,r in enumerate(base) if int(r[axis])==value]
                error=wrong.intersection(cohort)
                groups.append(dict(model=model,group=axis,value=value,queries=len(cohort),base_errors=len(error),
                    base_error_share_percent=100*len(error)/58,rescued=len(rescued.intersection(cohort)),harmed=len(harmed.intersection(cohort)),
                    base_mAP=100*statistics.mean(float(base[i]['AP']) for i in cohort),
                    model_mAP=100*statistics.mean(float(rows[i]['AP']) for i in cohort)))
    OUTPUT.mkdir()
    for filename,rows in [('error_queries.csv',detail),('all_camera_identity_groups.csv',groups)]:
        with (OUTPUT/filename).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(status='ACTUAL_SAVED_RGBNT100_NORMAL_ERROR_STRUCTURE_CPU_COMPLETE',queries=1715,
        original_errors=58,original_error_identities=sorted({int(base[i]['identity']) for i in wrong}),
        original_error_cameras=sorted({int(base[i]['camera']) for i in wrong}),models=summaries,
        source_paths={key:str(path) for key,path in sources.items()},base_path=str(base_path),
        new_NN=0,new_optimizer_updates=0,
        limits='Fixed normal benchmark-selected AS/AV checkpoints and labelled cohorts, not independent test confirmation, causal camera attribution, training-seed variance or deployable query selection. Zero-based camera convention. Ground-truth error identities are diagnostic only and must never determine training samples, model routing or new checkpoint selection. Consecutive or same-identity queries are not independent runs. Source metadata are aligned across all1715 queries; no sample removal.')
    (OUTPUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
