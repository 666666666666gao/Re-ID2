"""Saved full-query error cohorts and paired identity-cluster uncertainty; no NN."""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import csv,hashlib,json,math,random

PROJECT=Path(__file__).resolve().parent
ROOT=PROJECT/'results/r201ai_AG_selected_outlets_20261012/RGBNT100/full'
OUTPUT=PROJECT/'results/rgbnt100_AG_error_cohorts_CPU_20261013'
REPETITIONS=2000
BOOTSTRAP_SEED=20261013
FIELDS=('query_index','name','identity','camera','scene','valid','relevant_gallery','kept_gallery')


def rows(path):
    with path.open(encoding='utf-8',newline='') as stream:
        result=list(csv.DictReader(stream))
    assert len(result)==1715 and [int(r['query_index']) for r in result]==list(range(1715))
    return result


def metrics(values):
    n=len(values)
    return dict(mAP=100*math.fsum(float(r['AP']) for r in values)/n,
        mINP=100*math.fsum(float(r['INP']) for r in values)/n,
        **{f'Rank-{k}':100*sum(int(r[f'Rank-{k}']) for r in values)/n for k in (1,5,10,20)})


def table(path,values):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)


def role(base,full):
    if int(base['Rank-1'])==0:
        return 'rescued' if int(full['Rank-1']) else 'remaining_original_error'
    return 'newly_harmed' if int(full['Rank-1'])==0 else 'correct_preserved'


def main():
    assert not OUTPUT.exists()
    load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    manifest=load(PROJECT/'results/full_official_baselines_20261004/training/RGBNT100_demo_s42/official_split_manifest.json')['query']
    inputs={};summaries=[];groups=[];query_rows=[]
    reference=None
    for variant in ('frequency_shared','axis_shared'):
        folder=ROOT/('RGBNT100_r201ag_'+variant+'_s42')
        base=rows(folder/'base00.csv');full=rows(folder/'normal11.csv')
        for values in (base,full):
            assert all(tuple(a[k] for k in FIELDS)==tuple(b[k] for k in FIELDS) for a,b in zip(values,base))
            assert all(r['name']==m['name'] and int(r['identity'])==m['identity'] and int(r['camera'])==m['camera'] and int(r['scene'])==m['scene'] for r,m in zip(values,manifest))
        if reference is None:reference=base
        else:assert base==reference
        for name in ('base00.csv','normal11.csv'):
            f=folder/name;inputs[f.relative_to(PROJECT).as_posix()]=hashlib.sha256(f.read_bytes()).hexdigest()
        by_role=defaultdict(list);original_errors=[];by_id=defaultdict(list)
        for before,after in zip(base,full):
            label=role(before,after);by_role[label].append((before,after));by_id[before['identity']].append((before,after))
            if not int(before['Rank-1']):original_errors.append(before)
            query_rows.append(dict(variant=variant,**{k:before[k] for k in FIELDS},role=label,
                base_AP=before['AP'],full_AP=after['AP'],base_first_match=before['first_match'],full_first_match=after['first_match']))
        assert len(original_errors)==58 and len(by_id)==50
        for label,pairs in by_role.items():
            for grouping in ('identity','camera','identity_camera'):
                slots=defaultdict(list)
                for before,after in pairs:
                    key=(before['identity']+'_'+before['camera']) if grouping=='identity_camera' else before[grouping]
                    slots[key].append((before,after))
                for key,members in sorted(slots.items()):
                    groups.append(dict(variant=variant,role=label,grouping=grouping,group=key,queries=len(members),
                        base_mAP=metrics([a for a,_ in members])['mAP'],full_mAP=metrics([b for _,b in members])['mAP']))
        cluster=[(len(v),math.fsum(float(b['AP'])-float(a['AP']) for a,b in v),sum(int(b['Rank-1'])-int(a['Rank-1']) for a,b in v)) for _,v in sorted(by_id.items())]
        rng=random.Random(BOOTSTRAP_SEED);draws={'mAP':[],'Rank-1':[]}
        for _ in range(REPETITIONS):
            chosen=[cluster[rng.randrange(50)] for _ in range(50)];n=sum(c[0] for c in chosen)
            draws['mAP'].append(100*math.fsum(c[1] for c in chosen)/n)
            draws['Rank-1'].append(100*sum(c[2] for c in chosen)/n)
        percentile={}
        for k,values in draws.items():
            values.sort();percentile[k]=dict(lower=values[int(.025*(REPETITIONS-1))],upper=values[int(.975*(REPETITIONS-1))])
        bmetric,fmetric=metrics(base),metrics(full)
        rescued=len(by_role['rescued']);harmed=len(by_role['newly_harmed'])
        assert abs(fmetric['Rank-1']-bmetric['Rank-1']-100*(rescued-harmed)/1715)<1e-8
        summaries.append(dict(variant=variant,base=bmetric,full=fmetric,delta={k:fmetric[k]-bmetric[k] for k in bmetric},
            original_error_queries=58,original_error_identities=len({r['identity'] for r in original_errors}),
            original_error_identity_camera_groups=len({(r['identity'],r['camera']) for r in original_errors}),
            rank1_rescued=rescued,rank1_harmed=harmed,
            roles={key:dict(queries=len(v),identities=len({a['identity'] for a,_ in v}),identity_camera_groups=len({(a['identity'],a['camera']) for a,_ in v})) for key,v in by_role.items()},
            paired_identity_bootstrap_percentile95=percentile))
    OUTPUT.mkdir()
    table(OUTPUT/'error_group_counts.csv',groups);table(OUTPUT/'query_roles.csv',query_rows)
    result=dict(status='ACTUAL_SAVED_AG_FULL_QUERY_ERROR_COHORT_AND_IDENTITY_BOOTSTRAP_CPU_COMPLETE',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        models=2,query_rows=3430,training_runs=0,new_neural_calls=0,new_remote_queries=0,new_optimizer_updates=0,
        bootstrap_repetitions=REPETITIONS,bootstrap_RNG_seed=BOOTSTRAP_SEED,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs_sha256=inputs,
        summaries=summaries,limits='Fixed already-selected AG checkpoints; labels only for offline error analysis, never deployment gating or training test identities. Cluster bootstrap samples 50 test identities with paired queries and variable total query count; this is within-benchmark identity uncertainty, not training-seed variation, an independent test, a causal mechanism, or proof of the running AK result.')
    (OUTPUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],summaries=summaries),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
