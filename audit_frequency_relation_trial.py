"""Installed-GT CPU audit of M2 states/utility and identical historical sampling."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_shared_identity_states import audit_variant, installed_gt, recount, summarize, changes, METRICS


STAGES=('deployed','base_common','M_pre','M_post','F_pre','F_post','M_aux','F_aux')
VARIANTS=('axis_shared','frequency_shared','twins_shared')


def main():
    parser=argparse.ArgumentParser()
    for name in ('root','reference-root','data-root'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();root=Path(args.root);reference=Path(args.reference_root)
    terminal=json.loads((root/'controller_result.json').read_text())
    assert terminal['status']=='COMPLETE' and {r['variant'] for r in terminal['runs']}==set(VARIANTS) and len(terminal['runs'])==3
    output=root/'independent_cpu_audit.json';assert not output.exists()
    gt=installed_gt(Path(args.data_root),Path('splits.json'));runs={}
    for variant in VARIANTS:
        name='MSVR310_'+variant+'_s42';run=root/'original_mean/development'/name
        trained=json.loads((run/'result.json').read_text())
        assert trained['epochs']==50 and trained['arguments']['relation_weight']==.1
        orders=[json.loads(v) for v in (run/'batch_orders.jsonl').read_text().splitlines()]
        old=[json.loads(v) for v in (reference/'original_mean/development'/name/'batch_orders.jsonl').read_text().splitlines()]
        assert len(orders)==len(old)==trained['steps']
        assert all(all(a[k]==b[k] for k in ('epoch','step','names','partial_set')) for a,b in zip(orders,old))
        assert all(v['frequency_relation_weight']==.1 and v['frequency_relation_pairs']==4032 and
            v['frequency_relation_target_requires_grad'] is False and np.isfinite(v['frequency_relation_raw']) for v in orders)
        states=audit_variant(root/'original_mean/controlled_states',variant,gt)
        folder=root/'original_mean/utility'/name/'full';result=json.loads((folder/'result.json').read_text())
        assert result['status']=='COMPLETE' and result['metric_count']==392 and result['stages']==list(STAGES)
        assert result['normal_feature_max_error']==0 and result['previous_all49_deployed_exact'] and result['state_tensor_versions_unchanged']
        assert result['optimizer_updates']==result['official_test_uses']==0
        utility={}
        with np.load(folder/'raw_distances.npz') as raw:
            assert len(raw.files)==392+len(gt) and all(np.array_equal(raw[n],v) for n,v in gt.items())
            assert len(result['measurements'])==49
            for condition,reported in result['measurements'].items():
                assert set(reported)==set(STAGES);rows={}
                for stage in STAGES:
                    rows[stage]=recount(raw[condition+'_'+stage],gt);calculated=summarize(rows[stage]);actual=reported[stage]
                    assert max(abs(calculated[m]-actual[m]) for m in METRICS)<1e-8
                    cmc=[100*np.mean([r['first_match']<=k for r in rows[stage]]) for k in range(1,51)]
                    assert np.max(np.abs(np.asarray(cmc)-actual['CMC_1_to_50']))<1e-8
                    assert actual['query_count']==actual['valid_queries']==len(rows[stage]) and actual['invalid_queries']==0
                    for group_key in ('identity','camera','scene'):
                        groups=actual['groups'][group_key]
                        assert [g['value'] for g in groups]==sorted({r[group_key] for r in rows[stage]})
                        for group in groups:
                            subset=[r for r in rows[stage] if r[group_key]==group['value']]
                            assert len(subset)==group['queries'] and all(abs(summarize(subset)[m]-group[m])<1e-8 for m in METRICS)
                    with (folder/condition/(stage+'.csv')).open() as f:exported=list(csv.DictReader(f))
                    assert len(exported)==len(rows[stage])
                    for a,b in zip(exported,rows[stage]):
                        assert set(a)==set(b)
                        for n,v in b.items():assert abs(float(a[n])-v)<1e-12 if n in ('AP','INP') else a[n]==str(v)
                assert all(abs(summarize(rows['deployed'])[m]-states['conditions'][condition]['metrics']['11'][m])<1e-8 for m in METRICS)
                utility[condition]=dict(metrics={s:summarize(v) for s,v in rows.items()},F_projection=changes(rows['F_pre'],rows['F_post']),
                    M_projection=changes(rows['M_pre'],rows['M_post']))
        runs[variant]=dict(states=states,utility=utility,sampling_matches_original_M1=True,
            relation_raw_first=orders[0]['frequency_relation_raw'],relation_raw_last=orders[-1]['frequency_relation_raw'])
    assert len(list(root.rglob('*.pth')))==3 and all(p.name=='best.pth' for p in root.rglob('*.pth'))
    output.write_text(json.dumps(dict(status='PASS',cases=2058,perquery_count=2058*len(gt['query_indices']),runs=runs,
        optimizer_updates=0,official_test_uses=0,limits='SingleMSVR split/seed; independent GT audit is not method superiority.'),indent=2)+'\n')
    print('FREQUENCY_RELATION_CPU_PASS',flush=True)


if __name__ == '__main__':
    main()
