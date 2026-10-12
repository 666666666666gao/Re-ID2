"""Installed GT audit of one completed AH full5632 normal checkpoint."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from audit_full_official49 import installed_rows,recount,verify,METRICS


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);run=Path(parser.parse_args().run_dir)
    trained=json.loads((run/'result.json').read_text())
    assert trained['status']=='COMPLETE' and trained['epochs']==50 and trained['training_heldout_identities']==0
    assert trained['arguments']['dataset']=='MSVR310' and trained['descriptor_dim']==5632
    assert json.loads((run.parent/(run.name+'_exit.json')).read_text())['exit_code']==0
    splits={s:installed_rows(Path(trained['arguments']['data_root']),'MSVR310',s) for s in ('train','query','gallery')}
    manifest=json.loads((run/'official_split_manifest.json').read_text());assert all(splits[s]==manifest[s] for s in splits)
    ids={r['identity'] for r in splits['train']};assert ids.isdisjoint({r['identity'] for r in splits['query']+splits['gallery']})
    assert manifest['training_labels']=={str(pid):i for i,pid in enumerate(sorted(ids))}
    assert {s:len(v) for s,v in splits.items()}==dict(train=1032,query=591,gallery=1055)
    orders=[json.loads(line) for line in (run/'batch_orders.jsonl').read_text().splitlines()]
    assert {n for row in orders for n in row['names']}=={r['name'] for r in splits['train']}
    assert len(orders)==trained['steps']==trained['optimizer_steps']==sum(r['optimizer_updated'] for r in orders)==705
    assert trained['amp_skipped_steps']==0
    epochs=list(csv.DictReader((run/'epochs.csv').open(encoding='utf-8',newline='')))
    assert [int(r['epoch']) for r in epochs]==list(range(1,51))
    selected=max(epochs,key=lambda r:float(r['mAP']));assert int(selected['epoch'])==trained['best']['epoch']
    assert all(abs(float(selected[k])-trained['full_metrics'][k])<1e-8 for k in METRICS)
    with np.load(run/'best_official_arrays.npz') as arrays:
        for side in ('query','gallery'):
            for suffix,key in (('names','name'),('ids','identity'),('cameras','camera'),('scenes','scene')):
                assert arrays[side+'_'+suffix].tolist()==[r[key] for r in splits[side]]
            features=arrays[side+'_features'];assert features.shape==(len(splits[side]),5632) and np.isfinite(features).all()
            assert np.max(np.abs((features[:,:5120]**2).sum(1)-.75))<1e-5
            assert np.max(np.abs((features[:,5120:]**2).sum(1)-.25))<1e-5
        rows=recount(arrays['distances'],splits['query'],splits['gallery'],'MSVR310')
    reported=json.loads((run/'best_per_query.json').read_text())
    metrics,error=verify(rows,reported,run/'best_per_query.csv',1055)
    assert all(abs(metrics[k]-trained['full_metrics'][k])<1e-8 for k in METRICS)
    (run/'normal_cpu_audit.json').write_text(json.dumps(dict(status='PASS',dataset='MSVR310',descriptor_dim=5632,
        selected_epoch=trained['best']['epoch'],installed_split_counts=dict(train=1032,query=591,gallery=1055),six_metrics=metrics,
        max_metric_error=error,CMC50_and_per_query_and_groups=True,full_training_coverage=True,new_optimizer_updates=0,
        missing_evaluation=False,limits='Full normal condition of a benchmark-selected weight; not a missing evaluation or independent final test.'),indent=2)+'\n')
    print('AH_FULL_NORMAL_INSTALLED_GT_AUDIT_PASS',flush=True)


if __name__=='__main__':main()
