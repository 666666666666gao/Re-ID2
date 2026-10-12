"""CPU recount of closed RGBNT100 batch exposure to train-only centroid neighbors."""
from pathlib import Path
from datetime import datetime
import json


def main():
    repo=Path(__file__).resolve().parent
    root=repo/'results/r201ab_RGB_training_neighbors_20261012'
    bank=json.loads((root/'result.json').read_text())
    assert bank['status']=='COMPLETE_RGBNT100_TRAIN_ONLY_NEIGHBOR_BANK'
    labels=bank['training_labels'];by_name={r['name']:labels[str(r['identity'])] for r in bank['train_records_metadata']}
    assert len(by_name)==8675 and len(labels)==50
    nearest=bank['nearest_identity_labels'];seen=set();observations=[]
    for variant in ('axis_shared','frequency_shared'):
        run=repo/'results/r201x_RGB_auxiliary_fp32_20261012/RGBNT100/training'/('RGBNT100_r201x_'+variant+'_s42')
        data=json.loads((run/'result.json').read_text());assert data['status']=='COMPLETE' and data['epochs']==50 and data['optimizer_steps']==6563
        rows=[json.loads(line) for line in (run/'batch_orders.jsonl').read_text().splitlines()]
        assert len(rows)==6563
        orders=[];covered=0;anchors=0
        for row in rows:
            names=row['names'];pids=[by_name[n] for n in names];present=set(pids)
            assert len(names)==64 and len(present)==8 and all(pids.count(p)==8 for p in present)
            orders.append((row['epoch'],row['step'],names));seen.update(names)
            for pid in present:
                anchors+=1;covered+=bool((present-{pid}) & set(nearest[str(pid)][:3]))
        assert len(seen)==8675
        observations.append(dict(variant=variant,orders=orders,batch_identity_anchors=anchors,
            anchors_with_top3_centroid_neighbor=covered,top3_coverage_fraction=covered/anchors))
    assert observations[0]['orders']==observations[1]['orders']
    for row in observations:del row['orders']
    output=root/'coverage.json';assert not output.exists()
    result=dict(status='COMPLETE_CPU_TRAIN_ONLY_NEIGHBOR_EXPOSURE',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        train_records=8675,train_identities=50,paired_closed_X_orders_exact=True,
        observations=observations,new_neural_calls=0,new_optimizer_updates=0,
        limits='Only centroid-relative training negative exposure; no test identities or labels choose training samples. Low measured exposure would motivate a matched sampling test, not prove it causes the current58 test errors. Future sampling must keep P8K8, record multiset and update budget matched; current sampler unchanged.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
