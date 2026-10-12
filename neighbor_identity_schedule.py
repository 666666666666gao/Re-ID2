"""One deterministic future-group swap per batch; preserve the P8K8 epoch multiset."""
from collections import Counter


def coverage(batches, labels, neighbors):
    covered=0
    for batch in batches:
        present={labels[group[0]] for group in batch}
        assert len(present)==8
        covered+=sum(bool((present-{pid}) & set(neighbors[pid][:3])) for pid in present)
    return covered,8*len(batches)


def rewrite_epoch(records,labels,neighbors):
    assert records and len(records)%64==0
    batches=[[list(records[offset+group*8:offset+(group+1)*8]) for group in range(8)]
             for offset in range(0,len(records),64)]
    assert all(len(group)==8 and len({labels[key] for key in group})==1 for batch in batches for group in batch)
    before,anchors=coverage(batches,labels,neighbors);swaps=0
    for index,batch in enumerate(batches):
        anchor_slot=index%8;anchor=labels[batch[anchor_slot][0]]
        present={labels[group[0]] for group in batch}
        if (present-{anchor}) & set(neighbors[anchor][:3]):continue
        displaced_slot=(anchor_slot+7)%8;displaced=labels[batch[displaced_slot][0]]
        selected=None
        for target in neighbors[anchor][:3]:
            for later in range(index+1,len(batches)):
                other=batches[later]
                other_ids={labels[group[0]] for group in other}
                if target not in other_ids or displaced in other_ids:continue
                slot=next(i for i,group in enumerate(other) if labels[group[0]]==target)
                selected=(later,slot);break
            if selected is not None:break
        if selected is not None:
            later,slot=selected
            batch[displaced_slot],batches[later][slot]=batches[later][slot],batch[displaced_slot]
            swaps+=1
    result=[key for batch in batches for group in batch for key in group]
    assert len(result)==len(records) and Counter(result)==Counter(records)
    after,anchors_after=coverage(batches,labels,neighbors);assert anchors_after==anchors
    return result,dict(batches=len(batches),record_positions=len(records),identity_anchors=anchors,
        swaps=swaps,top3_covered_before=before,top3_covered_after=after,
        original_record_multiset_exact=True,P=8,K=8,new_random_draws=0)
