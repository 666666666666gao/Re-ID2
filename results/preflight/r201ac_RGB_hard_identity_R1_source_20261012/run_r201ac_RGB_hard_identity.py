"""AC: deterministic train-only near-identity grouping; the entire X update is reused."""
import hashlib,json,random
from pathlib import Path
import numpy as np
import torch
import experiment_data
from official_training_data import name
from neighbor_identity_schedule import rewrite_epoch
import run_r201x_RGB_auxiliary_fp32 as x

configuration=x.configuration
VARIANTS=x.VARIANTS
ORIGINAL_SAMPLER=experiment_data.RandomIdentitySampler
ORIGINAL_LOADER=experiment_data.make_loader
CURRENT={}
BANK=None
REVISION=('AC changes only the co-occurrence of existing RGBNT100 P8K8 identity groups relative completed X. '
    'Original sampler consumes its RNG first; deterministic train-centroid top3 future-group swaps preserve '
    'each epoch record multiset, batch/update budget and post-sampler RNG. Current images are reencoded. '
    'X FP32 second F auxiliary scorer, loss/BN2/partial0/architecture/teacher42/freshheads42/newAdam/additional50 '
    'and5120 interface unchanged. No test labels choose training negatives. Same ordinary/axis hard schedule. '
    'Full official train/Q/G, normal mAP-best earliest ties; no201NN/missing专项/49/seed search or guaranteed gain.')


class HardIdentitySampler(ORIGINAL_SAMPLER):
    def __init__(self,records,batch_size,num_instances):
        super().__init__(records,batch_size,num_instances)
        assert batch_size==64 and num_instances==8 and len(records)==8675
        observed={name(r):(r[1],r[2],r[3]) for r in records}
        expected={r['name']:(BANK['training_labels'][str(r['identity'])],r['camera'],r['scene'])
            for r in BANK['train_records_metadata']}
        assert observed==expected and len(observed)==len(records)

    def __iter__(self):
        original=list(super().__iter__())
        python_state=random.getstate();numpy_state=np.random.get_state();torch_state=torch.get_rng_state()
        labels={i:r[1] for i,r in enumerate(self.data_source)}
        neighbors={int(pid):ids for pid,ids in BANK['nearest_identity_labels'].items()}
        rewritten,stats=rewrite_epoch(original,labels,neighbors)
        assert random.getstate()==python_state and torch.equal(torch.get_rng_state(),torch_state)
        current_numpy=np.random.get_state()
        assert current_numpy[0]==numpy_state[0] and np.array_equal(current_numpy[1],numpy_state[1]) and current_numpy[2:]==numpy_state[2:]
        CURRENT.update(stats=stats,post_sampler_RNG_exact=True,cursor=0,
            names=[name(self.data_source[i]) for i in rewritten])
        return iter(rewritten)


def make_loader(records,cfg,training,seed):
    if training:
        CURRENT.clear();CURRENT['seed']=seed
    return ORIGINAL_LOADER(records,cfg,training,seed)


def build(args,cfg,classes,cameras):
    global BANK
    model=x.build(args,cfg,classes,cameras)
    bank_path=Path('runs/r201ab_RGB_training_neighbors_20261012/result.json')
    data=bank_path.read_bytes();BANK=json.loads(data)
    assert BANK['status']=='COMPLETE_RGBNT100_TRAIN_ONLY_NEIGHBOR_BANK' and BANK['train_records']==8675 and BANK['train_identities']==classes==50
    assert BANK['query_gallery_forwards']==BANK['optimizer_updates']==BANK['new_weights']==0 and BANK['original_teacher_state_exact']
    assert hashlib.sha256((Path(args.anchor_run_dir)/'best.pth').read_bytes()).hexdigest()==BANK['teacher_sha256']
    assert set(map(int,BANK['nearest_identity_labels']))==set(range(50))
    assert all(set(ids)==set(range(50))-{int(pid)} and len(ids)==49 for pid,ids in BANK['nearest_identity_labels'].items())
    experiment_data.RandomIdentitySampler=HardIdentitySampler
    model.anchor_record['method']=REVISION
    model.anchor_record['hard_identity_sampling']=dict(bank=str(bank_path),bank_sha256=hashlib.sha256(data).hexdigest(),
        training_only=True,P=8,K=8,top_neighbors=3,maximum_swaps_per_batch=1,parameters_added=0,
        original_sampler_RNG_consumed_first=True,epoch_record_multiset_preserved=True)
    return model


def step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant):
    cursor=CURRENT['cursor'];expected=CURRENT['names'][cursor:cursor+64]
    assert list(batch[-1])==expected
    detail=x.step(model,batch,optimizer,scaler,loss_fn,xent,retained,variant)
    assert detail['names']==expected
    detail['hard_identity_sampling']=dict(seed=CURRENT['seed'],batch_index=cursor//64,
        post_sampler_RNG_exact=CURRENT['post_sampler_RNG_exact'],**CURRENT['stats'])
    CURRENT['cursor']+=64
    return detail


if __name__=='__main__':
    runner=x.w.runner;runner.configuration=configuration;runner.assert_anchor_unchanged=x.w.assert_anchor_unchanged
    runner.make_loader=make_loader;runner.write_json=x.w.s.p.parent.parent.parent.k.parent.g.record
    runner.main(builder=build,update=step,variants=VARIANTS,method_revision=REVISION,
        partial_training='Unchanged X partial0/BN2/two backwards. Only train-only deterministic P8K8 identity grouping changes.')
