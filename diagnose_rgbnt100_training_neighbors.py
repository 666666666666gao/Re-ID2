"""Frozen original DeMo: training-only neighbor evidence, no optimizer or test inference."""
import argparse, hashlib, json, math, os
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from full_evaluation import extract
from official_training_data import full_records,metadata,name
from run_experiment import build,configuration,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--anchor-run-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='2'
    anchor=Path(args.anchor_run_dir);out=Path(args.output)
    trained=json.loads((anchor/'result.json').read_text())
    assert trained['status']=='COMPLETE' and trained['epochs']==50 and trained['descriptor_dim']==5120
    assert trained['arguments']['dataset']=='RGBNT100' and trained['arguments']['variant']=='demo' and trained['arguments']['seed']==42
    assert trained['training_heldout_identities']==0 and trained['classes']==50 and trained['train_records']==8675
    assert json.loads((anchor.parent/(anchor.name+'_exit.json')).read_text())['exit_code']==0
    arguments=argparse.Namespace(**trained['arguments']);cfg=configuration(arguments)
    assert cfg.dump()==trained['config']
    train,query,gallery,classes,cameras,manifest=full_records(arguments.data_root,'RGBNT100')
    assert manifest==json.loads((anchor/'official_split_manifest.json').read_text())
    assert len(train)==8675 and len(query)==1715 and len(gallery)==8575 and classes==50
    assert set(name(r) for r in train).isdisjoint(name(r) for r in query+gallery)
    weight=anchor/'best.pth';weight_sha=hashlib.sha256(weight.read_bytes()).hexdigest()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    model=build(arguments,cfg,classes,cameras)
    saved=torch.load(weight,map_location='cpu',weights_only=True)
    model.load_state_dict(saved,strict=True);model.eval();model.requires_grad_(False)
    batches=[]
    def count_forward(module,inputs,output):
        assert output.ndim==2 and output.shape[1]==5120
        batches.append(len(output))
    handle=model.register_forward_hook(count_forward)
    features,runtime=extract(model,train,cfg,42);handle.remove()
    assert features.shape==(8675,5120) and torch.isfinite(features).all()
    assert len(batches)==math.ceil(8675/64) and sum(batches)==8675
    assert all(torch.equal(v.detach().cpu(),saved[k]) for k,v in model.state_dict().items())
    assert hashlib.sha256(weight.read_bytes()).hexdigest()==weight_sha
    labels=torch.tensor([r[1] for r in train]);assert set(labels.tolist())==set(range(50))
    centers=torch.stack([features[labels==label].mean(0) for label in range(50)])
    centers=torch.nn.functional.normalize(centers,dim=1)
    distances=(2-2*centers@centers.T).numpy();np.fill_diagonal(distances,np.inf)
    neighbors=np.argsort(distances,axis=1,kind='stable')[:,:49]
    assert all(set(row.tolist())==set(range(50))-{label} for label,row in enumerate(neighbors))
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'official_split_manifest.json',manifest)
    raw=out/'training_features.npz'
    np.savez_compressed(raw,features=features.numpy(),centers=centers.numpy(),**metadata(train,'train'))
    result=dict(status='COMPLETE_RGBNT100_TRAIN_ONLY_NEIGHBOR_BANK',at=datetime.now().astimezone().isoformat(timespec='seconds'),
        dataset='RGBNT100',source='Original DeMo50 seed42 fixed normal mAP-best, no expert retraining',
        selected_teacher_epoch=trained['best']['epoch'],teacher_sha256=weight_sha,descriptor_dim=5120,
        train_records=8675,train_identities=50,training_heldout_identities=0,forward_batches=len(batches),
        record_forwards=sum(batches),query_gallery_forwards=0,optimizer_updates=0,new_weights=0,
        original_teacher_state_exact=True,runtime=runtime,training_labels=manifest['training_labels'],
        train_records_metadata=manifest['train'],nearest_identity_labels={str(i):row.tolist() for i,row in enumerate(neighbors)},
        limits='Centroid neighbors use official training images/labels only. Existing benchmark-selected teacher is disclosed. Graph is a sampling diagnostic, not a method score, test-error oracle selector, or proof that hard sampling improves unseen identity retrieval. Full training records are retained; no changes to live AA or protected K201.')
    write_json(out/'result.json',result)
    files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (raw,out/'result.json',out/'official_split_manifest.json')}
    print('TRAIN_NEIGHBOR_RESULT '+json.dumps(dict(status=result['status'],files=files,forward_batches=len(batches),record_forwards=sum(batches),optimizer_updates=0)),flush=True)


if __name__=='__main__':main()
