"""One best checkpoint, all7 banks, all49 pairs and all4 closed expert states."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_full_official49 import SETS
from experiment_data import make_loader
from full_evaluation import distance, full_metrics
from missing_evaluation import mask_images
from official_training_data import full_records, metadata
from original_identity_anchor import assert_anchor_unchanged
from rgbnt201_identity_outlet import build
from run_experiment import configuration, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run,out=Path(args.run_dir),Path(args.output)
    trained=json.loads((run/'result.json').read_text())
    assert trained['status']=='COMPLETE' and trained['epochs']==50 and trained['descriptor_dim']==5120
    assert trained['training_heldout_identities']==0
    assert json.loads((run.parent/(run.name+'_exit.json')).read_text())['exit_code']==0
    arguments=argparse.Namespace(**trained['arguments'])
    cfg=configuration(arguments)
    train,query,gallery,classes,cameras,manifest=full_records(arguments.data_root,arguments.dataset)
    assert (len(train),len(query),len(gallery))==(3951,836,836)
    assert json.loads((run/'official_split_manifest.json').read_text())==manifest
    torch.set_num_threads(4)
    model=build(arguments,cfg,classes,cameras)
    model.load_state_dict(torch.load(run/'best.pth',map_location='cuda',weights_only=True),strict=True)
    model.eval()
    assert_anchor_unchanged(model)
    versions={name:value._version for name,value in model.state_dict().items()}
    out.mkdir(exist_ok=False)
    started=time.time()
    arrays=metadata(query,'query') | metadata(gallery,'gallery')
    saved=np.load(run/'best_official_arrays.npz')
    assert all(np.array_equal(saved[key],value) for key,value in arrays.items())
    banks={}
    with torch.no_grad():
        for available,missing in SETS.items():
            parts={state:[] for state in ('00','10','01','11')}
            for images,_,cam,scene,_ in make_loader(query+gallery,cfg,False,arguments.seed):
                images={key:value.cuda(non_blocking=True) for key,value in images.items()}
                states=model(mask_images(images,missing),cam_label=cam.cuda(),view_label=scene.cuda(),return_states=True)
                for state,value in states.items():
                    assert value.shape[1]==5120 and torch.isfinite(value).all()
                    parts[state].append(F.normalize(value.float(),dim=1).cpu())
            banks[available]={}
            for state,values in parts.items():
                feature=torch.cat(values)
                assert list(feature.shape)==[1672,5120]
                banks[available][state]=(feature[:836],feature[836:])
    assert np.array_equal(banks['RNT']['11'][0].numpy(),saved['query_features'])
    assert np.array_equal(banks['RNT']['11'][1].numpy(),saved['gallery_features'])
    measurements,state_measurements={},{}
    for qs in SETS:
        for gs in SETS:
            condition='q_'+qs+'_g_'+gs
            state_measurements[condition]={}
            raw={}
            for state in ('00','10','01','11'):
                d=distance(banks[qs][state][0],banks[gs][state][1])
                stem=condition if state=='11' else condition+'_state'+state
                values=full_metrics(d,arrays['query_ids'],arrays['gallery_ids'],
                    arrays['query_cameras'],arrays['gallery_cameras'],arrays['query_names'],
                    arrays['query_cameras'],arrays['query_scenes'],out/stem)
                state_measurements[condition][state]=values
                raw['distances' if state=='11' else 'distances_'+state]=d
                if state=='11':
                    measurements[condition]=values
            if qs==gs=='RNT':
                assert np.array_equal(raw['distances'],saved['distances'])
            np.savez_compressed(out/(condition+'.npz'),**raw)
    assert versions=={name:value._version for name,value in model.state_dict().items()}
    assert_anchor_unchanged(model)
    write_json(out/'result.json',dict(status='COMPLETE',dataset='RGBNT201',variant=arguments.variant,
        model_arguments=vars(arguments),seed=42,selected_epoch=trained['best']['epoch'],
        measurements=measurements,state_measurements=state_measurements,state_cases=196,
        query_records=836,gallery_records=836,train_records=3951,training_heldout_identities=0,
        optimizer_updates=0,normal_feature_max_error=0,state_tensor_versions_unchanged=True,
        selection=trained['checkpoint_rule'],seconds=time.time()-started,
        limits='Frozen best chosen by full benchmark mAP; one seed. Fourstates reuse the same836 queries. '
               'No shared retrieval coordinates: the12 disjoint-source pairs lack cross-source identity comparison. '
               'Report ties/numerical behavior rather than claiming these cases solved.'))
    print('RGBNT201_IDENTITY_FULL49_FOURSTATE_COMPLETE',flush=True)


if __name__=='__main__':
    main()
