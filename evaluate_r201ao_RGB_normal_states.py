"""Full normal query/gallery states for the fixed AO checkpoint; no training."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from audit_full_official49 import installed_rows,recount,verify,METRICS
from experiment_data import make_loader
from full_evaluation import distance,full_metrics
from official_training_data import full_records,metadata
from run_experiment import write_json
import run_r201ao_RGB_hdm_query_guidance as recipe
from original_identity_anchor import assert_anchor_unchanged


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run=Path(args.run_dir);out=Path(args.output)
    trained=json.loads((run/'result.json').read_text())
    assert trained['status']=='COMPLETE' and trained['epochs']==50
    assert trained['steps']==trained['optimizer_steps']==6563 and trained['amp_skipped_steps']==trained['training_heldout_identities']==0
    assert trained['descriptor_dim']==5120 and json.loads((run/'normal_cpu_audit.json').read_text())['status']=='PASS'
    assert json.loads((run.parent/(run.name+'_exit.json')).read_text())['exit_code']==0
    arguments=argparse.Namespace(**trained['arguments']);assert arguments.dataset=='RGBNT100'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    cfg=recipe.configuration(arguments)
    train,query,gallery,classes,cameras,manifest=full_records(arguments.data_root,'RGBNT100')
    assert (len(train),len(query),len(gallery))==(8675,1715,8575)
    assert json.loads((run/'official_split_manifest.json').read_text())==manifest
    installed={key:installed_rows(Path(arguments.data_root),'RGBNT100',key) for key in ('train','query','gallery')}
    assert all(installed[key]==manifest[key] for key in installed)
    model=recipe.build(arguments,cfg,classes,cameras)
    model.load_state_dict(torch.load(run/'best.pth',map_location='cuda',weights_only=True),strict=True)
    model.eval();assert_anchor_unchanged(model)
    versions={name:value._version for name,value in model.state_dict().items()}
    out.mkdir(exist_ok=False);started=time.time();parts={key:[] for key in ('00','10','01','11')}
    with torch.no_grad():
        for images,_,cam,scene,_ in make_loader(query+gallery,cfg,False,arguments.seed):
            images={key:value.cuda(non_blocking=True) for key,value in images.items()}
            states=model(images,cam_label=cam.cuda(),view_label=scene.cuda(),return_states=True)
            assert set(states)==set(parts)
            for key,value in states.items():
                assert value.shape[1]==5120 and torch.isfinite(value).all()
                parts[key].append((value.float() if key=='00' else F.normalize(value.float(),dim=1)).cpu())
    features_by_state={key:torch.cat(values) for key,values in parts.items()}
    with np.load(run/'best_official_arrays.npz') as expected:
        assert np.array_equal(features_by_state['11'][:1715].numpy(),expected['query_features'])
        assert np.array_equal(features_by_state['11'][1715:].numpy(),expected['gallery_features'])
    arrays=metadata(query,'query')|metadata(gallery,'gallery');raw=dict(arrays);measurements={};audits={}
    for key,values in parts.items():
        features=torch.cat(values);assert features.shape==(10290,5120)
        d=distance(features[:1715],features[1715:])
        summary=full_metrics(d,arrays['query_ids'],arrays['gallery_ids'],arrays['query_cameras'],arrays['gallery_cameras'],
            arrays['query_names'],arrays['query_cameras'],arrays['query_scenes'],out/('state_'+key))
        rows=recount(d,installed['query'],installed['gallery'],'RGBNT100')
        _,error=verify(rows,summary,out/('state_'+key+'.csv'),8575)
        measurements[key]=summary;audits[key]=dict(max_metric_error=error,installed_query_rows=len(rows))
        raw['distances' if key=='11' else 'distances_'+key]=d
    assert all(abs(measurements['11'][key]-trained['full_metrics'][key])<1e-8 for key in METRICS)
    with np.load(run/'best_official_arrays.npz') as expected:assert np.array_equal(raw['distances'],expected['distances'])
    raw['base_query_features']=features_by_state['00'][:1715].numpy()
    raw['base_gallery_features']=features_by_state['00'][1715:].numpy()
    assert versions=={name:value._version for name,value in model.state_dict().items()};assert_anchor_unchanged(model)
    np.savez_compressed(out/'normal_states.npz',**raw)
    write_json(out/'result.json',dict(status='COMPLETE',dataset='RGBNT100',control=arguments.variant,source_run=str(run),
        selected_epoch=trained['best']['epoch'],measurements=measurements,installed_GT_audit=audits,
        train_records=8675,query_records=1715,gallery_records=8575,state_cases=4,neural_passes=1,
        optimizer_updates=0,checkpoint_writes=0,full11_six_metrics_equal_selected_normal=True,full11_features_distance_equal_selected_normal=True,
        state_tensor_versions_unchanged=True,seconds=time.time()-started,
        limits='Same fixed normal mAP-best checkpoint/all official QG; frozen original DeMo identity encoder retained; only AO source-specific HDM query guidance. '
            'Repeated queries are not independent samples; no missing/49/retrained ablation.'))
    print('AO_FULL_NORMAO_FOUR_STATES_COMPLETE',arguments.variant,flush=True)


if __name__=='__main__':main()
