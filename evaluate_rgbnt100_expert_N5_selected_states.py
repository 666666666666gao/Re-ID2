"""Four frozen normal states for the final selected N5 seed and its matched control."""
import argparse,csv,hashlib,json,time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from audit_full_official49 import installed_rows,recount,verify,METRICS
from experiment_data import make_loader
from full_evaluation import distance,full_metrics
from official_training_data import full_records,metadata
from original_identity_anchor import assert_anchor_unchanged
from run_experiment import write_json
import identity_coordinate_rgbnt100_expert_seeds as seed_factory
from rgbnt100_expert_seed_recipe import build_selected,recipe_module


def main():
    parser=argparse.ArgumentParser()
    for key in ('run-dir','output','five-selection'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();run=Path(args.run_dir);out=Path(args.output)
    selected=json.loads(Path(args.five_selection).read_text())
    assert selected['status']=='ALL_FIVE_COMPLETE_SELECTED'
    trained=json.loads((run/'result.json').read_text())
    assert trained['status']=='COMPLETE' and trained['epochs']==50 and trained['steps']==trained['optimizer_steps']>0
    assert trained['amp_skipped_steps']==trained['training_heldout_identities']==0 and trained['descriptor_dim']==5120
    assert json.loads((run/'normal_cpu_audit.json').read_text())['status']=='PASS'
    assert json.loads((run.parent/(run.name+'_exit.json')).read_text())['exit_code']==0
    arguments=argparse.Namespace(**trained['arguments']);key=arguments.expert_recipe
    selected_seed=selected['main_best_single']['axis_shared']['seed']
    assert arguments.dataset=='RGBNT100' and arguments.seed==selected_seed and selected_seed in (43,44,45,46)
    assert key==selected['recipe'] and arguments.variant in ('frequency_shared','axis_shared')
    restoration=json.loads((run/'selected_states_restore.json').read_text())
    for filename in ('best.pth','best_official_arrays.npz'):
        f=run/filename;row=restoration['files'][filename]
        assert f.stat().st_size==row['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256']
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    parent=recipe_module(key);seed_factory.install_fixed_teacher_expert_seed_builder()
    cfg=parent.configuration(arguments)
    train,query,gallery,classes,cameras,manifest=full_records(arguments.data_root,'RGBNT100')
    assert (len(train),len(query),len(gallery),classes)==(8675,1715,8575,50)
    assert json.loads((run/'official_split_manifest.json').read_text())==manifest
    installed={s:installed_rows(Path(arguments.data_root),'RGBNT100',s) for s in ('train','query','gallery')}
    assert all(installed[s]==manifest[s] for s in installed)
    model=build_selected(key,arguments,cfg,classes,cameras)
    model.load_state_dict(torch.load(run/'best.pth',map_location='cuda',weights_only=True),strict=True)
    model.eval();assert_anchor_unchanged(model)
    versions={name:value._version for name,value in model.state_dict().items()}
    out.mkdir(parents=True,exist_ok=False);started=time.time();parts={s:[] for s in ('00','10','01','11')}
    with torch.no_grad():
        for images,_,cam,scene,_ in make_loader(query+gallery,cfg,False,arguments.seed):
            images={name:value.cuda(non_blocking=True) for name,value in images.items()}
            states=model(images,cam_label=cam.cuda(),view_label=scene.cuda(),return_states=True)
            assert set(states)==set(parts)
            for state,value in states.items():
                assert value.shape[1]==5120 and torch.isfinite(value).all()
                parts[state].append((value.float() if state=='00' else F.normalize(value.float(),dim=1)).cpu())
    features={s:torch.cat(values) for s,values in parts.items()}
    with np.load(run/'best_official_arrays.npz') as cache:
        assert np.array_equal(features['11'][:1715].numpy(),cache['query_features'])
        assert np.array_equal(features['11'][1715:].numpy(),cache['gallery_features'])
    arrays=metadata(query,'query')|metadata(gallery,'gallery');raw=dict(arrays);measurements={};audits={}
    for state,feature in features.items():
        assert feature.shape==(10290,5120)
        d=distance(feature[:1715],feature[1715:])
        summary=full_metrics(d,arrays['query_ids'],arrays['gallery_ids'],arrays['query_cameras'],arrays['gallery_cameras'],
            arrays['query_names'],arrays['query_cameras'],arrays['query_scenes'],out/('state_'+state))
        rows=recount(d,installed['query'],installed['gallery'],'RGBNT100')
        _,error=verify(rows,summary,out/('state_'+state+'.csv'),8575)
        measurements[state]=summary;audits[state]=dict(max_metric_error=error,installed_query_rows=len(rows))
        raw['distances' if state=='11' else 'distances_'+state]=d
    assert all(abs(measurements['11'][m]-trained['full_metrics'][m])<1e-8 for m in METRICS)
    with np.load(run/'best_official_arrays.npz') as cache:assert np.array_equal(raw['distances'],cache['distances'])
    original=Path(arguments.anchor_run_dir)
    before=list(csv.DictReader((original/'best_per_query.csv').open(encoding='utf-8',newline='')))
    after=list(csv.DictReader((out/'state_00.csv').open(encoding='utf-8',newline='')))
    assert len(before)==len(after)==1715
    for x,y in zip(before,after):
        assert set(x)==set(y)
        assert all(abs(float(x[name])-float(y[name]))<1e-12 if name in ('AP','INP') else x[name]==y[name] for name in x)
    raw['base_query_features']=features['00'][:1715].numpy();raw['base_gallery_features']=features['00'][1715:].numpy()
    assert versions=={name:value._version for name,value in model.state_dict().items()};assert_anchor_unchanged(model)
    np.savez_compressed(out/'normal_states.npz',**raw)
    write_json(out/'result.json',dict(status='COMPLETE',recipe=key,seed=arguments.seed,variant=arguments.variant,
        selected_epoch=trained['best']['epoch'],query_records=1715,gallery_records=8575,state_cases=4,
        measurements=measurements,installed_GT_audit=audits,full11_features_distance_equal_selected_normal=True,
        original00_per_query_equal=True,state_tensor_versions_unchanged=True,optimizer_updates=0,checkpoint_writes=0,
        seconds=time.time()-started,limits='Final locked N5 axis seed and its same-seed ordinary control only. Fixed normal checkpoint, full official Q/G; frozen intervention is not a retrained ablation. Seed42 uses already completed parent states and must not be rerun through this tool. No missing or whole-method completion claim.'))
    print('RGBNT100_N5_SELECTED_FOUR_STATES_COMPLETE',key,arguments.seed,arguments.variant,flush=True)


if __name__=='__main__':main()
