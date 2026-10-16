"""Full official normal four states of a fixed head16-reference expert best; no update."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from audit_full_official49 import installed_rows,recount,verify,METRICS
from experiment_data import make_loader
from full_evaluation import distance,full_metrics
from official_training_data import full_records,metadata
from run_experiment import write_json
from run_msvr_head16_experts import build,configuration,assert_retained


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run=Path(args.run_dir);out=Path(args.output);trained=json.loads((run/'result.json').read_text())
    assert trained['status']=='COMPLETE' and trained['epochs']==50 and trained['descriptor_dim']==5632
    assert trained['steps']==705 and 0<trained['optimizer_steps']<=705
    assert trained['amp_skipped_steps']==705-trained['optimizer_steps'] and trained['training_heldout_identities']==0
    assert json.loads((run/'normal_cpu_audit.json').read_text())['status']=='PASS'
    assert json.loads((run.parent/(run.name+'_exit.json')).read_text())['exit_code']==0
    arguments=argparse.Namespace(**trained['arguments']);assert arguments.dataset=='MSVR310'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;cfg=configuration(arguments)
    train,query,gallery,classes,cameras,manifest=full_records(arguments.data_root,'MSVR310')
    assert (len(train),len(query),len(gallery))==(1032,591,1055)
    assert json.loads((run/'official_split_manifest.json').read_text())==manifest
    installed={s:installed_rows(Path(arguments.data_root),'MSVR310',s) for s in ('train','query','gallery')}
    assert all(installed[s]==manifest[s] for s in installed)
    model=build(arguments,cfg,classes,cameras)
    model.load_state_dict(torch.load(run/'best.pth',map_location='cuda',weights_only=True),strict=True);model.eval();assert_retained(model)
    versions={name:value._version for name,value in model.state_dict().items()}
    out.mkdir(exist_ok=False);started=time.time();parts={key:[] for key in ('00','10','01','11')}
    with torch.no_grad():
        for images,_,cam,scene,_ in make_loader(query+gallery,cfg,False,arguments.seed):
            images={key:value.cuda(non_blocking=True) for key,value in images.items()}
            states=model(images,cam_label=cam.cuda(),view_label=scene.cuda(),return_states=True);assert set(states)==set(parts)
            for key,value in states.items():
                assert value.shape[1]==5632 and torch.isfinite(value).all()
                parts[key].append(F.normalize(value.float(),dim=1).cpu())
    arrays=metadata(query,'query')|metadata(gallery,'gallery');raw=dict(arrays);measurements={};audits={}
    with np.load(run/'best_official_arrays.npz') as saved:normal={key:saved[key].copy() for key in saved.files}
    teacher=Path(arguments.anchor_run_dir)
    selected=json.loads((teacher/'result.json').read_text())
    expected_rows=list(csv.DictReader((teacher/'best_per_query.csv').open(encoding='utf-8',newline='')))
    for key,values in parts.items():
        features=torch.cat(values);assert features.shape==(1646,5632)
        q,g=features[:591],features[591:];d=distance(q,g)
        summary=full_metrics(d,arrays['query_ids'],arrays['gallery_ids'],arrays['query_scenes'],arrays['gallery_scenes'],
            arrays['query_names'],arrays['query_cameras'],arrays['query_scenes'],out/('state_'+key))
        _,error=verify(recount(d,installed['query'],installed['gallery'],'MSVR310'),summary,out/('state_'+key+'.csv'),1055)
        measurements[key]=summary;audits[key]=dict(max_metric_error=error,installed_query_rows=591)
        raw['distances' if key=='11' else 'distances_'+key]=d
        if key=='11':
            assert np.array_equal(q.numpy(),normal['query_features']) and np.array_equal(g.numpy(),normal['gallery_features']) and np.array_equal(d,normal['distances'])
        if key=='00':
            assert all(abs(summary[m]-selected['full_metrics'][m])<1e-8 for m in METRICS)
            assert list(csv.DictReader((out/'state_00.csv').open(encoding='utf-8',newline='')))==expected_rows
    assert all(abs(measurements['11'][m]-trained['full_metrics'][m])<1e-8 for m in METRICS)
    assert versions=={name:value._version for name,value in model.state_dict().items()};assert_retained(model)
    np.savez_compressed(out/'normal_states.npz',**raw)
    write_json(out/'result.json',dict(status='COMPLETE',dataset='MSVR310',control=arguments.variant,source_run=str(run),
        selected_epoch=trained['best']['epoch'],measurements=measurements,installed_GT_audit=audits,
        train_records=1032,query_records=591,gallery_records=1055,state_cases=4,neural_passes=1,
        optimizer_updates=0,checkpoint_writes=0,full11_features_distance_exact_selected_normal=True,
        frozen00_full_teacher_six_per_query_equal=True,state_tensor_versions_unchanged=True,seconds=time.time()-started,
        limits='Same normal mAP-best/full official QG; 00 retains existing full strongteacher. Actual 10/01/11 use private relation corrections plus the same public block. Frozen state intervention is not retrained ablation, missing robustness or multi-seed confirmation.'))
    print('AH_FULL_NORMAL_FOUR_STATES_COMPLETE',arguments.variant,flush=True)


if __name__=='__main__':main()
