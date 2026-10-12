"""Full normal query/gallery states for the fixed AE checkpoint; no training."""
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
import run_r201y_MS_K4 as y
import msvr_strong_staged_identity as stage


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run=Path(args.run_dir);out=Path(args.output)
    trained=json.loads((run/'result.json').read_text())
    assert trained['status']=='COMPLETE' and trained['epochs']==50
    assert trained['steps']==trained['optimizer_steps']==705 and trained['amp_skipped_steps']==trained['training_heldout_identities']==0
    assert trained['descriptor_dim']==5120 and json.loads((run/'normal_cpu_audit.json').read_text())['status']=='PASS'
    assert json.loads((run.parent/(run.name+'_exit.json')).read_text())['exit_code']==0
    arguments=argparse.Namespace(**trained['arguments']);assert arguments.dataset=='MSVR310'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    cfg=y.configuration(arguments)
    train,query,gallery,classes,cameras,manifest=full_records(arguments.data_root,'MSVR310')
    assert (len(train),len(query),len(gallery))==(1032,591,1055)
    assert json.loads((run/'official_split_manifest.json').read_text())==manifest
    installed={key:installed_rows(Path(arguments.data_root),'MSVR310',key) for key in ('train','query','gallery')}
    assert all(installed[key]==manifest[key] for key in installed)
    model=stage.build(arguments,cfg,classes,cameras)
    model.stage_epoch=trained['best']['epoch']
    model.adaptation_phase='relations' if model.staged_adaptation and model.stage_epoch>stage.HEAD_EPOCHS else 'outlets'
    model.generalFusion.requires_grad_(model.adaptation_phase=='relations')
    model.load_state_dict(torch.load(run/'best.pth',map_location='cuda',weights_only=True),strict=True)
    model.eval();stage.assert_retained(model)
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
    arrays=metadata(query,'query')|metadata(gallery,'gallery');raw=dict(arrays);measurements={};audits={}
    for key,values in parts.items():
        features=torch.cat(values);assert features.shape==(1646,5120)
        d=distance(features[:591],features[591:])
        summary=full_metrics(d,arrays['query_ids'],arrays['gallery_ids'],arrays['query_scenes'],arrays['gallery_scenes'],
            arrays['query_names'],arrays['query_cameras'],arrays['query_scenes'],out/('state_'+key))
        rows=recount(d,installed['query'],installed['gallery'],'MSVR310')
        _,error=verify(rows,summary,out/('state_'+key+'.csv'),1055)
        measurements[key]=summary;audits[key]=dict(max_metric_error=error,installed_query_rows=len(rows))
        raw['distances' if key=='11' else 'distances_'+key]=d
    assert all(abs(measurements['11'][key]-trained['full_metrics'][key])<1e-8 for key in METRICS)
    assert versions=={name:value._version for name,value in model.state_dict().items()};stage.assert_retained(model)
    np.savez_compressed(out/'normal_states.npz',**raw)
    write_json(out/'result.json',dict(status='COMPLETE',dataset='MSVR310',control=arguments.variant,source_run=str(run),
        selected_epoch=trained['best']['epoch'],measurements=measurements,installed_GT_audit=audits,
        train_records=1032,query_records=591,gallery_records=1055,state_cases=4,neural_passes=1,
        optimizer_updates=0,checkpoint_writes=0,full11_six_metrics_equal_selected_normal=True,
        state_tensor_versions_unchanged=True,seconds=time.time()-started,
        limits='Same fixed normal mAP-best checkpoint/all official QG. Frozen-axis00 retains stronger teacher private reference. '
            'Staged00 is current adapted base, not original or strong teacher. Noexpert00/10/01/11 coincide. '
            'Repeated queries are not independent samples; no missing/49/retrained ablation.'))
    print('AE_FULL_NORMAL_FOUR_STATES_COMPLETE',arguments.variant,flush=True)


if __name__=='__main__':main()
