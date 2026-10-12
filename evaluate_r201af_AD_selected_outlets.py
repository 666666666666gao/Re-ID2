"""Observe fixed AD RGBNT100 selected outlets; no parameters or inference path change."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from audit_full_official49 import recount, verify, METRICS
from audit_full_official_condition import installed_context
from audit_shared_identity_states import changes
from experiment_data import make_loader
from full_evaluation import distance, full_metrics
from official_training_data import full_records, metadata
from original_identity_anchor import assert_anchor_unchanged
from run_experiment import write_json
import run_r201ad_RGB_hard as recipe

PREFIX = 'R201AF_STREAM '
STAGES = ('base00', 'normal11', 'M_raw_pool', 'F_raw_pool', 'M_routed_pool',
          'F_routed_pool', 'M_projection_input', 'F_projection_input',
          'M_projection_output', 'F_projection_output', 'M_unit_direction',
          'F_unit_direction', 'common_context', 'M_projected_auxiliary', 'F_projected_auxiliary')


def emit(value):
    print(PREFIX + json.dumps(value), flush=True)


@torch.no_grad()
def extract(model, records, cfg, seed):
    parts = {stage: [] for stage in STAGES}
    captured = {}
    conditions = model.router.conditions

    def record_conditions(modality, frequency):
        captured['m0'], captured['f0'] = modality, frequency
        return conditions(modality, frequency)

    def projection_hook(key):
        def hook(module, inputs, output):
            assert key + '_input' not in captured
            captured[key + '_input'], captured[key + '_output'] = inputs[0], output
        return hook

    def calibrator_hook(module, inputs, output):
        assert 'base' not in captured
        captured['base'], captured['modality'], captured['frequency'] = inputs

    model.router.conditions = record_conditions
    hooks = [model.modality_projection.register_forward_hook(projection_hook('M')),
             model.frequency_projection.register_forward_hook(projection_hook('F')),
             model.calibrator.register_forward_hook(calibrator_hook)]
    count = 0
    for images, _, cam, scene, _ in make_loader(records, cfg, False, seed):
        captured.clear()
        images = {key: value.cuda(non_blocking=True) for key, value in images.items()}
        fused = model(images, cam_label=cam.cuda(), view_label=scene.cuda())
        n = len(fused)
        assert set(captured) == {'m0', 'f0', 'base', 'modality', 'frequency',
                                 'M_input', 'M_output', 'F_input', 'F_output'}
        eligible = torch.ones((n, 7), device=fused.device, dtype=torch.bool)
        available = torch.ones((n, 3), device=fused.device, dtype=torch.bool)
        assert captured['M_input'].shape == captured['F_input'].shape == (n, 7, 512)
        assert torch.equal(captured['modality'], captured['M_input'])
        values = dict(base00=captured['base'][:, :5120], normal11=fused,
            M_raw_pool=model.modality_expert.pool(captured['m0'], eligible).mean(1),
            F_raw_pool=model.frequency_expert.pool(captured['f0'], available, model.structured),
            M_routed_pool=captured['M_input'].mean(1), F_routed_pool=captured['frequency'],
            M_projection_input=captured['M_input'].flatten(1),
            F_projection_input=captured['F_input'].flatten(1),
            M_projection_output=captured['M_output'].flatten(1),
            F_projection_output=captured['F_output'].flatten(1),
            M_unit_direction=F.normalize(captured['M_output'].float(), dim=-1).flatten(1),
            F_unit_direction=F.normalize(captured['F_output'].float(), dim=-1).flatten(1),
            common_context=captured['base'][:, 5120:],
            M_projected_auxiliary=actual_auxiliary(model.modality_expert.pool_score, captured['M_output'], 'M'),
            F_projected_auxiliary=actual_auxiliary(model.frequency_expert.pool_score, captured['F_output'], 'F'))
        assert set(values) == set(STAGES)
        for key, value in values.items():
            assert torch.isfinite(value).all() and (value.float().norm(dim=1) > 0).all()
            parts[key].append(F.normalize(value.float(), dim=1).cpu())
        count += n
    assert count == len(records)
    model.router.conditions = conditions
    for hook in hooks:
        hook.remove()
    return {key: torch.cat(values) for key, values in parts.items()}



def actual_auxiliary(scorer, output, expert):
    # AD's installed 3D forward asserts training=True. Read exactly its FP32
    # bias-free linear arithmetic here, without toggling eval or BatchNorm.
    assert scorer.bias is None and not scorer.training
    units=F.normalize(output.float(), dim=-1)
    weights=F.linear(units, scorer.weight.float()).squeeze(-1).softmax(1)
    return (units*weights[...,None]*7).mean(1) if expert=='M' else (units*weights[...,None]).sum(1)


def main():
    parser=argparse.ArgumentParser()
    for key in ('run-dir','normal-arrays','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--mode',choices=('smoke','full'),required=True)
    args=parser.parse_args();run,normal,out=map(Path,(args.run_dir,args.normal_arrays,args.output))
    trained,dataset,installed=installed_context(run)
    assert dataset=='RGBNT100' and trained['descriptor_dim']==5120
    assert trained['method_revision'].startswith('AD hard sampling candidate')
    assert trained['parameters']==99585158 and trained['trainable_parameters']==1887366
    assert trained['arguments']['variant'] in ('frequency_shared','axis_shared')
    assert json.loads((run.parent/(run.name+'_exit.json')).read_text())['exit_code']==0
    assert json.loads((run/'normal_cpu_audit.json').read_text())['status']=='PASS'
    assert trained['amp_skipped_steps']==trained['training_heldout_identities']==0
    assert trained['training_coverage']==dict(eligible=8675,visited=8675,unvisited=[])
    arguments=argparse.Namespace(**trained['arguments']);cfg=recipe.configuration(arguments)
    train,query,gallery,classes,cameras,manifest=full_records(arguments.data_root,dataset)
    assert (len(train),len(query),len(gallery),classes)==(8675,1715,8575,50)
    assert manifest==json.loads((run/'official_split_manifest.json').read_text())
    arrays=metadata(query,'query')|metadata(gallery,'gallery')
    inputs={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (run/'best.pth',run/'result.json',normal)}
    with np.load(normal) as saved:
        assert all(np.array_equal(saved[key],value) for key,value in arrays.items())
        expected_q,expected_g,expected_d=(saved[key].copy() for key in ('query_features','gallery_features','distances'))
    torch.set_num_threads(4)
    model=recipe.build(arguments,cfg,classes,cameras)
    model.load_state_dict(torch.load(run/'best.pth',map_location='cuda',weights_only=True),strict=True)
    model.eval();assert_anchor_unchanged(model)
    assert all(dict(model.named_parameters())[n].requires_grad for n in ('modality_expert.pool_score.weight','frequency_expert.pool_score.weight'))
    versions={key:value._version for key,value in model.state_dict().items()}
    out.mkdir(parents=True,exist_ok=False);start=time.time()
    if args.mode=='smoke':
        records=query[:8]
        observed=extract(model,records,cfg,arguments.seed)
        with torch.no_grad():
            plain=[]
            for images,_,cam,scene,_ in make_loader(records,cfg,False,arguments.seed):
                images={key:value.cuda(non_blocking=True) for key,value in images.items()}
                plain.append(F.normalize(model(images,cam_label=cam.cuda(),view_label=scene.cuda()).float(),dim=1).cpu())
        assert torch.equal(observed['normal11'],torch.cat(plain))
        assert versions=={key:value._version for key,value in model.state_dict().items()}
        assert_anchor_unchanged(model)
        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in inputs.items())
        write_json(out/'result.json',dict(status='SMOKE_PASS',mode='smoke',dataset=dataset,variant=arguments.variant,
            observed_records=8,plain_parity_records=8,stages=list(STAGES),dimensions={k:v.shape[1] for k,v in observed.items()},
            hook_vs_unhooked_same_batch_exact=True,state_tensor_versions_unchanged=True,optimizer_updates=0,new_weights=0,
            original_inputs=inputs,seconds=time.time()-start,limits='Native eval only, no AP is reported on eight records. Small batches are compared to identical unhooked batches, not a differently sized cached batch.'))
        emit(dict(event='SMOKE_PASS',variant=arguments.variant,optimizer_updates=0))
        return
    # AD's evaluate() batches query + gallery once, then splits the cache.
    features=extract(model,query+gallery,cfg,arguments.seed)
    q={key:value[:len(query)] for key,value in features.items()}
    g={key:value[len(query):] for key,value in features.items()}
    assert np.array_equal(q['normal11'].numpy(),expected_q) and np.array_equal(g['normal11'].numpy(),expected_g)
    assert versions=={key:value._version for key,value in model.state_dict().items()}
    assert_anchor_unchanged(model);dimensions={key:value.shape[1] for key,value in q.items()}
    del model;torch.cuda.empty_cache()
    emit(dict(event='ALL_OFFICIAL_FEATURES_READY',records=10290,stages=15,optimizer_updates=0))
    raw,measured,audited,rows,errors={},{},{},{},[]
    for stage in STAGES:
        d=distance(q[stage],g[stage])
        if stage=='normal11':assert np.array_equal(d,expected_d)
        measured[stage]=full_metrics(d,arrays['query_ids'],arrays['gallery_ids'],arrays['query_cameras'],arrays['gallery_cameras'],
            arrays['query_names'],arrays['query_cameras'],arrays['query_scenes'],out/stage)
        rows[stage]=recount(d,installed['query'],installed['gallery'],dataset)
        audited[stage],error=verify(rows[stage],measured[stage],out/(stage+'.csv'),len(gallery));errors.append(error)
        raw['distance_'+stage]=d
    assert all(audited['normal11'][key]==trained['full_metrics'][key] for key in METRICS)
    comparisons={stage:changes(rows['base00'],rows[stage]) for stage in STAGES if stage!='base00'}
    for expert in ('M','F'):
        comparisons[expert+'_projection_vs_input']=changes(rows[expert+'_projection_input'],rows[expert+'_projection_output'])
        comparisons[expert+'_routing_vs_raw']=changes(rows[expert+'_raw_pool'],rows[expert+'_routed_pool'])
    # Oracle cohort coverage is diagnostic only, never a deployed selector.
    cohort={}
    for reference in ('base00','normal11'):
        wrong=[i for i,row in enumerate(rows[reference]) if row['Rank-1']==0]
        cohort[reference]=dict(errors=len(wrong),correct_by_stage={stage:sum(rows[stage][i]['Rank-1']==1 for i in wrong) for stage in STAGES},
            any_MF_direction_correct=sum(any(rows[s][i]['Rank-1']==1 for s in ('M_projection_input','F_projection_input','M_projection_output','F_projection_output','M_projected_auxiliary','F_projected_auxiliary')) for i in wrong))
    path=out/'raw.npz';np.savez_compressed(path,**raw,**arrays)
    with np.load(path) as saved:
        assert set(saved.files)==set(raw)|set(arrays)
        assert all(np.array_equal(saved[key],value) for key,value in (raw|arrays).items())
    file=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in inputs.items())
    write_json(out/'independent_cpu_audit.json',dict(status='PASS',measured=audited,changes=comparisons,
        max_sixmetric_error_pp=max(errors),installed_split_counts={key:len(value) for key,value in installed.items()}))
    write_json(out/'result.json',dict(status='COMPLETE',mode='full',dataset=dataset,variant=arguments.variant,
        selected_epoch=trained['best']['epoch'],model_arguments=trained['arguments'],stages=list(STAGES),dimensions=dimensions,
        measurements=measured,comparisons=comparisons,error_cohorts=cohort,normal_features_and_distance_exact=True,
        state_tensor_versions_unchanged=True,optimizer_updates=0,new_weights=0,new49=False,missing_evaluated=False,
        query_records=1715,gallery_records=8575,train_records=8675,original_inputs=inputs,
        raw=dict(path=str(path),file=file,preserved_pending_local_transfer=True),seconds=time.time()-start,
        limits='Two fixed hard AD mAP-best checkpoints, not new models or hyperparameter selection. All official GT, exact unchanged normal replay. Pools512,relationblocks3584 andnormal5120 compared in distinct groups; independent residual retrieval alone is not fusion value. Actual AD auxiliary pools reproduce the first11 FP32 per-relation units and trained bias-free scores before neck/classifier; raw pools are diagnostic, not current auxiliary targets. Error cohort branch oracle is offline diagnosis only. No training,missing,49,newseed or201NN.'))
    emit(dict(event='EVALUATION_COMPLETE',variant=arguments.variant,cases=15,optimizer_updates=0))


if __name__=='__main__':main()
