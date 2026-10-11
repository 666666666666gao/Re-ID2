"""Read fixed P/R expert evidence at successive outlets; no model or loss changes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
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
import run_r201p_auxiliary_balance as p_recipe
import run_r201r_primary_ce as r_recipe

PREFIX = 'OUTLET_STREAM '
STAGES = ('base00', 'normal11', 'M_raw_pool', 'F_raw_pool', 'M_routed_pool',
          'F_routed_pool', 'M_projection_input', 'F_projection_input',
          'M_projection_output', 'F_projection_output', 'M_unit_direction',
          'F_unit_direction', 'common_context')


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
            common_context=captured['base'][:, 5120:])
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


def main():
    parser = argparse.ArgumentParser()
    for key in ('run-dir', 'normal-arrays', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    run, normal, out = map(Path, (args.run_dir, args.normal_arrays, args.output))
    trained, dataset, installed = installed_context(run)
    assert dataset in ('MSVR310', 'RGBNT100') and trained['descriptor_dim'] == 5120
    assert json.loads((run.parent / (run.name + '_exit.json')).read_text())['exit_code'] == 0
    assert json.loads((run / 'normal_cpu_audit.json').read_text())['status'] == 'PASS'
    assert trained['amp_skipped_steps'] == trained['training_heldout_identities'] == 0
    assert trained['training_coverage'] == dict(eligible=trained['train_records'], visited=trained['train_records'], unvisited=[])
    recipe = {'MSVR310': r_recipe, 'RGBNT100': p_recipe}[dataset]
    revision = {'MSVR310': 'R201R single final-full-CE-weight factor', 'RGBNT100': 'R201P single auxiliary-identity-weight factor'}[dataset]
    assert trained['method_revision'].startswith(revision)
    arguments = argparse.Namespace(**trained['arguments'])
    cfg = recipe.configuration(arguments)
    train, query, gallery, classes, cameras, manifest = full_records(arguments.data_root, dataset)
    assert manifest == json.loads((run / 'official_split_manifest.json').read_text())
    arrays = metadata(query, 'query') | metadata(gallery, 'gallery')
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (run/'best.pth', run/'result.json', normal)}
    with np.load(normal) as saved:
        assert all(np.array_equal(saved[key], value) for key, value in arrays.items())
        expected_q, expected_g, expected_d = (saved[key].copy() for key in ('query_features', 'gallery_features', 'distances'))
    torch.set_num_threads(4)
    model = recipe.build(arguments, cfg, classes, cameras)
    model.load_state_dict(torch.load(run/'best.pth', map_location='cuda', weights_only=True), strict=True)
    model.eval()
    assert_anchor_unchanged(model)
    versions = {key: value._version for key, value in model.state_dict().items()}
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    features = extract(model, query + gallery, cfg, arguments.seed)
    assert np.array_equal(features['normal11'][:len(query)].numpy(), expected_q)
    assert np.array_equal(features['normal11'][len(query):].numpy(), expected_g)
    assert versions == {key: value._version for key, value in model.state_dict().items()}
    assert_anchor_unchanged(model)
    dimensions = {key: value.shape[1] for key, value in features.items()}
    del model
    torch.cuda.empty_cache()
    print('OUTLET_FEATURE_BANKS_CPU_ONLY', json.dumps(dict(records=len(query)+len(gallery), stages=len(STAGES), allocated=torch.cuda.memory_allocated(), optimizer_updates=0)), flush=True)
    selector = 'scenes' if dataset == 'MSVR310' else 'cameras'
    raw, measured, audited, rows, errors = {}, {}, {}, {}, []
    for stage in STAGES:
        d = distance(features[stage][:len(query)], features[stage][len(query):])
        if stage == 'normal11':
            assert np.array_equal(d, expected_d)
        measured[stage] = full_metrics(d, arrays['query_ids'], arrays['gallery_ids'],
            arrays['query_'+selector], arrays['gallery_'+selector], arrays['query_names'],
            arrays['query_cameras'], arrays['query_scenes'], out/stage)
        rows[stage] = recount(d, installed['query'], installed['gallery'], dataset)
        audited[stage], error = verify(rows[stage], measured[stage], out/(stage+'.csv'), len(gallery))
        errors.append(error)
        raw['distance_'+stage] = d
    assert all(audited['normal11'][key] == trained['full_metrics'][key] for key in METRICS)
    comparisons = {stage: changes(rows['base00'], rows[stage]) for stage in STAGES if stage != 'base00'}
    for expert in ('M', 'F'):
        comparisons[expert+'_projection_vs_input'] = changes(rows[expert+'_projection_input'], rows[expert+'_projection_output'])
        comparisons[expert+'_routing_vs_raw'] = changes(rows[expert+'_raw_pool'], rows[expert+'_routed_pool'])
    path = out/'raw.npz'
    np.savez_compressed(path, **raw, **arrays)
    with np.load(path) as saved:
        assert set(saved.files) == set(raw) | set(arrays)
        assert all(np.array_equal(saved[key], value) for key, value in (raw|arrays).items())
    file = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write_json(out/'independent_cpu_audit.json', dict(status='PASS', measured=audited, changes=comparisons, max_sixmetric_error_pp=max(errors), installed_split_counts={key:len(value) for key,value in installed.items()}))
    emit(dict(event='RAW_READY', condition='normal_outlets', path=str(path), file=file, installed_gt_cases=len(STAGES)))
    ack = json.loads(sys.stdin.readline())
    assert ack == dict(event='ARCHIVED', condition='normal_outlets', file=file)
    assert path.resolve().is_relative_to(out.resolve()) and hashlib.sha256(path.read_bytes()).hexdigest() == file['sha256']
    path.unlink()
    emit(dict(event='RAW_CLEARED', condition='normal_outlets'))
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path,sha in inputs.items())
    write_json(out/'result.json', dict(status='COMPLETE', dataset=dataset, variant=arguments.variant,
        selected_epoch=trained['best']['epoch'], model_arguments=trained['arguments'], stages=list(STAGES),
        dimensions=dimensions, measurements=measured, comparisons=comparisons, normal_features_and_distance_exact=True,
        state_tensor_versions_unchanged=True, optimizer_updates=0, new_weights=0, new49=False,
        query_records=len(query), gallery_records=len(gallery), train_records=len(train),
        original_inputs=inputs, raw=dict(file=file, local_verified_acknowledged=True, server_copy_cleared=True), seconds=time.time()-start,
        limits='Fixed normal benchmark-mAP-best, full official Q/G, no training/selection/missing/new49. Intermediate directions are diagnostic descriptors, not separately trained methods. 512D pools and3584D relation blocks are separate comparisons; residual independent discrimination alone does not prove fusion usefulness. P RGB100 and R MS use dataset-specific losses, cached K201 untouched. Hooks only observe one unchanged normal forward, state tensors and exact deployed normal features/distance checked.'))
    emit(dict(event='EVALUATION_COMPLETE', conditions=1, cases=len(STAGES), optimizer_updates=0))


if __name__ == '__main__':
    main()
