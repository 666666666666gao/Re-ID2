"""Reconstruct four explicitly retired normal archives; never train or reselect."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_full_official49 import METRICS, recount, verify
from audit_full_official_condition import installed_context
from evaluate_full_official49 import SETS
from evaluate_identity_coordinate_missing49_stream import extract_bank, STATES
from full_evaluation import distance, full_metrics
from official_training_data import full_records, metadata
from original_identity_anchor import assert_anchor_unchanged
from relation_local_identity_outlet import RelationLocalPIAxis
from run_experiment import configuration, write_json
from run_r201k_relation_local_pi import build, REVISION


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    return dict(count=len(values), mean=float(np.mean(values)), minimum=float(np.min(values)),
        maximum=float(np.max(values)), **{key:float(value) for key,value in
        zip(('p25','p50','p75','p90','p95','p99'), np.quantile(values, (.25,.5,.75,.9,.95,.99)))})


def main():
    parser = argparse.ArgumentParser()
    for key in ('jobs', 'name', 'output'):
        parser.add_argument('--' + key, required=True)
    cli = parser.parse_args()
    jobs = json.loads(Path(cli.jobs).read_text())['jobs']
    assert len(jobs) == 4 and len({j['name'] for j in jobs}) == 4
    job = next(j for j in jobs if j['name'] == cli.name)
    run, out = Path(job['run']), Path(cli.output)
    assert not out.exists()
    weights, record = run/'best.pth', run/'result.json'
    assert digest(weights) == job['checkpoint_sha256'] and digest(record) == job['result_sha256']
    trained, dataset, installed = installed_context(run)
    assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50
    assert dataset == job['dataset'] and trained['arguments']['variant'] == job['variant']
    assert trained['arguments']['seed'] == 42 and trained['best']['epoch'] == job['selected_epoch']
    assert trained['full_metrics'] == job['full_metrics'] and trained['method_revision'] == REVISION
    assert json.loads((run.parent/(run.name+'_exit.json')).read_text())['exit_code'] == 0
    assert json.loads((run/'normal_cpu_audit.json').read_text())['status'] == 'PASS'
    assert trained['descriptor_dim'] == 5120 and trained['amp_skipped_steps'] == 0
    args = argparse.Namespace(**trained['arguments'])
    cfg = configuration(args)
    _, query, gallery, classes, cameras, manifest = full_records(args.data_root, dataset)
    assert manifest == json.loads((run/'official_split_manifest.json').read_text())
    arrays = metadata(query, 'query') | metadata(gallery, 'gallery')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    model = build(args, cfg, classes, cameras)
    assert type(model) is RelationLocalPIAxis and model.anchor_record['method'] == REVISION
    model.load_state_dict(torch.load(weights, map_location='cuda', weights_only=True), strict=True)
    model.eval()
    assert_anchor_unchanged(model)
    versions = {key:value._version for key,value in model.state_dict().items()}
    # Same all-RNT feature extraction already used by the four-state native verifier.
    # Reading 00/10/01 adds controlled heads, not backbone passes or optimizer updates.
    bank = extract_bank(model, query+gallery, cfg, 42, SETS['RNT'])
    assert versions == {key:value._version for key,value in model.state_dict().items()}
    assert_anchor_unchanged(model)
    del model
    torch.cuda.empty_cache()
    out.mkdir(parents=True)
    selector = 'scenes' if dataset == 'MSVR310' else 'cameras'
    metrics, rows = {}, {}
    path = out/'best_official_arrays.npz'
    for state in STATES:
        features = bank[state]
        assert tuple(features.shape) == (len(query)+len(gallery), 5120)
        q, g = features[:len(query)], features[len(query):]
        distances = distance(q, g)
        target = out/('best_per_query' if state == '11' else 'state_'+state+'_per_query')
        metrics[state] = full_metrics(distances, arrays['query_ids'], arrays['gallery_ids'],
            arrays['query_'+selector], arrays['gallery_'+selector], arrays['query_names'],
            arrays['query_cameras'], arrays['query_scenes'], target)
        rows[state] = recount(distances, installed['query'], installed['gallery'], dataset)
        reported = json.loads(target.with_suffix('.json').read_text())
        values, error = verify(rows[state], reported, target.with_suffix('.csv'), len(gallery))
        assert error < 1e-8 and all(abs(values[key]-metrics[state][key]) < 1e-8 for key in METRICS)
        if state == '11':
            np.savez_compressed(path, **arrays, query_features=q.numpy(), gallery_features=g.numpy(), distances=distances)
    assert path.stat().st_size == job['normal_file']['bytes'] and digest(path) == job['normal_file']['sha256']
    assert all(abs(metrics['11'][key]-trained['full_metrics'][key]) < 1e-8 for key in METRICS)
    assert all(abs(metrics['00'][key]-trained['anchor']['selected_anchor_full_metrics'][key]) < 1e-8 for key in METRICS)
    assert all(digest(out/('best_per_query'+suffix)) == digest(run/('best_per_query'+suffix)) for suffix in ('.json','.csv'))
    base = bank['00'].double().numpy()
    reference_unit = base/np.linalg.norm(base,axis=1,keepdims=True)
    geometry, query_rows = {}, []
    for state in ('10','01','11'):
        feature = bank[state].double().numpy()
        shift = np.linalg.norm(feature-base, axis=1)
        feature_unit = feature/np.linalg.norm(feature,axis=1,keepdims=True)
        angles = np.degrees(2*np.arctan2(np.linalg.norm(feature_unit-reference_unit,axis=1),
            np.linalg.norm(feature_unit+reference_unit,axis=1)))
        geometry[state] = dict(query_shift=distribution(shift[:len(query)]),
            gallery_shift=distribution(shift[len(query):]), query_angle_deg=distribution(angles[:len(query)]),
            gallery_angle_deg=distribution(angles[len(query):]))
        for i,(actual,reference) in enumerate(zip(rows[state],rows['00'])):
            query_rows.append(dict(state=state,query_index=i,name=installed['query'][i]['name'],
                identity=installed['query'][i]['identity'],shift=float(shift[i]),angle_deg=float(angles[i]),
                delta_AP_pp=100*(actual['AP']-reference['AP']),
                delta_rank1=int(actual['Rank-1'])-int(reference['Rank-1'])))
    with (out/'query_direction_changes.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(query_rows[0]));writer.writeheader();writer.writerows(query_rows)
    assert digest(weights) == job['checkpoint_sha256'] and digest(record) == job['result_sha256']
    result=dict(status='ACTUAL_RETIRED_K_NORMAL_RECONSTRUCTED_ORIGINAL_SHA_AND_GT_EXACT',
        completed_at=datetime.now().isoformat(timespec='seconds'),name=job['name'],dataset=dataset,
        selected_epoch=job['selected_epoch'],normal_file=job['normal_file'],normal_path=str(path),
        checkpoint_sha256=job['checkpoint_sha256'],result_sha256=job['result_sha256'],
        four_state_metrics={state:{key:metrics[state][key] for key in METRICS} for state in STATES},
        geometry=geometry,query_direction_rows=len(query_rows),installed_GT_exact=True,
        normal_perquery_text_exact=True,model_state_versions_unchanged=True,new_optimizer_updates=0,
        feature_forward_batches=(len(query)+len(gallery)+63)//64,
        limits='Only four explicitly retired closed K42 normal inputs; same selected checkpoint, no training or reselection. Four-state geometry is selected-normal diagnostic, not late-epoch evidence, intervention, missing49 or independent causal proof. Exact original NPZ bytes/SHA required; no approximate reconstruction or fallback.')
    write_json(out/'result.json',result)
    print('K_NORMAL_RECONSTRUCTED '+json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
