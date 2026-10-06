"""Selected identity-coordinate checkpoint, complete normal input, four states."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from audit_full_official49 import installed_rows, recount, verify, METRICS
from experiment_data import make_loader
from full_evaluation import distance, full_metrics
from identity_coordinate_three_dataset import build
from official_training_data import full_records, metadata
from original_identity_anchor import assert_anchor_unchanged
from run_experiment import configuration, write_json

STATES = ('00', '10', '01', '11')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run, out = Path(args.run_dir), Path(args.output)
    trained = json.loads((run / 'result.json').read_text())
    assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50
    assert trained['training_heldout_identities'] == trained['amp_skipped_steps'] == 0
    assert trained['steps'] == trained['optimizer_steps'] and trained['descriptor_dim'] == 5120
    assert json.loads((run.parent / (run.name + '_exit.json')).read_text())['exit_code'] == 0
    assert json.loads((run / 'normal_cpu_audit.json').read_text())['status'] == 'PASS'
    arguments = argparse.Namespace(**trained['arguments'])
    assert arguments.freeze_identity_encoder == 1
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    cfg = configuration(arguments)
    train, query, gallery, classes, cameras, manifest = full_records(arguments.data_root, arguments.dataset)
    assert json.loads((run / 'official_split_manifest.json').read_text()) == manifest
    assert all(len(records) == trained[key + '_records'] for key, records in
               (('train', train), ('query', query), ('gallery', gallery)))
    installed = {key: installed_rows(Path(arguments.data_root), arguments.dataset, key)
                 for key in ('train', 'query', 'gallery')}
    assert all(installed[key] == manifest[key] for key in installed)
    model = build(arguments, cfg, classes, cameras)
    model.load_state_dict(torch.load(run / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    model.eval()
    assert_anchor_unchanged(model)
    versions = {name: value._version for name, value in model.state_dict().items()}
    out.mkdir(exist_ok=False)
    started = time.time()
    parts = {state: [] for state in STATES}
    with torch.no_grad():
        for images, _, cam, scene, _ in make_loader(query + gallery, cfg, False, arguments.seed):
            images = {key: value.cuda(non_blocking=True) for key, value in images.items()}
            states = model(images, cam_label=cam.cuda(), view_label=scene.cuda(), return_states=True)
            assert set(states) == set(STATES)
            for state, value in states.items():
                assert value.shape[1] == 5120 and torch.isfinite(value).all()
                parts[state].append(F.normalize(value.float(), dim=1).cpu())
    arrays = metadata(query, 'query') | metadata(gallery, 'gallery')
    selector = 'scenes' if arguments.dataset == 'MSVR310' else 'cameras'
    reported, audited, raw = {}, {}, dict(arrays)
    for state in STATES:
        features = torch.cat(parts[state])
        assert tuple(features.shape) == (len(query) + len(gallery), 5120)
        distances = distance(features[:len(query)], features[len(query):])
        summary = full_metrics(distances, arrays['query_ids'], arrays['gallery_ids'],
            arrays['query_' + selector], arrays['gallery_' + selector], arrays['query_names'],
            arrays['query_cameras'], arrays['query_scenes'], out / ('state_' + state))
        rows = recount(distances, installed['query'], installed['gallery'], arguments.dataset)
        _, error = verify(rows, summary, out / ('state_' + state + '.csv'), len(gallery))
        reported[state] = summary
        audited[state] = dict(max_metric_error=error, installed_query_rows=len(rows))
        raw['distances' if state == '11' else 'distances_' + state] = distances
    assert all(abs(reported['11'][key] - trained['full_metrics'][key]) < 1e-8 for key in METRICS)
    assert versions == {name: value._version for name, value in model.state_dict().items()}
    assert_anchor_unchanged(model)
    np.savez_compressed(out / 'normal_states.npz', **raw)
    write_json(out / 'result.json', dict(status='COMPLETE', dataset=arguments.dataset,
        variant=arguments.variant, source_run=str(run), selected_epoch=trained['best']['epoch'],
        seed=arguments.seed, measurements=reported, installed_GT_audit=audited,
        query_records=len(query), gallery_records=len(gallery), train_records=len(train),
        state_cases=4, condition='q_RNT_g_RNT', neural_passes=1,
        optimizer_updates=0, checkpoint_writes=0, state_tensor_versions_unchanged=True,
        full11_six_metrics_equal_selected_normal=True, seconds=time.time() - started,
        selection=trained['checkpoint_rule'],
        limits='Same selected checkpoint, four controlled inference states; no retraining ablation. '
               'The same queries repeat four times. No missing-modality evaluation or multi-seed claim. '
               'M/F-only use independent evidence without cross-condition messages, joint psi or I. '
               'This audit is executor CPU installed-GT recount, not independent research acceptance.'))
    print('FULL_NORMAL_FOURSTATE_INSTALLED_GT_COMPLETE', arguments.dataset, arguments.variant, flush=True)


if __name__ == '__main__':
    main()
