"""Fixed-best four-state inference; installed-GT audit and archive one pair at a time."""
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
from evaluate_full_official49 import SETS
from experiment_data import make_loader
from full_evaluation import distance, full_metrics
from identity_coordinate_three_dataset import build
from missing_evaluation import mask_images
from official_training_data import full_records, metadata
from original_identity_anchor import assert_anchor_unchanged
from run_experiment import configuration, write_json

PREFIX = 'IDENTITY_MISSING_STREAM '
STATES = ('00', '10', '01', '11')


def emit(value):
    print(PREFIX + json.dumps(value), flush=True)


@torch.no_grad()
def extract_bank(model, records, cfg, seed, missing):
    parts = {state: [] for state in STATES}
    for images, _, cam, scene, _ in make_loader(records, cfg, False, seed):
        images = {key: value.cuda(non_blocking=True) for key, value in images.items()}
        states = model(mask_images(images, missing), cam_label=cam.cuda(), view_label=scene.cuda(), return_states=True)
        assert set(states) == set(STATES)
        for state, value in states.items():
            assert value.shape[1] == 5120 and torch.isfinite(value).all()
            parts[state].append(F.normalize(value.float(), dim=1).cpu())
    return {state: torch.cat(values) for state, values in parts.items()}


def main():
    parser = argparse.ArgumentParser()
    for key in ('run-dir', 'normal-arrays', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--mode', choices=('native', 'full'), required=True)
    args = parser.parse_args()
    run, normal_path, out = map(Path, (args.run_dir, args.normal_arrays, args.output))
    trained, dataset, installed = installed_context(run)
    assert json.loads((run.parent / (run.name + '_exit.json')).read_text())['exit_code'] == 0
    assert trained['descriptor_dim'] == 5120 and trained['amp_skipped_steps'] == 0
    assert trained['training_coverage'] == dict(eligible=trained['train_records'], visited=trained['train_records'], unvisited=[])
    assert json.loads((run / 'normal_cpu_audit.json').read_text())['status'] == 'PASS'
    arguments = argparse.Namespace(**trained['arguments'])
    cfg = configuration(arguments)
    train, query, gallery, classes, cameras, manifest = full_records(arguments.data_root, dataset)
    assert manifest == json.loads((run / 'official_split_manifest.json').read_text())
    arrays = metadata(query, 'query') | metadata(gallery, 'gallery')
    protected = (run / 'best.pth', run / 'result.json', normal_path)
    inputs = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    with np.load(normal_path) as saved:
        assert all(np.array_equal(saved[key], value) for key, value in arrays.items())
        expected_query, expected_gallery, expected_distance = (saved[key].copy() for key in ('query_features', 'gallery_features', 'distances'))
    torch.set_num_threads(4)
    model = build(arguments, cfg, classes, cameras)
    model.load_state_dict(torch.load(run / 'best.pth', map_location='cuda', weights_only=True), strict=True)
    model.eval()
    assert_anchor_unchanged(model)
    versions = {key: value._version for key, value in model.state_dict().items()}
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    native_checks = {}
    if args.mode == 'native':
        for available, missing in SETS.items():
            sample = extract_bank(model, (query + gallery)[:8], cfg, arguments.seed, missing)
            assert all(tuple(value.shape) == (8, 5120) for value in sample.values())
            error = max(float((value.norm(dim=1) - 1).abs().max()) for value in sample.values())
            assert error < 1e-5
            native_checks[available] = dict(states=4, samples=8, finite=True, max_unit_norm_error=error)
    banks, bank_seconds = {}, {}
    sets = {'RNT': SETS['RNT']} if args.mode == 'native' else SETS
    for available, missing in sets.items():
        start = time.time()
        features = extract_bank(model, query + gallery, cfg, arguments.seed, missing)
        assert all(tuple(value.shape) == (len(query) + len(gallery), 5120) for value in features.values())
        banks[available] = {state: (value[:len(query)], value[len(query):]) for state, value in features.items()}
        bank_seconds[available] = time.time() - start
    assert np.array_equal(banks['RNT']['11'][0].numpy(), expected_query)
    assert np.array_equal(banks['RNT']['11'][1].numpy(), expected_gallery)
    assert versions == {key: value._version for key, value in model.state_dict().items()}
    assert_anchor_unchanged(model)
    scales = model.residual_scale.detach().cpu().tolist()
    del model
    torch.cuda.empty_cache()
    print('IDENTITY_MISSING_FEATURE_BANKS_CPU_ONLY', json.dumps(dict(available_sets=len(sets),
        cuda_allocated_bytes=torch.cuda.memory_allocated(), new_optimizer_updates=0)), flush=True)
    selector = 'scenes' if dataset == 'MSVR310' else 'cameras'
    measurements, conditions, archives, errors = {}, {}, {}, []
    peak_raw_bytes = 0
    for qs in sets:
        for gs in sets:
            condition = 'q_' + qs + '_g_' + gs
            folder = out / condition
            folder.mkdir()
            raw, values = {}, {}
            for state in STATES:
                distances = distance(banks[qs][state][0], banks[gs][state][1])
                if qs == gs == 'RNT' and state == '11':
                    assert np.array_equal(distances, expected_distance)
                values[state] = full_metrics(distances, arrays['query_ids'], arrays['gallery_ids'],
                    arrays['query_' + selector], arrays['gallery_' + selector], arrays['query_names'],
                    arrays['query_cameras'], arrays['query_scenes'], folder / ('state_' + state))
                raw['distance_' + state] = distances
            path = folder / 'raw.npz'
            np.savez_compressed(path, **raw, **arrays)
            audited, rows = {}, {}
            with np.load(path) as saved:
                for role in ('query', 'gallery'):
                    for field, key in (('ids', 'identity'), ('cameras', 'camera'), ('scenes', 'scene'), ('names', 'name')):
                        assert np.array_equal(saved[role + '_' + field], np.asarray([row[key] for row in installed[role]]))
                for state in STATES:
                    rows[state] = recount(saved['distance_' + state], installed['query'], installed['gallery'], dataset)
                    audited[state], error = verify(rows[state], values[state], folder / ('state_' + state + '.csv'), len(gallery))
                    errors.append(error)
            checked = dict(metrics=audited, full11_minus00=changes(rows['00'], rows['11']),
                full11_minus10=changes(rows['10'], rows['11']), full11_minus01=changes(rows['01'], rows['11']))
            if set(qs).isdisjoint(gs):
                checked['disjoint_private_distance'] = {state: dict(minimum=float(raw['distance_' + state].min()),
                    maximum=float(raw['distance_' + state].max()), max_abs_from2=float(np.max(np.abs(raw['distance_' + state] - 2)))) for state in STATES}
                checked['cross_source_identity_comparison_not_established'] = True
            file = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            peak_raw_bytes = max(peak_raw_bytes, file['bytes'])
            assert len(list(out.glob('q_*_g_*/raw.npz'))) == 1
            write_json(folder / 'condition_cpu_audit.json', dict(status='PASS', condition=condition, state_cases=4, measured=checked))
            emit(dict(event='RAW_READY', condition=condition, path=str(path), file=file, installed_gt_cases=4))
            ack = json.loads(sys.stdin.readline())
            assert ack == dict(event='ARCHIVED', condition=condition, file=file)
            assert path.resolve().is_relative_to(out.resolve()) and path.stat().st_size == file['bytes']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == file['sha256']
            path.unlink()
            assert not list(out.glob('q_*_g_*/raw.npz'))
            archives[condition] = dict(file=file, local_verified_acknowledged=True, server_copy_cleared=True)
            measurements[condition], conditions[condition] = values, checked
            emit(dict(event='RAW_CLEARED', condition=condition))
    expected = 1 if args.mode == 'native' else 49
    assert len(measurements) == len(conditions) == len(archives) == expected
    assert all(conditions['q_RNT_g_RNT']['metrics']['11'][key] == trained['full_metrics'][key] for key in METRICS)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == inputs[str(path)] for path in protected)
    write_json(out / 'independent_cpu_audit.json', dict(status='PASS', dataset=dataset, conditions=expected,
        state_cases=4 * expected, max_sixmetric_error_pp=max(errors), installed_split_counts={key: len(value) for key, value in installed.items()},
        measured=conditions, optimizer_updates=0, selection='Fixed full-normal selected best; missing conditions never reselect checkpoint'))
    write_json(out / 'stream_archive.json', dict(status='ALL_DECLARED_CONDITIONS_GT_AUDITED_LOCAL_VERIFIED_REMOTE_RAW_CLEARED',
        conditions=archives, max_live_raw_files=1, peak_live_raw_bytes=peak_raw_bytes))
    write_json(out / 'result.json', dict(status='COMPLETE', mode=args.mode, dataset=dataset, model_arguments=trained['arguments'],
        variant=arguments.variant, selected_epoch=trained['best']['epoch'], measurements=measurements,
        conditions=expected, state_cases=4 * expected, train_records=len(train), query_records=len(query), gallery_records=len(gallery),
        training_heldout_identities=0, optimizer_updates=0, new_weights=0, residual_scales=scales,
        normal_features_and_distance_exact=True, native_small_batch_checks=native_checks,
        bank_seconds=bank_seconds, seconds=time.time() - started,
        timing_includes_CPU_GT_and_archive_wait=True, original_inputs=inputs, state_tensor_versions_unchanged=True,
        limits='Benchmark-selected/one seed/original50+extra50. Four closed expert states use the same fixed query/gallery references. '
            'The12 disjoint-source cases lack common identity coordinates; ties/rounding do not prove cross-source identification. '
            'Repeated condition-query rows are not independent samples. No calibrated, causal or information-theoretic synergy claim.'))
    emit(dict(event='EVALUATION_COMPLETE', conditions=expected, cases=4 * expected, optimizer_updates=0))


if __name__ == '__main__':
    main()
