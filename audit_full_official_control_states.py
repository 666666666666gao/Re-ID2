"""Pure CPU installed-GT audit of full-official frozen control diagnostics."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_full_official49 import installed_rows, recount, verify
from audit_shared_identity_states import changes, METRICS


STATES = ('00', '10', '01', '11', 'base_private', 'base_shared')
TARGETS = ('delta_M_given_F', 'delta_F_given_M', 'empirical_interaction')


def verify_calibration(raw, reported, csv_path, queries, gallery, dataset, head_predictions):
    qids = np.asarray([r['identity'] for r in queries])
    gids = np.asarray([r['identity'] for r in gallery])
    selector = 'scene' if dataset == 'MSVR310' else 'camera'
    qexclude = np.asarray([r[selector] for r in queries])
    gexclude = np.asarray([r[selector] for r in gallery])
    similarity = raw['reference_similarity']
    assert similarity.shape == (len(queries), len(gallery)) and np.isfinite(similarity).all()
    positive = (qids[:, None] == gids[None]) & (qexclude[:, None] != gexclude[None])
    negative = qids[:, None] != gids[None]
    assert positive.any(1).all() and negative.any(1).all()
    pos = np.where(positive, similarity, np.inf).argmin(1)
    neg = np.where(negative, similarity, -np.inf).argmax(1)
    assert np.array_equal(pos, raw['positive_indices']) and np.array_equal(neg, raw['negative_indices'])
    scores, predictions = raw['raw_scores'], raw['predictions']
    assert scores.shape == (len(queries), 4) and predictions.shape == (len(queries), 3)
    assert np.isfinite(scores).all() and np.isfinite(predictions).all()
    assert np.array_equal(predictions, head_predictions[:len(queries)])
    target = np.stack((scores[:, 3] - scores[:, 2], scores[:, 3] - scores[:, 1],
        scores[:, 3] - scores[:, 1] - scores[:, 2] + scores[:, 0]), 1)
    assert np.array_equal(target, raw['targets'])
    recalculated = dict(target_mean=target.mean(0), target_std=target.std(0),
        MAE=np.abs(target - predictions).mean(0), zero_prediction_MAE=np.abs(target).mean(0))
    for key, values in recalculated.items():
        assert np.max(np.abs(values - reported[key])) < 1e-8
    with csv_path.open() as handle: rows = list(csv.DictReader(handle))
    assert len(rows) == len(queries)
    for i, row in enumerate(rows):
        assert int(row['query_index']) == i and int(row['gallery_positive_index']) == pos[i] and int(row['gallery_negative_index']) == neg[i]
        for j, state in enumerate(('00', '10', '01', '11')): assert abs(float(row['R' + state]) - float(scores[i, j])) < 1e-12
        for j, key in enumerate(TARGETS): assert abs(float(row[key]) - float(target[i, j])) < 1e-12
        for j, key in enumerate(('prediction_M', 'prediction_F', 'prediction_I')):
            assert abs(float(row[key]) - float(predictions[i, j])) < 1e-12
    return {key: values.tolist() for key, values in recalculated.items()}


def verify_heads(folder, queries, gallery, mode):
    heads, predictions = {}, {}
    records = [('query', i, row) for i, row in enumerate(queries)] + [('gallery', i, row) for i, row in enumerate(gallery)]
    for available in ('RNT', 'R', 'N', 'T', 'RN', 'RT', 'NT'):
        with np.load(folder / ('heads_' + available + '.npz')) as raw:
            for role, installed in (('query', queries), ('gallery', gallery)):
                for field, key in (('ids', 'identity'), ('cameras', 'camera'), ('scenes', 'scene'), ('names', 'name')):
                    assert np.array_equal(raw[role + '_' + field], np.asarray([r[key] for r in installed]))
            prediction, gates, control = (raw[k] for k in ('predictions', 'gates', 'control_logits'))
            assert prediction.shape == gates.shape == control.shape == (len(records), 3)
            assert all(np.isfinite(v).all() for v in (prediction, gates, control))
            assert ((gates >= 0) & (gates <= 1)).all()
            if mode == 'measurement_only': assert np.count_nonzero(control) == 0
            expected = 1 / (1 + np.exp(-(20 * prediction.astype(np.float64) + control.astype(np.float64))))
            assert np.max(np.abs(expected - gates)) < 1e-7
            with (folder / ('heads_' + available + '.csv')).open() as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == len(records)
            for i, ((role, index, record), row) in enumerate(zip(records, rows)):
                assert row['role'] == role and int(row['row']) == index
                assert all(row[k] == str(record[k]) for k in ('name', 'identity', 'camera', 'scene'))
                for values, prefix in ((prediction, 'prediction_'), (gates, 'gate_'), (control, 'control_')):
                    for j, suffix in enumerate(('M', 'F', 'I')): assert abs(float(row[prefix + suffix]) - float(values[i, j])) < 1e-12
            heads[available] = dict(prediction_mean=prediction.mean(0).tolist(), prediction_std=prediction.std(0).tolist(),
                gate_mean=gates.mean(0).tolist(), gate_std=gates.std(0).tolist(), control_mean=control.mean(0).tolist(),
                control_std=control.std(0).tolist(), sigmoid_formula_max_error=float(np.max(np.abs(expected - gates))))
            predictions[available] = prediction
    return heads, predictions


def main():
    parser = argparse.ArgumentParser()
    for key in ('run-dir', 'previous-frozen', 'diagnosis'): parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    run, previous, diagnosis = map(Path, (args.run_dir, args.previous_frozen, args.diagnosis))
    trained = json.loads((run / 'result.json').read_text())
    reported = json.loads((diagnosis / 'result.json').read_text())
    prior = json.loads((previous / 'result.json').read_text())
    assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50 and trained['training_heldout_identities'] == 0
    assert reported['status'] == prior['status'] == 'COMPLETE'
    assert reported['metric_count'] == 294 and reported['states'] == list(STATES)
    assert reported['model_arguments'] == prior['model_arguments'] == trained['arguments']
    assert reported['selected_epoch'] == prior['selected_epoch'] == trained['best']['epoch']
    assert reported['optimizer_updates'] == reported['new_weights'] == reported['training_heldout_identities'] == 0
    assert reported['normal_feature_max_error'] == 0 and reported['state_tensor_versions_unchanged']
    assert reported['previous_all49_full11_distances_metrics_perquery_exact']
    arguments = trained['arguments']
    dataset, data_root = arguments['dataset'], Path(arguments['data_root'])
    installed = {split: installed_rows(data_root, dataset, split) for split in ('train', 'query', 'gallery')}
    manifest = json.loads((run / 'official_split_manifest.json').read_text())
    assert all(installed[split] == manifest[split] for split in installed)
    assert len(installed['query']) == reported['query_records'] and len(installed['gallery']) == reported['gallery_records']
    heads, head_predictions = verify_heads(diagnosis, installed['query'], installed['gallery'], arguments['gate_gradient_mode'])
    conditions, errors = {}, []
    assert set(reported['measurements']) == set(prior['measurements']) and len(reported['measurements']) == 49
    for condition, measured in reported['measurements'].items():
        folder = diagnosis / condition
        values, rows = {}, {}
        with np.load(folder / 'raw.npz') as raw:
            for role in ('query', 'gallery'):
                for field, key in (('ids', 'identity'), ('cameras', 'camera'), ('scenes', 'scene'), ('names', 'name')):
                    assert np.array_equal(raw[role + '_' + field], np.asarray([r[key] for r in installed[role]]))
            for state in STATES:
                actual = recount(raw['distance_' + state], installed['query'], installed['gallery'], dataset)
                values[state], error = verify(actual, measured[state], folder / ('state_' + state + '.csv'), len(installed['gallery']))
                rows[state] = actual; errors.append(error)
            with np.load(previous / (condition + '.npz')) as old:
                assert np.array_equal(raw['distance_11'], old['distances'])
            assert all(values['11'][m] == prior['measurements'][condition][m] for m in METRICS)
            calibrated = verify_calibration(raw, reported['contributions'][condition], folder / 'contributions.csv',
                installed['query'], installed['gallery'], dataset, head_predictions[condition.split('_')[1]])
        conditions[condition] = dict(metrics=values, full11_minus00=changes(rows['00'], rows['11']),
            full11_minus10=changes(rows['10'], rows['11']), full11_minus01=changes(rows['01'], rows['11']), calibration=calibrated)
    normal = conditions['q_RNT_g_RNT']
    assert all(normal['metrics']['11'][m] == trained['full_metrics'][m] for m in METRICS)
    result = dict(status='PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT', model_arguments=arguments,
        cases=294, repeated_condition_query_rows=294 * len(installed['query']),
        query_records=len(installed['query']), gallery_records=len(installed['gallery']),
        max_sixmetric_error_pp=max(errors), conditions=conditions, normal=normal, heads=heads,
        full11_minus00_equal_condition_mean={m: float(np.mean([v['full11_minus00']['delta_pp'][m] for v in conditions.values()])) for m in METRICS},
        all49_harm=sum(v['full11_minus00']['Rank1_harm_queries'] for v in conditions.values()),
        all49_rescue=sum(v['full11_minus00']['Rank1_rescue_queries'] for v in conditions.values()),
        calibration_rows=49 * len(installed['query']), head_records=7 * (len(installed['query']) + len(installed['gallery'])),
        limits='Independent GT recount, raw-score arithmetic/reference legality, head sigmoid and CSV/group/CMC audit. Raw neural distances/scores are inputs, not independently regenerated NN outputs. Repeated conditions are not independent queries; empirical margins are not causal or information-theoretic synergy.')
    (diagnosis / 'independent_cpu_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print('FULL_OFFICIAL_CONTROL_STATES_ALL294_GT_CPU_PASS', flush=True)


if __name__ == '__main__': main()
