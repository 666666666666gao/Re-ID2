"""Audit one complete official condition before its raw archive is transferred."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_full_official49 import installed_rows, recount, verify
from audit_full_official_control_states import STATES, verify_calibration, verify_heads
from audit_shared_identity_states import changes, METRICS


def installed_context(run):
    trained = json.loads((run / 'result.json').read_text())
    assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50
    assert trained['training_heldout_identities'] == 0
    arguments = trained['arguments']
    dataset = arguments['dataset']
    installed = {s: installed_rows(Path(arguments['data_root']), dataset, s)
                 for s in ('train', 'query', 'gallery')}
    manifest = json.loads((run / 'official_split_manifest.json').read_text())
    assert all(installed[s] == manifest[s] for s in installed)
    assert all(len(installed[s]) == trained[s + '_records'] for s in installed)
    train_ids = {r['identity'] for r in installed['train']}
    assert train_ids.isdisjoint({r['identity'] for r in installed['query'] + installed['gallery']})
    return trained, dataset, installed


def audit_condition(folder, previous, measured, calibration, dataset, installed, prediction):
    values, rows, errors = {}, {}, []
    with np.load(folder / 'raw.npz') as raw:
        for role in ('query', 'gallery'):
            for field, key in (('ids', 'identity'), ('cameras', 'camera'), ('scenes', 'scene'), ('names', 'name')):
                assert np.array_equal(raw[role + '_' + field], np.asarray([r[key] for r in installed[role]]))
        for state in STATES:
            actual = recount(raw['distance_' + state], installed['query'], installed['gallery'], dataset)
            values[state], error = verify(actual, measured[state], folder / ('state_' + state + '.csv'), len(installed['gallery']))
            rows[state] = actual
            errors.append(error)
        with np.load(previous / (folder.name + '.npz')) as frozen:
            assert np.array_equal(raw['distance_11'], frozen['distances'])
        prior = json.loads((previous / 'result.json').read_text())
        assert all(values['11'][m] == prior['measurements'][folder.name][m] for m in METRICS)
        calibrated = verify_calibration(raw, calibration, folder / 'contributions.csv',
                                       installed['query'], installed['gallery'], dataset, prediction)
    with (folder / 'state_11.csv').open() as after, (previous / (folder.name + '.csv')).open() as before:
        assert after.read() == before.read()
    return dict(metrics=values, full11_minus00=changes(rows['00'], rows['11']),
                full11_minus10=changes(rows['10'], rows['11']),
                full11_minus01=changes(rows['01'], rows['11']), calibration=calibrated), max(errors)


def main():
    parser = argparse.ArgumentParser()
    for key in ('run-dir', 'previous-frozen', 'diagnosis', 'condition', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    run, previous, diagnosis = map(Path, (args.run_dir, args.previous_frozen, args.diagnosis))
    output = Path(args.output)
    assert not output.exists()
    trained, dataset, installed = installed_context(run)
    reported = json.loads((diagnosis / 'result.json').read_text())
    prior = json.loads((previous / 'result.json').read_text())
    existing = json.loads((diagnosis / 'independent_cpu_audit.json').read_text())
    assert reported['status'] == prior['status'] == 'COMPLETE'
    assert existing['status'] == 'PASS_FULL_OFFICIAL_CONTROL_STATES_INSTALLED_GT'
    assert reported['model_arguments'] == prior['model_arguments'] == trained['arguments']
    assert reported['selected_epoch'] == prior['selected_epoch'] == trained['best']['epoch']
    assert args.condition in reported['measurements'] and len(reported['measurements']) == 49
    heads, predictions = verify_heads(diagnosis, installed['query'], installed['gallery'], trained['arguments']['gate_gradient_mode'])
    assert heads == existing['heads']
    values, error = audit_condition(diagnosis / args.condition, previous,
                                    reported['measurements'][args.condition], reported['contributions'][args.condition],
                                    dataset, installed, predictions[args.condition.split('_')[1]])
    assert values == existing['conditions'][args.condition]
    proof = dict(status='PASS_ONE_FULL_OFFICIAL_CONDITION_EQUALS_EXISTING_ALL49_GT_AUDIT',
                 condition=args.condition, cases=6, dataset=dataset,
                 full_split_counts={s: len(r) for s, r in installed.items()},
                 training_heldout_identities=0, optimizer_updates=0, neural_execution=0,
                 exact_existing_condition_audit=True, exact_full11_frozen_distances=True,
                 max_sixmetric_error_pp=error, measured=values,
                 limits='Pure CPU equivalence for this condition only; no future streaming scheduler or neural inference is validated.')
    output.write_text(json.dumps(proof, indent=2) + '\n')
    print('FULL_OFFICIAL_ONE_CONDITION_EQUIVALENCE_PASS', args.condition, flush=True)


if __name__ == '__main__':
    main()
