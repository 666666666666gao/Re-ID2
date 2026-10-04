"""Independent CPU installed-GT recount of all588 frozen readout cases."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_full_official49 import installed_rows, recount, verify
from audit_shared_identity_states import METRICS


STAGES = ('deployed', 'base_common', 'M_pre', 'M_post', 'F_pre', 'F_post', 'M_aux', 'F_aux')
READOUTS = {stage: (stage, stage) for stage in STAGES}
READOUTS.update(F_post_to_common=('F_post', 'base_common'), common_to_F_post=('base_common', 'F_post'),
    M_post_to_common=('M_post', 'base_common'), common_to_M_post=('base_common', 'M_post'))
SETS = ('RNT', 'R', 'N', 'T', 'RN', 'RT', 'NT')


def main():
    parser = argparse.ArgumentParser()
    for name in ('run-dir', 'previous-frozen', 'diagnosis'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run, previous, diagnosis = map(Path, (args.run_dir, args.previous_frozen, args.diagnosis))
    trained = json.loads((run / 'result.json').read_text())
    reported = json.loads((diagnosis / 'result.json').read_text())
    prior = json.loads((previous / 'independent_cpu_audit.json').read_text())
    arguments = trained['arguments']
    assert trained['status'] == 'COMPLETE' and trained['epochs'] == 50 and trained['training_heldout_identities'] == 0
    assert reported['status'] == 'COMPLETE' and reported['metric_cases'] == 588
    assert reported['readouts'] == {key: list(value) for key, value in READOUTS.items()}
    assert reported['optimizer_updates'] == reported['new_weights'] == reported['training_heldout_identities'] == 0
    assert reported['model_arguments'] == prior['model_arguments'] == arguments
    assert reported['selected_epoch'] == prior['selected_epoch'] == trained['best']['epoch']
    assert prior['status'] == 'PASS' and prior['cases'] == 49
    splits = {s: installed_rows(Path(arguments['data_root']), arguments['dataset'], s) for s in ('train', 'query', 'gallery')}
    manifest = json.loads((run / 'official_split_manifest.json').read_text())
    assert all(splits[s] == manifest[s] for s in splits)
    assert (len(splits['train']), len(splits['query']), len(splits['gallery'])) == (1032, 591, 1055)
    assert reported['repeated_condition_query_rows'] == 588 * len(splits['query'])
    expected = {'q_' + q + '_g_' + g for q in SETS for g in SETS}
    assert set(reported['measurements']) == set(prior['conditions']) == expected
    conditions, errors = {}, []
    for condition in sorted(expected):
        folder = diagnosis / condition
        values = {}
        assert set(reported['measurements'][condition]) == set(READOUTS)
        with np.load(folder / 'raw.npz') as raw:
            for role in ('query', 'gallery'):
                for field, key in (('ids', 'identity'), ('cameras', 'camera'), ('scenes', 'scene'), ('names', 'name')):
                    assert np.array_equal(raw[role + '_' + field], np.asarray([r[key] for r in splits[role]]))
            for readout in READOUTS:
                rows = recount(raw[readout], splits['query'], splits['gallery'], arguments['dataset'])
                values[readout], error = verify(rows, reported['measurements'][condition][readout],
                    folder / (readout + '.csv'), len(splits['gallery']))
                errors.append(error)
                if readout == 'deployed':
                    assert all(abs(values[readout][m] - prior['conditions'][condition][m]) < 1e-8 for m in METRICS)
        conditions[condition] = values
    result = dict(status='PASS_FULL_OFFICIAL_READOUT_INSTALLED_GT', cases=588,
        repeated_condition_query_rows=588 * len(splits['query']), model_arguments=arguments, readouts=READOUTS,
        selected_epoch=trained['best']['epoch'], normal=conditions['q_RNT_g_RNT'], conditions=conditions,
        equal_condition_mean={stage: {m: float(np.mean([r[stage][m] for r in conditions.values()])) for m in METRICS}
            for stage in READOUTS}, max_sixmetric_error_pp=max(errors), optimizer_updates=0,
        limits='Installed official filename GT independently recounts every stored stage distance, CSV, CMC50 and group. '
            'Stored neural distances are not independently regenerated NN outputs; repeated conditions are not independent queries.')
    output = diagnosis / 'independent_cpu_audit.json'
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('FULL_OFFICIAL_READOUT_ALL588_CPU_PASS', arguments['gate_gradient_mode'], flush=True)


if __name__ == '__main__':
    main()
