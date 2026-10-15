"""Inspect retained MSVR310 nearest matches; no new inference or fitting."""
import array
import ast
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile

from analyze_identity_coordinate_three_normal import load, table
from analyze_rgbnt100_saved_error_margins import distribution, margins

PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / 'results/msvr_saved_error_cases_CPU_20261015'


def read_array(archive, name):
    with archive.open(name + '.npy') as stream:
        assert stream.read(8) == b'\x93NUMPY\x01\x00'
        size = struct.unpack('<H', stream.read(2))[0]
        header = ast.literal_eval(stream.read(size).decode('latin1'))
        assert header['fortran_order'] is False and sys.byteorder == 'little'
        data = stream.read()
    kind = header['descr']
    if kind == '<f4':
        values = array.array('f')
        values.frombytes(data)
    elif kind == '<i8':
        values = array.array('q')
        values.frombytes(data)
    else:
        assert kind == '<U20'
        values = [data[offset:offset + 80].decode('utf-32-le').rstrip('\0')
                  for offset in range(0, len(data), 80)]
    count = 1
    for dimension in header['shape']:
        count *= dimension
    assert len(values) == count
    return values, header['shape']


def main():
    assert not OUTPUT.exists()
    inputs = []
    manifest_path = PROJECT / 'results/full_official_baselines_20261004/training/MSVR310_demo_s42/official_split_manifest.json'
    manifest = load(manifest_path)
    inputs.append(manifest_path)
    assert len(manifest['train']) == 1032 and manifest['training_heldout_identities'] == 0
    query, gallery = manifest['query'], manifest['gallery']
    assert len(query) == 591 and len(gallery) == 1055
    for record in query + gallery:
        name = record['name']
        assert Path(name).name == name
        assert (int(name[:4]), int(name[11]), int(name[6:9])) == (
            record['identity'], record['camera'], record['scene'])
    ah_dir = PROJECT / 'results/r201ah_MS_full_identity_20261012/MSVR310/states/MSVR310_r201ah_axis_shared_s42'
    r_dir = PROJECT / 'results/selected_expert_outlets_20261011/MSVR310_r201r_axis_shared_s42'
    marker_path = ah_dir / 'normal_local_archive.json'
    result_path = r_dir / 'result.json'
    marker, r_result = load(marker_path), load(result_path)
    inputs.extend((marker_path, result_path))
    assert marker['status'] == 'NORMAL_LOCAL_SIZE_SHA_VERIFIED_REMOTE_CLEARED'
    archives = {
        'AH': dict(path=marker['local'], bytes=marker['bytes'], sha256=marker['sha256']),
        'R': dict(path='D:/Program Files/UserCache/gb/ReID2-experiment-artifacts/selected_expert_outlets_20261011/MSVR310_r201r_axis_shared_s42/raw.npz',
                  **r_result['raw']['file']),
    }
    arrays = {}
    for label, metadata in archives.items():
        path = Path(metadata['path'])
        assert path.stat().st_size == metadata['bytes']
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == metadata['sha256']
        with zipfile.ZipFile(path) as stored:
            for split, records in (('query', query), ('gallery', gallery)):
                for suffix, key in (('ids', 'identity'), ('names', 'name'),
                                    ('cameras', 'camera'), ('scenes', 'scene')):
                    values, shape = read_array(stored, split + '_' + suffix)
                    assert shape == (len(records),) and list(values) == [row[key] for row in records]
            names = [('strong', 'distances_00'), ('AH11', 'distances')] if label == 'AH' else [
                ('original', 'distance_base00'), ('R_common', 'distance_common_context'),
                ('R_final', 'distance_normal11')]
            for name, member in names:
                values, shape = read_array(stored, member)
                assert shape == (591, 1055)
                arrays[name] = values
    csv_paths = {
        'original': PROJECT / 'results/msvr_reference_blocks_p0_20261009/original_DeMo.csv',
        'strong': PROJECT / 'results/msvr_reference_blocks_p0_20261009/shared_fused.csv',
        'AH11': ah_dir / 'state_11.csv',
        'R_common': r_dir / 'common_context.csv',
        'R_final': r_dir / 'normal11.csv',
    }
    saved = {}
    for name, path in csv_paths.items():
        with path.open(encoding='utf-8-sig', newline='') as stream:
            saved[name] = list(csv.DictReader(stream))
        assert len(saved[name]) == 591
        inputs.append(path)
    details, rows = {}, []
    for i, record in enumerate(query):
        identity, scene = record['identity'], record['scene']
        positives = [j for j, g in enumerate(gallery) if g['identity'] == identity and g['scene'] != scene]
        negatives = [j for j, g in enumerate(gallery) if g['identity'] != identity]
        assert positives and negatives
        for name, distances in arrays.items():
            measured = margins(distances, i, positives, negatives, 1055)
            reference = saved[name][i]
            assert all(str(record[key]) == reference[key] for key in ('name', 'identity', 'camera', 'scene'))
            assert reference['valid'] == 'True'
            assert len(positives) == int(reference['relevant_gallery'])
            assert len(positives) + len(negatives) == int(reference['kept_gallery'])
            assert measured['first_positive_rank'] == int(reference['first_match'])
            correct = int(gallery[measured['top1_index']]['identity'] == identity)
            assert correct == int(reference['Rank-1'])
            details[name, i] = measured
            row = dict(model=name, query_index=i, **record, Rank1=correct,
                AP=float(reference['AP']), INP=float(reference['INP']), **measured)
            for role in ('top1', 'positive', 'negative'):
                g = gallery[measured[role + '_index']]
                row.update({role + '_' + key: g[key] for key in ('name', 'identity', 'camera', 'scene')})
            rows.append(row)
    comparisons, selections = [], []
    for before_name, after_name in (('original', 'strong'), ('strong', 'AH11'),
                                    ('strong', 'R_common'), ('R_final', 'R_common')):
        pair = before_name + '_to_' + after_name
        groups = {key: [] for key in ('both_wrong', 'rescued', 'harmed', 'both_correct')}
        for i in range(591):
            b, a = (int(saved[name][i]['Rank-1']) for name in (before_name, after_name))
            key = ('both_wrong', 'rescued', 'harmed', 'both_correct')[2 * b + a]
            groups[key].append(i)
        for cohort, indices in groups.items():
            before = [details[before_name, i]['margin'] for i in indices]
            after = [details[after_name, i]['margin'] for i in indices]
            comparisons.append(dict(comparison=pair, cohort=cohort, queries=len(indices),
                identities=len({query[i]['identity'] for i in indices}),
                before_margin=distribution(before), after_margin=distribution(after),
                margin_change=distribution([a-b for a,b in zip(after,before)])))
            if cohort == 'both_correct':
                continue
            used = set()
            for i in indices:
                if query[i]['identity'] in used:
                    continue
                used.add(query[i]['identity'])
                wrong_state = before_name if cohort == 'rescued' else after_name
                positive_state = before_name if cohort == 'harmed' else after_name
                wrong = gallery[details[wrong_state, i]['negative_index']]
                positive = gallery[details[positive_state, i]['positive_index']]
                selected = dict(case=len(selections) + 1, comparison=pair, cohort=cohort,
                    query_index=i, before_model=before_name, after_model=after_name,
                    before_margin=details[before_name, i]['margin'],
                    after_margin=details[after_name, i]['margin'])
                for role, record in (('query', query[i]), ('wrong', wrong), ('correct', positive)):
                    selected.update({role + '_' + key: record[key] for key in ('name', 'identity', 'camera', 'scene')})
                selections.append(selected)
                if len(used) == 3:
                    break
    OUTPUT.mkdir()
    table(OUTPUT / 'all_outlet_nearest_margins.csv', rows)
    (OUTPUT / 'selected_visual_cases.json').write_text(json.dumps(dict(
        selection='At most three lowest-query-index distinct identities per non-correct cohort of each specified comparison. Label-based offline diagnosis; no visual selection or model changes.',
        cases=selections), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report = dict(status='ACTUAL_SAVED_MSVR310_FIVE_OUTLET_NEAREST_MATCH_CPU_ANALYSIS_NO_NN',
        at=datetime.now().astimezone().isoformat(timespec='seconds'), dataset='MSVR310',
        query_count=591, gallery_count=1055, outlet_count=5,
        all_nearest_positive_ranks_and_Rank1_match_saved_CSV=True,
        installed_metadata_manifest_and_filename_exact=True,
        whole_cache_size_SHA_verified=archives, comparisons=comparisons,
        selected_visual_cases=len(selections),
        input_sha256={str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_neural_calls=0, new_optimizer_updates=0, new_fitting=0, training_progress_queries=0,
        margin_definition='nearest negative squared distance minus nearest positive squared distance; same-ID same-SCENE excluded as official MSVR310 scorer. Ties use gallery index; first match and Rank1 checked against every saved CSV row.',
        limits='Five retained outputs from three different benchmark-selected models. R common_context is a public identity block, not a frequency-expert output. R and AH differ in teacher/training/interface. Cross-model rescues, harms and margin changes are descriptive offline comparisons, not causal, deployable selection or new method scores. Distances have representation-dependent geometry; compare margin signs/changes without interpreting cross-model magnitude as calibrated confidence. No new AP/INP recomputation, image inspection, neural inference or progress polling.')
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], outlet_rows=len(rows),
        cohorts=[{k:c[k] for k in ('comparison','cohort','queries','identities')} for c in comparisons],
        selected_visual_cases=len(selections), new_neural_calls=0)), flush=True)


if __name__ == '__main__':
    main()
