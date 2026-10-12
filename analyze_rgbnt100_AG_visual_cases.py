"""Select fixed cached AG error examples for manual inspection; no new NN."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile

from analyze_r201o_saved_normal_geometry import values
from analyze_rgbnt100_common_distance_probe import names

PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / 'results/rgbnt100_AG_visual_cases_CPU_20261013'
MODEL = 'RGBNT100_r201ag_axis_shared_s42'


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    assert not OUTPUT.exists()
    folder = PROJECT / 'results/r201ai_AG_selected_outlets_20261012/RGBNT100/full' / MODEL
    base, full, frequency = [read_rows(folder / (stage + '.csv'))
        for stage in ('base00', 'normal11', 'F_unit_direction')]
    assert len(base) == len(full) == len(frequency) == 1715
    manifest = json.loads((PROJECT / 'results/full_official_baselines_20261004/training/RGBNT100_demo_s42/official_split_manifest.json').read_text(encoding='utf-8-sig'))
    query, gallery = manifest['query'], manifest['gallery']
    metadata = ('name', 'identity', 'camera', 'scene')
    for rows in (base, full, frequency):
        assert all(all(str(row[k]) == str(item[k]) for k in metadata) for row, item in zip(rows, query))
    cohorts = {
        'frequency_correct_fusion_wrong': [i for i in range(1715)
            if base[i]['Rank-1'] == '0' and full[i]['Rank-1'] == '0' and frequency[i]['Rank-1'] == '1'],
        'newly_harmed': [i for i in range(1715) if base[i]['Rank-1'] == '1' and full[i]['Rank-1'] == '0'],
        'rescued': [i for i in range(1715) if base[i]['Rank-1'] == '0' and full[i]['Rank-1'] == '1'],
        'remaining_original_error': [i for i in range(1715) if base[i]['Rank-1'] == '0' and full[i]['Rank-1'] == '0'],
    }
    # One example per role, distinct identities, deterministic score/index order.
    scoring = {
        'frequency_correct_fusion_wrong': lambda i: (float(base[i]['AP']), i),
        'newly_harmed': lambda i: (float(full[i]['AP']) - float(base[i]['AP']), i),
        'rescued': lambda i: (-(float(full[i]['AP']) - float(base[i]['AP'])), i),
        'remaining_original_error': lambda i: (-int(base[i]['first_match']), i),
    }
    selected, identities = [], set()
    for role, candidates in cohorts.items():
        i = next(i for i in sorted(candidates, key=scoring[role]) if query[i]['identity'] not in identities)
        selected.append((role, i))
        identities.add(query[i]['identity'])
    receipt = json.loads((PROJECT / 'results/preflight/r201ai_AG_selected_outlets_20261012_complete_intake.json').read_text(encoding='utf-8-sig'))['archives'][MODEL]
    raw = Path(receipt['local'])
    assert raw.stat().st_size == receipt['bytes']
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == receipt['sha256']
    with zipfile.ZipFile(raw) as archive:
        for side, records in (('query', query), ('gallery', gallery)):
            assert names(archive, side + '_names', len(records)) == [r['name'] for r in records]
            for suffix, key in (('ids', 'identity'), ('cameras', 'camera'), ('scenes', 'scene')):
                assert list(values(archive, side + '_' + suffix, (len(records),), '<i8', 'q')) == [r[key] for r in records]
        distances = {stage: values(archive, 'distance_' + stage, (1715, 8575), '<f4', 'f')
            for stage in ('base00', 'normal11', 'F_unit_direction')}
    cases = []
    for role, i in selected:
        stages = {}
        q = query[i]
        for stage, rows in (('base00', base), ('normal11', full), ('F_unit_direction', frequency)):
            offset = i * 8575
            order = sorted(range(8575), key=lambda j: distances[stage][offset + j])
            kept = [j for j in order if not (gallery[j]['identity'] == q['identity'] and gallery[j]['camera'] == q['camera'])]
            positive = next(j for j in kept if gallery[j]['identity'] == q['identity'])
            first = kept[0]
            assert kept.index(positive) + 1 == int(rows[i]['first_match'])
            stages[stage] = dict(AP=float(rows[i]['AP']), first_match=int(rows[i]['first_match']),
                first_gallery_index=first, first_gallery=gallery[first],
                nearest_positive_index=positive, nearest_positive=gallery[positive],
                first_distance=distances[stage][offset + first],
                positive_distance=distances[stage][offset + positive])
        cases.append(dict(role=role, query_index=i, query=q, stages=stages))
    OUTPUT.mkdir()
    record = dict(status='ACTUAL_CACHED_AG_FOUR_ERROR_CASES_SELECTED',
        at=datetime.now().astimezone().isoformat(timespec='seconds'), model=MODEL,
        cases=cases, cohort_sizes={k: len(v) for k, v in cohorts.items()}, raw=receipt,
        selection='One case per role, distinct identities; predeclared sorting by saved AP/first-match, then index.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_neural_calls=0, optimizer_updates=0,
        limitations='Four diagnostic examples from already benchmark-selected AG, not a representative error estimate, training input selection, deployment branch oracle, or evidence about running AK. Images remain private; only names/metadata and cached ranks are recorded.')
    (OUTPUT / 'result.json').write_bytes((json.dumps(record, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(dict(status=record['status'], cohort_sizes=record['cohort_sizes'],
        selected=[dict(role=c['role'], query=c['query']['name'], query_index=c['query_index']) for c in cases])), flush=True)


if __name__ == '__main__':
    main()
